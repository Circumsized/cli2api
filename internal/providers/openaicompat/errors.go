package openaicompat

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/caigee-cmd/cli2api/internal/accounts"
	"github.com/caigee-cmd/cli2api/internal/providers"
)

// maxBody caps how much of an upstream body is buffered. OpenAI-compatible
// endpoints return JSON, so anything larger is a misrouted response.
const maxBody = 16 << 20

func readLimited(r io.Reader) ([]byte, error) {
	return io.ReadAll(io.LimitReader(r, maxBody))
}

// classifier is the stateless ErrorClassifier implementation.
type classifier struct{}

// errorEnvelope is the OpenAI error shape.
type errorEnvelope struct {
	Error struct {
		Message string          `json:"message"`
		Type    string          `json:"type"`
		Code    json.RawMessage `json:"code"`
	} `json:"error"`
}

func (e errorEnvelope) code() string {
	raw := strings.TrimSpace(string(e.Error.Code))
	raw = strings.Trim(raw, `"`)
	return raw
}

// classifiedError maps an upstream failure onto the internal taxonomy so the
// pool applies the right cooldown and failover behaviour. The taxonomy matches
// the standard gateway split: a request the upstream rejected as malformed
// fails everywhere, while quota and rate limits are worth another target.
func classifiedError(status int, body []byte) *providers.Error {
	trimmed := strings.TrimSpace(string(body))
	var env errorEnvelope
	_ = json.Unmarshal(body, &env)
	message := strings.TrimSpace(env.Error.Message)
	if message == "" {
		message = trimmed
	}
	if len(message) > 400 {
		message = message[:400]
	}

	failover := true
	kind := accounts.KindUnavailable
	cooldown := 30 * time.Second

	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		kind = accounts.KindAuth
		cooldown = 30 * time.Second
	case status == http.StatusPaymentRequired:
		// 402 is the conventional "out of credit" answer on these endpoints.
		kind = accounts.KindQuota
		failover = false
		cooldown = accounts.NextLocalMidnightCooldown()
	case status == http.StatusTooManyRequests:
		kind = accounts.KindRateLimit
		cooldown = 60 * time.Second
	case status == http.StatusNotFound:
		// An unknown model is a request problem, not an account problem, so it
		// must not cool the account down or fail over.
		kind = accounts.KindModelNotAvailable
		failover = false
		cooldown = 0
	case status >= 400 && status < 500:
		kind = accounts.KindInvalidRequest
		failover = false
		cooldown = 0
	}

	// Body text overrides the status guess: several endpoints answer 429 for
	// exhausted credit and 400 for a model they do not host.
	lower := strings.ToLower(message)
	switch {
	case strings.Contains(lower, "insufficient") && strings.Contains(lower, "quota"),
		strings.Contains(lower, "quota") && strings.Contains(lower, "exceed"),
		strings.Contains(lower, "credit") && strings.Contains(lower, "exhaust"),
		strings.Contains(lower, "insufficient_quota"),
		strings.Contains(lower, "billing"):
		kind = accounts.KindQuota
		failover = false
		cooldown = accounts.NextLocalMidnightCooldown()
	case strings.Contains(lower, "rate limit"), strings.Contains(lower, "rate_limit"),
		strings.Contains(lower, "too many requests"):
		kind = accounts.KindRateLimit
		failover = true
		cooldown = 60 * time.Second
	case strings.Contains(lower, "model") && (strings.Contains(lower, "not found") ||
		strings.Contains(lower, "does not exist") || strings.Contains(lower, "unsupported") ||
		strings.Contains(lower, "no such")):
		kind = accounts.KindModelNotAvailable
		failover = false
		cooldown = 0
	}

	out := &providers.Error{
		Kind:     kind,
		Status:   status,
		Message:  message,
		Code:     env.code(),
		Type:     env.Error.Type,
		Cooldown: cooldown,
		Failover: &failover,
	}
	if out.Code == "" {
		out.Code = kind
	}
	return out
}

// completionEnvelope is the non-streaming completion shape.
type completionEnvelope struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int `json:"index"`
		Message      struct {
			Role             string          `json:"role"`
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        json.RawMessage `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// parseCompletion maps the wire response onto the neutral outcome. An empty
// choices array is treated as an error rather than a silent empty answer, so
// the caller can fail over instead of returning nothing to the client.
func parseCompletion(body []byte, requestedModel string) (providers.ChatOutcome, error) {
	var env completionEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		return providers.ChatOutcome{}, fmt.Errorf("openai-compat completion parse: %w", err)
	}
	if len(env.Choices) == 0 {
		return providers.ChatOutcome{}, &providers.Error{
			Kind:    accounts.KindUnavailable,
			Status:  502,
			Message: "upstream returned no choices",
			Code:    "empty_completion",
		}
	}
	choice := env.Choices[0]
	model := strings.TrimSpace(env.Model)
	if model == "" {
		model = requestedModel
	}
	out := providers.ChatOutcome{
		Model:            model,
		Content:          choice.Message.Content,
		ToolCalls:        choice.Message.ToolCalls,
		FinishReason:     choice.FinishReason,
		PromptTokens:     env.Usage.PromptTokens,
		CompletionTokens: env.Usage.CompletionTokens,
	}
	// Reasoning models disagree on the field name; take whichever is present.
	switch {
	case strings.TrimSpace(choice.Message.ReasoningContent) != "":
		out.Reasoning = choice.Message.ReasoningContent
	case strings.TrimSpace(choice.Message.Reasoning) != "":
		out.Reasoning = choice.Message.Reasoning
	}
	if out.PromptTokens > 0 || out.CompletionTokens > 0 {
		out.UsageSource = "upstream"
	}
	return out, nil
}
