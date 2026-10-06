// Package openaicompat implements a generic OpenAI-compatible provider.
//
// It exists so an account pool is never a single point of failure: when every
// WorkBuddy account is out of allowance, requests can fall back to any endpoint
// that speaks the OpenAI wire protocol (SiliconFlow, OpenRouter, a self-hosted
// vLLM, another proxy on the same host). Those endpoints need no reverse
// engineering, which is what makes this provider cheap compared to the
// provider-native ones.
//
// Only three capabilities are implemented — Chat, Models and Credential —
// because the protocol is standard: there is no browser login and no private
// quota API to probe.
package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/caigee-cmd/cli2api/internal/providers"
	"github.com/caigee-cmd/cli2api/internal/translate"
)

const (
	// ProviderID is the family name accounts are registered under.
	ProviderID = "openaicompat"
	// CredentialFormat is the stored credential shape.
	CredentialFormat = "openai-compat-v1"
	// DefaultRegion is the single logical region for this provider.
	DefaultRegion = "default"
)

// Credential is the stored payload: where to call and how to authenticate.
type Credential struct {
	// BaseURL is the endpoint root, e.g. https://api.siliconflow.cn/v1.
	BaseURL string `json:"base_url"`
	// APIKey is sent as a bearer token. Empty means the endpoint needs no auth.
	APIKey string `json:"api_key,omitempty"`
	// Models optionally pins the catalog when the endpoint cannot list models.
	// When set, the catalog probe is skipped entirely.
	Models []string `json:"models,omitempty"`
	// Aliases maps a public model ID onto this endpoint's native ID, letting a
	// request written for one provider be served by another. Cross-provider
	// fallback depends on this: the fallback endpoint almost never uses the
	// same model names, so without a mapping the chain has nothing to match.
	// Example: {"hy3": "THUDM/GLM-4-9B-0414"}.
	Aliases map[string]string `json:"aliases,omitempty"`
	// Label is a display hint for the console.
	Label string `json:"label,omitempty"`
}

// Ready reports whether the credential can address an endpoint.
func (c Credential) Ready() bool {
	return strings.TrimSpace(c.BaseURL) != ""
}

// DecodeCredential accepts both this package's canonical shape and a plain
// {base_url, api_key} object, so an operator can paste either.
func DecodeCredential(payload []byte) (Credential, error) {
	var cred Credential
	if err := json.Unmarshal(payload, &cred); err != nil {
		return Credential{}, fmt.Errorf("openai-compat credential parse: %w", err)
	}
	cred.BaseURL = strings.TrimRight(strings.TrimSpace(cred.BaseURL), "/")
	cred.APIKey = strings.TrimSpace(cred.APIKey)
	return cred, nil
}

// Store is the slice of the account store this provider needs.
type Store interface {
	LoadCredentialPayload(ctx context.Context, accountID string) (string, []byte, error)
	SaveCredentialPayload(ctx context.Context, accountID, format string, payload []byte) error
}

const requestTimeout = 120 * time.Second

// Client implements the provider capability interfaces.
type Client struct {
	store Store
	http  *http.Client
}

func NewClient(store Store) *Client {
	return &Client{
		store: store,
		http: &http.Client{
			Timeout: requestTimeout,
			// A 302 to a login document would hide the real upstream status.
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
	}
}

// Adapter returns the capability bundle for registry registration.
func (c *Client) Adapter() providers.Adapter {
	return providers.Adapter{
		ID:         ProviderID,
		Credential: credentialCodec{},
		Chat:       c,
		Models:     c,
		Classifier: classifier{},
	}
}

func (c *Client) credential(ctx context.Context, accountID string) (Credential, error) {
	_, payload, err := c.store.LoadCredentialPayload(ctx, accountID)
	if err != nil {
		return Credential{}, err
	}
	cred, err := DecodeCredential(payload)
	if err != nil {
		return Credential{}, err
	}
	if !cred.Ready() {
		return Credential{}, fmt.Errorf("openai-compat credential incomplete: base_url required")
	}
	return cred, nil
}

type credentialCodec struct{}

func (credentialCodec) Validate(payload []byte) error {
	cred, err := DecodeCredential(payload)
	if err != nil {
		return err
	}
	if !cred.Ready() {
		return fmt.Errorf("base_url is required")
	}
	return nil
}

// endpoint joins the credential base with a path, tolerating a base that
// already carries a /v1 suffix and one that does not.
func (c Credential) endpoint(path string) string {
	base := strings.TrimRight(strings.TrimSpace(c.BaseURL), "/")
	if !strings.HasSuffix(base, "/v1") && !strings.Contains(base, "/v1/") {
		base += "/v1"
	}
	return base + path
}

func (c Credential) applyAuth(req *http.Request) {
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
}

// chatBody converts the neutral request into the OpenAI wire shape. Fields are
// passed through as the caller sent them (they arrive as raw JSON), so a
// parameter this provider does not model is forwarded rather than dropped.
func chatBody(req translate.ChatRequest, stream bool) ([]byte, error) {
	body := map[string]any{
		"model":    req.Model,
		"messages": req.Messages,
		"stream":   stream,
	}
	for key, raw := range map[string]json.RawMessage{
		"temperature":       req.Temperature,
		"top_p":             req.TopP,
		"max_tokens":        req.MaxTokens,
		"stop":              req.Stop,
		"tools":             req.Tools,
		"tool_choice":       req.ToolChoice,
		"response_format":   req.ResponseFormat,
		"reasoning_effort":  req.ReasoningEffort,
	} {
		if len(raw) == 0 || string(raw) == "null" {
			continue
		}
		body[key] = raw
	}
	if req.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *req.ParallelToolCalls
	}
	if stream {
		// Ask for usage in the final chunk so token accounting survives the
		// relay instead of being estimated.
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	return json.Marshal(body)
}

func (c *Client) newChatRequest(ctx context.Context, cred Credential, req translate.ChatRequest, stream bool) (*http.Request, error) {
	payload, err := chatBody(req, stream)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, cred.endpoint("/chat/completions"), strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}
	cred.applyAuth(httpReq)
	return httpReq, nil
}

// ChatNonStream executes one non-streaming completion and maps the response
// back onto the neutral outcome.
func (c *Client) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	cred, err := c.credential(ctx, accountID)
	if err != nil {
		return providers.ChatOutcome{}, err
	}
	httpReq, err := c.newChatRequest(ctx, cred, req, false)
	if err != nil {
		return providers.ChatOutcome{}, err
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return providers.ChatOutcome{}, err
	}
	defer resp.Body.Close()
	body, err := readLimited(resp.Body)
	if err != nil {
		return providers.ChatOutcome{}, err
	}
	if resp.StatusCode >= 300 {
		return providers.ChatOutcome{}, classifiedError(resp.StatusCode, body)
	}
	return parseCompletion(body, req.Model)
}

// ChatStream opens the upstream SSE response for the API layer to relay. The
// caller owns the body.
func (c *Client) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	cred, err := c.credential(ctx, accountID)
	if err != nil {
		return nil, err
	}
	httpReq, err := c.newChatRequest(ctx, cred, req, true)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := readLimited(resp.Body)
		return nil, classifiedError(resp.StatusCode, body)
	}
	return resp, nil
}

// Models lists the endpoint catalog, or the pinned list when the credential
// carries one. Aliases are always added: they are how this endpoint advertises
// itself as able to serve another provider's model ID.
func (c *Client) Models(ctx context.Context, accountID string) ([]providers.ModelInfo, error) {
	cred, err := c.credential(ctx, accountID)
	if err != nil {
		return nil, err
	}
	var out []providers.ModelInfo
	seen := map[string]struct{}{}
	add := func(native, public string) {
		native = strings.TrimSpace(native)
		public = strings.TrimSpace(public)
		if native == "" || public == "" {
			return
		}
		if _, dup := seen[public]; dup {
			return
		}
		seen[public] = struct{}{}
		out = append(out, providers.ModelInfo{NativeModel: native, PublicModel: public})
	}
	// Aliases first: they are the entries that make cross-provider fallback
	// possible, so they must survive a catalog that also lists the same model.
	for public, native := range cred.Aliases {
		add(native, public)
	}
	if len(cred.Models) > 0 {
		for _, id := range cred.Models {
			add(id, id)
		}
		return out, nil
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, cred.endpoint("/models"), nil)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "application/json")
	cred.applyAuth(httpReq)
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := readLimited(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 300 {
		return nil, classifiedError(resp.StatusCode, body)
	}
	var env struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, fmt.Errorf("openai-compat models parse: %w", err)
	}
	for _, m := range env.Data {
		add(m.ID, m.ID)
	}
	return out, nil
}

// Classify maps OpenAI-compatible error bodies onto the internal taxonomy.
func (classifier) Classify(status int, body string) providers.ClassifiedError {
	err := classifiedError(status, []byte(body))
	failover := true
	if err.Failover != nil {
		failover = *err.Failover
	}
	out := providers.ClassifiedError{
		Kind:     err.Kind,
		Status:   err.Status,
		Failover: failover,
		Message:  err.Message,
	}
	if err.Cooldown > 0 {
		out.Cooldown = err.Cooldown.String()
	}
	return out
}
