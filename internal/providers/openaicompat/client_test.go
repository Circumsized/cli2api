package openaicompat

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/caigee-cmd/cli2api/internal/accounts"
	"github.com/caigee-cmd/cli2api/internal/providers"
	"github.com/caigee-cmd/cli2api/internal/translate"
)

type memStore struct {
	format  string
	payload []byte
}

func (s *memStore) LoadCredentialPayload(context.Context, string) (string, []byte, error) {
	if s.payload == nil {
		return "", nil, errors.New("no credential")
	}
	return s.format, s.payload, nil
}

func (s *memStore) SaveCredentialPayload(_ context.Context, _, format string, payload []byte) error {
	s.format, s.payload = format, payload
	return nil
}

func newTestClient(t *testing.T, handler http.HandlerFunc) (*Client, *memStore) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	store := &memStore{}
	cred, _ := json.Marshal(Credential{BaseURL: server.URL, APIKey: "k"})
	store.format, store.payload = CredentialFormat, cred
	client := NewClient(store)
	client.http = server.Client()
	return client, store
}

// newTestClientWithURL also returns the server URL, for tests that need to
// re-encode a credential pointing at it.
func newTestClientWithURL(t *testing.T, handler http.HandlerFunc) (*Client, *memStore, string) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	store := &memStore{}
	cred, _ := json.Marshal(Credential{BaseURL: server.URL, APIKey: "k"})
	store.format, store.payload = CredentialFormat, cred
	client := NewClient(store)
	client.http = server.Client()
	return client, store, server.URL
}

func TestChatNonStreamMapsOutcome(t *testing.T) {
	var gotAuth, gotPath string
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":    "x",
			"model": "qwen3-8b",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "hello"},
			}},
			"usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 3},
		})
	})
	out, err := client.ChatNonStream(context.Background(), "a1", translate.ChatRequest{
		Model: "qwen3-8b", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Content != "hello" || out.Model != "qwen3-8b" || out.PromptTokens != 7 || out.CompletionTokens != 3 {
		t.Fatalf("outcome=%+v", out)
	}
	if out.UsageSource != "upstream" {
		t.Fatalf("usage source=%q", out.UsageSource)
	}
	// The base URL must gain /v1 and the key must be sent as a bearer token.
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path=%q", gotPath)
	}
	if gotAuth != "Bearer k" {
		t.Fatalf("auth=%q", gotAuth)
	}
}

// TestChatNonStreamEmptyChoicesIsAnError pins that a 200 with no choices is a
// failure the caller can fail over on, not a silently empty answer.
func TestChatNonStreamEmptyChoicesIsAnError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "x", "choices": []any{}})
	})
	_, err := client.ChatNonStream(context.Background(), "a1", translate.ChatRequest{
		Model: "m", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("empty choices must be an error")
	}
}

func TestModelsUsesPinnedListWithoutCallingUpstream(t *testing.T) {
	var hits int
	client, store := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
	})
	// Point at an unreachable URL so any upstream call would fail the test.
	cred, _ := json.Marshal(Credential{BaseURL: "http://127.0.0.1:1", Models: []string{"m1", "m2"}})
	store.payload = cred
	models, err := client.Models(context.Background(), "a1")
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].PublicModel != "m1" {
		t.Fatalf("models=%+v", models)
	}
	if hits != 0 {
		t.Fatalf("a pinned catalog must not call upstream, hits=%d", hits)
	}
}

func TestClassifyTaxonomy(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		body       string
		wantKind   string
		wantFailov bool
	}{
		{"quota402", 402, `{"error":{"message":"insufficient balance"}}`, accounts.KindQuota, false},
		{"quota429", 429, `{"error":{"message":"You exceeded your current quota, please check your plan"}}`, accounts.KindQuota, false},
		{"rate429", 429, `{"error":{"message":"Rate limit reached for requests"}}`, accounts.KindRateLimit, true},
		{"auth401", 401, `{"error":{"message":"Invalid API key"}}`, accounts.KindAuth, true},
		{"model404", 404, `{"error":{"message":"The model does not exist"}}`, accounts.KindModelNotAvailable, false},
		{"bad400", 400, `{"error":{"message":"missing field messages"}}`, accounts.KindInvalidRequest, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := classifier{}.Classify(tc.status, tc.body)
			if got.Kind != tc.wantKind {
				t.Fatalf("kind=%q want %q", got.Kind, tc.wantKind)
			}
			if got.Failover != tc.wantFailov {
				t.Fatalf("failover=%v want %v", got.Failover, tc.wantFailov)
			}
		})
	}
}

// TestCredentialRequiresBaseURL pins the validation that keeps an unusable
// account out of the pool.
func TestCredentialRequiresBaseURL(t *testing.T) {
	if err := (credentialCodec{}).Validate([]byte(`{"api_key":"k"}`)); err == nil {
		t.Fatal("a credential without base_url must be rejected")
	}
	if err := (credentialCodec{}).Validate([]byte(`{"base_url":"https://example.com/v1"}`)); err != nil {
		t.Fatalf("valid credential rejected: %v", err)
	}
}

// TestAdapterExposesNoLogin pins the capability set: this provider authenticates
// with a key, so it must not advertise browser login.
func TestAdapterExposesNoLogin(t *testing.T) {
	adapter := NewClient(&memStore{}).Adapter()
	if adapter.ID != ProviderID {
		t.Fatalf("id=%q", adapter.ID)
	}
	for _, capability := range []string{"chat", "models", "credential", "classifier"} {
		if !adapter.Supports(capability) {
			t.Fatalf("missing capability %q", capability)
		}
	}
	if adapter.Supports("login") {
		t.Fatal("openaicompat must not advertise login")
	}
}

// TestAliasAdvertisesForeignPublicModel pins the mapping that makes
// cross-provider fallback possible: the endpoint must advertise the caller's
// public model ID while sending its own native ID upstream.
func TestAliasAdvertisesForeignPublicModel(t *testing.T) {
	var sentModel string
	client, store, baseURL := newTestClientWithURL(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"id": "native-x"}}})
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		sentModel, _ = body["model"].(string)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   sentModel,
			"choices": []map[string]any{{"finish_reason": "stop", "message": map[string]any{"content": "ok"}}},
		})
	})
	cred, _ := json.Marshal(Credential{
		BaseURL: baseURL,
		Aliases: map[string]string{"hy3": "THUDM/GLM-4-9B-0414"},
	})
	store.payload = cred

	models, err := client.Models(context.Background(), "a1")
	if err != nil {
		t.Fatal(err)
	}
	// The alias must be present alongside the endpoint's own catalog entry.
	var alias *providers.ModelInfo
	for i := range models {
		if models[i].PublicModel == "hy3" {
			alias = &models[i]
		}
	}
	if alias == nil {
		t.Fatalf("alias missing from catalog: %+v", models)
	}
	if alias.NativeModel != "THUDM/GLM-4-9B-0414" {
		t.Fatalf("alias must keep public and native distinct: %+v", *alias)
	}

	// The executor rewrites req.Model to the native ID before calling the
	// provider, so the provider sends whatever it was handed.
	if _, err := client.ChatNonStream(context.Background(), "a1", translate.ChatRequest{
		Model: "THUDM/GLM-4-9B-0414", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}); err != nil {
		t.Fatal(err)
	}
	if sentModel != "THUDM/GLM-4-9B-0414" {
		t.Fatalf("native id must reach upstream, got %q", sentModel)
	}
}
