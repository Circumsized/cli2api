package executor

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/caigee-cmd/cli2api/internal/accounts"
	"github.com/caigee-cmd/cli2api/internal/providers"
	"github.com/caigee-cmd/cli2api/internal/translate"
)

type fakeInProcessChat struct {
	calls    int
	provider string
}

func (f *fakeInProcessChat) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	f.calls++
	if req.Model != "glm-5.2" {
		return providers.ChatOutcome{}, errors.New("model unsupported")
	}
	return providers.ChatOutcome{Model: req.Model, Content: "OK", FinishReason: "stop"}, nil
}

func (f *fakeInProcessChat) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	f.calls++
	return nil, errors.New("stream unsupported in fake")
}

func TestSanitizeForItemUsesNativeCatalogSpelling(t *testing.T) {
	item := accounts.Item{ID: "t1", Provider: "trae", Models: []string{"DeepSeek-V4-Flash"}}
	got := sanitizeForItem(item, translate.ChatRequest{Model: "deepseek-v4-flash"})
	if got.Model != "DeepSeek-V4-Flash" {
		t.Fatalf("model=%q", got.Model)
	}
}

func TestInProcessProviderPinnedChatDoesNotTouchWorkers(t *testing.T) {
	pool := accounts.NewPool([]string{"http://127.0.0.1:1"}, []string{"qoder1"})
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Region: "cn", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "wb1", "")
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "OK" || result.AccountID != "wb1" || fake.calls != 1 {
		t.Fatalf("result=%+v calls=%d", result, fake.calls)
	}
}

func TestInProcessMixedCaseProviderExecutesRegisteredAdapter(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "WorkBuddy", Region: "CN", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "WorkBuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "WORKBUDDY")
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "OK" || result.AccountID != "wb1" || result.Provider != "workbuddy" || fake.calls != 1 {
		t.Fatalf("result=%+v calls=%d", result, fake.calls)
	}
}

func TestInProcessProviderFilterRoutesWithoutPin(t *testing.T) {
	pool := accounts.NewPool([]string{"http://127.0.0.1:1"}, []string{"qoder1"})
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Region: "cn", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "workbuddy")
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "OK" || result.AccountID != "wb1" || fake.calls != 1 {
		t.Fatalf("result=%+v calls=%d", result, fake.calls)
	}
}

func TestAPIKeyAllowlistBlocksOtherProviderFamily(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Region: "cn", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})
	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	ctx := WithAllowedProviders(context.Background(), []string{"qoder"})
	_, err := ex.ChatNonStream(ctx, translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "workbuddy")
	if err == nil {
		t.Fatal("expected qoder-only key to miss workbuddy")
	}
	if fake.calls != 0 {
		t.Fatalf("unexpected workbuddy calls=%d", fake.calls)
	}
}

func TestAPIKeyAllowlistKeepsBareModelInsideAllowedFamily(t *testing.T) {
	qoder := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"model":"glm-5.2","choices":[{"message":{"content":"qoder"},"finish_reason":"stop"}],"usage":{"source":"upstream"}}`)
	}))
	defer qoder.Close()
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "q1", URL: qoder.URL, Provider: "qoder", Runtime: "child_process"})
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Region: "cn", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})
	ex := NewChatExecutor(pool, "")
	ex.HTTPClient = qoder.Client()
	ex.Providers = registry
	ctx := WithAllowedProviders(context.Background(), []string{"qoder"})
	result, err := ex.ChatNonStream(ctx, translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if result.AccountID != "q1" || result.Provider != "qoder" || fake.calls != 0 {
		t.Fatalf("result=%+v workbuddy calls=%d", result, fake.calls)
	}
	if n := pool.LenRoute(accounts.RouteQuery{PublicModel: "glm-5.2", AllowedProviders: []string{"qoder"}}); n != 1 {
		t.Fatalf("qoder-only allowlist candidates = %d", n)
	}
}

func TestInProcessProviderOnlyAccountRoutesWithoutPin(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "qoder")
	if err == nil {
		t.Fatalf("expected no qoder account, got %+v", result)
	}

	result, err = ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if result.AccountID != "wb1" || fake.calls != 1 {
		t.Fatalf("result=%+v calls=%d", result, fake.calls)
	}
}

func TestInProcessProviderUnsupportedModelDoesNotFailoverToQoder(t *testing.T) {
	pool := accounts.NewPool([]string{"http://127.0.0.1:1"}, []string{"qoder1"})
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &fakeInProcessChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	_, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "unknown-model", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "wb1", "")
	if err == nil {
		t.Fatal("expected unsupported model error")
	}
}

type rateLimitedThenOKChat struct {
	calls int
}

func (f *rateLimitedThenOKChat) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	f.calls++
	if accountID == "wb1" {
		return providers.ChatOutcome{}, &providers.Error{Kind: accounts.KindRateLimit, Status: 429, Message: "soft_rate"}
	}
	return providers.ChatOutcome{Model: req.Model, Content: "OK-" + accountID, FinishReason: "stop"}, nil
}

func (f *rateLimitedThenOKChat) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	return nil, errors.New("stream unsupported in fake")
}

type systemObservingChat struct {
	sawSystem []bool
}

func (f *systemObservingChat) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	hasSystem := false
	for _, message := range req.Messages {
		if message.Role == "system" {
			hasSystem = true
		}
	}
	f.sawSystem = append(f.sawSystem, hasSystem)
	return providers.ChatOutcome{Model: req.Model, Content: "OK", FinishReason: "stop"}, nil
}

func (f *systemObservingChat) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	return nil, errors.New("stream unsupported in fake")
}

func TestInProcessDropSystemPromptStripsBeforeProvider(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process", DropSystemPrompt: true})
	registry := providers.NewRegistry()
	fake := &systemObservingChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{
			{Role: "system", Content: "third-party identity"},
			{Role: "user", Content: "hi"},
		},
	}, "", "workbuddy")
	if err != nil {
		t.Fatal(err)
	}
	if result.AccountID != "wb1" || len(fake.sawSystem) != 1 || fake.sawSystem[0] {
		t.Fatalf("result=%+v sawSystem=%v", result, fake.sawSystem)
	}
}

func TestInProcessKeepSystemPromptWhenFlagOff(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &systemObservingChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	_, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{
			{Role: "system", Content: "third-party identity"},
			{Role: "user", Content: "hi"},
		},
	}, "", "workbuddy")
	if err != nil {
		t.Fatal(err)
	}
	if len(fake.sawSystem) != 1 || !fake.sawSystem[0] {
		t.Fatalf("system prompt must reach provider when the flag is off: %v", fake.sawSystem)
	}
}

type contentRejectedChat struct {
	calls int
}

func (f *contentRejectedChat) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	f.calls++
	return providers.ChatOutcome{}, &providers.Error{Kind: accounts.KindInvalidRequest, Status: 400, Message: "sensitive content rejected"}
}

func (f *contentRejectedChat) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	return nil, errors.New("stream unsupported in fake")
}

func TestInProcessContentRejectionDoesNotFailover(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process"})
	pool.Upsert(accounts.Item{ID: "wb2", Provider: "workbuddy", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &contentRejectedChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	_, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "workbuddy")
	if err == nil {
		t.Fatal("expected the content rejection to surface to the caller")
	}
	if fake.calls != 1 {
		t.Fatalf("content rejection must not retry other accounts: calls=%d", fake.calls)
	}
	if item, ok := pool.ByID("wb1"); !ok || !item.DownUntil.IsZero() {
		t.Fatalf("rejected request must not put the account into cooldown: %+v", item)
	}
}

func TestInProcessFailoverRotatesAcrossWorkBuddyAccounts(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb1", Provider: "workbuddy", Runtime: "in_process"})
	pool.Upsert(accounts.Item{ID: "wb2", Provider: "workbuddy", Runtime: "in_process"})
	registry := providers.NewRegistry()
	fake := &rateLimitedThenOKChat{}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: fake})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	result, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "glm-5.2", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "workbuddy")
	if err != nil {
		t.Fatal(err)
	}
	if result.AccountID != "wb2" || result.Provider != "workbuddy" || result.Content != "OK-wb2" || fake.calls != 2 {
		t.Fatalf("result=%+v calls=%d", result, fake.calls)
	}
}

func TestAttemptsFollowProviderFilteredPool(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "q1", URL: "http://a", Provider: "qoder", Runtime: "child_process"})
	pool.Upsert(accounts.Item{ID: "q2", URL: "http://b", Provider: "qoder", Runtime: "child_process"})
	pool.Upsert(accounts.Item{ID: "w1", Provider: "workbuddy", Runtime: "in_process"})
	if got := pool.LenRoute(accounts.RouteQuery{ProviderFilter: "qoder"}); got != 2 {
		t.Fatalf("qoder candidates=%d", got)
	}
	if got := pool.LenRoute(accounts.RouteQuery{ProviderFilter: "workbuddy"}); got != 1 {
		t.Fatalf("workbuddy candidates=%d", got)
	}
	if got := pool.LenRoute(accounts.RouteQuery{ProviderFilter: "qoder", Excluded: map[string]struct{}{"q1": {}}}); got != 1 {
		t.Fatalf("excluded candidates=%d", got)
	}
}

func TestProviderPickFiltersByProviderFamily(t *testing.T) {
	pool := accounts.NewPool([]string{"http://a"}, []string{"q1"})
	pool.Upsert(accounts.Item{ID: "w1", Provider: "workbuddy", Runtime: "in_process"})
	item, ok := pool.PickRoute(accounts.RouteQuery{ProviderFilter: "workbuddy"})
	if !ok || item.ID != "w1" {
		t.Fatalf("workbuddy pick=%+v ok=%v", item, ok)
	}
	if _, ok := pool.PickRoute(accounts.RouteQuery{ProviderFilter: "cursor"}); ok {
		t.Fatal("unknown provider family must not pick an account")
	}
}

var _ = json.RawMessage{}

// recordingProviderChat records which provider family served each call, so a
// cross-provider chain can be asserted by the sequence of families tried.
type recordingProviderChat struct {
	seen []string
	// failWith, when set for a family, is returned for that family's accounts.
	failWith map[string]error
}

func (f *recordingProviderChat) ChatNonStream(ctx context.Context, accountID string, req translate.ChatRequest) (providers.ChatOutcome, error) {
	f.seen = append(f.seen, accountID)
	// An exact account ID wins; otherwise fall back to the family prefix so a
	// test can fail a whole family at once.
	if err, ok := f.failWith[accountID]; ok {
		return providers.ChatOutcome{}, err
	}
	family := accountID
	if idx := strings.Index(accountID, ":"); idx >= 0 {
		family = accountID[:idx]
	}
	if err, ok := f.failWith[family]; ok {
		return providers.ChatOutcome{}, err
	}
	return providers.ChatOutcome{Model: req.Model, Content: "OK-" + accountID, FinishReason: "stop"}, nil
}

func (f *recordingProviderChat) ChatStream(ctx context.Context, accountID string, req translate.ChatRequest) (*http.Response, error) {
	return nil, errors.New("stream unsupported in fake")
}

// families maps the recorded account IDs onto their provider families.
func (f *recordingProviderChat) families() []string {
	out := make([]string, 0, len(f.seen))
	for _, id := range f.seen {
		if idx := strings.Index(id, ":"); idx >= 0 {
			out = append(out, id[:idx])
		} else {
			out = append(out, id)
		}
	}
	return out
}

// TestBareModelFailsOverAcrossProviders is the cross-provider guarantee: when
// every account of the primary family is exhausted, a bare model ID must reach
// a second family instead of failing. Before the per-family budget existed the
// loop stopped inside the first family and never tried the fallback.
func TestBareModelFailsOverAcrossProviders(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb:1", Provider: "workbuddy", Runtime: "in_process",
		Models: []string{"hy3"}})
	pool.Upsert(accounts.Item{ID: "wb:2", Provider: "workbuddy", Runtime: "in_process",
		Models: []string{"hy3"}})
	pool.Upsert(accounts.Item{ID: "oc:1", Provider: "openaicompat", Runtime: "in_process",
		Models: []string{"hy3"}})

	registry := providers.NewRegistry()
	rec := &recordingProviderChat{failWith: map[string]error{
		"wb": &providers.Error{Kind: accounts.KindQuota, Status: 429, Message: "credits exhausted"},
	}}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: rec})
	registry.Register(providers.Adapter{ID: "openaicompat", Chat: rec})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	got, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "hy3", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "")
	if err != nil {
		t.Fatalf("cross-provider failover must succeed: %v (tried %v)", err, rec.seen)
	}
	if got.Provider != "openaicompat" {
		t.Fatalf("expected the fallback family to serve, got %+v (tried %v)", got, rec.seen)
	}
	// The primary family must be bounded, not scanned exhaustively.
	wbTries := 0
	for _, f := range rec.families() {
		if f == "wb" {
			wbTries++
		}
	}
	if wbTries > maxAttemptsPerProvider {
		t.Fatalf("primary family consumed %d attempts, cap is %d (tried %v)",
			wbTries, maxAttemptsPerProvider, rec.seen)
	}
}

// TestPrefixedModelStaysInsideItsProvider is the guard on the other side: an
// explicit provider prefix is the caller's stated intent, so it must never
// fail over to a different family even when the prefixed family is exhausted.
func TestPrefixedModelStaysInsideItsProvider(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	pool.Upsert(accounts.Item{ID: "wb:1", Provider: "workbuddy", Runtime: "in_process",
		Models: []string{"hy3"}})
	pool.Upsert(accounts.Item{ID: "oc:1", Provider: "openaicompat", Runtime: "in_process",
		Models: []string{"hy3"}})

	registry := providers.NewRegistry()
	rec := &recordingProviderChat{failWith: map[string]error{
		"wb": &providers.Error{Kind: accounts.KindQuota, Status: 429, Message: "credits exhausted"},
	}}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: rec})
	registry.Register(providers.Adapter{ID: "openaicompat", Chat: rec})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	_, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "workbuddy/hy3", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "workbuddy")
	if err == nil {
		t.Fatalf("a prefixed model must not escape its family (tried %v)", rec.seen)
	}
	for _, f := range rec.families() {
		if f != "wb" {
			t.Fatalf("prefixed request reached %q; only workbuddy was authorized (tried %v)", f, rec.seen)
		}
	}
}

// TestSingleFamilyStillReachesLaterAccounts guards a regression found live: the
// per-family attempt cap is only meaningful when another family exists. Applied
// to a single-family pool it stopped the chain after maxAttemptsPerProvider
// accounts, so a healthy third account was never tried and a request that used
// to succeed started returning 429.
func TestSingleFamilyStillReachesLaterAccounts(t *testing.T) {
	pool := accounts.NewPool(nil, nil)
	// Three workbuddy accounts: two exhausted, the last healthy. This mirrors
	// the real pool where ch and qiu are spent and github still has allowance.
	for _, id := range []string{"wb:ch", "wb:qiu"} {
		pool.Upsert(accounts.Item{ID: id, Provider: "workbuddy", Runtime: "in_process",
			ForceRoute: true, Models: []string{"hy3"},
			Quota: &accounts.QuotaSnapshot{Exceeded: true, Unit: "credits"}})
	}
	pool.Upsert(accounts.Item{ID: "wb:github", Provider: "workbuddy", Runtime: "in_process",
		Models: []string{"hy3"}})

	registry := providers.NewRegistry()
	rec := &recordingProviderChat{failWith: map[string]error{
		"wb:ch":  &providers.Error{Kind: accounts.KindQuota, Status: 429, Message: "credits exhausted"},
		"wb:qiu": &providers.Error{Kind: accounts.KindQuota, Status: 429, Message: "credits exhausted"},
	}}
	registry.Register(providers.Adapter{ID: "workbuddy", Chat: rec})

	ex := NewChatExecutor(pool, "")
	ex.Providers = registry
	got, err := ex.ChatNonStream(context.Background(), translate.ChatRequest{
		Model: "hy3", Messages: []translate.ChatMessage{{Role: "user", Content: "hi"}},
	}, "", "")
	if err != nil {
		t.Fatalf("a single-family pool must still reach its healthy account: %v (tried %v)", err, rec.seen)
	}
	if got.Content != "OK-wb:github" {
		t.Fatalf("expected the healthy account to serve, got %+v (tried %v)", got, rec.seen)
	}
}
