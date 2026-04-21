package yuanqi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-kratos/blades"
)

func TestToYuanqiMessagesInstructionAndTruncation(t *testing.T) {
	t.Parallel()
	messages := make([]*blades.Message, 0, 45)
	for i := range 45 {
		if i%2 == 0 {
			messages = append(messages, blades.UserMessage(fmt.Sprintf("u-%d", i)))
			continue
		}
		messages = append(messages, blades.AssistantMessage(fmt.Sprintf("a-%d", i)))
	}
	result := toYuanqiMessages(&blades.ModelRequest{
		Instruction: blades.SystemMessage("follow policy"),
		Messages:    messages,
	})
	if got, want := len(result), maxMessages; got != want {
		t.Fatalf("message length = %d, want %d", got, want)
	}
	if got := result[0].Role; got != "assistant" {
		t.Fatalf("first role after truncation = %s, want assistant", got)
	}
	foundInstruction := false
	for _, msg := range result {
		if msg.Role != "user" {
			continue
		}
		for _, c := range msg.Content {
			if strings.Contains(c.Text, "[system instruction]") {
				foundInstruction = true
			}
		}
	}
	if !foundInstruction {
		t.Fatal("expected instruction prefix in user message")
	}
}

func TestGenerateMapsResponse(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("authorization header = %q", got)
		}
		var payload chatRequest
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}
		if payload.Stream {
			t.Fatal("expected non-stream request")
		}
		if payload.AssistantID != "assistant-id" {
			t.Fatalf("assistant_id = %s", payload.AssistantID)
		}
		if payload.UserID != "user-id" {
			t.Fatalf("user_id = %s", payload.UserID)
		}
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"stop","message":{"content":"hello"}}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`))
	}))
	defer server.Close()

	model := NewModel("yuanqi", Config{
		BaseURL:     server.URL,
		AppKey:      "test-key",
		AssistantID: "assistant-id",
		UserID:      "user-id",
	}).(*Model)

	resp, err := model.Generate(context.Background(), &blades.ModelRequest{Messages: []*blades.Message{blades.UserMessage("hi")}})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if got, want := resp.Message.Text(), "hello"; got != want {
		t.Fatalf("text = %q, want %q", got, want)
	}
	if got, want := resp.Message.TokenUsage.TotalTokens, int64(5); got != want {
		t.Fatalf("total tokens = %d, want %d", got, want)
	}
}

func TestGenerateFinishReasonError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"sensitive","message":{"content":"blocked"}}]}`))
	}))
	defer server.Close()

	model := NewModel("yuanqi", Config{
		BaseURL:     server.URL,
		AppKey:      "test-key",
		AssistantID: "assistant-id",
		UserID:      "user-id",
	})
	_, err := model.Generate(context.Background(), &blades.ModelRequest{Messages: []*blades.Message{blades.UserMessage("hi")}})
	if !errors.Is(err, ErrSensitiveContent) {
		t.Fatalf("error = %v, want ErrSensitiveContent", err)
	}
}

func TestNewStreamingParsesSSE(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"你\"}}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"好\"}}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	model := NewModel("yuanqi", Config{
		BaseURL:     server.URL,
		AppKey:      "test-key",
		AssistantID: "assistant-id",
		UserID:      "user-id",
	})
	stream := model.NewStreaming(context.Background(), &blades.ModelRequest{Messages: []*blades.Message{blades.UserMessage("你好")}})
	var parts []string
	var statuses []blades.Status
	var finalUsage blades.TokenUsage
	for resp, err := range stream {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		parts = append(parts, resp.Message.Text())
		statuses = append(statuses, resp.Message.Status)
		if resp.Message.Status == blades.StatusCompleted {
			finalUsage = resp.Message.TokenUsage
		}
	}
	if got, want := len(parts), 3; got != want {
		t.Fatalf("messages = %d, want %d", got, want)
	}
	if got, want := parts[0], "你"; got != want {
		t.Fatalf("first delta = %q, want %q", got, want)
	}
	if got, want := parts[2], "你好"; got != want {
		t.Fatalf("final text = %q, want %q", got, want)
	}
	if got, want := statuses[0], blades.StatusIncomplete; got != want {
		t.Fatalf("first status = %s, want %s", got, want)
	}
	if got, want := statuses[2], blades.StatusCompleted; got != want {
		t.Fatalf("final status = %s, want %s", got, want)
	}
	if got, want := finalUsage.TotalTokens, int64(3); got != want {
		t.Fatalf("final usage total_tokens = %d, want %d", got, want)
	}
}

func TestResolveUserIDFromResolver(t *testing.T) {
	t.Parallel()
	model := NewModel("yuanqi", Config{
		AppKey:      "test-key",
		AssistantID: "assistant-id",
		UserID:      "fallback-user",
		UserIDResolver: func(ctx context.Context) string {
			return "ctx-user"
		},
	}).(*Model)
	payload, err := model.toRequestPayload(context.Background(), &blades.ModelRequest{Messages: []*blades.Message{blades.UserMessage("hi")}}, false)
	if err != nil {
		t.Fatalf("toRequestPayload error: %v", err)
	}
	if got, want := payload.UserID, "ctx-user"; got != want {
		t.Fatalf("user_id = %q, want %q", got, want)
	}
}
