package yuanqi

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-kratos/blades"
)

const (
	defaultBaseURL        = "https://yuanqi.tencent.com/openapi/v1/agent/chat/completions"
	defaultModelName      = "yuanqi-agent"
	defaultRequestTimeout = 60 * time.Second
	maxConcurrencyLimit   = 10
	maxMessages           = 40
)

var (
	ErrAppKeyRequired      = errors.New("yuanqi: app key is required")
	ErrAssistantIDRequired = errors.New("yuanqi: assistant ID is required")
	ErrUserIDRequired      = errors.New("yuanqi: user ID is required")
	ErrSensitiveContent    = errors.New("yuanqi: response blocked by moderation")
	ErrToolCallFailed      = errors.New("yuanqi: tool execution failed")
)

// UserIDResolver resolves user ID dynamically from context.
type UserIDResolver func(ctx context.Context) string

// Config holds Yuanqi model configuration.
type Config struct {
	BaseURL         string
	AppKey          string
	AssistantID     string
	UserID          string
	UserIDResolver  UserIDResolver
	CustomVariables map[string]string
	HTTPClient      *http.Client
	MaxConcurrency  int
}

// Model implements blades.ModelProvider for Yuanqi Agent API.
type Model struct {
	name    string
	config  Config
	client  *http.Client
	limiter chan struct{}
}

// NewModel creates a Yuanqi provider model.
func NewModel(name string, cfg Config) blades.ModelProvider {
	if name == "" {
		name = defaultModelName
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	if cfg.MaxConcurrency <= 0 || cfg.MaxConcurrency > maxConcurrencyLimit {
		cfg.MaxConcurrency = maxConcurrencyLimit
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultRequestTimeout}
	}
	return &Model{
		name:    name,
		config:  cfg,
		client:  client,
		limiter: make(chan struct{}, cfg.MaxConcurrency),
	}
}

// Name returns model name.
func (m *Model) Name() string {
	return m.name
}

// Generate executes a non-streaming Yuanqi request.
func (m *Model) Generate(ctx context.Context, req *blades.ModelRequest) (*blades.ModelResponse, error) {
	if err := m.validateConfig(ctx); err != nil {
		return nil, err
	}
	if err := m.acquire(ctx); err != nil {
		return nil, err
	}
	defer m.release()

	payload, err := m.toRequestPayload(ctx, req, false)
	if err != nil {
		return nil, err
	}
	response, err := m.send(ctx, payload)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, decodeHTTPError(response)
	}

	var body chatResponse
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		return nil, err
	}
	if len(body.Choices) == 0 {
		return nil, errors.New("yuanqi: empty response choices")
	}
	if err := mapFinishReasonError(body.Choices[0].FinishReason); err != nil {
		return nil, err
	}
	msg := blades.NewAssistantMessage(blades.StatusCompleted)
	msg.FinishReason = body.Choices[0].FinishReason
	if content := body.Choices[0].Message.Content; content != "" {
		msg.Parts = append(msg.Parts, blades.TextPart{Text: content})
	}
	msg.TokenUsage = toUsage(body.Usage)
	return &blades.ModelResponse{Message: msg}, nil
}

// NewStreaming executes a streaming Yuanqi request.
func (m *Model) NewStreaming(ctx context.Context, req *blades.ModelRequest) blades.Generator[*blades.ModelResponse, error] {
	return func(yield func(*blades.ModelResponse, error) bool) {
		if err := m.validateConfig(ctx); err != nil {
			yield(nil, err)
			return
		}
		if err := m.acquire(ctx); err != nil {
			yield(nil, err)
			return
		}
		defer m.release()

		payload, err := m.toRequestPayload(ctx, req, true)
		if err != nil {
			yield(nil, err)
			return
		}
		response, err := m.send(ctx, payload)
		if err != nil {
			yield(nil, err)
			return
		}
		defer response.Body.Close()

		if response.StatusCode < 200 || response.StatusCode >= 300 {
			yield(nil, decodeHTTPError(response))
			return
		}

		scanner := bufio.NewScanner(response.Body)
		scanner.Buffer(make([]byte, 0, 1024), 2*1024*1024)
		var (
			fullText     strings.Builder
			finalUsage   usage
			finishReason string
		)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			if strings.HasPrefix(line, "data:") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
			if line == "[DONE]" {
				break
			}
			var chunk streamResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				continue
			}
			if chunk.Usage.TotalTokens > 0 {
				finalUsage = chunk.Usage
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			choice := chunk.Choices[0]
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
			deltaText := strings.TrimSpace(choice.Delta.Content)
			if deltaText == "" {
				continue
			}
			fullText.WriteString(deltaText)
			msg := blades.NewAssistantMessage(blades.StatusIncomplete)
			msg.Parts = append(msg.Parts, blades.TextPart{Text: deltaText})
			msg.FinishReason = finishReason
			if !yield(&blades.ModelResponse{Message: msg}, nil) {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
			return
		}
		if err := mapFinishReasonError(finishReason); err != nil {
			yield(nil, err)
			return
		}
		final := blades.NewAssistantMessage(blades.StatusCompleted)
		if text := strings.TrimSpace(fullText.String()); text != "" {
			final.Parts = append(final.Parts, blades.TextPart{Text: text})
		}
		final.FinishReason = finishReason
		final.TokenUsage = toUsage(finalUsage)
		yield(&blades.ModelResponse{Message: final}, nil)
	}
}

func (m *Model) validateConfig(ctx context.Context) error {
	if strings.TrimSpace(m.config.AppKey) == "" {
		return ErrAppKeyRequired
	}
	if strings.TrimSpace(m.config.AssistantID) == "" {
		return ErrAssistantIDRequired
	}
	if strings.TrimSpace(m.resolveUserID(ctx)) == "" {
		return ErrUserIDRequired
	}
	return nil
}

func (m *Model) resolveUserID(ctx context.Context) string {
	if m.config.UserIDResolver != nil {
		if uid := strings.TrimSpace(m.config.UserIDResolver(ctx)); uid != "" {
			return uid
		}
	}
	return strings.TrimSpace(m.config.UserID)
}

func (m *Model) acquire(ctx context.Context) error {
	select {
	case m.limiter <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *Model) release() {
	<-m.limiter
}

func (m *Model) send(ctx context.Context, payload chatRequest) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.config.BaseURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.config.AppKey)
	return m.client.Do(httpReq)
}

func (m *Model) toRequestPayload(ctx context.Context, req *blades.ModelRequest, stream bool) (chatRequest, error) {
	messages := toYuanqiMessages(req)
	if len(messages) == 0 {
		messages = append(messages, chatMessage{
			Role: "user",
			Content: []chatContent{{
				Type: "text",
				Text: "",
			}},
		})
	}
	return chatRequest{
		AssistantID:     m.config.AssistantID,
		UserID:          m.resolveUserID(ctx),
		Stream:          stream,
		Messages:        messages,
		CustomVariables: cloneStringMap(m.config.CustomVariables),
	}, nil
}

func toYuanqiMessages(req *blades.ModelRequest) []chatMessage {
	if req == nil {
		return nil
	}
	instruction := ""
	if req.Instruction != nil {
		instruction = strings.TrimSpace(req.Instruction.Text())
	}
	messages := make([]chatMessage, 0, len(req.Messages))
	for _, msg := range req.Messages {
		if msg == nil {
			continue
		}
		if msg.Role != blades.RoleUser && msg.Role != blades.RoleAssistant {
			continue
		}
		content := toContent(msg.Parts)
		if len(content) == 0 {
			continue
		}
		messages = append(messages, chatMessage{
			Role:    string(msg.Role),
			Content: content,
		})
	}
	messages = normalizeAlternating(messages)
	if len(messages) > maxMessages {
		messages = messages[len(messages)-maxMessages:]
	}
	if instruction != "" {
		injectInstruction(&messages, instruction)
	}
	return messages
}

func injectInstruction(messages *[]chatMessage, instruction string) {
	prefix := "[system instruction]\n" + instruction + "\n\n"
	for i := range *messages {
		if (*messages)[i].Role != "user" {
			continue
		}
		for j := range (*messages)[i].Content {
			if (*messages)[i].Content[j].Type == "text" {
				(*messages)[i].Content[j].Text = prefix + (*messages)[i].Content[j].Text
				return
			}
		}
		(*messages)[i].Content = append([]chatContent{{Type: "text", Text: prefix}}, (*messages)[i].Content...)
		return
	}
	*messages = append([]chatMessage{{
		Role: "user",
		Content: []chatContent{{
			Type: "text",
			Text: prefix,
		}},
	}}, *messages...)
}

func normalizeAlternating(messages []chatMessage) []chatMessage {
	if len(messages) == 0 {
		return messages
	}
	result := make([]chatMessage, 0, len(messages))
	for _, msg := range messages {
		if len(result) == 0 {
			result = append(result, msg)
			continue
		}
		if result[len(result)-1].Role == msg.Role {
			continue
		}
		result = append(result, msg)
	}
	return result
}

func toContent(parts []blades.Part) []chatContent {
	content := make([]chatContent, 0, len(parts))
	for _, part := range parts {
		switch v := part.(type) {
		case blades.TextPart:
			if strings.TrimSpace(v.Text) == "" {
				continue
			}
			content = append(content, chatContent{Type: "text", Text: v.Text})
		case blades.FilePart:
			if v.URI == "" || v.MIMEType.Type() != "image" {
				continue
			}
			content = append(content, chatContent{
				Type: "image_url",
				ImageURL: &imageURL{
					Type: "image_url",
					URL:  v.URI,
				},
			})
		case blades.DataPart:
			if v.Name != "" {
				content = append(content, chatContent{Type: "text", Text: "[unsupported binary input: " + v.Name + "]"})
				continue
			}
			content = append(content, chatContent{Type: "text", Text: "[unsupported binary input]"})
		case blades.ToolPart:
			content = append(content, chatContent{Type: "text", Text: "[tool call context omitted]"})
		}
	}
	return content
}

func mapFinishReasonError(reason string) error {
	switch reason {
	case "sensitive":
		return ErrSensitiveContent
	case "tool_fail":
		return ErrToolCallFailed
	default:
		return nil
	}
}

func toUsage(u usage) blades.TokenUsage {
	return blades.TokenUsage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
	}
}

func cloneStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func decodeHTTPError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if len(body) == 0 {
		return fmt.Errorf("yuanqi: status %d", resp.StatusCode)
	}
	return fmt.Errorf("yuanqi: status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
}

type chatRequest struct {
	AssistantID     string            `json:"assistant_id"`
	UserID          string            `json:"user_id"`
	Stream          bool              `json:"stream,omitempty"`
	Messages        []chatMessage     `json:"messages"`
	CustomVariables map[string]string `json:"custom_variables,omitempty"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content []chatContent `json:"content"`
}

type chatContent struct {
	Type     string    `json:"type,omitempty"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	Type string `json:"type,omitempty"`
	URL  string `json:"url,omitempty"`
}

type chatResponse struct {
	Choices []choice `json:"choices"`
	Usage   usage    `json:"usage"`
}

type streamResponse struct {
	Choices []streamChoice `json:"choices"`
	Usage   usage          `json:"usage"`
}

type choice struct {
	FinishReason string        `json:"finish_reason"`
	Message      choiceMessage `json:"message"`
}

type choiceMessage struct {
	Content string `json:"content"`
}

type streamChoice struct {
	FinishReason string      `json:"finish_reason"`
	Delta        streamDelta `json:"delta"`
}

type streamDelta struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}
