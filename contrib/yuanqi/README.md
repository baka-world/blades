# Yuanqi Provider

`yuanqi` provides a `blades.ModelProvider` implementation for Tencent Yuanqi agent API.

## Features

- Non-streaming and streaming chat completions
- `assistant_id` / `AppKey` auth mapping
- Optional `UserIDResolver(ctx)`
- Message mapping for text and image URL input
- Provider-level concurrency limiting (max 10)

## Usage

```go
model := yuanqi.NewModel("yuanqi-agent", yuanqi.Config{
    AppKey:      os.Getenv("YUANQI_APPKEY"),
    AssistantID: os.Getenv("YUANQI_ASSISTANT_ID"),
    UserID:      "demo-user",
})

agent, _ := blades.NewAgent("demo", blades.WithModel(model))
```
