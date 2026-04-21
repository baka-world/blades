package main

import (
	"context"
	"log"
	"os"

	"github.com/go-kratos/blades"
	"github.com/go-kratos/blades/contrib/yuanqi"
)

func main() {
	model := yuanqi.NewModel("yuanqi-agent", yuanqi.Config{
		AppKey:      os.Getenv("YUANQI_APPKEY"),
		AssistantID: os.Getenv("YUANQI_ASSISTANT_ID"),
		UserID:      os.Getenv("YUANQI_USER_ID"),
	})

	agent, err := blades.NewAgent(
		"Yuanqi Agent",
		blades.WithModel(model),
		blades.WithInstruction("You are a helpful assistant."),
	)
	if err != nil {
		log.Fatal(err)
	}

	runner := blades.NewRunner(agent)
	output, err := runner.Run(context.Background(), blades.UserMessage("你好"))
	if err != nil {
		log.Fatal(err)
	}
	log.Println("output:", output.Text())
}
