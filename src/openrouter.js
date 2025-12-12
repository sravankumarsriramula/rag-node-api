import { OpenRouter } from "@openrouter/sdk";

const openRouter = new OpenRouter({
  apiKey: "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",
});

console.log(openRouter)

const completion = await openRouter.chat.send({
  model: "kwaipilot/kat-coder-pro:free",
  messages: [
    {
      role: "user",
      content: "What is the meaning of life?",
    },
  ],
  stream: false,
});

console.log(completion.choices[0].message.content);
