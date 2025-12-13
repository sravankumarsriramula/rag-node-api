import dotenv from "dotenv";
dotenv.config();

import { OpenRouter } from "@openrouter/sdk";

export async function openRouterLLM(question, context) {
    const body = {
          messages: [
            {
              role: "system",
              content:
                "You are a helpful assistant for an Expodite. " +
                "Answer only using the provided context. If the context " +
                'does not contain the answer, say exactly: "No matching information found."',
            },
            {
              role: "user",
              content: `
    Use ONLY the following context to answer the user's question.
    If the context does not contain the answer, reply exactly:
    "No matching information found."
    
    CONTEXT:
    ${context}
    
    QUESTION:
    ${question}
              `.trim(),
            },
          ],
          temperature: 0.1,
        };
        const openRouter = new OpenRouter({
          apiKey:
            process.env.OPENROUTER_API_KEY
          });
    
        const completion = await openRouter.chat.send({
          model: "kwaipilot/kat-coder-pro:free", 
          messages: [
            {
              role: "user",
              content: JSON.stringify(body),
            },
          ],
          stream: false,
        });
     
        return completion.choices[0].message.content;
}