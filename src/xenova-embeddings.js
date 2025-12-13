import ollama from "ollama";
import { CONFIG } from "./config.js";

export async function embedDocumentText(text) {
  const res = await ollama.embeddings({
    model: CONFIG.EMBED_MODEL,
    prompt: `search_document: ${text}`
  });

  return res.embedding;
}

export async function embedQueryText(text) {
  const res = await ollama.embeddings({
    model: CONFIG.EMBED_MODEL,
    prompt: `search_query: ${text}`
  });

  return res.embedding;
}
