import dotenv from "dotenv";
dotenv.config();
 
import { pipeline } from "@xenova/transformers";

let embedder = null;

async function loadEmbedder() {
  if (!embedder) { 
    embedder = await pipeline("feature-extraction", process.env.QDRANT_EMBED_MODEL);
  }
  return embedder;
}


export async function embedText(text) {
  if (!text || !text.trim()) {
    throw new Error("Cannot embed empty text");
  }

  const model = await loadEmbedder();
  const output = await model(text, {
    pooling: "mean",
    normalize: true,
  });

    //vector array
    return  Array.from(output.data);  
}