import { QdrantClient } from "@qdrant/js-client-rest";
import { htmlToText } from "html-to-text";
import crypto from "crypto";
import dotenv from "dotenv";

dotenv.config();

const COLLECTION = process.env.QDRANT_COLLECTION;
const EMBED_MODEL = process.env.QDRANT_EMBED_MODEL;

// ----------------------------
// Qdrant Cloud Client
// ----------------------------
const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
});

// ----------------------------
// Clean and chunk HTML
// ----------------------------
function cleanHTML(html) {
  return htmlToText(html, {
    preserveNewlines: true,
    wordwrap: false,
  });
}

function chunk(text, limit = 800) {
  const words = text.split(/\s+/);
  let out = [];
  let buff = [];

  for (const w of words) {
    buff.push(w);
    if (buff.join(" ").length > limit) {
      out.push(buff.join(" "));
      buff = [];
    }
  }
  if (buff.length) out.push(buff.join(" "));
  return out;
}

// ----------------------------
// Use Qdrant Cloud Embeddings
// ----------------------------
async function embedQdrant(text) {
  const response = await fetch(`${process.env.QDRANT_URL}/v1/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "api-key": process.env.QDRANT_API_KEY,
    },
    body: JSON.stringify({
      model: process.env.QDRANT_EMBED_MODEL,
      input: text
    })
  });

  const content = await response.text();

  if (!response.ok) {
    console.error("❌ Qdrant Embedding Error Response:", content);
    throw new Error(content);
  }

  const data = JSON.parse(content);
  return data.data[0].embedding;
}


// ----------------------------
// Store document chunks
// ----------------------------
export async function embedDocument({ id, text, metadata = {} }) {
  const cleaned = cleanHTML(text);
  const chunks = chunk(cleaned);

  const points = [];

  for (let i = 0; i < chunks.length; i++) {
    const vector = await embedQdrant(chunks[i]);

    points.push({
      id: crypto.randomUUID(),
      vector,
      payload: {
        docId: id,
        chunkIndex: i,
        text: chunks[i],
        ...metadata,
      },
    });
  }

  await qdrant.upsert(COLLECTION, {
    points,
    wait: true,
  });

  return { success: true, chunks: chunks.length };
}

// ----------------------------
// Retrieval + generation
// ----------------------------
export async function ask(question, generateAnswer) {
  const qVector = await embedQdrant(question);

  const near = await qdrant.search(COLLECTION, {
    limit: 5,
    vector: qVector,
  });

  const context = near.map((p) => p.payload.text).join("\n\n");

  const answer = await generateAnswer(question, context);

  return {
    answer,
    sources: near,
  };
}
