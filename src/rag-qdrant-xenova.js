import dotenv from "dotenv";
dotenv.config();

import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@xenova/transformers";
import { convert } from "html-to-text";
import { randomUUID } from "crypto";
import OpenAI from "openai";
import { OpenRouter } from "@openrouter/sdk";
// ------------------------
// ENV + QDRANT CLIENT
// ------------------------
const {
  QDRANT_URL,
  QDRANT_API_KEY,
  QDRANT_COLLECTION,
  QDRANT_EMBED_MODEL,
  GROQ_API_KEY,
  GROQ_MODEL,
} = process.env;

if (!QDRANT_URL) throw new Error("QDRANT_URL is not set");
if (!QDRANT_COLLECTION) throw new Error("QDRANT_COLLECTION is not set");
if (!QDRANT_EMBED_MODEL) throw new Error("QDRANT_EMBED_MODEL is not set");
if (!GROQ_API_KEY) throw new Error("GROQ_API_KEY is not set");

console.log(`Initializing QdrantClient with URL: "${QDRANT_URL}"`);

const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY || undefined,
});

const EMBED_MODEL = QDRANT_EMBED_MODEL;

// ------------------------
// XENOVA EMBEDDING MODEL
// ------------------------
let embedder = null;

async function loadEmbedder() {
  if (!embedder) {
    console.log("⏳ Loading Xenova embedding model:", EMBED_MODEL);
    embedder = await pipeline("feature-extraction", EMBED_MODEL);
    console.log("✅ Xenova model loaded");
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

  const vector = Array.from(output.data);
  // console.log("🔎 Embedding vector LENGTH:", vector.length);
  return vector;
}

// ------------------------
// HTML → CLEAN TEXT
// ------------------------
function cleanHTML(htmlOrText) {
  if (!htmlOrText) return "";

  // If it's plain text, convert doesn't hurt; if HTML, we'll strip tags.
  return convert(htmlOrText, {
    wordwrap: false,
    selectors: [
      { selector: "img", format: "skip" },
      { selector: "script", format: "skip" },
      { selector: "style", format: "skip" },
    ],
  })
    .replace(/\s+/g, " ")
    .trim();
}

// ------------------------
// SIMPLE SENTENCE CHUNKING
// ------------------------
function chunkText(text, maxLength = 1200) {
  const sentences = text.split(/(?<=[.?!])\s+/);
  const chunks = [];
  let current = "";

  for (const sentence of sentences) {
    if (!sentence.trim()) continue;

    if ((current + " " + sentence).length > maxLength) {
      if (current.trim().length) chunks.push(current.trim());
      current = sentence;
    } else {
      current += " " + sentence;
    }
  }

  if (current.trim().length) chunks.push(current.trim());

  return chunks;
}

// ------------------------
// ENSURE COLLECTION
// ------------------------
async function ensureCollection() {
  const collections = await qdrant.getCollections();
  const exists = collections.collections.some(
    (c) => c.name === QDRANT_COLLECTION
  );

  if (!exists) {
    console.log(`📦 Creating Qdrant collection "${QDRANT_COLLECTION}"...`);
    await qdrant.createCollection(QDRANT_COLLECTION, {
      vectors: {
        size: 384, // all-MiniLM-L6-v2 dimension; adjust if you use another model
        distance: "Cosine",
      },
    });
    console.log("✅ Collection created");
  }
}

// Call once at startup
ensureCollection().catch((err) => {
  console.error("❌ Error ensuring collection:", err);
});

// ------------------------
// EMBED DOCUMENT → QDRANT
// ------------------------
/**
 * Embed a document (HTML or plain text) and store into Qdrant.
 * @param {Object} params
 * @param {string|number} params.id - Your document ID (e.g. DB id)
 * @param {string} params.text - HTML or plain text content
 * @param {Object} [params.meta] - Optional metadata (title, type, tenant_id, etc.)
 */
export async function embedDocument({ id, text, meta = {} }) {
  try {
    if (!id) throw new Error("Document 'id' is required");
    if (!text) throw new Error("Document 'text' is required");

    const cleaned = cleanHTML(text);
    if (!cleaned) throw new Error("Document content is empty after cleaning");

    const chunks = chunkText(cleaned);
    console.log(`📄 Document ${id} split into ${chunks.length} chunks`);

    const points = [];

    for (const chunk of chunks) {
      const vector = await embedText(chunk);

      points.push({
        id: randomUUID(),
        vector,
        payload: {
          doc_id: String(id),
          content: chunk,
          ...meta,
        },
      });
    }

    console.log()

    console.log(`⬆️ Uploading ${points.length} points to Qdrant…`);

    await qdrant.upsert(QDRANT_COLLECTION, {
      wait: true,
      points,
    });

    console.log("✅ Qdrant upsert finished");

    return { success: true, chunks: points.length };
  } catch (err) {
    console.error("❌ Qdrant Embed Error:", err);
    throw err;
  }
}

// ------------------------
// SEARCH CHUNKS FROM QDRANT
// ------------------------
export async function queryChunks(query, topK = 5) {
  const vector = await embedText(query);

  const results = await qdrant.search(QDRANT_COLLECTION, {
    vector,
    limit: topK,
    // score_threshold: 0.0, // filter out weak matches
  });

  return results;
}

// ------------------------
// FORMAT CONTEXT FOR GROQ
// ------------------------
function buildContext(matches) {
  if (!matches || matches.length === 0) return "";

  return matches
    .map(
      (m, i) => `
Chunk ${i + 1} (score: ${m.score?.toFixed(3) ?? "n/a"}):

${m.payload?.content ?? ""}
`
    )
    .join("\n--------------------\n");
}

// ------------------------
// CALL GROQ LLM
// ------------------------
async function generateAnswer(question, context) {
  try {
    const model = GROQ_MODEL || "llama-3.1-8b-instant";

    if (!context || !context.trim()) {
      // No context at all → don't even call Groq
      return "No matching information found.";
    }

    const body = {
      model,
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

    // const response = await fetch(
    //   "https://api.groq.com/openai/v1/chat/completions",
    //   {
    //     method: "POST",
    //     headers: {
    //       "Content-Type": "application/json",
    //       Authorization: `Bearer ${GROQ_API_KEY}`,
    //     },
    //     body: JSON.stringify(body),
    //   }
    // );

    // const openai = new OpenAI({
    //   baseURL: "https://api.deepseek.com",
    //   apiKey: "sk-8b511ee579404d59ab095b7049e41454",
    // });

    // const completion = await openai.chat.completions.create({
    //   messages: [{ role: "system", content: body }],
    //   model: "deepseek-chat",
    // });
    // console.log(completion);

    const openRouter = new OpenRouter({
      apiKey:
        process.env.OPENROUTER_API_KEY ||
        "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",
    });

    const completion = await openRouter.chat.send({
      model: "kwaipilot/kat-coder-pro:free",
      // model: "tngtech/deepseek-r1t2-chimera:free",
      messages: [
        {
          role: "user",
          content: JSON.stringify(body),
        },
      ],
      stream: false,
    });

    // console.log(completion);
    // console.log(completion.choices);

    // if (!response.ok) {
    //   const errText = await response.text();
    //   throw new Error("Groq API Error: " + errText);
    // }

    // const data = await response.json();
    // return data.choices[0].message.content.trim();

    return completion.choices[0].message.content;
  } catch (error) {
    console.error("❌ Groq LLM Error:", error);
    throw error;
  }
}

// ------------------------
// ASK QUESTION → FULL RAG
// ------------------------
export async function askQuestion(question, topK = 5) {
  const matches = await queryChunks(question, topK);

  if (!matches || matches.length === 0) {
    return {
      answer: "No matching information found.",
      matches: [],
    };
  }

  const context = buildContext(matches);
  const answer = await generateAnswer(question, context);

  return {
    answer,
    matches,
  };
}

// ------------------------
// RAW DOCUMENT QUERY
// ------------------------
export async function queryDocuments(query, topK = 5) {
  const vector = await embedText(query);

  return await qdrant.search(QDRANT_COLLECTION, {
    vector,
    limit: topK,
  });
}
