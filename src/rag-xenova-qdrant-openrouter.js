import dotenv from "dotenv";
dotenv.config();

import { QdrantClient } from "@qdrant/js-client-rest";
// import { pipeline } from "@xenova/transformers";
// import { convert } from "html-to-text"; 
// import { OpenRouter } from "@openrouter/sdk";


import { chunkText } from './utils/chunk.js';
import { embedText } from './embeddings/xenova.js';
import { openRouterLLM } from './llms/openrouter.js'; 

// register the vector type with node-postgres
 
// ------------------------
// ENV + QDRANT CLIENT
// ------------------------
const {
  QDRANT_URL,
  QDRANT_API_KEY,
  QDRANT_COLLECTION,
  QDRANT_EMBED_MODEL,  
} = process.env;

if (!QDRANT_URL) throw new Error("QDRANT_URL is not set");
if (!QDRANT_COLLECTION) throw new Error("QDRANT_COLLECTION is not set");
if (!QDRANT_EMBED_MODEL) throw new Error("QDRANT_EMBED_MODEL is not set"); 
 
 
const qdrant = new QdrantClient({
  url: QDRANT_URL,
  apiKey: QDRANT_API_KEY || undefined,
});
 
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

${m?.payload?.content ?? ""}
`
    )
    .join("\n--------------------\n");
}

// ------------------------
// CALL GROQ LLM
// ------------------------
async function generateAnswer(question, context) {
  try { 
       return await openRouterLLM(question,context); 
  } catch (error) {
    console.error("❌ OPENROUTER LLM Error:", error);
    throw error;
  }
}

// ------------------------
// ASK QUESTION → FULL RAG
// ------------------------
export async function askQuestion(question, topK = 5) {
  const matches = await queryChunks(question, topK);
// console.log({matches})
  if (!matches || matches.length === 0) {
    return {
      answer: "No matching information found.",
      matches: [],
    };
  }

  const context = buildContext(matches);

  // console.log({context})
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


