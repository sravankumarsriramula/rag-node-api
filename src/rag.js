// rag.js
import fs from "fs";
import path from "path";
import ollama from "ollama";
import { htmlToText } from "html-to-text";

// -------------------------------
// CONFIG
// -------------------------------
const VECTOR_DB_PATH = "./data/vector_store.json";
const EMBED_MODEL = "nomic-embed-text";
const LLM_MODEL = process.env.OLLAMA_MODEL || "llama3.2:3b";

// -------------------------------
// 1️⃣ Chunk Text
// -------------------------------
function chunkText(text, size = 800, overlap = 120) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(text.length, start + size);
    chunks.push(text.slice(start, end));
    start += size - overlap;
  }

  return chunks;
}

// -------------------------------
// 2️⃣ Convert HTML → Clean Text
// -------------------------------
function extractTextFromHTML(html) {
  return htmlToText(html, {
    wordwrap: false,
    selectors: [{ selector: "img", format: "skip" }],
  });
}

// -------------------------------
// 3️⃣ Local JSON Vector DB Helpers
// -------------------------------
function ensureDB() {
  const dir = path.dirname(VECTOR_DB_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  if (!fs.existsSync(VECTOR_DB_PATH)) {
    fs.writeFileSync(VECTOR_DB_PATH, JSON.stringify({ docs: [] }, null, 2));
  }
}

function loadDB() {
  ensureDB();
  return JSON.parse(fs.readFileSync(VECTOR_DB_PATH, "utf-8"));
}

function saveDB(db) {
  ensureDB();
  fs.writeFileSync(VECTOR_DB_PATH, JSON.stringify(db, null, 2));
}

// -------------------------------
// 4️⃣ Embedding with OLLAMA
// -------------------------------
async function embed(text, mode = "search_document") {
  const prompt = `${mode}: ${text}`;
  const res = await ollama.embeddings({
    model: EMBED_MODEL,
    prompt,
  });
  return res.embedding;
}

// -------------------------------
// 5️⃣ Cosine Similarity
// -------------------------------
function cosineSim(a, b) {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] ** 2;
    nb += b[i] ** 2;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function searchSimilar(queryEmbedding, k = 5) {
  const db = loadDB();

  const scored = db.docs.map((doc) => ({
    ...doc,
    score: cosineSim(queryEmbedding, doc.embedding),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

// -------------------------------
// 6️⃣ EMBED HTML OR TEXT DOCUMENT
// -------------------------------
export async function embedDocument({ id, text, metadata = {} }) {
  if (!text || !text.trim()) throw new Error("Document text is empty");

  // Auto-detect HTML
  let content = text;
  if (text.trim().startsWith("<")) {
    content = extractTextFromHTML(text);
  }

  const chunks = chunkText(content);

  const db = loadDB();
  const docs = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    const embedding = await embed(chunk, "search_document");

    docs.push({
      id: `${id}_${i}`,
      text: chunk,
      embedding,
      metadata: { ...metadata, chunkIndex: i },
    });
  }

  db.docs.push(...docs);
  saveDB(db);

  return {
    success: true,
    totalChunks: docs.length,
  };
}

// -------------------------------
// 7️⃣ ASK QUESTION (RAG)
// -------------------------------
export async function ask(question) {
  if (!question || !question.trim())
    throw new Error("Question cannot be empty");

  const queryEmbedding = await embed(question, "search_query");

  const results = searchSimilar(queryEmbedding, 5);

  const context =
    results
      .map((r, i) => `### Chunk ${i + 1} (score: ${r.score.toFixed(3)})\n${r.text}`)
      .join("\n\n") || "No matching documents found.";

  const prompt = `
You are an import-export expert AI.

Use only the context below to answer.
If the answer is not present in the context, reply:
"I don't know based on the provided documents."

CONTEXT:
${context}

QUESTION:
${question}
`;

  const response = await ollama.chat({
    model: LLM_MODEL,
    messages: [{ role: "user", content: prompt }],
  });

  return {
    answer: response.message.content.trim(),
    sources: results,
  };
}
