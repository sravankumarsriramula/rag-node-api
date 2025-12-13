import express from "express";
import cors from "cors";
import dotenv from "dotenv";
dotenv.config();

// import {
//   embedDocument,
//   queryDocuments,
//   askQuestion,
// } from "./rag-qdrant-xenova.js"; 

// import {
//   embedDocument,
//   queryDocuments,
//   askQuestion,
// } from "./rag-xenova-qdrant-openrouter.js"; 


import {
  embedDocument,
  queryDocuments,
  askQuestion,
} from "./rag-xenova-pgvector-openrouter.js"; 

const app = express();
app.use(express.json({ limit: "20mb" }));
app.use(cors());

// console.log("Loaded Qdrant URL:", `"${process.env.QDRANT_URL}"`);
// console.log(
//   "Loaded API Key:",
//   process.env.QDRANT_API_KEY
//     ? `"${process.env.QDRANT_API_KEY.slice(0, 6)}..."`
//     : "(none)"
// );

// -------------------------
// HEALTH CHECK
// -------------------------
app.get("/ping", (req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

// -------------------------
// EMBED DOCUMENT
// -------------------------
// Body: { id, text, meta? }
app.post("/embed", async (req, res) => {
  try {
    const { id, text, meta } = req.body;

    if (!id || !text) {
      return res.status(400).json({
        error: "Missing 'id' or 'text' in request body",
      });
    }

    const result = await embedDocument({ id, text, meta });

    res.json({
      success: true,
      message: "Document embedded",
      details: result,
    });
  } catch (err) {
    console.error("Embed error:", err);
    res.status(500).json({ error: err.message || "Embed error" });
  }
});

// -------------------------
// ASK QUESTION (RAG)
// -------------------------
// Body: { question, topK? }
app.post("/ask", async (req, res) => {
  try {
    const { question, topK } = req.body;

    if (!question || !question.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    const result = await askQuestion(question, topK || 5);

    res.json({
      success: true,
      answer: result.answer,
      matches: result.matches,
    });
  } catch (err) {
    console.error("Ask error:", err);
    res.status(500).json({ error: err.message || "Ask error" });
  }
});

// -------------------------
// RAW VECTOR SEARCH
// -------------------------
// Body: { query, topK? }
app.post("/search", async (req, res) => {
  try {
    const { query, topK } = req.body;

    if (!query || !query.trim()) {
      return res.status(400).json({ error: "Query is required" });
    }

    const matches = await queryDocuments(query, topK || 5);

    res.json({
      success: true,
      matches,
    });
  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: err.message || "Search error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () =>
  console.log(`🚀 RAG API running on http://localhost:${PORT}`)
);
