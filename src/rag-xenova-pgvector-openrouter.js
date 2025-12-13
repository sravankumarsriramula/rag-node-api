import dotenv from "dotenv";
dotenv.config();
   
import * as cheerio from "cheerio";
import { randomUUID } from "crypto";
import { Client } from "pg";
import { registerType } from "pgvector/pg";
import { toSql } from "pgvector";

import { chunkText } from './utils/chunk.js';
import { embedText } from './embeddings/xenova.js';
import { openRouterLLM } from './llms/openrouter.js'; 

const client = new Client({
  connectionString: process.env.POSTGRES_URL,
});
await client.connect();

// register the vector type with node-postgres
await registerType(client); 

function cleanHTML(html) { 
    const $ = cheerio.load(html);
    return $.text().replace(/\s+/g, " ").trim();
} 
 
// ------------------------
// EMBED DOCUMENT → PGVECTOR
// ------------------------ 
export async function embedDocument({ id, text, meta = {} }) {
  try {
    if (!id) throw new Error("Document 'id' is required");
    if (!text) throw new Error("Document 'text' is required");

    const cleaned = cleanHTML(text);
    if (!cleaned) throw new Error("Document content is empty after cleaning");

    const chunks = chunkText(cleaned);
    console.log(`📄 Document ${id} split into ${chunks.length} chunks`);


    for (const chunk of chunks) {
      const vector = await embedText(chunk);
      
      const vectorSql = toSql(vector);

      await client.query(

      `INSERT INTO admin.documents_embeddings (id, content, embedding)
       VALUES ($1, $2, $3)`,
      [randomUUID(),chunk, vectorSql]
      );
    } 
    console.log("✅ PGVector document insert finished");
    return { success: true, chunks: chunks.length };
  } catch (err) {
    console.error("❌ PGVector Embed Error:", err);
    throw err;
  }
}

// ------------------------
// SEARCH CHUNKS FROM PGVector
// ------------------------
export async function queryChunks(query, topK = 5) {
  const results = await searchPgVector(query, topK);
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

${m?.content ?? ""}
`
    )
    .join("\n--------------------\n");
}

// ------------------------
// CALL LLM
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
 const results = await searchPgVector(query, topK);
  return results;
}

async function searchPgVector(query, topK = 5) {
  const vector = await embedText(query); 

  const vectorSql = toSql(vector);

  const { rows } = await client.query(
    `
    SELECT 
      id,
      content,
      (embedding <-> $1::vector) AS score
    FROM admin.documents_embeddings
    ORDER BY embedding <-> $1::vector
    LIMIT $2;
    `,
    [vectorSql, topK]
  ); 
  return rows;
}


