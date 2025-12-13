import dotenv from "dotenv";
dotenv.config();
 
import { pipeline } from "@xenova/transformers";
import { convert } from "html-to-text"; 
import { parse } from "node-html-parser";
import { OpenRouter } from "@openrouter/sdk";

import * as cheerio from "cheerio";
import { randomUUID } from "crypto";
import { Client } from "pg";
import { registerType } from "pgvector/pg";
import { toSql } from "pgvector";

const EMBED_MODEL = process.env.QDRANT_EMBED_MODEL; 

const client = new Client({
  connectionString: process.env.POSTGRES_URL,
});
await client.connect();

// register the vector type with node-postgres
await registerType(client);
 
// ------------------------
// XENOVA EMBEDDING MODEL
// ------------------------
let embedder = null;

async function loadEmbedder() {
  if (!embedder) {
    // console.log("⏳ Loading Xenova embedding model:", EMBED_MODEL);
    embedder = await pipeline("feature-extraction", EMBED_MODEL);
    // console.log("✅ Xenova model loaded");
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

// ------------------------
// HTML → CLEAN TEXT
// ------------------------
// function cleanHTML(htmlOrText) {
//   if (!htmlOrText) return "";

//   // If it's plain text, convert doesn't hurt; if HTML, we'll strip tags.
//   return convert(htmlOrText, {
//     wordwrap: false,
//     selectors: [
//       { selector: "img", format: "skip" },
//       { selector: "script", format: "skip" },
//       { selector: "style", format: "skip" },
//     ],
//   })
//     .replace(/\s+/g, " ")
//     .trim();
// }



function cleanHTML(html) {
//   const root = parse(html, {
//     comment: false,
//     blockTextElements: {
//       script: false,
//       noscript: false,
//       style: false,
//     },
//   });

//   return root.innerText
//     .replace(/\s+/g, " ")
//     .trim();

    const $ = cheerio.load(html);
    return $.text().replace(/\s+/g, " ").trim();
}
// ------------------------
// SIMPLE SENTENCE CHUNKING
// ------------------------
function chunkText(text, maxLength = 2000) {
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


    for (const chunk of chunks) {
      const vector = await embedText(chunk);
     
      
    //   const vectorString = `[${vector.join(",")}]`;
      const vectorSql = toSql(vector);

      await client.query(

      `INSERT INTO admin.documents_embeddings (id, content, embedding)
       VALUES ($1, $2, $3)`,
      [randomUUID(),chunk, vectorSql]
      );
    } 
    console.log("✅ Qdrant upsert finished");
    return { success: true, chunks: chunks.length };
  } catch (err) {
    console.error("❌ Qdrant Embed Error:", err);
    throw err;
  }
}

// ------------------------
// SEARCH CHUNKS FROM QDRANT
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
// CALL GROQ LLM
// ------------------------
async function generateAnswer(question, context) {
  try { 

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
console.log({matches});
  if (!matches || matches.length === 0) {
    return {
      answer: "No matching information found.",
      matches: [],
    };
  }

  const context = buildContext(matches);
console.log({context});
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

async function searchPgVector(query, topK = 5) {
  const vector = await embedText(query);
//   const vectorString = `[${vector.join(",")}]`;

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


