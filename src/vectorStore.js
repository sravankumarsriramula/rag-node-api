import fs from "fs";
import path from "path";
import { CONFIG } from "./config.js";

function ensureStore() {
  const dir = path.dirname(CONFIG.VECTOR_STORE);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  if (!fs.existsSync(CONFIG.VECTOR_STORE)) {
    fs.writeFileSync(
      CONFIG.VECTOR_STORE,
      JSON.stringify({ docs: [] }, null, 2)
    );
  }
}

export function loadStore() {
  ensureStore();
  return JSON.parse(fs.readFileSync(CONFIG.VECTOR_STORE, "utf-8"));
}

export function saveStore(store) {
  ensureStore();
  fs.writeFileSync(CONFIG.VECTOR_STORE, JSON.stringify(store, null, 2));
}

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

export function addDocuments(docs) {
  const store = loadStore();
  store.docs.push(...docs);
  saveStore(store);
}

export function searchSimilar(queryEmbedding, k = 5) {
  const store = loadStore();
  const scored = store.docs.map((d) => ({
    ...d,
    score: cosineSim(queryEmbedding, d.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
