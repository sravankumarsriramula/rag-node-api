import dotenv from "dotenv";
dotenv.config();

export const CONFIG = {
  PORT: process.env.PORT || 3000,
  OLLAMA_MODEL: process.env.OLLAMA_MODEL || "llama3.2",
  VECTOR_STORE: process.env.VECTOR_STORE || "./data/vector_store.json"
};
