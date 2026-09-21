import { withRetry } from "./withRetry.js";

const EMBEDDING_MODEL = "gemini-embedding-001";
const EMBEDDING_DIMENSIONS = 256;

let chunkIndexPromise = null;

export function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function embed(ai, contents) {
  const response = await withRetry(() =>
    ai.models.embedContent({
      model: EMBEDDING_MODEL,
      contents,
      config: { outputDimensionality: EMBEDDING_DIMENSIONS },
    })
  );
  return response.embeddings.map((embedding) => embedding.values);
}

// Chunk embeddings are computed once and cached in memory — the "index"
// build step of a RAG pipeline, kept separate from the per-query search.
function getChunkIndex(ai, chunks) {
  if (!chunkIndexPromise) {
    chunkIndexPromise = embed(
      ai,
      chunks.map((chunk) => `${chunk.title}\n${chunk.content}`)
    ).catch((error) => {
      chunkIndexPromise = null;
      throw error;
    });
  }
  return chunkIndexPromise;
}

export async function retrieveByEmbedding(ai, chunks, question, topK = 3) {
  const [chunkVectors, [queryVector]] = await Promise.all([
    getChunkIndex(ai, chunks),
    embed(ai, [question]),
  ]);

  return chunks
    .map((chunk, i) => ({
      ...chunk,
      score: cosineSimilarity(queryVector, chunkVectors[i]),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export const ragMeta = {
  retrieval: "cosine similarity over Gemini embeddings",
  embeddingModel: `${EMBEDDING_MODEL} (${EMBEDDING_DIMENSIONS}-dim)`,
};
