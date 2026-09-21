import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { portfolioChunks } from "./data/portfolioData.js";
import { retrieveByEmbedding, ragMeta } from "./rag.js";
import { withRetry } from "./withRetry.js";

dotenv.config();

if (!process.env.GEMINI_API_KEY) {
  console.warn(
    "WARNING: GEMINI_API_KEY is not set. Chat requests will fail until it's added to .env."
  );
}

process.on("uncaughtException", (error) => {
  console.error("FATAL — uncaughtException (server will exit):", error);
});

process.on("unhandledRejection", (reason) => {
  console.error("FATAL — unhandledRejection (server will exit):", reason);
});

process.on("exit", (code) => {
  console.log(`Process exiting with code ${code}`);
});

const app = express();

const allowedOrigins = [
  "https://anshika-gupta-software-developer.vercel.app",
  "http://localhost:5173",
];

app.use(
  cors({
    origin: allowedOrigins,
  })
);
app.use(express.json());

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

// Small talk never has an answer "in the portfolio," so it's handled before
// retrieval runs at all — otherwise the RAG prompt has nothing relevant to
// point to and falls back to "I don't have that information."
const SMALL_TALK = [
  {
    pattern: /^(hi+|hello+|hey+|yo+|sup|howdy|greetings|namaste|hola)[\s!.,?]*$/i,
    reply:
      "Hey! I'm Anshika's portfolio assistant — ask me about her skills, projects, experience, or how to reach her.",
  },
  {
    pattern: /^(good\s?(morning|afternoon|evening|day))[\s!.,?]*$/i,
    reply:
      "Hello! Ask me anything about Anshika's skills, projects, or experience.",
  },
  {
    pattern: /^(how\s?(are\s?you|r\s?u)|what'?s\s?up)[\s!.,?]*$/i,
    reply:
      "Doing well, thanks! I'm here to answer questions about Anshika's work — try asking about a project or her tech stack.",
  },
  {
    pattern: /^(who\s?are\s?you|what\s?(are\s?you|can\s?you\s?do))[\s!.,?]*$/i,
    reply:
      "I'm a small RAG pipeline over Anshika's portfolio — I embed your question, retrieve the most relevant chunks by cosine similarity, and use Gemini to answer from them. Ask me about her experience, projects, or stack.",
  },
  {
    pattern: /^(thanks|thank\s?you|thx|ty)[\s!.,?]*$/i,
    reply: "You're welcome! Let me know if you'd like to know more.",
  },
  {
    pattern: /^(bye|goodbye|see\s?ya|see\s?you|cya)[\s!.,?]*$/i,
    reply: "Thanks for stopping by — feel free to reach out to Anshika directly from the contact section!",
  },
];

function matchSmallTalk(question) {
  const trimmed = question.trim();
  const match = SMALL_TALK.find(({ pattern }) => pattern.test(trimmed));
  return match?.reply;
}

// Fallback retrieval used only if the embedding call fails (quota, network,
// bad key) so the assistant degrades instead of breaking.
function retrieveByKeyword(question) {
  const words = question.toLowerCase().split(" ");

  return portfolioChunks
    .map((chunk) => {
      const text = `${chunk.title} ${chunk.content}`.toLowerCase();
      const score = words.reduce(
        (total, word) => (text.includes(word) ? total + 1 : total),
        0
      );
      return { ...chunk, score };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

async function retrieveRelevantChunks(question) {
  try {
    const chunks = await retrieveByEmbedding(ai, portfolioChunks, question);
    return { chunks, retrieval: "embedding" };
  } catch (error) {
    console.error("Embedding retrieval failed, falling back to keyword match:", error.message);
    return { chunks: retrieveByKeyword(question), retrieval: "keyword-fallback" };
  }
}

app.post("/api/chat", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const smallTalkReply = matchSmallTalk(question);
    if (smallTalkReply) {
      return res.json({
        answer: smallTalkReply,
        sources: [],
        meta: {
          retrieval: "small talk (no retrieval needed)",
          embeddingModel: null,
          generationModel: null,
          retrievalMs: 0,
        },
      });
    }

    const started = Date.now();
    const { chunks: relevantChunks, retrieval } = await retrieveRelevantChunks(
      question
    );
    const retrievalMs = Date.now() - started;

    const context = relevantChunks
      .map((chunk) => `${chunk.title}: ${chunk.content}`)
      .join("\n\n");

    const prompt = `
You are Anshika Gupta's portfolio AI assistant, speaking with recruiters and hiring managers.

Answer using the context below when the question is genuinely about Anshika's
background, skills, projects, or experience. Only say
"I don't have that information in Anshika's portfolio yet." when the question
is clearly asking about her and the context doesn't cover it.

If the message is a greeting, small talk, or general conversation rather than
a real question about Anshika, respond naturally and briefly invite them to
ask about her skills, projects, or experience — do not say you "don't have
that information."

Keep the answer short, clear, and recruiter-friendly.

Context:
${context}

Question:
${question}
`;

    const response = await withRetry(() =>
      ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
      })
    );

    res.json({
      answer: response.text,
      sources: relevantChunks.map((chunk) => ({
        title: chunk.title,
        score: Number(chunk.score.toFixed(3)),
      })),
      meta: {
        retrieval:
          retrieval === "embedding"
            ? ragMeta.retrieval
            : "keyword overlap (fallback)",
        embeddingModel: retrieval === "embedding" ? ragMeta.embeddingModel : null,
        generationModel: "gemini-2.5-flash",
        retrievalMs,
      },
    });
  } catch (error) {
    console.error(error);
    const isOverloaded = error?.status === 503 || error?.status === 429;
    res.status(isOverloaded ? 503 : 500).json({
      error: isOverloaded
        ? "The AI model is busy right now — please try again in a moment."
        : "Something went wrong while generating response",
    });
  }
});

const PORT = process.env.PORT || 5000;

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(
      `Port ${PORT} is already in use — another process (maybe an older run of this same server) is still running. ` +
        `Find it with: lsof -i :${PORT}  then stop it with: kill -9 <PID>`
    );
  } else {
    console.error("Server failed to start:", error);
  }
  process.exit(1);
});
