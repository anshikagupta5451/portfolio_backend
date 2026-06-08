import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import { portfolioChunks } from "./data/portfolioData.js";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

function retrieveRelevantChunks(question) {
  const words = question.toLowerCase().split(" ");

  return portfolioChunks
    .map((chunk) => {
      const text = `${chunk.title} ${chunk.content}`.toLowerCase();

      const score = words.reduce((total, word) => {
        if (text.includes(word)) {
          return total + 1;
        }
        return total;
      }, 0);

      return {
        ...chunk,
        score,
      };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

app.post("/api/chat", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    const relevantChunks = retrieveRelevantChunks(question);

    const context = relevantChunks
      .map((chunk) => `${chunk.title}: ${chunk.content}`)
      .join("\n\n");

    const prompt = `
You are Anshika Gupta's portfolio AI assistant.

Answer only using the context below.
If the answer is not available, say:
"I don't have that information in Anshika's portfolio yet."

Keep the answer short, clear, and recruiter-friendly.

Context:
${context}

Question:
${question}
`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
    });

    res.json({
      answer: response.text,
      sources: relevantChunks.map((chunk) => chunk.title),
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: "Something went wrong while generating response",
    });
  }
});

app.listen(process.env.PORT || 5000, () => {
  console.log(`Server running on port ${process.env.PORT || 5000}`);
});