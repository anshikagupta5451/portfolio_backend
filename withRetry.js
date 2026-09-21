// Gemini's free tier intermittently returns 503 "model overloaded" or 429
// "rate limited" — both transient. Retrying with a short backoff clears most
// of them without the caller ever seeing a failure.
const RETRYABLE_STATUS = new Set([429, 503]);

function getStatus(error) {
  if (typeof error?.status === "number") return error.status;
  const match = /"code"\s*:\s*(\d+)/.exec(error?.message ?? "");
  return match ? Number(match[1]) : undefined;
}

export async function withRetry(fn, { retries = 2, baseDelayMs = 500 } = {}) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const status = getStatus(error);
      const isLastAttempt = attempt === retries;
      if (!RETRYABLE_STATUS.has(status) || isLastAttempt) {
        throw error;
      }
      const delay = baseDelayMs * 2 ** attempt;
      console.warn(
        `Gemini call failed with status ${status}, retrying in ${delay}ms (attempt ${attempt + 1}/${retries})`
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}
