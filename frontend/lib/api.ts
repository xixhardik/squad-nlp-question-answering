/**
 * Typed client for the FastAPI inference backend.
 *
 * The interfaces here mirror `backend/app/schemas.py` field for field. They are
 * hand-written rather than generated so the mismatch shows up as a TypeScript
 * error at build time if either side drifts; the backend also serves
 * `/openapi.json`, which is the authoritative contract.
 *
 * Nothing in this module has a hard-coded backend address. Every request goes
 * through `apiUrl()` from `./config`, which throws when `NEXT_PUBLIC_API_URL`
 * is unset rather than silently defaulting to localhost.
 */

import { apiUrl } from "@/lib/config";

/** Request body for `POST /predict`. Mirrors `PredictRequest`. */
export interface PredictRequest {
  question: string;
  context: string;
}

/** One ranked alternative inside `PredictionResponse.n_best`. */
export interface NBestEntry {
  answer: string;
  char_start: number;
  char_end: number;
  score: number;
}

/**
 * Response body for `POST /predict`. Mirrors `PredictionResponse`.
 *
 * `score` is deliberately not named "confidence". Its meaning is carried by
 * `score_type`, which the backend sets to `uncalibrated_span_probability`: a
 * single softmax over the pooled valid candidate spans from every sliding
 * window. That is a proper distribution over the hypotheses considered, but it
 * is not calibrated.
 */
export interface PredictionResponse {
  answer: string;
  char_start: number;
  char_end: number;
  score: number;
  score_type: string;
  latency_ms: number;
  num_windows: number;
  model_id: string;
  truncated: boolean;
  has_answer: boolean;
  n_best: NBestEntry[];
}

/** Response body for `GET /health`. Mirrors `HealthResponse`. */
export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  phase: string;
  model_loaded: boolean;
  model_id: string | null;
}

/**
 * Input limits enforced by the backend (`Settings.max_question_chars` and
 * `Settings.max_context_chars`). Mirrored here to give immediate feedback in
 * the browser. The server remains the authority; these are not a substitute
 * for its validation.
 */
export const MAX_QUESTION_CHARS = 512;
export const MAX_CONTEXT_CHARS = 20_000;

/** A failed backend call, carrying the HTTP status when there was one. */
export class ApiError extends Error {
  /** HTTP status, or `null` when the request never completed. */
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Pull a human-readable message out of a FastAPI error body.
 *
 * FastAPI uses `detail` for both of its error shapes, with different types:
 * `HTTPException` produces a string, while request-validation failures produce
 * an array of objects. Both are handled so a 422 from Pydantic is as readable
 * as a 503 raised by the route.
 */
function extractDetail(body: unknown): string | null {
  if (typeof body !== "object" || body === null) {
    return null;
  }

  const detail = (body as { detail?: unknown }).detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item !== "object" || item === null) {
          return null;
        }
        const entry = item as { msg?: unknown; loc?: unknown };
        const message = typeof entry.msg === "string" ? entry.msg : null;
        if (message === null) {
          return null;
        }
        const field = Array.isArray(entry.loc)
          ? entry.loc.filter((part) => typeof part === "string").at(-1)
          : undefined;
        return field ? `${field}: ${message}` : message;
      })
      .filter((message): message is string => message !== null);

    return messages.length > 0 ? messages.join("; ") : null;
  }

  return null;
}

/** Human-readable fallback for statuses whose body carried no usable detail. */
function fallbackMessage(status: number): string {
  if (status === 503) {
    return "The model is not loaded on the backend. Set QAS_MODEL_PATH and restart it.";
  }
  if (status === 404) {
    return "The backend responded 404. Check that NEXT_PUBLIC_API_URL points at the FastAPI service.";
  }
  if (status >= 500) {
    return `The backend failed with status ${status}.`;
  }
  return `The backend rejected the request with status ${status}.`;
}

/**
 * Ask the backend to answer a question about a passage.
 *
 * @param payload - Question and context, matching `PredictRequest`.
 * @param signal - Abort signal, so a superseded request can be cancelled.
 * @returns The parsed `PredictionResponse`.
 * @throws {ApiError} On a non-2xx response, an unreachable backend, an
 *   unparseable body, or an unconfigured `NEXT_PUBLIC_API_URL`.
 */
export async function requestPrediction(
  payload: PredictRequest,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  let endpoint: string;
  try {
    endpoint = apiUrl("/predict");
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "The backend URL is not configured.",
    );
  }

  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (error) {
    // Rethrow aborts untouched so the caller can distinguish "superseded" from
    // "failed" and avoid showing an error for a request it cancelled itself.
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      `Could not reach the backend at ${endpoint}. Check that it is running and that CORS allows this origin.`,
    );
  }

  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    throw new ApiError(
      extractDetail(body) ?? fallbackMessage(response.status),
      response.status,
    );
  }

  if (typeof body !== "object" || body === null || !("answer" in body)) {
    throw new ApiError(
      "The backend returned a body that does not match PredictionResponse.",
      response.status,
    );
  }

  return body as PredictionResponse;
}
