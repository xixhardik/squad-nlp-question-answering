/**
 * Frontend runtime configuration.
 *
 * The backend URL is read from the environment and is deliberately NOT given a
 * hard-coded `http://localhost:8000` fallback. A silent localhost default is
 * convenient in development but becomes a defect anywhere else: the app would
 * appear to work while quietly pointing at nothing.
 *
 * When the variable is unset, `apiBaseUrl` is `null` and the UI says so.
 * Set it in `frontend/.env.local` (see `frontend/.env.example`).
 */

/** Base URL of the FastAPI backend, or `null` when not configured. */
export const apiBaseUrl: string | null =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") || null;

/** Whether a backend URL has been configured. */
export const isApiConfigured = apiBaseUrl !== null;

/**
 * Build an absolute URL for a backend endpoint.
 *
 * @param path - Endpoint path beginning with `/`, e.g. `/health`.
 * @returns The absolute URL.
 * @throws If `NEXT_PUBLIC_API_URL` is not configured.
 */
export function apiUrl(path: string): string {
  if (apiBaseUrl === null) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not configured. Copy frontend/.env.example to " +
        "frontend/.env.local and set it to the FastAPI backend URL.",
    );
  }
  return `${apiBaseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}
