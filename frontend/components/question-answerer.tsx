"use client";

/**
 * The question answering interface.
 *
 * A Client Component because it owns form state, an in-flight request and an
 * abort controller. It is nested inside the Server Component page, which is the
 * boundary the Next.js docs recommend: static content stays on the server and
 * only the interactive island ships JavaScript.
 *
 * Every number shown here comes from the backend response. There is no mock
 * path and no fabricated answer: when the API is unreachable or the model is
 * not loaded, the UI reports that instead of inventing a result.
 */

import { useId, useRef, useState } from "react";

import {
  ApiError,
  MAX_CONTEXT_CHARS,
  MAX_QUESTION_CHARS,
  type PredictionResponse,
  requestPrediction,
} from "@/lib/api";
import { isApiConfigured } from "@/lib/config";

/**
 * A completed prediction together with the exact context it was computed from.
 *
 * The context is captured at submit time rather than read from the textarea at
 * render time. `char_start`/`char_end` index the passage that was actually sent,
 * so highlighting the live textarea value would put the marker in the wrong
 * place the moment the user edits the field after a result comes back.
 */
interface Outcome {
  prediction: PredictionResponse;
  context: string;
}

/** Render the answer span highlighted inside the passage it was found in. */
function HighlightedContext({ prediction, context }: Outcome) {
  const { char_start: start, char_end: end, has_answer: hasAnswer } = prediction;

  // Clamp defensively. A span outside the passage would mean the backend and
  // this component disagree about the context, and silently rendering a blank
  // highlight would hide that.
  const safeStart = Math.max(0, Math.min(start, context.length));
  const safeEnd = Math.max(safeStart, Math.min(end, context.length));
  const spanIsRenderable = hasAnswer && safeEnd > safeStart;

  if (!spanIsRenderable) {
    return (
      <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-400">
        {context}
      </p>
    );
  }

  return (
    <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-slate-400">
      {context.slice(0, safeStart)}
      <mark className="rounded bg-sky-400/25 px-0.5 font-medium text-sky-100 ring-1 ring-inset ring-sky-400/40">
        {context.slice(safeStart, safeEnd)}
      </mark>
      {context.slice(safeEnd)}
    </p>
  );
}

/** One label/value pair in the diagnostics grid. */
function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg bg-slate-900/70 px-3 py-2.5">
      <dt className="text-[0.6875rem] font-medium uppercase tracking-wider text-slate-500">
        {label}
      </dt>
      <dd className="mt-1 break-words font-mono text-sm text-slate-200">{value}</dd>
      {hint ? <p className="mt-1 text-[0.6875rem] text-slate-500">{hint}</p> : null}
    </div>
  );
}

export default function QuestionAnswerer() {
  const contextId = useId();
  const questionId = useId();

  const [context, setContext] = useState("");
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<Outcome | null>(null);

  const inFlight = useRef<AbortController | null>(null);

  const contextTooLong = context.length > MAX_CONTEXT_CHARS;
  const questionTooLong = question.length > MAX_QUESTION_CHARS;
  const canSubmit =
    isApiConfigured &&
    context.trim().length > 0 &&
    question.trim().length > 0 &&
    !contextTooLong &&
    !questionTooLong &&
    !isLoading;

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }

    // Supersede any request still running, so a fast double submit cannot let
    // an older response overwrite a newer one.
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    const submittedContext = context;
    const submittedQuestion = question;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const prediction = await requestPrediction(
        { question: submittedQuestion, context: submittedContext },
        controller.signal,
      );
      setOutcome({ prediction, context: submittedContext });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return; // Superseded by a newer submit; leave the UI to that request.
      }
      setOutcome(null);
      setErrorMessage(
        error instanceof ApiError
          ? error.message
          : "Something went wrong while contacting the backend.",
      );
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null;
        setIsLoading(false);
      }
    }
  }

  return (
    <section aria-labelledby="ask-heading" className="mt-10">
      <h2 id="ask-heading" className="text-lg font-semibold tracking-tight text-white">
        Ask a question
      </h2>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
        Paste a passage, ask something answerable from it, and the model returns
        the exact span of that passage which answers the question.
      </p>

      {!isApiConfigured ? (
        <div
          role="status"
          className="mt-5 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm leading-relaxed text-amber-200"
        >
          <span className="font-semibold">Backend not configured.</span> Set{" "}
          <code className="font-mono text-xs">NEXT_PUBLIC_API_URL</code> in{" "}
          <code className="font-mono text-xs">frontend/.env.local</code> and
          restart the dev server. The form stays disabled until then, because a
          fabricated answer would be worse than no answer.
        </div>
      ) : null}

      <form onSubmit={handleSubmit} className="mt-6 space-y-5">
        <div>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <label htmlFor={contextId} className="text-sm font-medium text-slate-200">
              Context
            </label>
            <span
              className={`font-mono text-xs ${
                contextTooLong ? "text-rose-400" : "text-slate-500"
              }`}
            >
              {context.length.toLocaleString()} / {MAX_CONTEXT_CHARS.toLocaleString()}
            </span>
          </div>
          <textarea
            id={contextId}
            value={context}
            onChange={(event) => setContext(event.target.value)}
            rows={8}
            spellCheck={false}
            placeholder="Paste the passage that contains the answer."
            aria-invalid={contextTooLong}
            className="mt-2 w-full resize-y rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-3 text-sm leading-relaxed text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:opacity-50"
          />
          {contextTooLong ? (
            <p className="mt-1.5 text-xs text-rose-400">
              The backend rejects contexts longer than{" "}
              {MAX_CONTEXT_CHARS.toLocaleString()} characters.
            </p>
          ) : null}
        </div>

        <div>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <label htmlFor={questionId} className="text-sm font-medium text-slate-200">
              Question
            </label>
            <span
              className={`font-mono text-xs ${
                questionTooLong ? "text-rose-400" : "text-slate-500"
              }`}
            >
              {question.length.toLocaleString()} / {MAX_QUESTION_CHARS.toLocaleString()}
            </span>
          </div>
          <input
            id={questionId}
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What does the passage say about ...?"
            aria-invalid={questionTooLong}
            className="mt-2 w-full rounded-lg border border-slate-800 bg-slate-900/70 px-3.5 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 disabled:opacity-50"
          />
          {questionTooLong ? (
            <p className="mt-1.5 text-xs text-rose-400">
              The backend rejects questions longer than {MAX_QUESTION_CHARS} characters.
            </p>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="submit"
            disabled={!canSubmit}
            aria-busy={isLoading}
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-sky-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            {isLoading ? (
              <>
                <span
                  aria-hidden="true"
                  className="size-3.5 animate-spin rounded-full border-2 border-slate-950/30 border-t-slate-950"
                />
                Asking…
              </>
            ) : (
              "Ask Question"
            )}
          </button>
          {isLoading ? (
            <span aria-live="polite" className="text-xs text-slate-400">
              Running tokenization, the forward pass and span decoding…
            </span>
          ) : null}
        </div>
      </form>

      {errorMessage !== null ? (
        <div
          role="alert"
          className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/5 p-4"
        >
          <p className="text-sm font-semibold text-rose-300">Request failed</p>
          <p className="mt-1.5 text-sm leading-relaxed text-rose-200/90">
            {errorMessage}
          </p>
        </div>
      ) : null}

      {outcome !== null ? (
        <div className="mt-8" aria-live="polite">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
            Result
          </h3>

          {outcome.prediction.has_answer ? (
            <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/50 p-5">
              <p className="text-[0.6875rem] font-medium uppercase tracking-wider text-slate-500">
                Extracted answer
              </p>
              <p className="mt-2 text-xl font-semibold leading-snug text-white">
                {outcome.prediction.answer}
              </p>
              <p className="mt-2 font-mono text-xs text-slate-500">
                context[{outcome.prediction.char_start}:{outcome.prediction.char_end}]
              </p>
            </div>
          ) : (
            <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-5">
              <p className="text-sm font-semibold text-amber-300">No answer span</p>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-300">
                Every candidate span was rejected, so the model returned no
                answer rather than guessing. This usually means the passage does
                not contain the answer.
              </p>
            </div>
          )}

          <dl className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Metric
              label="has_answer"
              value={String(outcome.prediction.has_answer)}
            />
            <Metric
              label="latency"
              value={`${outcome.prediction.latency_ms.toFixed(2)} ms`}
              hint="Tokenization, forward pass and decoding"
            />
            <Metric label="num_windows" value={String(outcome.prediction.num_windows)} />
            <Metric
              label="truncated"
              value={outcome.prediction.truncated ? "truncated" : "not truncated"}
              hint={
                outcome.prediction.truncated
                  ? "Context exceeded one window; candidates pooled across windows"
                  : "Context fitted in a single window"
              }
            />
            <Metric
              label="score"
              value={outcome.prediction.score.toFixed(6)}
              hint="Not a confidence value"
            />
            <Metric label="score_type" value={outcome.prediction.score_type} />
          </dl>

          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            <span className="font-medium text-slate-400">On the score:</span> its
            meaning is given by{" "}
            <code className="font-mono">{outcome.prediction.score_type}</code> — a
            single softmax over the pooled valid candidate spans from every
            window. That is a proper distribution over the hypotheses the model
            considered, but it is <strong>not calibrated</strong> and is not a
            confidence: fine-tuned transformers are systematically overconfident,
            so 0.9 does not mean a 90% chance of being correct.
          </p>

          <div className="mt-6">
            <h4 className="text-sm font-semibold text-slate-200">
              Answer located in the context
            </h4>
            <div className="mt-2 max-h-72 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <HighlightedContext
                prediction={outcome.prediction}
                context={outcome.context}
              />
            </div>
          </div>

          {outcome.prediction.n_best.length > 1 ? (
            <details className="mt-5 rounded-xl border border-slate-800 bg-slate-900/40">
              <summary className="cursor-pointer px-4 py-3 text-sm font-medium text-slate-300">
                Ranked alternatives ({outcome.prediction.n_best.length})
              </summary>
              <ol className="border-t border-slate-800 px-4 py-3 text-sm">
                {outcome.prediction.n_best.map((entry, index) => (
                  <li
                    key={`${entry.char_start}-${entry.char_end}-${index}`}
                    className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-1.5"
                  >
                    <span className="min-w-0 break-words text-slate-300">
                      <span className="mr-2 font-mono text-xs text-slate-600">
                        {index + 1}
                      </span>
                      {entry.answer || <em className="text-slate-500">empty span</em>}
                    </span>
                    <span className="font-mono text-xs text-slate-500">
                      {entry.score.toFixed(6)} · [{entry.char_start}:{entry.char_end}]
                    </span>
                  </li>
                ))}
              </ol>
            </details>
          ) : null}

          <p className="mt-4 font-mono text-xs text-slate-600">
            model_id: {outcome.prediction.model_id}
          </p>
        </div>
      ) : null}
    </section>
  );
}
