import { apiBaseUrl, isApiConfigured } from "@/lib/config";

/**
 * Phase 1 landing page.
 *
 * Scope is deliberately narrow: identify the project, explain what extractive
 * question answering is, and report build status honestly.
 *
 * Deliberately absent:
 *  - no context or question inputs (Phase 14)
 *  - no answer card, no score, no latency, no highlighted span
 *  - no network calls
 *
 * No model has been trained, so any answer shown here would be fabricated.
 * Rendering an empty answer card would misrepresent how far along the project is.
 */

const PIPELINE_STAGES = [
  "Context + Question",
  "Tokenizer (offset mapping retained)",
  "Transformer encoder",
  "Start logits + End logits",
  "Candidate span enumeration",
  "Validity filtering",
  "Span scoring",
  "Token span → character span",
  "Answer = context[char_start:char_end]",
] as const;

const PHASES = [
  { id: 0, name: "Architecture and repository audit", state: "done" },
  { id: 1, name: "Project foundation and reproducible environment", state: "done" },
  { id: 2, name: "Lightning L4 GPU environment validation", state: "next" },
  { id: 3, name: "SQuAD dataset exploration", state: "todo" },
  { id: 4, name: "Tokenization and answer-span preprocessing", state: "todo" },
  { id: 6, name: "DistilBERT smoke training", state: "todo" },
  { id: 7, name: "Model training experiments (4 candidates)", state: "todo" },
  { id: 11, name: "Model comparison and selection", state: "todo" },
  { id: 13, name: "FastAPI inference backend", state: "todo" },
  { id: 14, name: "Question answering interface", state: "todo" },
] as const;

function StateBadge({ state }: { state: "done" | "next" | "todo" }) {
  const styles = {
    done: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
    next: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
    todo: "bg-slate-500/10 text-slate-400 ring-slate-500/20",
  } as const;
  const labels = { done: "complete", next: "next", todo: "planned" } as const;
  return (
    <span
      className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${styles[state]}`}
    >
      {labels[state]}
    </span>
  );
}

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-4xl px-5 py-12 sm:px-8 sm:py-16">
      <header>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400">
          SQuAD 1.1 · Transformer span prediction
        </p>
        <h1 className="mt-3 text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Extractive Question Answering System
        </h1>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-slate-300">
          Ask a question about a passage and the model returns the exact span of
          that passage which answers it. The answer is{" "}
          <strong className="font-semibold text-white">extracted</strong>, never
          generated, so it cannot contain text that is absent from the source.
        </p>
      </header>

      <section
        aria-labelledby="status-heading"
        className="mt-10 rounded-xl border border-amber-500/30 bg-amber-500/5 p-5"
      >
        <h2 id="status-heading" className="text-sm font-semibold text-amber-300">
          Under construction — Phase 1 of 17 complete
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">
          This is the project foundation only. No model has been trained yet, so
          the question answering interface is not present and no accuracy figures
          are shown. Nothing on this page is a model output.
        </p>
      </section>

      <section aria-labelledby="pipeline-heading" className="mt-12">
        <h2
          id="pipeline-heading"
          className="text-lg font-semibold tracking-tight text-white"
        >
          How extractive QA works
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Rather than producing new text, the model predicts two distributions
          over the input tokens: where the answer starts and where it ends. The
          answer is a position, which is why the result can be highlighted inside
          the original passage.
        </p>
        <ol className="mt-6 space-y-px">
          {PIPELINE_STAGES.map((stage, index) => (
            <li
              key={stage}
              className="flex items-center gap-3 rounded-md bg-slate-900/60 px-4 py-2.5 text-sm text-slate-300"
            >
              <span className="w-5 shrink-0 text-right font-mono text-xs text-slate-500">
                {index + 1}
              </span>
              <span>{stage}</span>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="roadmap-heading" className="mt-12">
        <h2
          id="roadmap-heading"
          className="text-lg font-semibold tracking-tight text-white"
        >
          Build progress
        </h2>
        <ul className="mt-5 space-y-2">
          {PHASES.map((phase) => (
            <li
              key={phase.id}
              className="flex items-center justify-between gap-4 rounded-md bg-slate-900/60 px-4 py-2.5"
            >
              <span className="flex min-w-0 items-baseline gap-3 text-sm">
                <span className="shrink-0 font-mono text-xs text-slate-500">
                  {String(phase.id).padStart(2, "0")}
                </span>
                <span className="truncate text-slate-300">{phase.name}</span>
              </span>
              <StateBadge state={phase.state} />
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="config-heading" className="mt-12">
        <h2
          id="config-heading"
          className="text-lg font-semibold tracking-tight text-white"
        >
          Configuration
        </h2>
        <dl className="mt-5 overflow-hidden rounded-md bg-slate-900/60 text-sm">
          <div className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-center sm:gap-4">
            <dt className="w-56 shrink-0 font-mono text-xs text-slate-500">
              NEXT_PUBLIC_API_URL
            </dt>
            <dd className="min-w-0 break-all">
              {isApiConfigured ? (
                <span className="text-slate-300">{apiBaseUrl}</span>
              ) : (
                <span className="text-amber-300">
                  not configured — copy{" "}
                  <code className="font-mono text-xs">.env.example</code> to{" "}
                  <code className="font-mono text-xs">.env.local</code>
                </span>
              )}
            </dd>
          </div>
        </dl>
        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          The backend URL comes from the environment with no localhost fallback in
          source. This page does not call the API; connectivity is wired up in
          Phase 15.
        </p>
      </section>

      <footer className="mt-16 border-t border-slate-800 pt-6 text-xs text-slate-500">
        Trained on SQuAD 1.1 · Next.js frontend, FastAPI backend, PyTorch model
      </footer>
    </main>
  );
}
