# QAS-NLP — Extractive Question Answering on SQuAD 1.1

A from-scratch extractive question answering system: fine-tune a Transformer
encoder to locate the answer to a question **inside** a supplied passage, then
serve it through a local API and web interface.

> **Status: Phase 1 of 17 complete — project foundation.**
> No model has been trained yet. **This document contains no accuracy figures,
> because none have been measured.** Every metric published here later will be
> generated from saved experiment records, never typed by hand.

---

## Problem statement

Given a context passage and a question about it, return the exact span of the
passage that answers the question.

```
Context:  "The Amazon rainforest ... The majority of the forest is contained
           within Brazil, with 60 percent of the rainforest, followed by Peru
           with 13 percent."

Question: "Which country contains the majority of the Amazon rainforest?"

Answer:   "Brazil"        characters 290-296 of the context
```

The answer is **extracted**, not generated. The system cannot return text that is
absent from the context, which makes its output verifiable by construction: every
answer is a pointer into the source passage.

## What "extractive" means here

A generative model answers by producing new tokens, and can state things the
source never said. An extractive model instead predicts **two probability
distributions over the input tokens**: one for where the answer starts, one for
where it ends. The answer is a *span*, identified by position.

```
Context + Question
        |
        v  tokenizer (offset mapping retained)
        |
        v  Transformer encoder
        |
   start logits  +  end logits
        |
        v  candidate span enumeration
        v  validity filtering  (end >= start, inside the context,
        |                       within max answer length)
        v  span scoring
        |
        v  best token span
        v  token span -> character span
        |
        v  answer text = context[char_start:char_end]
```

Two consequences drive the whole design:

1. **Hallucination is structurally impossible.** The output is a pair of indices.
2. **Character offsets are the real product.** They let the UI highlight the exact
   answer inside the original passage, which is what visibly demonstrates that
   the system is extractive.

### The hard part: character offsets versus token positions

SQuAD annotates answers as `{"text": "Brazil", "answer_start": 290}`, where
`answer_start` is a **character** offset. A Transformer needs **token** indices.
Converting between them is where extractive QA implementations usually go wrong,
and the failure is silent: training loss decreases normally while the labels are
subtly incorrect.

This project treats that conversion as the primary correctness risk. It is
handled with the tokenizer's `offset_mapping`, `sequence_ids()` and
`overflow_to_sample_mapping`, and it is covered by a dedicated test suite. See
[`docs/dataset-policy.md`](docs/dataset-policy.md).

A concrete example of why this needs tests — measured across the four candidate
tokenizers for the answer `"Brazil"` at characters `(290, 296)`:

| Model | Tokenizer | Raw offsets | Slice of raw context | `decode()` |
|---|---|---|---|---|
| `distilbert-base-uncased` | WordPiece | (290, 296) | `'Brazil'` | `'brazil'` |
| `bert-base-uncased` | WordPiece | (290, 296) | `'Brazil'` | `'brazil'` |
| `roberta-base` | BPE | (290, 296) | `'Brazil'` | `' Brazil'` |
| `microsoft/deberta-v3-base` | SentencePiece | **(289, 296)** | **`' Brazil'`** | `'Brazil'` |

DeBERTa's SentencePiece offsets include the preceding space. SQuAD normalization
strips whitespace, so **Exact Match and F1 would not detect this** — it would
surface only as an answer highlighted one character early in the UI.
`qa_core.spans.tighten_char_span` exists for exactly this reason, and
`tests/test_spans.py` locks the behaviour down.

## Architecture

```
LOCAL PC (Windows, CPU-only)        LIGHTNING AI STUDIO (NVIDIA L4)
  authoring + fast tests              training + evaluation
  FastAPI :8000                              |
  Next.js :3000                              |
        ^                                    v
        |                            artifacts/runs/<run_id>/
        |    lightning cp                    |
        +------------------------------------+
                                             |
                    git (code, configs, reports -- never weights)
```

The load-bearing decision:

```
                  qa_core   (no torch, no fastapi)
                 /        \
    training evaluation    inference backend
```

`qa_core` holds all span-decoding logic and is imported by **both** the
evaluation pipeline and the serving backend. If those two paths could diverge,
reported metrics would not describe the deployed system. The boundary is enforced
mechanically: `tests/test_qa_core_isolation.py` runs a clean interpreter and
fails if importing `qa_core` pulls in torch, transformers, datasets or fastapi.

## Repository structure

```
.
├── constraints.txt          SINGLE SOURCE OF VERSION TRUTH (all pins)
├── pyproject.toml           packaging, pytest and ruff configuration
├── requirements-dev.txt     pytest, httpx, ruff
│
├── src/
│   ├── qa_core/             shared span logic. STDLIB ONLY, no torch
│   │   ├── normalize.py       official SQuAD answer normalization
│   │   ├── metrics.py         Exact Match, token-level F1
│   │   ├── spans.py           char span validation, tightening, extraction
│   │   └── schemas.py         dataclass contracts
│   ├── qa_torch/            torch-dependent code
│   │   └── device.py          device abstraction, CUDA diagnostics
│   └── qa_ml/               training/evaluation orchestration
│       ├── config.py          typed YAML experiment configuration
│       ├── paths.py           repository-root and directory resolution
│       └── logging_utils.py   logging setup
│
├── ml/
│   ├── configs/             experiment YAML (base + smoke + 4 experiments)
│   ├── requirements*.txt    ML deps; -cpu for local, -gpu for the L4
│   └── scripts/
│       └── check_environment.py   diagnostics and the pre-training CUDA gate
│
├── backend/                 FastAPI service (Phase 1: GET /health only)
│   ├── app/
│   └── tests/
│
├── frontend/                Next.js + TypeScript interface
│
├── tests/                   test suite for the src/ packages
├── docs/                    dependency rationale, dataset policy, workflow
├── reports/                 COMMITTED: metrics JSON, generated comparisons
├── artifacts/               git-ignored: checkpoints, logs, tokenized caches
├── models/                  git-ignored: selected checkpoint for serving
└── data/                    git-ignored: Hugging Face dataset cache
```

Deviations from a conventional layout, and why:

- **`src/` packages instead of loose `ml/` subdirectories.** Makes `qa_core`'s
  independence enforceable by import rules rather than by convention.
- **`reports/` committed, `artifacts/` not.** Metrics JSON is small, diffable and
  is the evidence trail. Checkpoints are neither.
- **One `tests/` root for all `src/` packages**, plus `backend/tests/`. Splitting
  tests for the same package across two roots would be arbitrary.

## Model candidates

Four candidates will be compared **on measured results only**. Parameter counts
below are read from Hub safetensors metadata, not recalled:

| Exp | Model | Parameters | Tokenizer | Vocab | Role |
|---|---|---:|---|---:|---|
| A | `distilbert-base-uncased` | 66,985,530 | WordPiece | 30,522 | Baseline; 6 layers |
| B | `bert-base-uncased` | 110,106,428 | WordPiece | 30,522 | Depth ablation vs A (same tokenizer) |
| C | `roberta-base` | 124,697,947 | BPE | 50,265 | Different pretraining + tokenizer family |
| D | `microsoft/deberta-v3-base` | *to be measured* | SentencePiece | 128,100 | Optional, budget-gated |

A → B is the only clean pairwise comparison: identical tokenizer and vocabulary,
so the difference isolates encoder depth (6 vs 12 layers). B → C changes
pretraining *and* tokenizer simultaneously; that confound is acknowledged rather
than glossed over.

Experiment D publishes no safetensors metadata on the Hub, so its parameter count
must be measured locally before being reported. From its `config.json`, the
embedding matrix alone is 128,100 × 768 = 98,380,800 parameters — more than
DistilBERT's entire model.

Selection criteria, all measured by us: Exact Match, F1, validation loss,
training time, inference latency, parameter count, model size, peak GPU memory,
throughput. The largest model does not win by default.

## Local setup

Prerequisites: Python 3.12, Node.js 20+, Git.

```powershell
# 1. Create an isolated environment (leaves any existing envs untouched)
conda create -y -n qas-nlp python=3.12
conda activate qas-nlp

# 2. Install pinned dependencies (CPU torch locally)
pip install -r ml/requirements-cpu.txt
pip install -r backend/requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# 3. Verify the environment
python ml/scripts/check_environment.py --tensor-test

# 4. Run the test suite
pytest
```

On Windows, set `PYTHONIOENCODING=utf-8` first. Without it, tools that print
non-cp1252 characters (`lightning`, `hf`, `tqdm`) crash with
`UnicodeEncodeError` in PowerShell.

### Run the backend

```powershell
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

Then: <http://localhost:8000/health> · docs at <http://localhost:8000/docs>

### Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Then: <http://localhost:3000>

## Testing

```powershell
pytest                              # everything
pytest tests/                       # src/ packages only
pytest backend/tests/               # API only
pytest -m "not slow"                # skip slow tests
ruff check .                        # lint
```

What is covered in Phase 1 — all real behaviour, no placeholder assertions:

| Area | Verifies |
|---|---|
| `test_normalize.py` | SQuAD normalization: case, punctuation, articles, whitespace, ordering |
| `test_metrics.py` | EM and F1, multiset token overlap, multi-gold maximum, empty-string edges |
| `test_spans.py` | Span validation, whitespace tightening, per-tokenizer offset regressions |
| `test_config.py` | YAML loading, inheritance, strict key rejection, hash determinism, controlled-variable equality across experiments |
| `test_device.py` | Device resolution, no hard-coded `cuda:0`, no silent CPU downgrade |
| `test_qa_core_isolation.py` | `qa_core` imports no heavy dependency (clean subprocess) |
| `test_project_structure.py` | Layout, path robustness, `.gitignore` correctness, installed-versus-pinned versions |
| `test_check_environment.py` | Diagnostics run on CPU, report CUDA truthfully, gate correctly |
| `backend/tests/test_health.py` | `/health` contract, CORS allow-list, absence of `/predict` |

## Phase plan

| Phase | Description | Status |
|---:|---|---|
| 0 | Architecture and repository audit | Complete |
| **1** | **Project foundation and reproducible environment** | **Complete** |
| 2 | Lightning L4 GPU environment validation | Not started |
| 3 | SQuAD dataset exploration | Not started |
| 4 | Tokenization and answer-span preprocessing | Not started |
| 5 | Preprocessing unit tests | Not started |
| 6 | DistilBERT smoke training | Not started |
| 7 | DistilBERT full training | Not started |
| 8 | DistilBERT evaluation | Not started |
| 9 | BERT-base experiment | Not started |
| 10 | Optional third-model experiment | Not started |
| 11 | Model comparison | Not started |
| 12 | Final inference engine | Not started |
| 13 | FastAPI backend (full) | Not started |
| 14 | Next.js frontend (full) | Not started |
| 15 | Frontend/backend integration | Not started |
| 16 | End-to-end local testing | Not started |
| 17 | Documentation and final audit | Not started |

Each phase has a quality gate that must pass, with recorded evidence, before the
next begins.

## Results

*Empty by design.* Populated from `reports/*/metrics.json` once training has
actually run. The comparison table will be generated by tooling, so hand-typed
numbers cannot appear here.

<!-- BEGIN GENERATED: comparison -->
<!-- No experiments have been run. -->
<!-- END GENERATED: comparison -->

## On confidence

The API reports a `score` alongside `score_type`, and the distinction is
deliberate.

The usual approach multiplies `softmax(start)[i]` by `softmax(end)[j]`. That is a
product of two independent marginals, **not** a distribution over spans:
probability mass sits on spans that get rejected as invalid, and with sliding
windows each window has its own normalizer, so scores are not comparable across
windows.

Instead, valid candidate spans are pooled across **all** windows of an example
and a **single** softmax is applied over that pooled set. That yields a proper
distribution over the hypotheses actually considered.

It is still **uncalibrated**, and is labelled
`uncalibrated_span_probability`. Fine-tuned transformers are systematically
overconfident; a score of 0.9 does not mean a 90% chance of being correct.
Relabelling to `temperature_scaled` requires actually fitting a temperature on
held-out data and measuring the calibration error.

## Future deployment

Out of scope for now, and no deployment-specific code exists. The architecture
keeps it straightforward later: Next.js → Vercel, FastAPI → any container host,
model → Hugging Face Hub. All inference logic sits in `qa_core` plus a thin
FastAPI layer, so the serving target can change without touching QA logic.

## Documentation

- [`docs/dependencies.md`](docs/dependencies.md) — every version pin and its justification
- [`docs/dataset-policy.md`](docs/dataset-policy.md) — SQuAD handling and offset preservation rules
- [`docs/lightning-workflow.md`](docs/lightning-workflow.md) — the GPU training workflow

## License

MIT
