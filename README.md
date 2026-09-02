# QAS-NLP — Extractive Question Answering on SQuAD 1.1

A from-scratch extractive question answering system: fine-tune a Transformer
encoder to locate the answer to a question **inside** a supplied passage, then
serve it through a local API and web interface.

> **Status: trained, selected and running locally.**
> All four candidates were fine-tuned on an NVIDIA L4. `microsoft/deberta-v3-base` was
> selected on measured results: **EM 88.35 / F1 93.93** on the full SQuAD 1.1 validation
> split (10,570 examples). That checkpoint now serves locally through FastAPI on
> `:8000` with the Next.js interface on `:3000`; see
> [Local deployment](#local-deployment).
>
> The GPU environment is no longer required. Lightning AI was the training environment
> only — local Windows is the inference and frontend environment.

See [`docs/pipeline.md`](docs/pipeline.md) for the full technical walkthrough.

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
  FastAPI :8000                              |          [COMPLETE]
  Next.js :3000                              |
        ^                                    v
        |                            artifacts/runs/<run_id>/
        |    lightning cp                    |
        +------------------------------------+
                                             |
                    git (code, configs, reports -- never weights)
```

The split is by capability, not preference: this machine has no CUDA device, so
training here is a non-option rather than a slow option. The division is now
settled in one direction — **training is finished**, the selected checkpoint has
been copied into `models/`, and inference runs entirely locally. Lightning is only
needed again if a model is retrained.

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
│   ├── qa_core/             shared span logic. STDLIB ONLY, no torch, no numpy
│   │   ├── normalize.py       official SQuAD answer normalization
│   │   ├── metrics.py         Exact Match, token-level F1
│   │   ├── spans.py           char span validation, tightening, extraction
│   │   ├── alignment.py       SQuAD char offsets -> token start/end positions
│   │   ├── postprocess.py     explicit n-best span decoding across windows
│   │   └── schemas.py         dataclass contracts
│   ├── qa_torch/            torch- and Hugging Face-dependent code
│   │   ├── device.py          device abstraction, CUDA diagnostics, precision plan
│   │   ├── loader.py          tokenizer / AutoModelForQuestionAnswering loading
│   │   ├── features.py        explicit sliding-window feature building
│   │   ├── engine.py          batched forward passes collecting start/end logits
│   │   └── inference.py       the reusable single-question engine
│   └── qa_ml/               training/evaluation orchestration
│       ├── config.py          typed YAML experiment configuration
│       ├── data.py            SQuAD loading, schema and offset verification
│       ├── preprocess.py      dataset feature building and column separation
│       ├── train.py           Trainer-backed training
│       ├── evaluate.py        logits -> decoded text -> EM/F1
│       ├── experiment.py      run directories and provenance records
│       ├── seeding.py         reproducible seeding
│       ├── environment.py     environment and git capture
│       ├── cli.py             prepare / train / evaluate / predict
│       ├── paths.py           repository-root and directory resolution
│       └── logging_utils.py   logging setup
│
├── ml/
│   ├── configs/             experiment YAML (base + smoke + 4 experiments)
│   ├── requirements*.txt    ML deps; -cpu for local, -gpu for the L4
│   └── scripts/
│       └── check_environment.py   diagnostics and the pre-training CUDA gate
│
├── backend/                 FastAPI service: GET /health, POST /predict
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

> **`models/` must never be committed.** The `.gitignore` rule is `models/*` with
> only `!models/.gitkeep` negated back in, so the *entire* directory is excluded —
> not just weight extensions. That matters: a rule limited to `*.safetensors` would
> still let `config.json` and `tokenizer.json` through. The same applies to
> `artifacts/*`. Weights are distributed through the Lightning model registry or the
> Hugging Face Hub, never through Git.

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
| D | `microsoft/deberta-v3-base` | 183,833,090 | SentencePiece | 128,100 | **Selected**; served locally |

A → B is the only clean pairwise comparison: identical tokenizer and vocabulary,
so the difference isolates encoder depth (6 vs 12 layers). B → C changes
pretraining *and* tokenizer simultaneously; that confound is acknowledged rather
than glossed over.

Experiment D publishes no safetensors metadata on the Hub, so its parameter count
had to be measured rather than quoted. Measured by instantiating
`DebertaV2ForQuestionAnswering` from its published `config.json`:
**183,833,090** parameters, confirmed again by reading the tensor shapes out of the
served checkpoint's own safetensors header (200 tensors, all `F32`). The embedding
matrix alone is 128,100 × 768 = 98,380,800 parameters — 53.5% of the model, and
more than DistilBERT in its entirety.

One consequence worth recording: `microsoft/deberta-v3-base` publishes **fp16**
weights, while A, B and C publish fp32. Loading fp16 and then training makes those
tensors the optimizer's master weights, and AdamW at `eps=1e-8` cannot survive that
— `exp_avg_sq` underflows fp16 to zero and the update divides by zero. `load_qa_model`
therefore pins `dtype=torch.float32` at load time; `tests/test_loader.py` locks the
behaviour down.

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

## Local deployment

The full system runs on this machine with no GPU and no cloud dependency. Inference
is CPU-only: measured **620–700 ms per request** warm, for both a 51-character and a
1,756-character context, with the first request after startup around 2 s while
threads warm up. Model load takes a couple of seconds and happens once at startup,
not per request.

### 1. Place the trained checkpoint

The serving checkpoint lives in `models/`, which is git-ignored. It is produced by
training (`artifacts/runs/<run_id>/model/`) and copied down with `lightning cp`; see
[`docs/lightning-workflow.md`](docs/lightning-workflow.md).

```
models/deberta-v3-base-squad/
├── config.json                 DebertaV2ForQuestionAnswering
├── model.safetensors           735,356,752 B — 183,833,090 params, all F32
├── tokenizer.json              serialized fast tokenizer
├── tokenizer_config.json
└── training_args.bin           written by Trainer; not read at inference
```

Verify the copy before trusting it. A truncated 700 MB transfer otherwise surfaces
later as a confusing load error:

```powershell
Get-FileHash "models\deberta-v3-base-squad\model.safetensors" -Algorithm SHA256
# f42b44e4904f132c18512c70c5c438a6b2daaf551d127b3f6213ff208a39de1f
```

Because `save_pretrained` wrote `tokenizer.json`, no SentencePiece→fast conversion
happens at load time, so `sentencepiece`/`protobuf` are not strictly needed to serve
this directory. They stay pinned because building features from the base model does
need them.

### 2. Point the backend at it

`QAS_MODEL_PATH` is read by `backend/app/config.py` (`Settings`, env prefix `QAS_`)
and consumed once at startup in `lifespan()`, which constructs a single
`ExtractiveQAEngine` reused for every request.

```powershell
$env:QAS_MODEL_PATH = "D:\QAS NLP\models\deberta-v3-base-squad"
```

Or persist it in a git-ignored `.env` at the repository root — copy `.env.example`
to `.env`. Do not quote the value; `pydantic-settings` would keep the quotes as part
of the string. The value may be a local directory **or** a Hugging Face repo id.

Behaviour when it is not set is deliberate rather than accidental:

| `QAS_MODEL_PATH` | `/health` | `POST /predict` |
|---|---|---|
| unset | `model_loaded: false` | **503**, `"QA model is not loaded. Configure QAS_MODEL_PATH."` |
| set and loadable | `model_loaded: true`, `model_id` echoes the path | **200** with a prediction |
| set but unloadable | startup **fails loudly** | — |

A wrong path is never served silently as a working model.

### 3. Run the backend

Start it **from the repository root**:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

`--app-dir backend` puts `backend/` on `sys.path` so `app.main` resolves; `qa_core`
and `qa_torch` resolve through the editable install (`pip install -e .`). Note that
`--app-dir` does **not** change the working directory, and `Settings.env_file` is the
relative path `.env` — so `.env` is read from wherever you launched the process.
Launching from `backend/` would silently look for `backend/.env` instead.

Then: <http://localhost:8000/health> · interactive docs at <http://localhost:8000/docs>

### 4. Run the frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local     # sets NEXT_PUBLIC_API_URL
npm run dev
```

Then: <http://localhost:3000>

`NEXT_PUBLIC_API_URL` has no hard-coded localhost fallback in source. If it is unset
the interface says so and keeps the form disabled, rather than appearing to work
while pointing at nothing. `.env.local` is git-ignored and must not be committed.

### The API

`POST /predict` — request and response are defined in `backend/app/schemas.py` and
published at `/openapi.json`, which is the authoritative contract.

```jsonc
// request  (PredictRequest)  -- both fields required, min length 1
{ "question": "Who developed the theory of relativity?",
  "context":  "Albert Einstein developed the theory of relativity." }
```

```jsonc
// response (PredictionResponse) -- an actual response from the local service
{ "answer": "Albert Einstein",
  "char_start": 0,
  "char_end": 15,
  "score": 0.997386,
  "score_type": "uncalibrated_span_probability",  // see "On confidence"
  "latency_ms": 668.2,
  "num_windows": 1,
  "model_id": "D:\\QAS NLP\\models\\deberta-v3-base-squad",
  "truncated": false,        // true when the context needed >1 window
  "has_answer": true,        // false when every candidate span was rejected
  "n_best": [ /* 10 ranked alternatives, best first */ ] }
```

`char_start`/`char_end` index the **submitted context verbatim**, which is what lets
the interface highlight the answer in place. Input limits are enforced server-side:
512 characters for the question, 20,000 for the context, both returning 422.

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"Who developed the theory of relativity?\",\"context\":\"Albert Einstein developed the theory of relativity.\"}'
```

Verified end to end on this machine: `/health` reports `model_loaded: true`, the
request above returns `"Albert Einstein"`, and the frontend highlights that span
inside the passage it was found in.

## Testing

```powershell
pytest                              # everything
pytest tests/                       # src/ packages only
pytest backend/tests/               # API only
pytest -m "not slow"                # skip slow tests
ruff check .                        # lint
```

**701 passed, 17 skipped** (all skips are CUDA-only tests on this CPU-only machine).

| Area | Verifies |
|---|---|
| `test_normalize.py` | SQuAD normalization: case, punctuation, articles, whitespace, ordering |
| `test_metrics.py` | EM and F1, multiset token overlap, multi-gold maximum, empty-string edges |
| `test_spans.py` | Span validation, whitespace tightening, per-tokenizer offset regressions |
| `test_alignment.py` | Character-to-token alignment, partial-token answers, answers outside a window |
| `test_features.py` | Real-tokenizer feature building across WordPiece/BPE, **window coverage regression**, labels decoding back to gold answers |
| `test_postprocess.py` | Span decoding: shortlisting, validity filtering, pooled softmax, determinism |
| `test_evaluate.py` | Feature-to-example regrouping, EM/F1 from logits, **cross-check against the `evaluate` library** |
| `test_data.py` | SQuAD schema, offset verification, SQuAD 2.0 rejection |
| `test_train_setup.py` | Config-to-`TrainingArguments` adapter, `warmup_ratio` → float `warmup_steps`, CUDA gate |
| `test_experiment.py` | Run directories never overwritten, complete provenance records |
| `test_seeding.py` | Reproducible Python/NumPy/torch streams |
| `test_gpu_smoke.py` | CUDA detection, device placement, forward/backward/optimizer step, bf16, peak memory |
| `test_inference.py` | End-to-end engine, long-context windowing, **evaluation/inference decoding parity** |
| `test_config.py` | YAML loading, inheritance, strict key rejection, hash determinism, controlled-variable equality |
| `test_device.py` | Device resolution, no hard-coded `cuda:0`, no silent CPU downgrade |
| `test_qa_core_isolation.py` | `qa_core` imports no heavy dependency (clean subprocess) |
| `test_project_structure.py` | Layout, path robustness, `.gitignore` correctness, installed-versus-pinned versions |
| `test_check_environment.py` | Diagnostics run on CPU, report CUDA truthfully, gate correctly |
| `test_loader.py` | Checkpoint round-trip, legacy LayerNorm key mapping, **fp16 checkpoints load as fp32 master weights**, AdamW underflow regression, non-finite checkpoint rejection |
| `backend/tests/test_health.py` | `/health` contract, `POST /predict` contract and validation, CORS allow-list, OpenAPI schema |

## Pipeline commands

```powershell
# Validate the dataset and report feature/alignment statistics
python -m qa_ml prepare --config smoke.yaml --build-features

# Train (refuses CPU by default; --allow-cpu only for tiny smoke configs)
python -m qa_ml train --config experiment_a_distilbert.yaml --cross-check

# Evaluate a checkpoint
python -m qa_ml evaluate --config experiment_a_distilbert.yaml \
    --model artifacts/runs/<run_id>/model

# Answer one question
python -m qa_ml predict --config experiment_a_distilbert.yaml \
    --model artifacts/runs/<run_id>/model \
    --question "Which country contains the majority of the Amazon rainforest?" \
    --context "The Amazon rainforest ... contained within Brazil ..."
```

Against the served local checkpoint, without starting the API:

```powershell
python -m qa_ml predict --config experiment_d_deberta_v3.yaml `
    --model "models\deberta-v3-base-squad" `
    --question "Who developed the theory of relativity?" `
    --context "Albert Einstein developed the theory of relativity."
```

This uses the same `ExtractiveQAEngine` and the same `decode_spans` call as the
backend, so it is a way to check the checkpoint independently of HTTP.

Any config value can be overridden without editing files:
`--set training.num_train_epochs=3 --set training.per_device_train_batch_size=32`

## Phase plan

Phase numbers below match the commit history (`git log --oneline`), which is the
authoritative record.

| Phase | Description | Status |
|---:|---|---|
| 0 | Architecture and repository audit | Complete |
| 1 | Project foundation and reproducible environment | Complete |
| 2 | Dataset, preprocessing, training, evaluation, decoding, CLI | Complete |
| 2.1 | Tokenizer and checkpoint integrity audit | Complete |
| 2.2 | DeBERTa tokenizer dependency pins and dataset evidence | Complete |
| 3–7 | L4 GPU validation, four training experiments, comparison and selection | Complete |
| 13 | FastAPI inference backend (`/health`, `/predict`) | Complete |
| **14** | **Question answering interface (Next.js)** | **Complete** |
| — | Local deployment and documentation | Complete |
| — | Error analysis, inference optimization, hosted deployment | Not started |

Each phase has a quality gate that must pass, with recorded evidence, before the
next begins. One gate caught a real defect worth recording: the fp16 master-weight
failure described under [Model candidates](#model-candidates) was found by the
checkpoint-integrity check, not by a metric.

## Results

Selected model: **`microsoft/deberta-v3-base`**, evaluated on the full SQuAD 1.1
validation split — **10,570 examples / 10,756 features**. The denominator is quoted
because a metric without its sample size is an anecdote.

| Metric | Training run | Persisted checkpoint, re-evaluated |
|---|---:|---:|
| Exact Match | 88.35 | 88.3538 |
| F1 | 93.93 | 93.9110 |

The second column is the deployment-relevant one: it is the checkpoint in `models/`
re-scored after being written to disk and reloaded, rather than the model still held
in memory at the end of training. Exact Match agrees to four decimal places; F1
differs by 0.019. Both are reported rather than only the better one, and the cause of
that delta has not been established, so it is not explained away here.

Decoding is shared code, not a reimplementation: evaluation and the serving backend
both call `qa_core.postprocess.decode_spans` with the same parameters, and
`tests/test_inference.py` asserts that parity. Without it, a published metric would
not describe the deployed system.

**Provenance caveat, stated plainly.** These figures come from the experiment records
produced on the GPU environment, which are **not committed to this repository** —
`reports/` currently holds only `README.md` and `dataset_preparation.json`. They are
therefore not independently verifiable from a fresh clone. Committing
`reports/<run_id>/metrics.json` for each experiment would close that gap and let the
comparison table below be generated rather than written.

<!-- BEGIN GENERATED: comparison -->
<!-- Not generated: per-run metrics.json files are not yet committed to reports/. -->
<!-- END GENERATED: comparison -->

### Dataset preparation evidence

`reports/dataset_preparation.json` **is** committed, and is the full-split
preparation record for the selected model (`sample_capped: false`):

| Split | Examples | Features | Aligned | Answer outside window | No aligned window |
|---|---:|---:|---:|---:|---:|
| train | 87,599 | 88,316 | 87,734 (99.34%) | 582 | **0** |
| validation | 10,570 | 10,756 | 10,610 (98.64%) | 146 | **0** |

Zero examples lost every window, and `no_context_tokens` and `degenerate_answer` are
both zero on both splits — the character-to-token alignment described above holds
across the entire dataset, not just the test fixtures.

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

## Hosted deployment

Not done, and no deployment-specific code exists. Local serving works end to end
(see [Local deployment](#local-deployment)); hosting it is a separate concern.

The architecture keeps it straightforward: Next.js → Vercel, FastAPI → any container
host, model → Hugging Face Hub. Two properties make the serving target swappable
without touching QA logic — all decoding lives in `qa_core`, and nothing is
hard-coded: the backend reads `QAS_MODEL_PATH` and `QAS_ALLOWED_ORIGINS` from the
environment, and the frontend reads `NEXT_PUBLIC_API_URL` with no localhost fallback
in source.

The one thing that would need attention first is CPU inference latency at ~184M
parameters, which is acceptable for a single local user and would not be for
concurrent traffic.

## Documentation

- [`docs/pipeline.md`](docs/pipeline.md) — the full technical walkthrough, tokenization to metrics
- [`docs/dependencies.md`](docs/dependencies.md) — every version pin and its justification
- [`docs/dataset-policy.md`](docs/dataset-policy.md) — SQuAD handling and offset preservation rules
- [`docs/lightning-workflow.md`](docs/lightning-workflow.md) — the GPU training workflow and how the
  checkpoint was produced and transferred. Historical: training is complete and this
  workflow is not needed for local inference.

## License

MIT
