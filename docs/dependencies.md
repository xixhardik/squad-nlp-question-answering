# Dependency selection and rationale

Every version in [`constraints.txt`](../constraints.txt) was chosen deliberately.
This document records why. Versions were verified against PyPI release metadata
on 2026-08-29, not recalled from memory.

## How the layering works

```
constraints.txt          <- single source of version truth (all pins live here)
├── requirements-dev.txt        -c constraints.txt   pytest, httpx, ruff
├── ml/requirements.txt         -c ../constraints.txt  HF stack, no torch build
│   ├── ml/requirements-cpu.txt  -r requirements.txt + torch   (local dev)
│   └── ml/requirements-gpu.txt  -r requirements.txt + torch   (Lightning L4)
└── backend/requirements.txt    -c ../constraints.txt  fastapi, uvicorn, pydantic
```

Every requirements file references `constraints.txt` with `-c`. That is what
prevents the ML environment and the backend environment from resolving different
versions of a shared package. `tests/test_project_structure.py` asserts the
reference exists in each file, and separately asserts the *installed* versions
match the pins, so environment drift fails a test instead of silently changing
behaviour.

## PyTorch: `torch==2.13.0`

The pin is identical for local development and for the GPU environment. That
looks surprising, so it is worth being explicit about why it works. From PyPI
release metadata for 2.13.0, `cp312` wheels:

| Platform | Wheel | Size | Build |
|---|---|---:|---|
| Windows x86_64 | `torch-2.13.0-cp312-cp312-win_amd64.whl` | 122.1 MB | CPU-only |
| Linux x86_64 | `torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl` | 526.6 MB | CUDA-enabled |

So `pip install torch==2.13.0` yields a CPU build on the Windows dev machine and
a CUDA build on the Linux Lightning Studio, with no `--index-url` juggling and no
platform branching in the requirements files.

This matters beyond convenience. **Tokenizer and library versions must be
identical across both machines.** Offset mappings are the foundation of the whole
preprocessing pipeline; if `tokenizers` differed between local and remote, the
alignment tests verified locally would say nothing about the model actually
trained on the GPU. Keeping one pin set with only the torch *build* varying is
what makes local verification meaningful.

CUDA availability is never assumed. It is asserted by
`python ml/scripts/check_environment.py --require-cuda`, the Phase 2 gate.

## Hugging Face stack

| Package | Pin | Why |
|---|---|---|
| `transformers` | `5.16.1` | Current 5.x release. **v5 is required by this project**, see below. |
| `tokenizers` | `0.23.1` | `transformers 5.16.1` declares `tokenizers<0.24.0,>=0.23.1`. Pinned at the floor of that window. |
| `datasets` | `5.0.1` | SQuAD 1.1 is parquet-backed on the Hub, so no dataset loading script is involved. |
| `evaluate` | `0.4.6` | Used **only** to cross-check our own EM/F1. Never the primary implementation. |
| `accelerate` | `1.14.0` | `transformers.Trainer` delegates to it. Declares `torch>=2.0.0`, satisfied by 2.13.0. |

### Why transformers v5 specifically

The `question-answering` pipeline was **removed** in v5. Verified by execution:

```
>>> pipeline("question-answering")
KeyError: "Unknown task question-answering, available tasks are
['any-to-any', 'audio-classification', ..., 'document-question-answering', ...]"
```

That removal is a feature for this project. The requirement is to implement
start/end span decoding explicitly, so the shortcut was never going to be used.
Pinning v5 makes the shortcut unavailable rather than merely discouraged.

The trade-off, recorded honestly: we lose `pipeline("question-answering")` as a
convenient independent oracle for cross-checking our decoder. The mitigation is
to cross-check EM/F1 against `evaluate.load("squad")` and against hand-verified
examples.

### v5 API differences that affect this project

The migration guide on the `main` branch runs ahead of the released package, so
the surface was introspected directly rather than trusted. Confirmed on 5.7.0:

| Symbol | Status |
|---|---|
| `TrainingArguments.eval_strategy` | present |
| `TrainingArguments.evaluation_strategy` | **removed** |
| `TrainingArguments.warmup_ratio`, `warmup_steps` | both present (`warmup_step` does not exist) |
| `TrainingArguments.use_cpu` | present (`no_cuda` removed) |
| `TrainingArguments.group_by_length` | **removed** |
| `Trainer(processing_class=...)` | present (`tokenizer=` removed) |
| `AutoModelForQuestionAnswering` | present |

This project's own config vocabulary keeps `evaluation_strategy` (see
`ml/configs/base.yaml`). The rename to `eval_strategy` is handled at the Trainer
boundary by an adapter, so a library rename cannot silently invalidate committed
experiment records. This is documented in `src/qa_ml/config.py`.

## Backend

| Package | Pin | Why |
|---|---|---|
| `fastapi` | `0.141.1` | Declares `pydantic>=2.9.0` and `starlette>=0.46.0`, both satisfied. |
| `uvicorn` | `0.52.4` | ASGI server for local development. |
| `pydantic` | `2.13.5` | v2 for the validation and JSON-schema generation the API contract relies on. |
| `pydantic-settings` | `2.15.0` | Environment-driven settings, so no configuration is hard-coded. |

`torch` and `transformers` are deliberately **absent** from
`backend/requirements.txt` in Phase 1. The backend loads no model yet, so
installing half a gigabyte of ML dependencies to serve one health endpoint would
be waste. When `POST /predict` arrives they are added, pinned through the same
`constraints.txt`, which structurally prevents the serving tokenizer from
drifting away from the training tokenizer.

## Test and lint

| Package | Pin | Why |
|---|---|---|
| `pytest` | `9.1.1` | Test runner. The `pythonpath` ini option used in `pyproject.toml` needs >= 7. |
| `httpx` | `0.28.1` | Required by `fastapi.testclient`; Starlette's `TestClient` is built on it. |
| `ruff` | `0.16.5` | Lint and import sorting in one tool. |

## Python version

**3.12.** Both `transformers 5.16.1` and `datasets 5.0.1` declare
`requires-python >= 3.10`. 3.12 is what the local machine already runs (3.12.4),
which keeps the local and Lightning interpreters on the same minor version, and
it has complete wheel coverage for every pin above. 3.13 was not chosen because
wheel availability across this stack has not been verified.

## What is intentionally not pinned

Purely transitive packages (`numpy`, `pyarrow`, `safetensors`,
`huggingface-hub`, `starlette`, `dill`, `fsspec`) are left to resolve within the
windows their parents declare. Over-pinning them invites unsatisfiable
resolutions on a different platform for no reproducibility gain, because the
exact resolved graph is captured separately in `requirements.lock.txt`.

- `constraints.txt` = **intent**, hand-maintained and justified.
- `requirements.lock.txt` = **exact reality**, generated with `pip freeze`.
