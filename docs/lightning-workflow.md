# Lightning AI workflow

> **Status: training is complete and this workflow is no longer part of the normal
> development loop.**
>
> All four experiments were run on the L4, `microsoft/deberta-v3-base` was selected,
> and its checkpoint has been copied to `models/deberta-v3-base-squad/` and verified
> by SHA-256 against the Studio source. Inference, the frontend and all further
> development now run locally on Windows with no GPU and no cloud dependency — see
> **Local deployment** in the [README](../README.md).
>
> This document is retained because it is the record of how the checkpoint was
> produced and how to transfer one. Return to it only to retrain or to run a new
> experiment.

Training runs on a **Lightning AI Studio with an NVIDIA L4**. The local Windows
machine handles authoring, testing, and — now that training is finished — inference
and the web interface.

## Division of environments

| | Lightning AI Studio (Linux, NVIDIA L4) | Local (Windows, CPU-only) |
|---|---|---|
| Role | GPU training and evaluation | Inference, frontend, Git workflow |
| Status | **Complete; not required for inference** | **Active** |
| Runs | `qa_ml train`, full-split evaluation | FastAPI `:8000`, Next.js `:3000`, `pytest` |
| Holds | `artifacts/runs/<run_id>/` | `models/deberta-v3-base-squad/` |

Nothing about local inference depends on Lightning being reachable.

Verified locally: `nvidia-smi` is not on PATH and `torch.cuda.is_available()`
returns `False`, so the local machine has no usable GPU. Attempting to train here
is not a slow option, it is a non-option.

Verified tooling: `lightning` CLI **v2026.8.5** with `lightning_sdk 2026.8.5` is
already installed locally.

## The pipeline

```
LOCAL PC  (Windows, CPU-only)
    | author code, run fast CPU tests
    v
Git repository  (code, configs, reports -- never weights or data)
    | git push / git pull
    v
Lightning AI Studio  (Linux, NVIDIA L4)
    | clone or pull the repository
    | pip install -r ml/requirements-gpu.txt
    | python ml/scripts/check_environment.py --require-cuda --tensor-test   <-- GATE
    | run smoke test on a tiny subset
    | run training
    v
Experiment artifacts  (artifacts/runs/<run_id>/)
    | lightning model  -> register every checkpoint
    | git commit reports/<run_id>/  -> metrics travel as reviewable text
    v
LOCAL PC
    | lightning cp -> pull the selected checkpoint into models/
    v
Local inference  (FastAPI + Next.js on localhost)
    |
    v
Hugging Face Hub  (OPTIONAL, final model only, model storage/distribution)
```

## What Kiro can and cannot do

Stated plainly, because it determines how the remaining phases are run.

**Kiro has no access to your Lightning AI account.** It runs as a local process
on the Windows machine. It holds no Lightning credentials, cannot call the
Lightning API on your behalf, and cannot see inside a Studio.

| Actor | Can do | Cannot do |
|---|---|---|
| **Kiro (local)** | Read/write repo files, run local `pytest`, run local `git`, run the local FastAPI and Next.js servers | Authenticate to Lightning or Hugging Face; see the Studio filesystem; launch GPU jobs |
| **You** | `lightning login`, create/attach the Studio, run training, `lightning cp`, `hf auth login` | — |
| **Git** | Carries code and `reports/` in both directions | Must never carry weights, datasets or caches |

### Two ways to give Kiro visibility into remote runs

**Option A — paste-back (default, no extra credentials).** You run the Studio
command and paste the output into the chat. Kiro interprets logs, diagnoses
failures and patches code locally. This requires no cloud credentials of any kind,
which is why it is the default.

**Option B — VS Code Remote-SSH (optional, more fluid).** After
`lightning ssh`, attach VS Code to the Studio over SSH. Kiro's file and terminal
tools then operate on the *remote* filesystem, so it can read logs and run
commands on the L4 directly. This still uses **your** SSH identity; no API key is
handed to Kiro.

Recommendation: Option A for the short, output-oriented gate phases; consider
Option B for the training phases where log tailing gets repetitive.

## One-time setup

Commands **you** run. Nothing here is automated on your behalf.

### 1. Local console fix (Windows)

```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

Verified necessary: without it, `lightning --help` crashes with
`UnicodeEncodeError: 'charmap' codec can't encode character '\u26a1'` because the
PowerShell console is cp1252 and the CLI banner contains an emoji. The CLI itself
is fine; only the console encoding is at fault. The same fix prevents identical
failures from `hf` and from `tqdm` progress bars.

### 2. Authenticate and pick hardware

```powershell
lightning login       # interactive browser OAuth; no token enters the repo
lightning machine     # confirm an L4 instance type is available to you
```

### 3. GitHub remote

Already configured — `origin` points at the project repository and `main` tracks it.
This step is only relevant when setting up a fresh clone:

```powershell
git remote add origin <your-repo-url>
git push -u origin main
```

### 4. Prepare the Studio

Create a Studio in the Lightning web UI, attach the **L4** GPU, then inside it:

```bash
git clone <your-repo-url> ~/qas-nlp
cd ~/qas-nlp
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements-gpu.txt -r requirements-dev.txt
pip install -e .

# HARD GATE - must pass before any training
python ml/scripts/check_environment.py --require-cuda --tensor-test \
    --json --output artifacts/env-l4.json
```

The Studio needs read access to the repository: use a deploy key or an HTTPS
token **entered by you in the Studio**. Never commit either.

## Per-experiment cycle

```
LOCAL   edit code
        pytest                       # fast, CPU, tiny fixtures
        git commit && git push

STUDIO  git pull
        python ml/scripts/check_environment.py --require-cuda
        # ... smoke test, then the real run (later phases)
        lightning model upload artifacts/runs/<run_id>/checkpoint
        git add reports/<run_id> && git commit && git push

LOCAL   git pull                     # metrics arrive as reviewable text
        lightning cp <studio-path> ./models/<name>
```

### Interactive Studio versus `lightning job`

| Phase | Mode | Why |
|---|---|---|
| Validation, dataset inspection, smoke tests | **Interactive Studio** | Short, needs a REPL and fast iteration |
| Full training runs | **`lightning job`** | Survives laptop sleep and disconnects; one clean log per run |

A long interactive training run over SSH is fragile. Batch submission is the
defence against losing hours of GPU time to a dropped connection.

## Git discipline

1. **Never edit source directly in the Studio.** That creates untracked
   divergence and voids reproducibility. Fix locally, commit, `git pull` in the
   Studio.
2. Every run records `git rev-parse HEAD` **and a dirty-tree flag**. A run from a
   dirty tree is marked non-reproducible in its own metadata.
3. Branch per experiment (`exp/a-distilbert`), merge to `main` once its gate
   passes.

## Artifact destinations

| Artifact | Destination | Mechanism | In git? |
|---|---|---|---|
| `metrics.json`, resolved config, `env.json` | `reports/<run_id>/` | `git commit` from the Studio | **Yes** — small, diffable, the evidence trail |
| Every checkpoint | Lightning model registry | `lightning model` | No |
| Selected checkpoint | local `models/` | `lightning cp` | No |
| Final chosen model | Hugging Face Hub | `hf upload` (optional) | No |

**Completed transfer, for the record.** `artifacts/runs/<run_id>/model/` →
`models/deberta-v3-base-squad/`, 735,356,752 bytes of `model.safetensors` verified
bit-for-bit:

```
SHA-256  f42b44e4904f132c18512c70c5c438a6b2daaf551d127b3f6213ff208a39de1f
```

Verify any future transfer the same way — `sha256sum` on the Studio, `Get-FileHash`
on Windows. A partial copy of a 700 MB file otherwise fails much later, as an
obscure load error rather than as a transfer problem.

**Rule: nothing valuable ends a session living only in Studio-local storage.**
Register checkpoints with `lightning model` before detaching the GPU. A lost run
costs the GPU budget twice.

## Cost discipline

The GPU allowance is finite (~8–9 hours). Protecting it shapes the workflow:

- All `qa_core` development happens **locally on CPU** against tiny fixtures.
  That is where the correctness risk lives, and it needs no GPU whatsoever.
- Attach the L4 only for environment validation, smoke tests and real training.
  Detach the moment a run finishes.
- Forecast each run's cost from a measured calibration figure **before** launching
  it, and skip an experiment whose forecast does not fit. A documented omission is
  a legitimate result; an invented number is not.

## Known local environment issues

| Issue | Status | Action |
|---|---|---|
| `hf` CLI crashes with `TypeError: Typer.__init__() got an unexpected keyword argument 'suggest_commands'` | Base conda env has `typer 0.15.2`, too old for `huggingface_hub 1.13.0` | Not fixed in `base`, which is shared with other projects. The project env installs a compatible set. Not needed until model upload. |
| `huggingface-cli` reports itself deprecated and non-functional | Expected | Use `hf` instead. |
| PowerShell `UnicodeEncodeError` on emoji output | Diagnosed and fixed | Set `PYTHONIOENCODING=utf-8` (see step 1). |
| No local GPU | Expected and by design | Train on Lightning. |
