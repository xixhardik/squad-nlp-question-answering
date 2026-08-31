# The extractive QA pipeline, end to end

How a SQuAD example becomes a trained model and a highlighted answer span. Every
number quoted here was measured on this codebase; nothing is copied from a paper.

---

## 1. The task

Given a context passage and a question, return the **exact span of the passage** that
answers it.

```
Context:  "The Amazon rainforest ... The majority of the forest is contained
           within Brazil, with 60 percent of the rainforest, followed by Peru
           with 13 percent."
Question: "Which country contains the majority of the Amazon rainforest?"
Answer:   "Brazil"   <- characters [113, 119) of the context
```

The answer is a **location**, not generated text. Two consequences shape everything
downstream:

1. **Hallucination is structurally impossible.** The model's output is a pair of token
   indices; the answer text is a slice of the input.
2. **Character offsets are the real product.** They let a UI highlight the answer
   inside the original passage, which is what visibly demonstrates the system is
   extractive rather than generative.

---

## 2. Dataset

`rajpurkar/squad` on the Hugging Face Hub. Public, ungated, parquet-backed, so no
token and no dataset loading script.

| split | examples | parquet |
|---|---:|---:|
| train | 87,599 | ~14.5 MB |
| validation | 10,570 | ~1.8 MB |

### Schema

| field | type | notes |
|---|---|---|
| `id` | `str` | unique; the key for regrouping windows back into examples |
| `title` | `str` | source Wikipedia article |
| `context` | `str` | the passage. **Used verbatim, never modified.** |
| `question` | `str` | the question |
| `answers.text` | `list[str]` | one entry for train, **several** for validation |
| `answers.answer_start` | `list[int]` | **character** offsets into `context` |

The train/validation asymmetry is not cosmetic. Measured on a 128-example validation
sample: **mean 3.29 gold answers per example, max 5**. That is why Exact Match and F1
take the **maximum** over golds, not the mean; any annotator-accepted answer is
correct.

### The one inviolable rule

`answer_start` is a character offset into the raw context. Therefore the context is
never stripped, lowercased, unicode-normalised or whitespace-collapsed before offsets
are used. Any such edit shifts every subsequent offset and silently corrupts the
labels — training loss still falls, and only the final metrics look inexplicably poor.

`qa_ml.data.verify_answer_offsets` checks the annotation itself by asserting
`context[start:start+len(text)] == text`. Measured on 256 train and 128 validation
examples: **100.000% exact, 0 whitespace-only, 0 real mismatches**.

SQuAD 2.0 is **refused**, not adapted to. It adds unanswerable questions with empty
`answers.text`, which need a null-answer threshold this pipeline does not implement.
`qa_ml.data.assert_squad_v1` detects them and raises.

---

## 3. Tokenization and sliding windows

### The problem

A Transformer has a fixed maximum sequence length (384 here). SQuAD contexts routinely
exceed it. Truncating loses answers; splitting requires care.

### Why the standard approach was rejected

Nearly every SQuAD tutorial uses `return_overflowing_tokens=True` with a `stride` and
lets the tokenizer tile the context. **On this project's pinned stack that silently
loses data.** Measured with `transformers 5.16.1` / `tokenizers 0.23.1`,
`bert-base-uncased`, `max_length=384`, `stride=128`:

| context tokens | windows | window token counts | context covered |
|---:|---:|---|---:|
| 300 | 1 | `[300]` | 100% |
| 600 | 2 | `[373, 139]` | **64%** |
| 1000 | 2 | `[373, 139]` | **38%** |
| 2000 | 2 | `[373, 139]` | **19%** |

The window sizes are **identical regardless of context length**: the tokenizer returns
the first window plus exactly *one* overflow window, so total coverage is capped at
roughly `max_length` context tokens. Confirmed to originate in the Rust backend rather
than the Python wrapper by calling `backend_tokenizer.enable_truncation(...)` directly
and observing `Encoding.overflowing` has length 1. `stride=0` and `stride=64` produce
byte-identical output, which is the tell that tiling is not happening at all.

For extractive QA this is a correctness bug, not an inefficiency: an answer past the
cap is unlabelable at training time and unreachable at evaluation time. It would
surface only as unexplained Exact Match loss on long contexts.

### What this project does instead

`qa_torch.features.SquadFeatureBuilder` windows explicitly:

1. **Tokenize the context alone** to get its token count and per-token character
   offsets.
2. **Compute the per-window budget:**
   `max_seq_length - num_special_tokens_to_add(pair=True) - question_tokens`.
   The special-token count is *queried*, never assumed — measured: **3** for BERT and
   DistilBERT (`[CLS] q [SEP] c [SEP]`), **4** for RoBERTa (`<s> q </s></s> c </s>`).
3. **Slide over context token indices** with `step = budget - doc_stride`, converting
   each token window to a character range. Windows never split a token.
4. **Re-encode `(question, context[char_start:char_end])` per window**, letting the
   tokenizer place special tokens correctly for its own family, then shift the returned
   context offsets by `char_start` to make them global.

Step 4 keeps the tokenizer responsible for everything model-specific while this code
controls only the tiling. Regression tests in `tests/test_features.py`
(`TestWindowCoverageRegression`) assert full coverage, no gaps, that window count grows
with context length, and that an answer late in a long context still receives a real
label.

### The three tokenizer outputs that make alignment possible

| output | purpose |
|---|---|
| `return_offsets_mapping=True` | `(char_start, char_end)` per token — the bridge between SQuAD's character offsets and the model's token indices |
| `sequence_ids(i)` | `None` = special token, `0` = question, `1` = context |
| per-window char shift | makes offsets index the original context, not the window slice |

**`sequence_ids` is mandatory, never index arithmetic.** Separator layout differs per
family: RoBERTa's `sequence_ids` head is `[None, 0, ..., 0, None, None, 1, 1, ...]` —
two separators. Any code assuming a single `[SEP]` breaks on RoBERTa.

### Question truncation

`truncation="only_second"` protects the question at the context's expense, so a
pathologically long question would crowd the context out entirely.
`prepare_question` therefore left-strips the question and truncates it at a **token
boundary located via offset mapping**, applied as a character slice — so the surviving
text is an exact prefix, not an approximate decode round-trip.

---

## 4. Character-to-token span alignment

`qa_core.alignment.align_answer_to_tokens` converts `(answer_char_start,
answer_char_end)` into `(token_start, token_end)` for one window.

```
1. find the context token range [first, last] from sequence_ids
2. if the window does not FULLY contain the answer -> label at [CLS] (0, 0)
3. walk forward from `first` to the last token starting at or before
   answer_char_start  -> token_start
4. walk backward from `last` to the first token ending at or after
   answer_char_end    -> token_end
```

Containment uses inequalities, not equality, because answers routinely begin or end
part-way through a token: `"Brazil,"` may tokenize as one token covering `13..20`
while the answer covers only `13..19`.

### Answers outside a window are reported, never hidden

With sliding windows most windows genuinely do not contain the answer, and labelling
those at `[CLS]` is correct. But an answer contained in **no** window is a data
problem. `AlignmentStatus` distinguishes four outcomes — `ALIGNED`,
`ANSWER_OUTSIDE_WINDOW`, `NO_CONTEXT_TOKENS`, `DEGENERATE_ANSWER` — and
`AlignmentReport.examples_with_no_aligned_feature` counts the examples that matter.

Measured on the 256/128-example smoke split at `max_seq_length=256`:

```
train:       256 examples -> 258 features (1.0078/example)
             aligned 256 (99.22%)   outside window 2   NO aligned window 0
validation:  128 examples -> 129 features (1.0078/example)
             aligned 128 (99.22%)   outside window 1   NO aligned window 0
```

Zero examples with no aligned window is the property that matters.

### Never decode token ids to get answer text

Measured across the candidate tokenizers for the answer `"Brazil"` at characters
`(290, 296)`:

| model | tokenizer | raw offsets | slice of raw context | `decode()` |
|---|---|---|---|---|
| distilbert-base-uncased | WordPiece | (290, 296) | `'Brazil'` | `'brazil'` |
| bert-base-uncased | WordPiece | (290, 296) | `'Brazil'` | `'brazil'` |
| roberta-base | BPE | (290, 296) | `'Brazil'` | `' Brazil'` |
| microsoft/deberta-v3-base | SentencePiece | **(289, 296)** | **`' Brazil'`** | `'Brazil'` |

Two independent failure modes: `decode()` loses original casing on uncased models and
can add a leading space on byte-level BPE; and SentencePiece offsets *include* the
preceding whitespace, so the recovered span starts one character early.

The second is the dangerous one — SQuAD normalization strips whitespace, so **Exact
Match and F1 would be unaffected** and the defect would surface only as an answer
highlighted one character early in the UI. `qa_core.spans.tighten_char_span` strips
surrounding whitespace from every recovered span, making offsets exact and
tokenizer-independent.

---

## 5. Model and start/end logits

`AutoModelForQuestionAnswering` puts a single linear layer (`qa_outputs`, shape
`hidden_size x 2`) on a pretrained encoder. For each token it emits two scores:
"is this the answer's first token" and "is this its last token".

```
input_ids  ->  encoder  ->  hidden states [seq_len, hidden]
                               |
                          qa_outputs (Linear -> 2)
                               |
                 start_logits [seq_len],  end_logits [seq_len]
```

Verified output keys: `['start_logits', 'end_logits']`, shape `(batch, seq_len)`.
Passing `start_positions`/`end_positions` adds a `loss` key (cross-entropy over
positions, averaged across the two).

Model is configurable via `model_name`; nothing about the architecture is hard-coded.
Measured parameter counts:

| model | parameters | note |
|---|---:|---|
| `distilbert-base-uncased` + QA head | **66,364,418** | measured locally |
| `bert-base-uncased` (Hub total) | 110,106,428 | from safetensors metadata |
| `roberta-base` (Hub total) | 124,697,947 | from safetensors metadata |
| `microsoft/deberta-v3-base` | *unmeasured* | publishes no safetensors metadata |

**`microsoft/deberta-v3-base` cannot currently be loaded in this environment.** Its
tokenizer needs `sentencepiece` (or `tiktoken`) to convert the SPM model, and neither
is installed. Experiment D is optional and budget-gated, so `constraints.txt` was
deliberately **not** modified; see "Known limitations".

---

## 6. Answer decoding

`qa_core.postprocess.decode_spans`. Written by hand: transformers v5 removed the
`question-answering` pipeline, and the point of the project is to show what it hid.

```
start_logits + end_logits (per window)
   |
   1. shortlist  top n_best_size starts and ends per window
   |             (400 pairs at n_best_size=20, vs 147,456 at max_seq_length=384)
   |
   2. reject     offset is None (special/question token)   <- blocks question text
   |             end < start
   |             end - start + 1 > max_answer_length
   |
   3. score      start_logit + end_logit   (log space: stable, monotonic)
   |
   4. recover    offsets[start][0] .. offsets[end][1]  -> slice the RAW context
   |             then tighten whitespace
   |
   5. normalise  ONE softmax over candidates pooled from ALL windows
   v
best span + ranked n-best
```

### Why the pooled softmax

The conventional approach is `softmax(start)[i] * softmax(end)[j]`. That is a product
of two independent marginals and **not** a distribution over spans:

- probability mass sits on pairs rejected in step 2, so what remains does not sum to 1
  over the candidates actually considered;
- with sliding windows each window has its own denominator, so a score from window 0
  is not comparable with one from window 1 — and taking the maximum across windows
  compares numbers on different scales.

One softmax over the pooled candidate set fixes both. `tests/test_postprocess.py`
asserts pooled scores sum to 1.0 across multiple windows.

### The score is uncalibrated, and labelled as such

Fine-tuned transformers are systematically overconfident: 0.9 does not mean a 90%
chance of being correct. The value therefore travels with
`score_type: "uncalibrated_span_probability"` rather than being called "confidence".
Relabelling to `temperature_scaled` would require actually fitting a temperature on
held-out data and measuring calibration error.

Visible in the smoke run: the undertrained model's top candidates all scored
0.012–0.016 — a nearly flat distribution, which is the honest output for a model that
has learned little.

---

## 7. Exact Match and F1

`qa_core.metrics`, implemented from the SQuAD v1.1 methodology rather than imported, so
the behaviour is visible and version-stable.

### Normalization (order matters)

```
1. lowercase
2. strip punctuation
3. remove the articles a / an / the
4. collapse whitespace
```

Step 3 runs after step 2 so `"the,"` still loses its article. Step 4 runs last because
steps 2 and 3 leave gaps.

### Exact Match

1.0 if the normalized prediction equals any normalized gold answer, else 0.0.
All-or-nothing: `"within Brazil"` against gold `"Brazil"` scores 0.

### Token-level F1

Multiset token overlap, so repeats are handled correctly:

```
overlap   = Counter(pred) & Counter(gold)
precision = overlap / len(pred)
recall    = overlap / len(gold)
F1        = 2PR / (P + R)
```

`"New York New York"` vs `"New York"` scores 2/3, not 1.0 — with sets it would
wrongly score 1.0.

Edge case matching the official script: if either side normalizes to empty, F1 is 1.0
only when *both* are empty.

Per example, both metrics take the **maximum over gold answers**; the corpus score is
the unweighted mean, times 100.

### Metrics come from text, never from logits

Comparing token positions against gold token positions would reward a model for being
right about the wrong thing. A span that decodes to the correct *text* from different
token indices is correct; one that hits the labelled indices but decodes to the wrong
text is not. Text is also what makes the numbers comparable with published results.

### Independently verified

Our implementation is the source of truth; `evaluate` is a cross-check only. Measured
on the smoke run:

```
ours:      EM = 0.7812   F1 = 9.0898
evaluate:  EM = 0.7812   F1 = 9.0898
delta:     EM = 5e-05    F1 = 1.6e-05     agrees = true
```

A disagreement is logged as a warning, never silently resolved.

---

## 8. Training

`transformers.Trainer` (backed by Accelerate) handles checkpointing, resumption, mixed
precision, gradient accumulation and LR scheduling — exactly the parts a bespoke loop
tends to get subtly wrong. The parts this project is meant to demonstrate are
hand-written and live in `qa_core` / `qa_torch`.

### EM and F1 during training

Eval loss is a poor selection signal here because it scores token positions while the
task is judged on decoded text. A `compute_metrics` closure over the validation bundle
decodes real answers and returns EM and F1, which `metric_for_best_model: f1` then
selects on. Validation features carry **both** labels and offsets from one windowing
pass, so loss and text metrics come from identical features.

### Config translation (transformers 5.16.1)

The project's config vocabulary is its own; two mappings are not one-to-one:

| our config | TrainingArguments | why |
|---|---|---|
| `evaluation_strategy` | `eval_strategy` | old name removed in v5 |
| `warmup_ratio: 0.1` | `warmup_steps=0.1` (float) | `warmup_ratio` removed; `warmup_steps` is typed `float` and a value < 1 is a fraction of total steps |

Verified: `TrainingArguments(warmup_steps=0.1).get_warmup_steps(1000) == 100` and
`get_warmup_steps(5000) == 500`. Also removed in 5.16.1: `overwrite_output_dir`.

Keeping our names stable means a library rename cannot invalidate committed experiment
records.

### Precision on the L4

`precision: auto` resolves at **runtime**, never at config load:

| device | resolves to | reason |
|---|---|---|
| CUDA + `is_bf16_supported()` | **bf16** | same exponent range as fp32, so no gradient loss scaling — removes a class of silent divergence where fp16 gradients underflow |
| CUDA without bf16 | fp16 | older hardware still benefits; Trainer handles loss scaling |
| CPU | fp32 | mixed precision does not help here |

The L4 is Ada generation (compute capability 8.9) and is expected to report bf16
support, but this is **queried, not assumed**. An explicit `fp32`/`fp16`/`bf16` is
honoured so a benchmark can pin one setting.

### Baseline configuration for the L4

`ml/configs/base.yaml`, chosen to complete reliably rather than to saturate the GPU —
a run that OOMs at hour two wastes more time than a larger batch saves.

| parameter | value | rationale |
|---|---|---|
| `max_seq_length` | 384 | standard SQuAD setting |
| `doc_stride` | 128 | must exceed typical answer length |
| `per_device_train_batch_size` | 16 | comfortable on ~22 GiB at 384 tokens |
| `per_device_eval_batch_size` | 64 | evaluation stores no activations |
| `learning_rate` | 3e-5 | fixed across experiments for a controlled comparison |
| `num_train_epochs` | 2 | conventional SQuAD budget |
| `warmup_ratio` | 0.1 | |
| `weight_decay` | 0.01 | |
| `precision` | auto → bf16 on L4 | |
| `metric_for_best_model` | f1 | |

Experiment D halves the micro-batch and doubles accumulation, keeping the **effective
batch size equal** so the comparison stays fair. A test asserts this equality across
all four experiment configs.

### Reproducibility

Every run records: timestamp, git commit **and dirty flag**, Python/torch/transformers/
tokenizers/datasets versions, GPU details, the full resolved config, its hash, the
seed, dataset identity and version, split sizes, offset verification, feature and
alignment counts, precision plan and reason, runtime, throughput, peak GPU memory,
validation loss, EM, F1, and the checkpoint path.

`is_reproducible` is `false` when the working tree was dirty, because the recorded
commit then does not describe the code that ran. Seeding covers Python, NumPy and
torch, and is applied **before** the model is built — `qa_outputs` is randomly
initialised at load time, so seeding afterwards would leave the starting weights
unreproducible.

A run directory is **never silently overwritten**: `create_run_directory` uses
`mkdir(exist_ok=False)` and the error lists the four ways forward. Verified by
executing it.

---

## 9. Commands

All commands assume the `qas-nlp` environment is active and are run from the
repository root.

### Local (CPU) — development and smoke checks

```bash
# Environment diagnostics
python ml/scripts/check_environment.py --tensor-test

# Full test suite
pytest -q

# Fast unit tests only (no downloads)
pytest -q -m "not network and not slow"

# GPU smoke test (skips CUDA cases on CPU)
pytest tests/test_gpu_smoke.py -v

# Validate the dataset and report feature/alignment statistics
python -m qa_ml prepare --config smoke.yaml --build-features

# Tiny end-to-end training run on CPU, to prove the path executes
python -m qa_ml train --config smoke.yaml --allow-cpu --cross-check \
    --set output.run_name=cpu-smoke-001 \
    --set preprocessing.max_seq_length=256 --set preprocessing.doc_stride=64
```

### Lightning AI Studio (NVIDIA L4)

```bash
# 1. Set up
git clone <repo-url> ~/qas-nlp && cd ~/qas-nlp
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements-gpu.txt -r requirements-dev.txt
pip install -e .

# 2. HARD GATE - must pass before committing GPU budget
python ml/scripts/check_environment.py --require-cuda --tensor-test \
    --json --output artifacts/env-l4.json

# 3. GPU smoke test: forward, backward, optimizer step, bf16
pytest tests/test_gpu_smoke.py -v

# 4. Full test suite on the GPU machine
pytest -q

# 5. Validate the full dataset
python -m qa_ml prepare --config experiment_a_distilbert.yaml --verify-all

# 6. Short smoke training on GPU before the real run
python -m qa_ml train --config smoke.yaml --num-proc 4

# 7. Baseline experiment (DistilBERT)
python -m qa_ml train --config experiment_a_distilbert.yaml \
    --num-proc 4 --cross-check

# 8. Evaluate the resulting checkpoint
python -m qa_ml evaluate --config experiment_a_distilbert.yaml \
    --model artifacts/runs/<run_id>/model \
    --predictions reports/<run_id>-predictions.json

# 9. Answer a single question
python -m qa_ml predict --config experiment_a_distilbert.yaml \
    --model artifacts/runs/<run_id>/model \
    --question "Which country contains the majority of the Amazon rainforest?" \
    --context-file passage.txt
```

Resume an interrupted run:

```bash
python -m qa_ml train --config experiment_a_distilbert.yaml \
    --run-id <existing_run_id> --resume
```

Change training length without editing source:

```bash
python -m qa_ml train --config experiment_a_distilbert.yaml \
    --set training.num_train_epochs=3 \
    --set training.per_device_train_batch_size=32
```

---

## 10. Output artifacts

```
artifacts/runs/<run_id>/
├── experiment.json          full record: config, environment, git, seed, GPU,
│                            precision, dataset, preprocessing, training, evaluation
├── metrics.json             headline numbers, for the comparison table
├── config.resolved.json     the exact merged configuration
├── predictions.json         per-example n-best + EM/F1, for error analysis
├── checkpoints/             Trainer checkpoints (resumable; includes optimizer state)
└── model/                   selected weights + tokenizer, for serving
```

`run_id` is `<experiment>-<model>-<config_hash>-<UTC timestamp>`, so a checkpoint is
always traceable to the configuration that produced it.

---

## 11. Two warnings you will see in training logs

Both were audited. Neither is a defect, but the second one guards a real hazard.

### "Token indices sequence length is longer than the specified maximum sequence length (718 > 512)"

**Origin:** `SquadFeatureBuilder.window_char_ranges()`, the deliberate full-context
tokenization that has no `truncation` or `max_length` because it needs every token's
character offset in order to tile the passage.

**Attribution, established by running each call site in an isolated process** (the
warning fires only once per tokenizer instance, so in-process checks are contaminated):

| call site | warns |
|---|---|
| `window_char_ranges()` | **yes** |
| `encode_windows()` — the path that feeds the model | no |
| `_question_token_count()` | no |

**Why it is harmless:** the encoding it complains about is never given to a model. It
exists only to read `offset_mapping`. Every tensor that reaches the model comes from
`encode_windows`, with `truncation="only_second", max_length=max_seq_length`. Measured
on the 25 longest real SQuAD validation contexts (up to 789 tokens): 75 windows
produced, maximum feature length exactly 384, never over, and all `start_positions` /
`end_positions` in range.

**How often it triggers:** on the SQuAD validation split, context token lengths run
min 30 / mean 161.0 / max 789. **49 contexts (0.46%) exceed 512 tokens** and trigger
the warning; **161 (1.52%) exceed 384** and therefore need more than one window.

**Fix:** `verbose=False` at that one call site, with a comment explaining why.
Verified to change the returned `input_ids` and `offset_mapping` not at all.
`tests/test_features.py::TestNoOverLengthFeatureReachesTheModel` pins both that no
feature exceeds `max_seq_length` and that the warning does not return.

### BERT reload: missing `LayerNorm.weight`/`.bias`, unexpected `LayerNorm.gamma`/`.beta`

**Origin:** `bert-base-uncased` publishes its LayerNorm parameters under the legacy
TensorFlow names. The Hub checkpoint contains `LayerNorm.gamma` and `LayerNorm.beta`
and **no** `LayerNorm.weight` at all.

`transformers` maps those names in **both** directions, and the naming follows the
checkpoint's history rather than the architecture:

- `from_pretrained` maps `.gamma` → `.weight` on load
- `save_pretrained` writes `.gamma`/`.beta` back out for a model that came from such a
  checkpoint (a model built from `BertConfig` saves modern names instead)

**Verified: nothing is lost through the path this project uses.**

| load path | missing LayerNorm | unexpected `.gamma`/`.beta` | LayerNorm restored |
|---|---:|---:|---|
| `from_pretrained` (used by `load_qa_model`) | 0 | 0 | **yes** |
| raw `load_state_dict(strict=False)` | 50 | 50 | **no** |

Loading `bert-base-uncased` reports only `qa_outputs.weight`/`.bias` as missing — the
expected fresh QA head — and no LayerNorm keys, from either the `.safetensors` or the
`.bin` file. The loaded `bert.embeddings.LayerNorm.weight` has mean 0.849, not all
ones, and matches the raw checkpoint tensor exactly.

A full save/reload round trip with every parameter perturbed first showed **0 of 199
parameters drifted, 0 LayerNorm at default values, and bit-identical logits**.

**The real hazard:** a load path that bypasses the mapping silently leaves those 50
tensors unloaded. Nothing crashes; the model simply is not the one that was saved.
`Trainer._load_best_model` contains both `from_pretrained` and `load_state_dict`
branches, which is the likely source of the warning when
`load_best_model_at_end: true`.

**Fix:** `qa_torch.loader.verify_checkpoint_integrity` runs after every
`trainer.save_model`, reloads the checkpoint and raises unless all parameters are
bit-identical. Each run also records `best_model_checkpoint`, `best_metric`,
`best_global_step` and `final_global_step`, so the effect of best-model restoration is
auditable from the experiment record instead of assumed. Covered by
`tests/test_loader.py`.

---

## 12. Known limitations

1. **`microsoft/deberta-v3-base` (Experiment D) cannot be loaded** in this environment:
   its tokenizer requires `sentencepiece` or `tiktoken`. Experiment D is optional and
   budget-gated, so `constraints.txt` was deliberately left unchanged. Enabling it is a
   one-line addition plus a justified constraints update.
2. **No GPU results yet.** Every number in this document was produced on CPU or is a
   measured property of the code. The L4 has not run training; the GPU-specific tests
   are skip-guarded and untested on real hardware.
3. **The smoke run's metrics are meaningless as accuracy.** EM 0.78 / F1 9.09 came from
   256 training examples over 1 epoch. It demonstrates that the path executes, nothing
   more, and must never be quoted as a result.
4. **`compute_metrics` assumes evaluation-loop output order** matches dataset order.
   True for single-device evaluation; a distributed eval loop would need explicit
   index tracking. The logit/feature count is asserted rather than trusted.
5. **The score is uncalibrated.** No temperature has been fitted and no calibration
   error measured.
6. **`num_proc` is left at `None` on Windows**, where worker processes are spawned
   rather than forked and startup dominates. Use `--num-proc 4` on Linux.
