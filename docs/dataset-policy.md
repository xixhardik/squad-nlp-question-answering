# Dataset policy

## What we use

**SQuAD 1.1**, from the Hugging Face Hub repository
[`rajpurkar/squad`](https://huggingface.co/datasets/rajpurkar/squad).

Verified from the Hub datasets API (not recalled):

| Split | Examples | Parquet size |
|---|---:|---:|
| `train` | 87,599 | ~14.5 MB |
| `validation` | 10,570 | ~1.8 MB |

The repository is public and ungated, so **no Hugging Face token is required** to
download it.

SQuAD 2.0 (`train` 130,319 / `validation` 11,873) is deliberately out of scope
for now. It adds unanswerable questions, which require a null-answer threshold
and change the evaluation protocol. Starting there would mean debugging two new
things at once.

## Never committed to git

Raw data is not stored in version control. `.gitignore` covers `data/*` and
`*.arrow`, and `tests/test_project_structure.py` asserts those rules actually
match real candidate paths.

The reason is not only file size. SQuAD is small enough that committing it would
be technically possible. It is excluded because a copy in git is a copy that can
silently diverge from the canonical dataset, and then a metric can no longer be
traced to a known revision of known data.

Instead, `data/` holds the Hugging Face cache, which is reproducible from the
Hub at any time.

## Reproducibility

Dataset identity is part of the experiment config, not hidden in code:

```yaml
data:
  dataset_name: rajpurkar/squad
  dataset_version: main
  train_split: train
  validation_split: validation
```

`dataset_version` is `main` during development. **Before any run whose numbers
will be reported, it should be pinned to a commit sha**, so the data behind a
published metric is unambiguous. `dataset_name` and `dataset_version` are both
folded into `ExperimentConfig.config_hash()`, so changing either produces a
different run id and cannot be confused with an earlier result.

## The dataset is never modified

This is the single most important rule in this document.

SQuAD gives each answer as `{"text": ..., "answer_start": ...}`, where
`answer_start` is a **character offset into the raw context string**.

Therefore the context must never be stripped, lowercased, unicode-normalized,
whitespace-collapsed or otherwise altered before offsets are computed. Any such
edit shifts every offset after the edit point and silently corrupts the training
labels. The corruption is quiet: loss still decreases, and only the metrics look
inexplicably poor.

Concretely:

- Load the dataset and pass `context` to the tokenizer **verbatim**.
- Derive token positions with `return_offsets_mapping=True` plus
  `sequence_ids()`, never by string searching.
- Recover answer text by slicing the original context, never by decoding token
  ids. `qa_core.spans.extract_answer_text` is the only sanctioned path.

## Expected schema

Asserted at load time in a later phase rather than assumed:

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Unique example id; the key for grouping windows back to examples |
| `title` | `str` | Source Wikipedia article title |
| `context` | `str` | The passage. **Used verbatim.** |
| `question` | `str` | The question |
| `answers.text` | `list[str]` | One entry for train; several for validation |
| `answers.answer_start` | `list[int]` | **Character** offsets into `context` |

The train/validation asymmetry matters for evaluation: dev examples carry several
annotator-accepted answers, and an example's score is the **maximum** over them,
not the mean. `qa_core.metrics` implements that.

## Sliding windows

Contexts longer than `max_seq_length` are split into overlapping windows using
`doc_stride`, so one example can produce several tokenizer features. Two rules
follow, both enforced in a later phase:

1. Features must be mappable back to their source example via
   `overflow_to_sample_mapping`, because metrics are computed per example.
2. Examples whose answer is not fully contained in **any** window must be
   **counted and reported**, never silently dropped. Silently dropping them would
   quietly shrink the denominator and inflate the reported score.

## Local caching

The cache lives under `data/` by default. Override with `HF_HOME` (see
`.env.example`) to place it on another volume. The local `D:` drive has ~78 GB
free, which is ample: the raw dataset is ~16 MB, and the tokenized feature cache
is the larger artifact but still modest.
