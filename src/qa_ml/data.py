"""Loading and validating the SQuAD 1.1 dataset.

Source
------
`rajpurkar/squad <https://huggingface.co/datasets/rajpurkar/squad>`_ on the Hugging
Face Hub: public, ungated, parquet-backed, so no access token and no dataset loading
script is involved.

Verified split sizes:

========== ========= ==============
split      examples  parquet size
========== ========= ==============
train      87,599    ~14.5 MB
validation 10,570    ~1.8 MB
========== ========= ==============

Expected schema
---------------
============================ ============ ================================================
field                        type         notes
============================ ============ ================================================
``id``                       ``str``      unique; the key for regrouping windows
``title``                    ``str``      source Wikipedia article
``context``                  ``str``      the passage. **Used verbatim.**
``question``                 ``str``      the question
``answers.text``             ``list[str]`` one entry for train, several for validation
``answers.answer_start``     ``list[int]`` **character** offsets into ``context``
============================ ============ ================================================

The train/validation asymmetry matters: dev examples carry several
annotator-accepted answers, and an example's score is the **maximum** over them.

Why SQuAD 2.0 is not accepted silently
--------------------------------------
SQuAD 2.0 adds unanswerable questions with empty ``answers.text``. Training on it
without a null-answer threshold produces a model that appears to work while being
systematically wrong on a third of the data. :func:`assert_squad_v1` therefore checks
for empty answers and refuses rather than adapting quietly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qa_ml.config import DataConfig

logger = logging.getLogger(__name__)

__all__ = [
    "EXPECTED_SPLIT_SIZES",
    "REQUIRED_COLUMNS",
    "SQUAD_V1_DATASET_ID",
    "DatasetLoadError",
    "DatasetSchemaError",
    "OffsetVerificationReport",
    "SplitSummary",
    "assert_squad_schema",
    "assert_squad_v1",
    "load_squad_split",
    "load_squad_splits",
    "summarize_split",
    "verify_answer_offsets",
]

#: The canonical SQuAD 1.1 repository on the Hugging Face Hub.
SQUAD_V1_DATASET_ID = "rajpurkar/squad"

#: Verified example counts, asserted at load time so a silent dataset swap is caught.
EXPECTED_SPLIT_SIZES = {"train": 87599, "validation": 10570}

#: Columns every SQuAD split must provide.
REQUIRED_COLUMNS = ("id", "title", "context", "question", "answers")


class DatasetLoadError(RuntimeError):
    """Raised when the dataset cannot be downloaded or read."""


class DatasetSchemaError(RuntimeError):
    """Raised when a loaded dataset does not match the expected SQuAD schema."""


@dataclass(frozen=True, slots=True)
class OffsetVerificationReport:
    """Result of checking that ``answer_start`` offsets actually locate the answer.

    The single most important dataset check in the project. If
    ``context[answer_start:answer_start + len(text)] != text`` then the annotation
    itself is inconsistent, and no amount of correct tokenization can recover the
    right label.

    Attributes:
        checked: Examples inspected.
        exact_matches: Examples where the slice equals the answer text exactly.
        whitespace_only_mismatches: Examples matching after stripping whitespace.
            Harmless: span tightening handles these.
        mismatches: Examples that do not match even after stripping. These are real
            annotation problems.
        examples_with_no_answer: Examples with an empty answer list. Non-zero means
            the data is not SQuAD 1.1.
        mismatch_samples: Up to a few offending examples, for inspection.
    """

    checked: int
    exact_matches: int
    whitespace_only_mismatches: int
    mismatches: int
    examples_with_no_answer: int
    mismatch_samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exact_match_rate(self) -> float:
        """Share of checked examples whose offsets are exactly correct."""
        if self.checked == 0:
            return 0.0
        return self.exact_matches / self.checked

    @property
    def usable_rate(self) -> float:
        """Share usable after whitespace tightening."""
        if self.checked == 0:
            return 0.0
        return (self.exact_matches + self.whitespace_only_mismatches) / self.checked

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for experiment records."""
        return {
            "checked": self.checked,
            "exact_matches": self.exact_matches,
            "whitespace_only_mismatches": self.whitespace_only_mismatches,
            "mismatches": self.mismatches,
            "examples_with_no_answer": self.examples_with_no_answer,
            "exact_match_rate": round(self.exact_match_rate, 6),
            "usable_rate": round(self.usable_rate, 6),
        }


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """Descriptive statistics for one dataset split.

    Attributes:
        split: Split name.
        num_examples: Number of examples.
        num_titles: Distinct source articles.
        context_chars: ``(min, mean, max)`` context length in characters.
        question_chars: ``(min, mean, max)`` question length in characters.
        answer_chars: ``(min, mean, max)`` answer length in characters.
        answers_per_example: ``(min, mean, max)`` gold answers per example.
    """

    split: str
    num_examples: int
    num_titles: int
    context_chars: tuple[int, float, int]
    question_chars: tuple[int, float, int]
    answer_chars: tuple[int, float, int]
    answers_per_example: tuple[int, float, int]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        def triple(values: tuple[int, float, int]) -> dict[str, float]:
            return {"min": values[0], "mean": round(values[1], 2), "max": values[2]}

        return {
            "split": self.split,
            "num_examples": self.num_examples,
            "num_titles": self.num_titles,
            "context_chars": triple(self.context_chars),
            "question_chars": triple(self.question_chars),
            "answer_chars": triple(self.answer_chars),
            "answers_per_example": triple(self.answers_per_example),
        }


def _stats(values: Sequence[int]) -> tuple[int, float, int]:
    """Return ``(min, mean, max)`` for a non-empty sequence of integers."""
    if not values:
        return (0, 0.0, 0)
    return (min(values), sum(values) / len(values), max(values))


def load_squad_split(
    data_config: DataConfig,
    split: str,
    *,
    cache_dir: str | None = None,
    apply_sample_cap: bool = True,
    seed: int = 42,
) -> Any:
    """Load one SQuAD split and validate it.

    Args:
        data_config: Dataset settings, including the repository id and revision.
        split: Either ``"train"`` or ``"validation"``, or a raw split expression.
        cache_dir: Override the datasets cache location.
        apply_sample_cap: Honour ``max_train_samples`` / ``max_eval_samples``.
        seed: Seed used when a sample cap requires subsetting.

    Returns:
        The loaded ``datasets.Dataset``.

    Raises:
        DatasetLoadError: If the dataset cannot be downloaded or read.
        DatasetSchemaError: If the loaded split does not match the SQuAD schema.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - datasets is a required dep
        raise DatasetLoadError(
            "The `datasets` package is required to load SQuAD. Install the ML "
            "requirements: pip install -r ml/requirements-cpu.txt"
        ) from exc

    if data_config.dataset_name != SQUAD_V1_DATASET_ID:
        logger.warning(
            "dataset_name is %r, not the canonical SQuAD 1.1 id %r. This is allowed "
            "but is recorded in the experiment metadata, and the SQuAD 1.1 checks "
            "below still apply.",
            data_config.dataset_name,
            SQUAD_V1_DATASET_ID,
        )

    logger.info(
        "Loading %s split=%s revision=%s",
        data_config.dataset_name,
        split,
        data_config.dataset_version,
    )
    try:
        dataset = load_dataset(
            data_config.dataset_name,
            split=split,
            revision=data_config.dataset_version,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise DatasetLoadError(
            f"Could not load {data_config.dataset_name!r} split {split!r} at revision "
            f"{data_config.dataset_version!r}.\n"
            "Check that:\n"
            "  - this machine can reach huggingface.co (no token is needed; the "
            "dataset is public and ungated)\n"
            "  - the split name is 'train' or 'validation'\n"
            "  - the revision exists (use 'main' if unsure)\n"
            "  - there is free disk space for the datasets cache (set HF_HOME to "
            "relocate it)\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    assert_squad_schema(dataset, split)

    expected = EXPECTED_SPLIT_SIZES.get(split)
    if expected is not None and len(dataset) != expected:
        logger.warning(
            "Split %r has %d examples but %d were expected for SQuAD 1.1. The dataset "
            "revision may differ from the one this project was validated against.",
            split,
            len(dataset),
            expected,
        )

    if apply_sample_cap:
        cap = (
            data_config.max_train_samples
            if split.startswith("train")
            else data_config.max_eval_samples
        )
        if cap is not None and cap < len(dataset):
            # Shuffle before slicing: SQuAD rows are grouped by article, so a raw
            # first-N slice would come from only a handful of titles and would not be
            # representative. Seeded, so the subset is reproducible.
            dataset = dataset.shuffle(seed=seed).select(range(cap))
            logger.info("Capped split %r to %d examples (seed=%d).", split, cap, seed)

    return dataset


def load_squad_splits(
    data_config: DataConfig,
    *,
    cache_dir: str | None = None,
    seed: int = 42,
    splits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load the train and validation splits together.

    Args:
        data_config: Dataset settings.
        cache_dir: Override the datasets cache location.
        seed: Seed used when a sample cap requires subsetting.
        splits: Which logical splits to load. Defaults to both.

    Returns:
        Mapping of ``"train"``/``"validation"`` to the loaded datasets.
    """
    wanted = tuple(splits) if splits is not None else ("train", "validation")
    resolved = {
        "train": data_config.train_split,
        "validation": data_config.validation_split,
    }
    return {
        name: load_squad_split(
            data_config, resolved[name], cache_dir=cache_dir, seed=seed
        )
        for name in wanted
    }


def assert_squad_schema(dataset: Any, split: str = "unknown") -> None:
    """Verify a dataset exposes the SQuAD columns with the expected shapes.

    Args:
        dataset: A ``datasets.Dataset``.
        split: Split name, used in error messages.

    Raises:
        DatasetSchemaError: If a column is missing, the split is empty, or the
            ``answers`` field is not the expected ``{text, answer_start}`` structure.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise DatasetSchemaError(
            f"Split {split!r} is missing required column(s): {', '.join(missing)}.\n"
            f"Found: {', '.join(dataset.column_names)}.\n"
            "Expected the SQuAD schema: id, title, context, question, answers."
        )

    if len(dataset) == 0:
        raise DatasetSchemaError(f"Split {split!r} contains no examples.")

    first = dataset[0]
    answers = first["answers"]
    if not isinstance(answers, dict):
        raise DatasetSchemaError(
            f"Split {split!r}: `answers` should be a mapping with 'text' and "
            f"'answer_start' keys, got {type(answers).__name__}."
        )
    for key in ("text", "answer_start"):
        if key not in answers:
            raise DatasetSchemaError(
                f"Split {split!r}: `answers` is missing {key!r}. "
                f"Found keys: {sorted(answers)}."
            )
    if not isinstance(answers["text"], list) or not isinstance(answers["answer_start"], list):
        raise DatasetSchemaError(
            f"Split {split!r}: `answers.text` and `answers.answer_start` must both be "
            "lists, because a validation example carries several accepted answers."
        )

    logger.debug("Schema check passed for split %r (%d examples).", split, len(dataset))


def assert_squad_v1(dataset: Any, split: str = "unknown", *, sample_size: int = 2000) -> None:
    """Refuse a dataset that looks like SQuAD 2.0.

    SQuAD 2.0 marks unanswerable questions with an empty ``answers.text``. Those rows
    need a null-answer threshold that this pipeline does not implement, so they are
    rejected loudly instead of being trained on as if answerable.

    Args:
        dataset: A ``datasets.Dataset``.
        split: Split name, used in error messages.
        sample_size: How many examples to inspect.

    Raises:
        DatasetSchemaError: If any inspected example has no answer.
    """
    limit = min(sample_size, len(dataset))
    empty_indices = [
        index for index in range(limit) if not dataset[index]["answers"]["text"]
    ]
    if empty_indices:
        raise DatasetSchemaError(
            f"Split {split!r} contains {len(empty_indices)} example(s) with no answer "
            f"in the first {limit} rows (e.g. index {empty_indices[0]}).\n"
            "That is the signature of SQuAD 2.0, which adds unanswerable questions.\n"
            "This pipeline implements SQuAD 1.1 extractive QA and has no null-answer "
            "threshold, so training on v2 would silently mishandle every unanswerable "
            "question. Set data.dataset_name to 'rajpurkar/squad'."
        )


def verify_answer_offsets(
    dataset: Any,
    *,
    sample_size: int | None = None,
    max_samples_recorded: int = 5,
) -> OffsetVerificationReport:
    """Check that ``answer_start`` offsets actually locate the answer text.

    Args:
        dataset: A ``datasets.Dataset``.
        sample_size: How many examples to inspect. ``None`` checks all of them.
        max_samples_recorded: How many mismatching examples to keep for inspection.

    Returns:
        The :class:`OffsetVerificationReport`. Mismatches are counted and sampled
        rather than raised on, because the correct response depends on the rate: a
        handful of noisy annotations is expected, while a large fraction means the
        context is being modified somewhere before offsets are used.
    """
    total = len(dataset)
    limit = total if sample_size is None else min(sample_size, total)

    exact = 0
    whitespace_only = 0
    mismatched = 0
    no_answer = 0
    samples: list[dict[str, Any]] = []

    for index in range(limit):
        row = dataset[index]
        answers = row["answers"]
        texts = answers["text"]
        starts = answers["answer_start"]
        if not texts or not starts:
            no_answer += 1
            continue

        context = row["context"]
        text = texts[0]
        start = int(starts[0])
        sliced = context[start : start + len(text)]

        if sliced == text:
            exact += 1
        elif sliced.strip() == text.strip():
            whitespace_only += 1
        else:
            mismatched += 1
            if len(samples) < max_samples_recorded:
                samples.append(
                    {
                        "id": row["id"],
                        "answer_text": text,
                        "answer_start": start,
                        "context_slice": sliced,
                    }
                )

    return OffsetVerificationReport(
        checked=limit,
        exact_matches=exact,
        whitespace_only_mismatches=whitespace_only,
        mismatches=mismatched,
        examples_with_no_answer=no_answer,
        mismatch_samples=samples,
    )


def summarize_split(dataset: Any, split: str) -> SplitSummary:
    """Compute descriptive statistics for a split.

    Args:
        dataset: A ``datasets.Dataset``.
        split: Split name.

    Returns:
        The :class:`SplitSummary`.
    """
    contexts = [len(text) for text in dataset["context"]]
    questions = [len(text) for text in dataset["question"]]
    answers = dataset["answers"]
    answer_lengths = [len(item["text"][0]) if item["text"] else 0 for item in answers]
    answer_counts = [len(item["text"]) for item in answers]

    return SplitSummary(
        split=split,
        num_examples=len(dataset),
        num_titles=len(set(dataset["title"])),
        context_chars=_stats(contexts),
        question_chars=_stats(questions),
        answer_chars=_stats(answer_lengths),
        answers_per_example=_stats(answer_counts),
    )


def build_reference_answers(dataset: Any) -> dict[str, list[str]]:
    """Build the ``example_id -> gold answers`` mapping used for scoring.

    Args:
        dataset: A ``datasets.Dataset`` with ``id`` and ``answers`` columns.

    Returns:
        Mapping of example id to its list of accepted answer strings. Validation
        examples contribute several; Exact Match and F1 take the maximum over them.
    """
    return {
        row_id: list(answers["text"])
        for row_id, answers in zip(dataset["id"], dataset["answers"], strict=True)
    }
