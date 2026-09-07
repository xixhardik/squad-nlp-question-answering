r"""How big is this dataset, in the units that decide a training run.

Why character counts are not enough
-----------------------------------
:mod:`qa_gen.statistics` measures context, question and answer lengths in characters, and says
so: tokenising needs a tokenizer, which needs a download and a heavy dependency, in a package
that has neither. Characters are fine for spotting a truncated passage. They are useless for the
questions this module exists to answer:

- Does a rendered example fit in 1024 tokens, or is the target being cut off?
- How many optimiser steps is one epoch, at batch 1 and accumulation 8?
- How long will one epoch take, given a measured seconds-per-step?

All three need the *real* tokenizer, applying the *real* chat template, to the *real* rendered
record. A characters-divided-by-four estimate would be wrong in the one direction that matters:
Qwen's vocabulary handles English prose at roughly 3.6 characters per token but JSON punctuation
far less efficiently, so the completion -- which is entirely JSON -- is systematically
underestimated. Truncation silently removes the end of the target, which is the part that
carries the answer.

Tokenizer only, never weights
-----------------------------
Nothing here loads a model. :func:`load_sizing_tokenizer` calls the tokenizer loader and stops.
That is a few megabytes and a second or two, against roughly 8 GiB and a minute for the
checkpoint, and it means sizing can run anywhere -- including on a laptop, once the tokenizer is
cached, and including while a GPU is busy with something else.

Percentiles, not just extremes
------------------------------
:class:`TokenLengthSummary` reports p50 and p95 alongside the minimum, mean and maximum.
:class:`qa_gen.statistics.LengthSummary` reports only min, mean, max and count, which is the
right shape for a character sanity check and the wrong one for choosing a sequence length: a
maximum of 4000 tokens says nothing about whether raising ``max_seq_length`` is worth it, while
"p95 is 780 and the max is 4000" says the tail is a handful of outliers that should be filtered
rather than accommodated. The Phase 17A type is left alone; this is a separate, richer one.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qa_gen.config import GenerationExperimentConfig
from qa_gen.splitting import SplitName

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_STEP_PLANS",
    "DatasetSizing",
    "SizingError",
    "SplitSizing",
    "StepEstimate",
    "TokenLengthSummary",
    "build_step_estimates",
    "estimate_step_count",
    "estimate_tokenization_seconds",
    "load_sizing_tokenizer",
    "measure_record_lengths",
    "split_names",
    "summarize_token_lengths",
]

#: Batch, accumulation and epoch combinations reported by default. The first is the measured
#: L4 configuration; the rest are the plausible next steps up from it, so the report answers
#: "what would this cost" without being re-run per guess.
DEFAULT_STEP_PLANS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, 8, 1),
    (1, 8, 2),
    (1, 8, 3),
    (2, 8, 2),
    (4, 4, 2),
)

#: Percentiles reported for every length distribution.
_PERCENTILES: tuple[int, ...] = (50, 95)


class SizingError(RuntimeError):
    """Raised when a dataset cannot be sized.

    Separate from :class:`qa_gen_runtime.sources.SourceLoadError` because the causes call for
    different responses: a corpus that will not load is an input problem, whereas a tokenizer
    that will not load is an environment problem.
    """


def _token_ids(result: Any, *, what: str) -> list[int]:
    """Extract a flat list of token ids from whatever ``apply_chat_template`` returned.

    This exists because of a real bug, and the bug is worth recording. transformers 5.x
    defaults ``apply_chat_template(..., tokenize=True)`` to ``return_dict=True``, so it returns
    a ``BatchEncoding`` rather than a list of ints. ``len()`` of that mapping is the number of
    keys -- ``input_ids`` and ``attention_mask``, so **2** -- not the sequence length. A first
    Phase 17C sizing run over 1,996 real SQuAD records therefore reported a prompt of exactly
    2 tokens and a completion of exactly 0 for every single example, and a truncation rate of
    zero, because 2 is comfortably under 1024.

    Taking a length from a shape you did not verify is the whole failure. So the ids are
    extracted explicitly here rather than by calling ``len()`` on the return value, every
    branch is handled, and anything unrecognised raises instead of producing a number.

    Args:
        result: Whatever the tokenizer returned.
        what: What was being tokenized, for the error message.

    Returns:
        The token ids.

    Raises:
        SizingError: If ids cannot be extracted. A raise beats a plausible-looking length: the
            symptom of getting this wrong is a report full of confident, meaningless numbers.
    """
    payload: Any = result
    if hasattr(payload, "keys"):
        if "input_ids" not in payload:
            raise SizingError(
                f"tokenizing {what} returned a mapping with keys {sorted(payload.keys())} and "
                "no 'input_ids'. The sizing measurement cannot proceed without the token ids."
            )
        payload = payload["input_ids"]
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise SizingError(
            f"tokenizing {what} returned {type(result).__name__}, which is not a sequence of "
            "token ids. Expected a list of ints, or a mapping carrying 'input_ids'."
        )
    if payload and isinstance(payload[0], Sequence) and not isinstance(payload[0], str):
        # A batch of one. Some processors return lists of lists even for a single example,
        # which TRL normalises the same way.
        payload = payload[0]
    try:
        return [int(item) for item in payload]
    except (TypeError, ValueError) as exc:
        raise SizingError(
            f"tokenizing {what} returned a sequence that is not token ids: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _percentile(ordered: Sequence[int], percentile: float) -> int:
    """Return a percentile of an already-sorted sequence, by nearest rank.

    Nearest rank rather than linear interpolation, because these are token counts and an
    interpolated 780.5 tokens is not a thing. The convention is the one ``numpy`` calls
    ``"lower"``, chosen so a reported p95 is always a length some real example actually has.

    Args:
        ordered: Values, ascending. Must be non-empty.
        percentile: The percentile, 0-100.

    Returns:
        The value at that rank.
    """
    if not ordered:
        return 0
    rank = math.ceil(percentile / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


@dataclass(frozen=True, slots=True)
class TokenLengthSummary:
    """A token-length distribution, in the shape a sequence-length decision needs.

    Attributes:
        count: How many values were measured.
        minimum: Shortest.
        mean: Arithmetic mean, rounded.
        p50: Median, by nearest rank.
        p95: 95th percentile, by nearest rank.
        maximum: Longest.
        total: Sum of all lengths. Carried because throughput is tokens per second, not
            examples per second, so an epoch's cost is a function of this rather than of the
            example count.
    """

    count: int = 0
    minimum: int = 0
    mean: float = 0.0
    p50: int = 0
    p95: int = 0
    maximum: int = 0
    total: int = 0

    @classmethod
    def from_values(cls, values: Sequence[int]) -> TokenLengthSummary:
        """Summarize a sequence of token counts.

        Args:
            values: The counts. An empty sequence yields an all-zero summary rather than an
                error, so a report's shape does not depend on whether a split was populated.

        Returns:
            The summary.
        """
        if not values:
            return cls()
        ordered = sorted(values)
        return cls(
            count=len(ordered),
            minimum=ordered[0],
            mean=round(sum(ordered) / len(ordered), 2),
            p50=_percentile(ordered, 50),
            p95=_percentile(ordered, 95),
            maximum=ordered[-1],
            total=sum(ordered),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "count": self.count,
            "min": self.minimum,
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class StepEstimate:
    """What one training configuration would cost, in steps.

    Attributes:
        train_examples: Examples in the training split.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        epochs: Passes over the training split.
        effective_batch_size: Examples per optimiser step.
        steps_per_epoch: Optimiser steps in one epoch.
        total_steps: Optimiser steps across every epoch.
        warmup_steps: Steps of warmup, from the configured ratio.
        estimated_seconds: Wall clock at :attr:`seconds_per_step`, or ``None`` when no
            measured rate was supplied.
        seconds_per_step: The measured rate this estimate assumed.
    """

    train_examples: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    effective_batch_size: int
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int = 0
    estimated_seconds: float | None = None
    seconds_per_step: float | None = None

    @property
    def label(self) -> str:
        """A compact identifier, e.g. ``"b1xa8xe2"``."""
        return (
            f"b{self.batch_size}"
            f"xa{self.gradient_accumulation_steps}"
            f"xe{self.epochs}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "label": self.label,
            "train_examples": self.train_examples,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
            "effective_batch_size": self.effective_batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "seconds_per_step": self.seconds_per_step,
            "estimated_seconds": self.estimated_seconds,
            "estimated_hours": (
                round(self.estimated_seconds / 3600.0, 2)
                if self.estimated_seconds is not None
                else None
            ),
        }


def estimate_step_count(
    train_examples: int,
    *,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    epochs: int = 1,
    warmup_ratio: float = 0.0,
    seconds_per_step: float | None = None,
) -> StepEstimate:
    """Compute the optimiser-step count for one configuration.

    The arithmetic transformers itself uses: one optimiser step consumes
    ``batch_size * gradient_accumulation_steps`` examples, and a partial final batch still
    counts as a step, hence ``ceil``. Worth computing here rather than in a reader's head,
    because the wrong version of it -- dividing by batch size and forgetting accumulation --
    understates the run by the accumulation factor.

    Args:
        train_examples: Size of the training split.
        batch_size: Micro-batch per device.
        gradient_accumulation_steps: Micro-batches per optimiser step.
        epochs: Passes over the training split.
        warmup_ratio: Warmup as a share of total steps, converted the same way
            :func:`qa_gen_runtime.trainer.resolve_warmup_steps` converts it.
        seconds_per_step: A measured rate, to turn steps into wall clock. ``None`` leaves the
            time estimate absent rather than guessing one.

    Returns:
        The :class:`StepEstimate`.

    Raises:
        SizingError: If any of the three counts is not positive. A zero would produce a
            division error or a run that trains on nothing.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("epochs", epochs),
    ):
        if value <= 0:
            raise SizingError(f"{name} must be a positive integer, got {value}.")

    effective = batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(train_examples / effective)) if train_examples else 0
    total = steps_per_epoch * epochs
    warmup = max(1, round(total * warmup_ratio)) if warmup_ratio > 0 and total else 0
    return StepEstimate(
        train_examples=train_examples,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        epochs=epochs,
        effective_batch_size=effective,
        steps_per_epoch=steps_per_epoch,
        total_steps=total,
        warmup_steps=warmup,
        seconds_per_step=seconds_per_step,
        estimated_seconds=(
            round(total * seconds_per_step, 1) if seconds_per_step is not None else None
        ),
    )


@dataclass(frozen=True, slots=True)
class SplitSizing:
    """Token-length measurements for one split.

    Attributes:
        split: Which split.
        examples: Examples measured.
        prompt_tokens: Prompt-half length distribution.
        completion_tokens: Completion-half length distribution.
        total_tokens: Full rendered sequence length distribution.
        truncated: Sequences at or above ``max_seq_length``, whose target is being cut.
        max_seq_length: The limit compared against.
        records_missing_template_kwargs: Records with no ``chat_template_kwargs`` column. On
            Qwen3 that means the measured completion still contains the empty
            ``<think></think>`` block, because the block only moves into the prompt when the
            reasoning flag is passed. A non-zero count on a real run means the records were
            built without a tokenizer, and the completion lengths are overstated by four
            tokens each.
    """

    split: str
    examples: int = 0
    prompt_tokens: TokenLengthSummary = field(default_factory=TokenLengthSummary)
    completion_tokens: TokenLengthSummary = field(default_factory=TokenLengthSummary)
    total_tokens: TokenLengthSummary = field(default_factory=TokenLengthSummary)
    truncated: int = 0
    max_seq_length: int = 0
    records_missing_template_kwargs: int = 0

    @property
    def truncation_rate(self) -> float:
        """Share of sequences that would be truncated, rounded."""
        if not self.examples:
            return 0.0
        return round(self.truncated / self.examples, 4)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "split": self.split,
            "examples": self.examples,
            "max_seq_length": self.max_seq_length,
            "prompt_tokens": self.prompt_tokens.as_dict(),
            "completion_tokens": self.completion_tokens.as_dict(),
            "total_tokens": self.total_tokens.as_dict(),
            "truncated": self.truncated,
            "truncation_rate": self.truncation_rate,
            "records_missing_template_kwargs": self.records_missing_template_kwargs,
        }


@dataclass(frozen=True, slots=True)
class DatasetSizing:
    """Token-length measurements across every split, plus step estimates.

    Attributes:
        tokenizer_id: The tokenizer that produced these numbers.
        max_seq_length: The limit compared against.
        measured: Whether a real tokenizer was used. ``False`` means the length fields are
            absent rather than estimated -- there is no character-based fallback, deliberately.
        splits: Per-split measurements.
        overall: Measurements across every split combined.
        step_estimates: One entry per requested training configuration.
        sampled: Examples per split actually measured, when sampling was requested.
        notes: Anything worth recording.
    """

    tokenizer_id: str = ""
    max_seq_length: int = 0
    measured: bool = False
    splits: dict[str, SplitSizing] = field(default_factory=dict)
    overall: SplitSizing | None = None
    step_estimates: tuple[StepEstimate, ...] = ()
    sampled: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "step_estimates", tuple(self.step_estimates))
        object.__setattr__(self, "notes", tuple(self.notes))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "tokenizer_id": self.tokenizer_id,
            "max_seq_length": self.max_seq_length,
            "measured": self.measured,
            "sampled": self.sampled,
            "splits": {name: sizing.as_dict() for name, sizing in sorted(self.splits.items())},
            "overall": self.overall.as_dict() if self.overall else None,
            "step_estimates": [estimate.as_dict() for estimate in self.step_estimates],
            "notes": list(self.notes),
        }


def load_sizing_tokenizer(config: GenerationExperimentConfig) -> Any:
    """Load the tokenizer, and nothing else.

    Args:
        config: The experiment configuration, for the tokenizer id and revision.

    Returns:
        The tokenizer.

    Raises:
        SizingError: If it cannot be loaded. The message says that only the tokenizer was
            wanted, because "could not load Qwen3-4B" reads like a weights problem and would
            send someone looking at their GPU.
    """
    from qa_gen_runtime.loader import load_tokenizer

    try:
        return load_tokenizer(config.model)
    except Exception as exc:  # noqa: BLE001 - transformers raises a wide family here
        raise SizingError(
            f"could not load the tokenizer for {config.model.effective_tokenizer_id!r} at "
            f"revision {config.model.revision!r}: {type(exc).__name__}: {exc}\n"
            "Only the tokenizer is needed for sizing -- no model weights are loaded -- so this "
            "is a network, cache or revision problem rather than a GPU one."
        ) from exc


def measure_record_lengths(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    config: GenerationExperimentConfig,
    *,
    split: str = "train",
) -> SplitSizing:
    """Tokenize rendered records and summarize their lengths.

    The prompt and completion halves are tokenized exactly the way TRL does during dataset
    preparation -- the prompt with ``add_generation_prompt=True``, the whole conversation
    without -- so the reported completion length is the number of tokens the loss will actually
    cover. Deriving it any other way would produce a figure that disagrees with the trainer,
    which is the failure Phase 17B.2 spent an inspection run diagnosing.

    Args:
        records: Conversational TRL records from
            :func:`qa_gen_runtime.dataset.build_training_records`.
        tokenizer: The real tokenizer.
        config: The experiment configuration, for the chat handling and the length limit.
        split: Which split these records belong to, for the report.

    Returns:
        The :class:`SplitSizing`.

    Raises:
        SizingError: If a record is not the conversational shape, or the tokenizer has no chat
            template. Both would otherwise produce lengths measured on something other than
            what gets trained.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise SizingError(
            f"the tokenizer for {config.model.model_id!r} has no chat template, so a rendered "
            "length cannot be measured. Sizing a conversational dataset without one would "
            "measure a format the model is not trained on."
        )

    limit = config.model.max_seq_length
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    truncated = 0
    missing_template_kwargs = 0

    for position, record in enumerate(records):
        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, list) or not isinstance(completion, list):
            raise SizingError(
                f"record {position} in split {split!r} is not the conversational shape: "
                f"expected list-valued 'prompt' and 'completion', got "
                f"{type(prompt).__name__} and {type(completion).__name__}. Sizing measures "
                "the records the trainer receives, so the shape has to match."
            )
        # Only the record's own column, with no fall back to a config-derived default. TRL
        # reads nothing else, so a default here would measure a boundary the trainer will not
        # use -- which is precisely the disagreement Phase 17B.2 diagnosed.
        column = record.get("chat_template_kwargs")
        if not column:
            missing_template_kwargs += 1
        extra = dict(column or {})
        # return_dict is passed explicitly, exactly as TRL v0.29.1 passes it: False for the
        # prompt-only rendering, True for the combined one. transformers 5.x defaults it to
        # True, so omitting it returns a BatchEncoding whose len() is the number of keys.
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                [dict(message) for message in prompt],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **extra,
            ),
            what=f"the prompt of record {position} in split {split!r}",
        )
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                [dict(message) for message in [*prompt, *completion]],
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                **extra,
            ),
            what=f"record {position} in split {split!r}",
        )
        prompt_length = len(prompt_ids)
        total_length = len(full_ids)
        prompt_lengths.append(prompt_length)
        completion_lengths.append(max(0, total_length - prompt_length))
        total_lengths.append(total_length)
        if total_length > limit:
            truncated += 1

    return SplitSizing(
        split=split,
        examples=len(records),
        prompt_tokens=TokenLengthSummary.from_values(prompt_lengths),
        completion_tokens=TokenLengthSummary.from_values(completion_lengths),
        total_tokens=TokenLengthSummary.from_values(total_lengths),
        truncated=truncated,
        max_seq_length=limit,
        records_missing_template_kwargs=missing_template_kwargs,
    )


def summarize_token_lengths(per_split: dict[str, SplitSizing]) -> SplitSizing:
    """Combine per-split measurements into one overall distribution.

    The summaries are recombined from their totals and extremes rather than from the raw
    lengths, which are not retained. Means and extremes come out exact; the percentiles are
    the *widest* of the per-split percentiles rather than a true pooled percentile, and that
    is what the note in the report says. Retaining every length to compute an exact pooled
    p95 would mean holding a hundred thousand integers to refine a planning figure by a token
    or two.

    Args:
        per_split: Per-split measurements.

    Returns:
        A :class:`SplitSizing` labelled ``"overall"``.
    """
    populated = [sizing for sizing in per_split.values() if sizing.examples]
    if not populated:
        return SplitSizing(split="overall")

    def combine(pick: str) -> TokenLengthSummary:
        summaries = [getattr(sizing, pick) for sizing in populated]
        count = sum(summary.count for summary in summaries)
        total = sum(summary.total for summary in summaries)
        return TokenLengthSummary(
            count=count,
            minimum=min(summary.minimum for summary in summaries),
            mean=round(total / count, 2) if count else 0.0,
            p50=max(summary.p50 for summary in summaries),
            p95=max(summary.p95 for summary in summaries),
            maximum=max(summary.maximum for summary in summaries),
            total=total,
        )

    return SplitSizing(
        split="overall",
        examples=sum(sizing.examples for sizing in populated),
        prompt_tokens=combine("prompt_tokens"),
        completion_tokens=combine("completion_tokens"),
        total_tokens=combine("total_tokens"),
        truncated=sum(sizing.truncated for sizing in populated),
        max_seq_length=max(sizing.max_seq_length for sizing in populated),
        records_missing_template_kwargs=sum(
            sizing.records_missing_template_kwargs for sizing in populated
        ),
    )


def estimate_tokenization_seconds(
    example_count: int, *, records_per_second: float = 900.0
) -> dict[str, Any]:
    """Estimate how long tokenizing a corpus will take.

    A planning figure, and labelled as one. The rate is a default rather than a measurement:
    the fast Rust tokenizer processes single sequences at a few thousand per second on a modern
    CPU, and this pipeline tokenizes each record twice -- once for the prompt, once for the
    whole conversation. 900 records per second is a deliberately conservative single-process
    figure. The sizing report overwrites it with the rate actually observed, so the estimate is
    only ever used before the first run.

    Args:
        example_count: Records to tokenize.
        records_per_second: Assumed throughput.

    Returns:
        A mapping with the assumption stated alongside the result, so no reader mistakes it for
        a measurement.

    Raises:
        SizingError: If the rate is not positive.
    """
    if records_per_second <= 0:
        raise SizingError(
            f"records_per_second must be positive, got {records_per_second}."
        )
    seconds = example_count / records_per_second
    return {
        "example_count": example_count,
        "assumed_records_per_second": records_per_second,
        "estimated_seconds": round(seconds, 1),
        "estimated_minutes": round(seconds / 60.0, 2),
        "measured": False,
        "note": (
            "planning estimate, not a measurement: each record is tokenized twice (prompt, "
            "then prompt plus completion) and the assumed rate is a conservative "
            "single-process figure for the fast tokenizer"
        ),
    }


def build_step_estimates(
    train_examples: int,
    *,
    plans: Sequence[tuple[int, int, int]] = DEFAULT_STEP_PLANS,
    warmup_ratio: float = 0.0,
    seconds_per_step: float | None = None,
) -> tuple[StepEstimate, ...]:
    """Compute a step estimate for each requested configuration.

    Args:
        train_examples: Size of the training split.
        plans: ``(batch_size, gradient_accumulation_steps, epochs)`` triples.
        warmup_ratio: Warmup share, applied to every plan.
        seconds_per_step: A measured rate, to turn steps into wall clock.

    Returns:
        One estimate per plan, in the order given, duplicates removed.

    Raises:
        SizingError: If a plan holds a non-positive value.
    """
    seen: set[tuple[int, int, int]] = set()
    estimates: list[StepEstimate] = []
    for batch_size, accumulation, epochs in plans:
        key = (batch_size, accumulation, epochs)
        if key in seen:
            continue
        seen.add(key)
        estimates.append(
            estimate_step_count(
                train_examples,
                batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                epochs=epochs,
                warmup_ratio=warmup_ratio,
                seconds_per_step=seconds_per_step,
            )
        )
    return tuple(estimates)


def split_names() -> tuple[str, ...]:
    """Return the split names in reporting order."""
    return tuple(member.value for member in SplitName)
