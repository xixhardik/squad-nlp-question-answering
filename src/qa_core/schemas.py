"""Plain dataclass contracts shared by training, evaluation and inference.

Deliberately ``dataclasses`` rather than Pydantic models. ``qa_core`` must stay
dependency-free, and these objects are internal contracts rather than an HTTP
boundary. The FastAPI layer defines its own Pydantic schemas and converts at the
edge, which keeps the wire format free to evolve without touching core logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AnswerSpan", "EvaluationSummary", "ScoreType"]


class ScoreType:
    """Allowed values for the ``score_type`` field of a prediction.

    The project does not present a raw softmax value as calibrated confidence.
    ``UNCALIBRATED_SPAN_PROBABILITY`` denotes a probability obtained by a single
    softmax over the pooled set of *valid* candidate spans gathered from every
    sliding window of an example. That is a proper distribution over the
    hypotheses actually considered, and it is comparable across windows, but it
    is **not** calibrated: fine-tuned transformers are systematically
    overconfident, so a score of 0.9 does not mean a 90% chance of being right.

    ``TEMPERATURE_SCALED`` may only be used once a temperature has actually been
    fitted on held-out data and the resulting calibration error measured.
    """

    UNCALIBRATED_SPAN_PROBABILITY = "uncalibrated_span_probability"
    TEMPERATURE_SCALED = "temperature_scaled"


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    """One candidate answer span located in an original context string.

    Attributes:
        text: Answer text obtained by slicing the context. Never produced by
            decoding token ids.
        char_start: Inclusive character offset into the original context.
        char_end: Exclusive character offset into the original context.
        score: Span score. Interpretation is given by ``score_type``.
        score_type: One of the :class:`ScoreType` constants.
        token_start: Token index within the feature window, for debugging.
        token_end: Token index within the feature window, for debugging.
        window_index: Which sliding window produced this span, for debugging.
    """

    text: str
    char_start: int
    char_end: int
    score: float
    score_type: str = ScoreType.UNCALIBRATED_SPAN_PROBABILITY
    token_start: int | None = None
    token_end: int | None = None
    window_index: int | None = None

    def __post_init__(self) -> None:
        """Reject spans that cannot be valid, so bad data fails fast."""
        if self.char_start < 0 or self.char_end < 0:
            raise ValueError(
                f"Character offsets must be non-negative, got "
                f"({self.char_start}, {self.char_end})."
            )
        if self.char_start > self.char_end:
            raise ValueError(
                f"char_start ({self.char_start}) must not exceed "
                f"char_end ({self.char_end})."
            )

    @property
    def length(self) -> int:
        """Length of the span in characters."""
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate metrics for one evaluation pass over a dataset split.

    Attributes:
        exact_match: Exact Match as a percentage in ``[0, 100]``.
        f1: Token-level F1 as a percentage in ``[0, 100]``.
        total_examples: Number of examples scored. This is the metric's
            denominator and is recorded so a result can never be quoted without
            the sample size it came from.
        total_features: Number of tokenizer features the examples expanded into
            after sliding-window overflow. Exceeds ``total_examples`` whenever
            contexts are longer than the model's maximum sequence length.
        extra: Additional measured values (validation loss, latency, throughput)
            attached by the training pipeline.
    """

    exact_match: float
    f1: float
    total_examples: int
    total_features: int | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for experiment records."""
        payload: dict[str, object] = {
            "exact_match": round(self.exact_match, 4),
            "f1": round(self.f1, 4),
            "total_examples": self.total_examples,
        }
        if self.total_features is not None:
            payload["total_features"] = self.total_features
        if self.extra:
            payload["extra"] = {k: round(v, 6) for k, v in self.extra.items()}
        return payload
