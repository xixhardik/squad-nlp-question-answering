"""Exact Match and token-level F1 for extractive question answering.

Follows the SQuAD v1.1 evaluation methodology:

- A dev example carries **several** acceptable gold answers. An example's score
  is the **maximum** over golds, not the mean, because any annotator-accepted
  answer is correct.
- F1 is computed over multiset token overlap, so repeated tokens are handled
  correctly (``"new york new york"`` vs ``"new york"`` must not score 1.0).
- The corpus score is the unweighted mean of per-example scores, multiplied by
  100 to give the conventional percentage.

Empty-string handling matches the official script: if either side normalizes to
empty, F1 is 1.0 only when *both* are empty, else 0.0. Computing precision or
recall on an empty token list would divide by zero.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from qa_core.normalize import get_answer_tokens, normalize_answer
from qa_core.schemas import EvaluationSummary

__all__ = [
    "compute_squad_metrics",
    "exact_match_score",
    "token_f1_score",
]


def _exact_match_single(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when the two strings match after normalization, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def _token_f1_single(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between one prediction and one gold answer."""
    pred_tokens = get_answer_tokens(prediction)
    gold_tokens = get_answer_tokens(ground_truth)

    # Official edge case: with an empty side, F1 is defined by equality alone.
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    # Multiset intersection: Counter & Counter keeps the minimum count per token.
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truths: Sequence[str]) -> float:
    """Best Exact Match of ``prediction`` against any gold answer.

    Args:
        prediction: Predicted answer string.
        ground_truths: All annotator-accepted answers for the example.

    Returns:
        ``1.0`` if the prediction exactly matches any gold answer after
        normalization, otherwise ``0.0``. Returns ``0.0`` when no golds are
        supplied, since correctness cannot be established.
    """
    if not ground_truths:
        return 0.0
    return max(_exact_match_single(prediction, gold) for gold in ground_truths)


def token_f1_score(prediction: str, ground_truths: Sequence[str]) -> float:
    """Best token-level F1 of ``prediction`` against any gold answer.

    Args:
        prediction: Predicted answer string.
        ground_truths: All annotator-accepted answers for the example.

    Returns:
        F1 in ``[0.0, 1.0]``, maximized over the gold answers. Returns ``0.0``
        when no golds are supplied.
    """
    if not ground_truths:
        return 0.0
    return max(_token_f1_single(prediction, gold) for gold in ground_truths)


def compute_squad_metrics(
    predictions: Mapping[str, str],
    references: Mapping[str, Sequence[str]],
) -> EvaluationSummary:
    """Aggregate Exact Match and F1 over a dataset split.

    Args:
        predictions: Maps SQuAD example id to the predicted answer string.
        references: Maps SQuAD example id to its list of gold answer strings.

    Returns:
        An :class:`~qa_core.schemas.EvaluationSummary` with ``exact_match`` and
        ``f1`` as percentages in ``[0, 100]``.

    Raises:
        ValueError: If ``references`` is empty, or if any reference id has no
            corresponding prediction. A silently missing prediction would
            inflate the score by shrinking the denominator, so it is rejected
            rather than skipped.
    """
    if not references:
        raise ValueError("`references` is empty; nothing to evaluate.")

    missing = [key for key in references if key not in predictions]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} reference id(s) have no prediction (e.g. {preview}). "
            "Every reference must be predicted, otherwise the reported metric is "
            "computed over a smaller denominator and is not comparable."
        )

    total = len(references)
    em_sum = 0.0
    f1_sum = 0.0
    for example_id, golds in references.items():
        prediction = predictions[example_id]
        em_sum += exact_match_score(prediction, golds)
        f1_sum += token_f1_score(prediction, golds)

    return EvaluationSummary(
        exact_match=100.0 * em_sum / total,
        f1=100.0 * f1_sum / total,
        total_examples=total,
    )
