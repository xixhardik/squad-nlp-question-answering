"""Official SQuAD answer normalization.

Reimplemented from the SQuAD v1.1 evaluation methodology rather than taken from
a library, so the exact behaviour is visible, testable and version-stable. The
``evaluate`` package is used only as an independent cross-check in a later
phase, never as the primary implementation.

Normalization order is significant and must not be rearranged:

    1. lowercase
    2. strip punctuation
    3. remove the articles ``a`` / ``an`` / ``the``
    4. collapse whitespace

Step 3 runs after step 2 so that a token such as ``"the,"`` still has its
article removed. Step 4 runs last because steps 2 and 3 both leave gaps behind.
"""

from __future__ import annotations

import re
import string

__all__ = ["get_answer_tokens", "normalize_answer"]

# Compiled once at import time: normalization runs on every prediction of every
# evaluation pass (10,570 dev examples x N candidate spans).
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)

# str.translate with a table is markedly faster than a per-character filter.
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize an answer string for Exact Match and F1 comparison.

    Args:
        text: Raw answer string, either predicted or a gold annotation.

    Returns:
        The normalized string. Returns ``""`` for input that normalizes away
        entirely (for example ``"the"`` or ``"..."``).

    Examples:
        >>> normalize_answer("The Amazon Rainforest.")
        'amazon rainforest'
        >>> normalize_answer("  BRAZIL  ")
        'brazil'
        >>> normalize_answer("an apple, a day")
        'apple day'
    """
    lowered = text.lower()
    without_punctuation = lowered.translate(_PUNCTUATION_TABLE)
    without_articles = _ARTICLES_RE.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def get_answer_tokens(text: str) -> list[str]:
    """Normalize ``text`` and split it into whitespace-delimited tokens.

    These are *scoring* tokens for the F1 metric, deliberately unrelated to the
    model's subword tokenizer. SQuAD F1 is defined over whitespace tokens of the
    normalized string, so using subword tokens here would silently change the
    metric.

    Args:
        text: Raw answer string.

    Returns:
        List of normalized tokens; empty when the string normalizes away.

    Examples:
        >>> get_answer_tokens("The Amazon Rainforest.")
        ['amazon', 'rainforest']
        >>> get_answer_tokens("the")
        []
    """
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()
