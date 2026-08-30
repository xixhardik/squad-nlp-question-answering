"""Core extractive question answering logic.

This package is the **single source of truth** for answer span handling. It is
imported by *both* the training/evaluation pipeline and the inference layer, so
that reported metrics describe exactly the system that is served.

Hard architectural rule
-----------------------
``qa_core`` must never import ``torch``, ``transformers``, ``datasets`` or
``fastapi``. It depends on the standard library only -- not even numpy.

That constraint exists for four reasons:

1. Its test suite runs in milliseconds and needs no GPU, no network and no model
   download, so span-decoding correctness is cheap to verify exhaustively.
2. The inference layer can import it without pulling in training machinery.
3. It makes the shared-implementation boundary mechanically enforceable rather
   than a matter of discipline. ``tests/test_qa_core_isolation.py`` asserts it.
4. Every function here takes plain Python data (lists of offset tuples, lists of
   floats), which means synthetic fixtures are enough to test the hard parts:
   character/token alignment and span decoding.

Modules
-------
- :mod:`qa_core.normalize`   - official SQuAD answer normalization
- :mod:`qa_core.metrics`     - Exact Match and token-level F1
- :mod:`qa_core.spans`       - character span validation, tightening, extraction
- :mod:`qa_core.alignment`   - SQuAD character offsets to token start/end positions
- :mod:`qa_core.postprocess` - n-best candidate span decoding across windows
- :mod:`qa_core.schemas`     - plain dataclass contracts
"""

from qa_core.alignment import (
    CLS_TOKEN_SPAN,
    AlignmentResult,
    AlignmentStatus,
    align_answer_to_tokens,
    find_context_token_range,
    mask_non_context_offsets,
)
from qa_core.metrics import (
    compute_squad_metrics,
    exact_match_score,
    token_f1_score,
)
from qa_core.normalize import get_answer_tokens, normalize_answer
from qa_core.postprocess import DecodedAnswer, WindowLogits, decode_spans
from qa_core.schemas import AnswerSpan, EvaluationSummary, ScoreType
from qa_core.spans import (
    InvalidSpanError,
    extract_answer_text,
    tighten_char_span,
    validate_char_span,
)

__version__ = "0.2.0"

__all__ = [
    "CLS_TOKEN_SPAN",
    "AlignmentResult",
    "AlignmentStatus",
    "AnswerSpan",
    "DecodedAnswer",
    "EvaluationSummary",
    "InvalidSpanError",
    "ScoreType",
    "WindowLogits",
    "__version__",
    "align_answer_to_tokens",
    "compute_squad_metrics",
    "decode_spans",
    "exact_match_score",
    "extract_answer_text",
    "find_context_token_range",
    "get_answer_tokens",
    "mask_non_context_offsets",
    "normalize_answer",
    "tighten_char_span",
    "token_f1_score",
    "validate_char_span",
]
