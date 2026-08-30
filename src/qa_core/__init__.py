"""Core extractive question answering logic.

This package is the **single source of truth** for answer span handling. It is
imported by *both* the training/evaluation pipeline and the production inference
backend, so that reported metrics describe exactly the system that is served.

Hard architectural rule
-----------------------
``qa_core`` must never import ``torch``, ``transformers``, ``datasets`` or
``fastapi``. It depends on the standard library only.

That constraint exists for three reasons:

1. Its test suite runs in milliseconds and needs no GPU, no network and no
   model download, so span-decoding correctness is cheap to verify.
2. The FastAPI backend can import it without pulling in training machinery.
3. It makes the shared-implementation boundary mechanically enforceable rather
   than a matter of discipline. ``tests/test_qa_core_isolation.py`` asserts it.

Implemented in Phase 1
----------------------
- :mod:`qa_core.normalize` - official SQuAD answer normalization
- :mod:`qa_core.metrics`   - Exact Match and token-level F1
- :mod:`qa_core.spans`     - character span validation and tightening
- :mod:`qa_core.schemas`   - plain dataclass contracts

Arriving in later phases
------------------------
- ``qa_core.alignment``   - SQuAD character offsets -> token start/end positions
- ``qa_core.windows``     - sliding-window features and feature/example mapping
- ``qa_core.postprocess`` - n-best candidate span decoding across windows
"""

from qa_core.metrics import (
    compute_squad_metrics,
    exact_match_score,
    token_f1_score,
)
from qa_core.normalize import get_answer_tokens, normalize_answer
from qa_core.schemas import AnswerSpan, EvaluationSummary
from qa_core.spans import (
    InvalidSpanError,
    extract_answer_text,
    tighten_char_span,
    validate_char_span,
)

__version__ = "0.1.0"

__all__ = [
    "AnswerSpan",
    "EvaluationSummary",
    "InvalidSpanError",
    "__version__",
    "compute_squad_metrics",
    "exact_match_score",
    "extract_answer_text",
    "get_answer_tokens",
    "normalize_answer",
    "tighten_char_span",
    "token_f1_score",
    "validate_char_span",
]
