"""Question paper domain models, validation and assembly.

This package is the foundation of the question paper generator. It defines what a
question and a paper *are*, what makes one valid, and how a batch of questions becomes
a structured paper. It generates nothing.

Relationship to the extractive QA system
----------------------------------------
Independent of it, and deliberately so. ``qa_core``, ``qa_torch`` and the
DeBERTa-v3-base SQuAD checkpoint behind ``POST /predict`` are untouched by this
package. The points of contact are :mod:`qa_paper.fingerprint` and
:mod:`qa_paper.content`, both of which *reuse* ``qa_core``'s normalizer rather than
adding a second one that could drift from it.

:mod:`qa_paper.fingerprint` is worth a note. Duplicate detection needs
:func:`qa_core.normalize.normalize_answer`, but that function deletes all punctuation --
correct for scoring an answer span, wrong for a question, because it makes
``What is 2 + 2?`` and ``What is 2 - 2?`` the same string. So a question-specific layer
protects the symbols first and then delegates. ``qa_core`` is not modified: the EM and F1
of the trained model depend on it exactly as written.

Hard architectural rules
------------------------
1. **No FastAPI, no torch, no transformers.** This is domain logic. It must be usable
   from a CLI, a worker or a test with no web or ML stack present.
   ``tests/test_qa_paper_isolation.py`` asserts it, in the same way
   ``tests/test_qa_core_isolation.py`` guards ``qa_core``.
2. **No network access and no model calls.** Generation is an interface here, not an
   implementation.
3. **The generation layer is replaceable.** Everything concrete depends on
   :class:`qa_paper.interfaces.QuestionGenerator`, a ``Protocol``, so an adapter for an
   external API, a local model or a fine-tuned model can be added without touching
   assembly or validation.

Two conventions worth knowing before reading further
----------------------------------------------------
**Blueprints raise, questions report.** A blueprint is human-authored configuration, so
:meth:`qa_paper.blueprint.PaperBlueprint.validate` raises immediately, matching
:class:`qa_ml.config.ExperimentConfig`. Questions arrive from a generator, so
:func:`qa_paper.validation.validate_question` returns a
:class:`~qa_paper.validation.ValidationReport` instead. Raising there would discard a
whole batch over one bad option list and make "detects empty question text" untestable.

**Marks are integers.** Floating-point marks make ``sum(marks) == total_marks``
unreliable, and inconsistent totals are exactly what this system exists to catch. See
:mod:`qa_paper.blueprint` for the reasoning and the migration path.

Modules
-------
- :mod:`qa_paper.enums`       - question types and difficulty vocabularies
- :mod:`qa_paper.fingerprint` - question-specific normalization for duplicate detection
- :mod:`qa_paper.grounding`   - source provenance data model
- :mod:`qa_paper.questions`   - a question and its per-type payloads
- :mod:`qa_paper.blueprint`   - the specification a paper is generated from
- :mod:`qa_paper.paper`       - sections, the assembled paper, the answer key
- :mod:`qa_paper.assembly`    - grouping questions into a paper
- :mod:`qa_paper.validation`  - question, batch and paper checks
- :mod:`qa_paper.serialization` - rebuilding domain objects from plain mappings
- :mod:`qa_paper.interfaces`  - the replaceable generator and content-source contracts
- :mod:`qa_paper.content`     - loading, chunking and retrieving source material

:mod:`qa_paper.content` is a subpackage with its own import surface rather than being
re-exported here. Ingestion is a distinct concern from the paper domain, and flattening
thirty more names into this namespace would obscure which of them a caller assembling a
paper actually needs.
"""

from qa_paper.assembly import (
    AssemblyResult,
    assemble_paper,
    build_answer_key,
    default_section_title,
)
from qa_paper.blueprint import BlueprintError, PaperBlueprint, SectionPlan
from qa_paper.enums import Difficulty, DifficultyPolicy, QuestionType
from qa_paper.fingerprint import normalize_question_text, question_fingerprint
from qa_paper.grounding import ContentGrounding, SourceSpan
from qa_paper.interfaces import (
    ContentSource,
    GenerationRequest,
    GenerationResult,
    GeneratorCapabilities,
    GeneratorError,
    QuestionGenerator,
    SourcePassage,
)
from qa_paper.paper import AnswerKey, AnswerKeyEntry, QuestionPaper, Section
from qa_paper.questions import (
    PAYLOAD_TYPES,
    CaseScenarioPayload,
    FillBlankPayload,
    MatchFollowingPayload,
    MatchPair,
    McqPayload,
    Question,
    TrueFalsePayload,
)
from qa_paper.serialization import (
    SerializationError,
    answer_key_from_dict,
    blueprint_from_dict,
    grounding_from_dict,
    paper_from_dict,
    payload_from_dict,
    question_from_dict,
    section_from_dict,
    section_plan_from_dict,
)
from qa_paper.validation import (
    MINIMUM_MCQ_OPTIONS,
    IssueCode,
    Severity,
    ValidationError,
    ValidationIssue,
    ValidationReport,
    find_duplicate_questions,
    validate_paper,
    validate_question,
    validate_questions,
)

__version__ = "0.1.0"

__all__ = [
    "MINIMUM_MCQ_OPTIONS",
    "PAYLOAD_TYPES",
    "AnswerKey",
    "AnswerKeyEntry",
    "AssemblyResult",
    "BlueprintError",
    "CaseScenarioPayload",
    "ContentGrounding",
    "ContentSource",
    "Difficulty",
    "DifficultyPolicy",
    "FillBlankPayload",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorCapabilities",
    "GeneratorError",
    "IssueCode",
    "MatchFollowingPayload",
    "MatchPair",
    "McqPayload",
    "PaperBlueprint",
    "Question",
    "QuestionGenerator",
    "QuestionPaper",
    "QuestionType",
    "Section",
    "SectionPlan",
    "SerializationError",
    "Severity",
    "SourcePassage",
    "SourceSpan",
    "TrueFalsePayload",
    "ValidationError",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "answer_key_from_dict",
    "assemble_paper",
    "blueprint_from_dict",
    "build_answer_key",
    "default_section_title",
    "find_duplicate_questions",
    "grounding_from_dict",
    "normalize_question_text",
    "paper_from_dict",
    "payload_from_dict",
    "question_fingerprint",
    "question_from_dict",
    "section_from_dict",
    "section_plan_from_dict",
    "validate_paper",
    "validate_question",
    "validate_questions",
]
