"""Enforces the architectural boundary around :mod:`qa_paper`.

Mirrors ``tests/test_qa_core_isolation.py``. ``qa_paper`` is domain logic for the
question paper generator, and it must stay usable from a CLI, a background worker or a
test with no web server and no ML stack installed. Phase 15 states that as a
requirement; this file makes it mechanical rather than aspirational, so a later phase
cannot casually import FastAPI into a schema module.

Checks run in a **subprocess** with a clean interpreter. In-process assertions would be
meaningless, because pytest collects sibling tests that import torch and fastapi, so
``sys.modules`` is already polluted before any assertion runs.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

FORBIDDEN_MODULES = ("torch", "transformers", "datasets", "fastapi", "evaluate", "pydantic")

SUBMODULES = (
    "enums",
    "fingerprint",
    "grounding",
    "questions",
    "blueprint",
    "paper",
    "assembly",
    "validation",
    "serialization",
    "interfaces",
    "content",
    "content.cleaning",
    "content.documents",
    "content.loaders",
    "content.chunking",
    "content.topics",
    "content.retrieval",
    "content.sources",
)


def _run_in_clean_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    """Execute ``code`` in a fresh interpreter and capture its output."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


class TestQaPaperHasNoHeavyDependencies:
    """qa_paper must import without pulling in the ML or web stack."""

    def test_importing_qa_paper_does_not_import_forbidden_modules(self):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_paper

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            "qa_paper imported a forbidden heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "CLEAN" in result.stdout

    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_each_submodule_is_independently_clean(self, submodule):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_paper.{submodule}

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            f"qa_paper.{submodule} leaked a heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    @pytest.mark.parametrize("module", ["qa_paper", "qa_paper.content"])
    def test_only_the_standard_library_and_qa_core_are_used(self, module):
        """The single permitted internal dependency is ``qa_core``, itself stdlib-only.

        ``qa_paper.fingerprint`` and ``qa_paper.content`` both reuse
        :func:`qa_core.normalize.normalize_answer` rather than adding a second
        normalizer that could drift from the first.

        ``qa_paper.content`` is checked separately because it is the module most likely
        to acquire a dependency: a PDF reader, a Markdown parser or an embedding
        library. ``markdown-it-py`` happens to be importable in this environment as a
        transitive dependency of ``rich``, and is deliberately not used -- it is absent
        from ``constraints.txt``, so depending on it would break the moment an unrelated
        package stopped pulling it in.
        """
        result = _run_in_clean_interpreter(
            f"""
            import sys

            before = set(sys.modules)
            import {module}  # noqa: F401
            new = set(sys.modules) - before

            third_party = sorted(
                name for name in new
                if not name.startswith(("qa_paper", "qa_core", "_", "encodings"))
                and "." not in name
                and name not in sys.stdlib_module_names
            )
            if third_party:
                print("THIRD_PARTY:" + ",".join(third_party))
                raise SystemExit(1)
            print("STDLIB_ONLY")
            """
        )
        assert result.returncode == 0, (
            f"{module} pulled in a third-party package.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "STDLIB_ONLY" in result.stdout

    @pytest.mark.parametrize("module", ["qa_paper", "qa_paper.content"])
    def test_no_network_module_is_imported(self, module):
        """Generation is an interface here. Nothing in this phase may open a socket.

        ``qa_paper.content`` is included because ingestion is where a URL loader or a
        remote embedding call would plausibly appear.
        """
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import {module}  # noqa: F401

            network = sorted(
                m for m in ("socket", "http", "urllib.request", "ssl", "httpx", "requests")
                if m in sys.modules
            )
            if network:
                print("NETWORK:" + ",".join(network))
                raise SystemExit(1)
            print("OFFLINE")
            """
        )
        assert result.returncode == 0, (
            f"{module} imported a networking module.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OFFLINE" in result.stdout


class TestQaPaperPublicApi:
    """The package must re-export the surface later phases will import."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "QuestionType",
            "Difficulty",
            "DifficultyPolicy",
            "Question",
            "McqPayload",
            "FillBlankPayload",
            "MatchFollowingPayload",
            "MatchPair",
            "TrueFalsePayload",
            "CaseScenarioPayload",
            "ContentGrounding",
            "SourceSpan",
            "PaperBlueprint",
            "SectionPlan",
            "BlueprintError",
            "QuestionPaper",
            "Section",
            "AnswerKey",
            "AnswerKeyEntry",
            "QuestionGenerator",
            "GenerationRequest",
            "GenerationResult",
            "GeneratorCapabilities",
            "GeneratorError",
            "ContentSource",
            "SourcePassage",
            "IssueCode",
            "Severity",
            "ValidationIssue",
            "ValidationReport",
            "ValidationError",
            "validate_question",
            "validate_questions",
            "validate_paper",
            "find_duplicate_questions",
            "normalize_question_text",
            "question_fingerprint",
            "assemble_paper",
            "build_answer_key",
            "question_from_dict",
            "paper_from_dict",
            "blueprint_from_dict",
        ],
    )
    def test_symbol_is_exported(self, symbol):
        import qa_paper

        assert hasattr(qa_paper, symbol)
        assert symbol in qa_paper.__all__

    @pytest.mark.parametrize(
        "symbol",
        [
            "SourceType",
            "SourceDocument",
            "ContentChunk",
            "ContentLoader",
            "ContentChunker",
            "ContentRetriever",
            "TextLoader",
            "MarkdownLoader",
            "ParagraphChunker",
            "BM25Retriever",
            "RetrievedChunk",
            "ContentCorpus",
            "DocumentContentSource",
            "ContentLoadError",
            "UnsupportedSourceError",
            "DEFERRED_SOURCE_TYPES",
            "clean_text",
            "load_document",
            "document_from_text",
            "extract_topics",
            "label_chunks",
            "document_from_dict",
            "chunk_from_dict",
        ],
    )
    def test_content_symbol_is_exported(self, symbol):
        """The content package is its own import surface, not re-exported by qa_paper."""
        from qa_paper import content

        assert hasattr(content, symbol)
        assert symbol in content.__all__

    def test_content_is_not_flattened_into_the_top_level_namespace(self):
        """Ingestion is a separate concern; thirty more names here would obscure it."""
        import qa_paper

        assert "ContentChunk" not in qa_paper.__all__
        assert "SourceDocument" not in qa_paper.__all__

    def test_version_is_declared(self):
        import qa_paper

        assert isinstance(qa_paper.__version__, str)
        assert qa_paper.__version__

    def test_all_entries_are_unique(self):
        """A repeated export is a merge artefact and hides which one is intended."""
        import qa_paper

        assert len(qa_paper.__all__) == len(set(qa_paper.__all__))

    def test_all_follows_the_same_ordering_convention_as_qa_core(self):
        """Ruff's isort-style ``__all__`` order: SCREAMING_CASE constants, then the rest.

        Asserted against the rule rather than ``sorted()``, because a plain sort is not
        the convention here -- ``qa_core.__all__`` begins ``CLS_TOKEN_SPAN`` before
        ``AlignmentResult`` and would fail such a check.
        """
        import qa_core
        import qa_paper

        def isort_style(names: tuple[str, ...] | list[str]) -> list[str]:
            constants = sorted(name for name in names if name.isupper())
            rest = sorted(name for name in names if not name.isupper())
            return constants + rest

        assert qa_core.__all__ == isort_style(qa_core.__all__), (
            "the reference convention changed; update this test deliberately"
        )
        assert qa_paper.__all__ == isort_style(qa_paper.__all__)

    def test_content_all_follows_the_same_convention(self):
        from qa_paper import content

        def isort_style(names: tuple[str, ...] | list[str]) -> list[str]:
            constants = sorted(name for name in names if name.isupper())
            rest = sorted(name for name in names if not name.isupper())
            return constants + rest

        assert content.__all__ == isort_style(content.__all__)
        assert len(content.__all__) == len(set(content.__all__))


class TestExistingQaSystemUntouched:
    """Phase 15 must not have altered the extractive QA path.

    A cheap regression guard: ``qa_paper`` is additive, so the QA surface that the
    trained DeBERTa checkpoint and ``POST /predict`` depend on must still import and
    expose the same names.
    """

    def test_qa_core_still_exposes_its_decoding_surface(self):
        import qa_core

        for symbol in ("decode_spans", "tighten_char_span", "compute_squad_metrics"):
            assert hasattr(qa_core, symbol)

    def test_qa_paper_does_not_shadow_or_patch_qa_core(self):
        """Importing qa_paper must not rebind anything in qa_core."""
        result = _run_in_clean_interpreter(
            """
            import qa_core
            before = id(qa_core.normalize_answer)
            import qa_paper  # noqa: F401
            after = id(qa_core.normalize_answer)
            print("SAME" if before == after else "PATCHED")
            """
        )
        assert "SAME" in result.stdout, result.stderr

    def test_inference_engine_module_still_imports(self):
        """Guards the ``ExtractiveQAEngine`` entry point the backend depends on."""
        from qa_torch.inference import ExtractiveQAEngine, PredictionResult

        assert hasattr(ExtractiveQAEngine, "answer")
        assert "answer" in PredictionResult.__dataclass_fields__
