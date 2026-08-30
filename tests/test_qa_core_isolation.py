"""Enforces the architectural boundary around :mod:`qa_core`.

``qa_core`` is imported by both the training/evaluation pipeline and the
inference backend. That shared implementation is what guarantees reported metrics
describe the served system. To keep the boundary real rather than aspirational,
``qa_core`` must not depend on torch, transformers, datasets or fastapi.

Checks run in a **subprocess** with a clean interpreter. Doing it in-process
would be meaningless: pytest collects sibling tests that import torch, so
``sys.modules`` is already polluted by the time any assertion runs.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

FORBIDDEN_MODULES = ("torch", "transformers", "datasets", "fastapi", "evaluate")


def _run_in_clean_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    """Execute ``code`` in a fresh interpreter and capture its output."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


class TestQaCoreHasNoHeavyDependencies:
    """qa_core must import without pulling in the ML or web stack."""

    def test_importing_qa_core_does_not_import_forbidden_modules(self):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_core

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            "qa_core imported a forbidden heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "CLEAN" in result.stdout

    @pytest.mark.parametrize(
        "submodule", ["normalize", "metrics", "spans", "schemas"]
    )
    def test_each_submodule_is_independently_clean(self, submodule):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_core.{submodule}

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            f"qa_core.{submodule} leaked a heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_qa_core_uses_only_the_standard_library(self):
        """Phase 1 qa_core has zero third-party imports, not even numpy.

        numpy will be needed once logit post-processing arrives. Until then the
        stricter guarantee is worth asserting, because it means the span
        correctness suite can run anywhere with no install step at all.
        """
        result = _run_in_clean_interpreter(
            """
            import sys

            before = set(sys.modules)
            import qa_core  # noqa: F401
            new = set(sys.modules) - before

            third_party = sorted(
                name for name in new
                if not name.startswith(("qa_core", "_", "encodings"))
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
            "qa_core pulled in a third-party package.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "STDLIB_ONLY" in result.stdout


class TestQaCorePublicApi:
    """The package must re-export the shared surface that other layers import."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "normalize_answer",
            "get_answer_tokens",
            "exact_match_score",
            "token_f1_score",
            "compute_squad_metrics",
            "tighten_char_span",
            "extract_answer_text",
            "validate_char_span",
            "AnswerSpan",
            "EvaluationSummary",
            "InvalidSpanError",
        ],
    )
    def test_symbol_is_exported(self, symbol):
        import qa_core

        assert hasattr(qa_core, symbol)
        assert symbol in qa_core.__all__

    def test_version_is_declared(self):
        import qa_core

        assert isinstance(qa_core.__version__, str)
        assert qa_core.__version__
