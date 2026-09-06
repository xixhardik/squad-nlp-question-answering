"""Enforces the architectural boundary around :mod:`qa_gen`.

Mirrors ``tests/test_qa_core_isolation.py`` and ``tests/test_qa_paper_isolation.py``. Phase 17A
states that the generative training foundation must be specifiable without the ML stack, without
a GPU, without a network and without a 4B checkpoint. This file makes that mechanical rather
than aspirational, so Phase 17B cannot casually import ``trl`` into a schema module and
Phase 17A's cheap, fast test suite cannot silently become a slow one.

The list of forbidden modules is longer here than for the other packages, because this is the
package most likely to acquire them: it is *about* training a transformer, and the obvious next
step for every module in it is to reach for ``transformers``.

Checks run in a **subprocess** with a clean interpreter. In-process assertions would be
meaningless, because pytest collects sibling tests that import torch and fastapi, so
``sys.modules`` is already polluted before any assertion runs.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

#: Every heavy dependency this package must not pull in. ``trl``, ``peft`` and
#: ``bitsandbytes`` are named even though they are not installed: naming them means the check
#: keeps working after Phase 17B adds them to the environment, which is exactly when it starts
#: to matter.
FORBIDDEN_MODULES = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "peft",
    "bitsandbytes",
    "accelerate",
    "evaluate",
    "fastapi",
    "pydantic",
    "yaml",
    "numpy",
    "pandas",
    "huggingface_hub",
)

SUBMODULES = (
    "examples",
    "prompts",
    "config",
    "adapters",
    "splitting",
    "validation",
    "statistics",
    "metadata",
    "serialization",
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


def code_string_literals(submodule: str) -> list[str]:
    """Return every string literal in one module's code, excluding docstrings.

    The companion to :func:`code_identifiers` for checks about what a module *emits*. The
    docstrings are excluded because several of them name the things the module deliberately
    avoids -- ``qa_gen.prompts`` explains at length that it contains no Qwen-specific markup,
    and a naive text search would flag that explanation as a violation of itself.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(f"qa_gen.{submodule}")))

    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_nodes.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


def code_identifiers(submodule: str) -> set[str]:
    """Return every name and attribute referenced in one module's executable code.

    Parsed rather than grepped. A substring search over the source text trips over the
    docstrings, which name ``AutoModelForQuestionAnswering`` and ``random.seed()`` precisely
    to explain why they are *not* used -- so a text search would flag the very comments that
    document the constraint. The AST sees only real references.
    """
    import ast
    import importlib
    import inspect

    tree = ast.parse(inspect.getsource(importlib.import_module(f"qa_gen.{submodule}")))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            names.update(alias.name for alias in node.names)
    return names


class TestQaGenHasNoHeavyDependencies:
    """qa_gen must import without pulling in the ML or web stack."""

    def test_importing_qa_gen_does_not_import_forbidden_modules(self):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_gen

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            "qa_gen imported a forbidden heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "CLEAN" in result.stdout

    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_each_submodule_is_independently_clean(self, submodule):
        result = _run_in_clean_interpreter(
            f"""
            import sys
            import qa_gen.{submodule}

            forbidden = {FORBIDDEN_MODULES!r}
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("CLEAN")
            """
        )
        assert result.returncode == 0, (
            f"qa_gen.{submodule} leaked a heavy dependency.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_qa_gen_uses_only_the_standard_library_qa_core_and_qa_paper(self):
        """The two permitted internal dependencies are themselves stdlib-only.

        ``qa_paper`` supplies the question-type and difficulty vocabulary and the fingerprint
        normalizer; ``qa_core`` supplies the SQuAD normalizer and the EM/F1 metrics. Reusing
        both rather than reimplementing them is what keeps the generative and extractive halves
        of the project describing the same things.
        """
        result = _run_in_clean_interpreter(
            """
            import sys

            before = set(sys.modules)
            import qa_gen  # noqa: F401
            new = set(sys.modules) - before

            third_party = sorted(
                name for name in new
                if not name.startswith(("qa_gen", "qa_paper", "qa_core", "_", "encodings"))
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
            "qa_gen pulled in a third-party package.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "STDLIB_ONLY" in result.stdout

    def test_no_network_module_is_imported(self):
        """No dataset download, no model download, no socket."""
        result = _run_in_clean_interpreter(
            """
            import sys
            import qa_gen  # noqa: F401

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
            f"qa_gen imported a networking module.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OFFLINE" in result.stdout

    def test_the_full_offline_pipeline_runs_in_a_clean_interpreter(self):
        """Adapt, validate, split, summarise and prompt, with nothing but the stdlib present."""
        result = _run_in_clean_interpreter(
            """
            import sys
            from qa_gen import (
                DEFAULT_TEMPLATE,
                DeterministicGroupSplitter,
                adapt_records,
                compute_statistics,
                experiment_config_from_dict,
                validate_dataset,
            )

            context = (
                "The mitochondrion generates most of the cell's supply of adenosine "
                "triphosphate, used as a source of chemical energy for the cell."
            )
            records = [
                {
                    "id": f"r{i}",
                    "title": f"Article {i % 3}",
                    "context": context + f" Paragraph {i}.",
                    "question": f"What generates ATP, variant {i}?",
                    "answers": {
                        "text": ["adenosine triphosphate"],
                        "answer_start": [(context + f" Paragraph {i}.").index(
                            "adenosine triphosphate"
                        )],
                    },
                }
                for i in range(9)
            ]
            examples = list(adapt_records("squad-qg", records))
            assert validate_dataset(examples).ok
            splits = DeterministicGroupSplitter().split(examples, seed=42, group_by="topic")
            assert len(splits) == 9
            assert splits.leaked_group_keys() == frozenset()
            assert compute_statistics(examples).total_examples == 9
            assert DEFAULT_TEMPLATE.render(examples[0]).completion
            assert experiment_config_from_dict({"name": "smoke"}).config_hash()

            forbidden = ("torch", "transformers", "datasets", "trl", "peft")
            leaked = sorted(m for m in forbidden if m in sys.modules)
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                raise SystemExit(1)
            print("PIPELINE_OK")
            """
        )
        assert result.returncode == 0, (
            f"the offline pipeline failed or leaked.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PIPELINE_OK" in result.stdout


class TestNoTrainingOrModelLoading:
    """Phase 17A specifies; it does not execute."""

    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_no_module_references_a_training_primitive(self, submodule):
        """No optimiser, no loop, no backward pass anywhere in the package."""
        referenced = code_identifiers(submodule)
        forbidden = {
            "backward",
            "step",
            "zero_grad",
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "from_pretrained",
            "get_peft_model",
            "prepare_model_for_kbit_training",
            "SFTTrainer",
            "SFTConfig",
            "Trainer",
            "TrainingArguments",
            "cuda",
            "generate",
        }
        leaked = sorted(referenced & forbidden)
        assert not leaked, f"qa_gen.{submodule} references training primitive(s) {leaked}"

    def test_the_base_model_is_named_but_never_resolved(self):
        """The config states which weights to use; nothing checks that they exist."""
        from qa_gen import DEFAULT_BASE_MODEL, GeneratorModelConfig

        assert DEFAULT_BASE_MODEL == "Qwen/Qwen3-4B"
        config = GeneratorModelConfig()
        config.validate()
        assert config.model_id == DEFAULT_BASE_MODEL

    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_no_module_reads_the_environment_or_a_model_cache(self, submodule):
        referenced = code_identifiers(submodule)
        forbidden = {"os", "environ", "getenv", "HF_HOME", "HF_TOKEN", "TRANSFORMERS_CACHE"}
        leaked = sorted(referenced & forbidden)
        assert not leaked, f"qa_gen.{submodule} references {leaked}"

    @pytest.mark.parametrize("submodule", SUBMODULES)
    def test_no_module_touches_the_filesystem(self, submodule):
        """Phase 17A produces objects; writing them to disk belongs to 17B."""
        referenced = code_identifiers(submodule)
        forbidden = {
            "open",
            "read_text",
            "write_text",
            "read_bytes",
            "write_bytes",
            "mkdir",
            "Path",
            "pathlib",
            "shutil",
        }
        leaked = sorted(referenced & forbidden)
        assert not leaked, f"qa_gen.{submodule} touches the filesystem via {leaked}"


class TestQaGenPublicApi:
    """The package must re-export the surface Phase 17B will import."""

    @pytest.mark.parametrize(
        "symbol",
        [
            "QuestionGenerationExample",
            "QuestionGenerationTarget",
            "QuestionGenerationDatasetConfig",
            "GeneratorModelConfig",
            "LoRAConfig",
            "TrainingConfig",
            "EvaluationConfig",
            "TrainingRunMetadata",
            "EvaluationMetrics",
            "GenerationExperimentConfig",
            "DatasetMetadata",
            "DatasetAdapter",
            "AdapterSpec",
            "AdapterError",
            "UnknownAdapterError",
            "SquadQuestionGenerationAdapter",
            "LmqgSquadQagAdapter",
            "LearningQAdapter",
            "EducationalMcqAdapter",
            "PromptTemplate",
            "RenderedPrompt",
            "DEFAULT_TEMPLATE",
            "DEFAULT_BASE_MODEL",
            "TARGET_JSON_FIELDS",
            "DeterministicGroupSplitter",
            "DatasetSplits",
            "SplitRatios",
            "SplitName",
            "GroupAssignment",
            "DatasetValidationReport",
            "DatasetIssueCode",
            "DatasetStatistics",
            "adapt_records",
            "adapter_for",
            "registered_sources",
            "compute_statistics",
            "compute_dataset_fingerprint",
            "find_duplicate_examples",
            "validate_dataset",
            "validate_example",
            "experiment_config_from_dict",
            "example_from_dict",
            "target_from_json",
            "score_predictions",
        ],
    )
    def test_symbol_is_exported(self, symbol):
        import qa_gen

        assert hasattr(qa_gen, symbol)
        assert symbol in qa_gen.__all__

    def test_version_is_declared(self):
        import qa_gen

        assert isinstance(qa_gen.__version__, str)
        assert qa_gen.__version__

    def test_all_entries_are_unique(self):
        import qa_gen

        assert len(qa_gen.__all__) == len(set(qa_gen.__all__))

    def test_all_follows_the_project_ordering_convention(self):
        """SCREAMING_CASE constants first, then everything else, each sorted."""
        import qa_gen

        def isort_style(names: tuple[str, ...] | list[str]) -> list[str]:
            constants = sorted(name for name in names if name.isupper())
            rest = sorted(name for name in names if not name.isupper())
            return constants + rest

        assert qa_gen.__all__ == isort_style(qa_gen.__all__)

    def test_every_exported_symbol_resolves(self):
        import qa_gen

        for name in qa_gen.__all__:
            assert getattr(qa_gen, name, None) is not None, f"{name} is exported but None"


class TestSharedVocabularyIsNotDuplicated:
    """qa_gen borrows the paper domain's vocabulary rather than declaring its own."""

    def test_question_type_is_the_qa_paper_enum(self):
        from qa_gen import examples
        from qa_paper import QuestionType

        assert examples.QuestionType is QuestionType

    def test_difficulty_is_the_qa_paper_enum(self):
        from qa_gen import examples
        from qa_paper import Difficulty

        assert examples.Difficulty is Difficulty

    def test_grounding_is_the_qa_paper_type(self):
        from qa_gen import examples
        from qa_paper import ContentGrounding

        assert examples.ContentGrounding is ContentGrounding

    def test_the_mcq_minimum_is_the_qa_paper_constant(self):
        from qa_gen import validation
        from qa_paper import MINIMUM_MCQ_OPTIONS

        assert validation.MINIMUM_MCQ_OPTIONS == MINIMUM_MCQ_OPTIONS

    def test_qa_gen_defines_no_competing_question_type_enum(self):
        """A second, identical-looking enumeration would drift and break the paper assembler."""
        import importlib
        import inspect
        from enum import Enum

        from qa_paper import Difficulty, QuestionType

        for submodule in SUBMODULES:
            module = importlib.import_module(f"qa_gen.{submodule}")
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, Enum) or obj in (QuestionType, Difficulty):
                    continue
                values = {member.value for member in obj}
                assert values != {m.value for m in QuestionType}, (
                    f"qa_gen.{submodule}.{obj.__name__} duplicates QuestionType"
                )
                assert values != {m.value for m in Difficulty}, (
                    f"qa_gen.{submodule}.{obj.__name__} duplicates Difficulty"
                )


class TestExistingSystemUntouched:
    """Phase 17A is additive. The extractive path must be unchanged.

    A cheap regression guard: ``qa_gen`` adds a package, so the surface the trained
    DeBERTa checkpoint and ``POST /predict`` depend on must still import and expose the same
    names, and importing ``qa_gen`` must not rebind anything in either shared package.
    """

    def test_qa_core_still_exposes_its_decoding_surface(self):
        import qa_core

        for symbol in ("decode_spans", "tighten_char_span", "compute_squad_metrics"):
            assert hasattr(qa_core, symbol)

    def test_qa_paper_still_exposes_its_assembly_surface(self):
        import qa_paper

        for symbol in ("assemble_paper", "validate_paper", "build_answer_key"):
            assert hasattr(qa_paper, symbol)

    def test_qa_gen_does_not_shadow_or_patch_qa_core(self):
        result = _run_in_clean_interpreter(
            """
            import qa_core
            before = id(qa_core.normalize_answer)
            import qa_gen  # noqa: F401
            after = id(qa_core.normalize_answer)
            print("SAME" if before == after else "PATCHED")
            """
        )
        assert "SAME" in result.stdout, result.stderr

    def test_qa_gen_does_not_shadow_or_patch_qa_paper(self):
        result = _run_in_clean_interpreter(
            """
            import qa_paper
            before = (id(qa_paper.validate_question), id(qa_paper.QuestionType))
            import qa_gen  # noqa: F401
            after = (id(qa_paper.validate_question), id(qa_paper.QuestionType))
            print("SAME" if before == after else "PATCHED")
            """
        )
        assert "SAME" in result.stdout, result.stderr

    def test_inference_engine_module_still_imports(self):
        """Guards the ``ExtractiveQAEngine`` entry point the backend depends on."""
        from qa_torch.inference import ExtractiveQAEngine, PredictionResult

        assert hasattr(ExtractiveQAEngine, "answer")
        assert "answer" in PredictionResult.__dataclass_fields__

    def test_the_backend_predict_route_is_unchanged(self):
        """qa_gen must not have touched the served application's routes."""
        from app.main import app

        paths = {getattr(route, "path", None) for route in app.routes}
        assert {"/health", "/predict"} <= paths
