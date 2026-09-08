r"""Tests for the Phase 17B.2 smoke harness.

What these tests can and cannot prove
-------------------------------------
Be clear about it, because the harness's entire purpose is to make a claim about real CUDA
execution and these tests run on a laptop with no GPU, no ``peft``, no ``trl``, no
``bitsandbytes`` and no ``datasets``.

**Proved here.** The corpus is deterministic and validates against the Phase 17A rules. The
target JSON begins with the prefix the inspection checks for, and round-trips through the
canonical parser. The bounded overrides produce the intended schedule and leave the shipped
configuration file alone. ``--inspect-only`` never reaches ``trainer.train()``, and exactly one
function in the harness contains that call. The mask inspection accepts a correct mask and
rejects one offset in either direction. The output audit detects a missing adapter and copied
base weights. Every report shape serializes to JSON.

**Not proved here.** That Qwen3-4B loads in 4-bit NF4. That ``paged_adamw_8bit`` constructs.
That the loss decreases. That the peak VRAM is anywhere near 3.943 GiB. That TRL's real
tokenizer produces a mask in the place these tests' fake tokenizer puts it. Every one of those
is what the Lightning run is for, and no assertion in this file should be read as evidence for
any of them.

The trainer stub is a mimic, not a mock
---------------------------------------
:func:`_render_qwen3` reimplements the two branches of the real Qwen3-4B chat template that
matter here, read from ``Qwen/Qwen3-4B``'s ``tokenizer_config.json``: the assistant turn, which
emits ``<think>\n\n</think>\n\n`` in front of the content unconditionally, and the
generation prompt, which emits the same block only when ``enable_thinking`` is explicitly
false. :func:`_tokenize_record` then applies TRL v0.29.1's own arithmetic --
``completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) -
len(prompt_ids))`` -- over a prefix-stable atom segmentation, so ``prompt_ids`` is a token-exact
prefix of ``prompt_completion_ids`` exactly when the prompt string is a string prefix of the
full rendering, which is the real invariant.

An earlier version of this mimic chunked the rendering per message and emitted no think block.
It agreed with the harness and disagreed with the Studio, which is the specific way a stub can
be worse than no test at all. :class:`TestQwen3ThinkingBlock` reproduces the failure the
Studio actually reported and pins the fix.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qa_gen import TARGET_JSON_FIELDS, experiment_config_from_dict, target_from_json
from qa_gen.validation import validate_dataset
from qa_gen_runtime import smoke as smoke_module
from qa_gen_runtime import smoke_data as smoke_data_module
from qa_gen_runtime.chat import REASONING_TEMPLATE_FLAG
from qa_gen_runtime.config_io import load_experiment_config
from qa_gen_runtime.dataset import (
    CHAT_TEMPLATE_KWARGS_COLUMN,
    build_training_records,
    resolve_record_format,
)
from qa_gen_runtime.loader import LoadedModel
from qa_gen_runtime.outputs import RunPaths
from qa_gen_runtime.precision import resolve_precision
from qa_gen_runtime.smoke import (
    EXPECTED_COMPLETION_PREFIX,
    SMOKE_BATCH_SIZE,
    SMOKE_GRADIENT_ACCUMULATION_STEPS,
    SMOKE_MAX_STEPS,
    SMOKE_REPORT_FILENAME,
    THINKING_MARKERS,
    OutputAudit,
    RecordInspection,
    SmokeHarnessError,
    SmokeMeasurements,
    SmokeMode,
    SmokeReport,
    audit_output,
    build_parser,
    first_tokenized_record,
    format_report,
    inspect_tokenized_record,
    main,
    resolve_mode,
    run_smoke,
    smoke_training_overrides,
)
from qa_gen_runtime.smoke_data import (
    SMOKE_EXAMPLE_COUNT,
    SMOKE_SOURCE,
    build_smoke_examples,
    describe_smoke_corpus,
    validate_smoke_examples,
)
from qa_gen_runtime.trainer import plan_trainer_arguments


def smoke_config_path() -> str:
    """Path to the shipped smoke configuration."""
    from qa_ml.paths import find_repo_root

    return str(find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml")


def bounded_config():
    """The shipped configuration with the harness's bounded overrides applied."""
    return load_experiment_config(smoke_config_path(), overrides=smoke_training_overrides())


# ---------------------------------------------------------------------------
# A tokenizer and a trainer that mimic TRL closely enough to inspect
# ---------------------------------------------------------------------------


class _TokenTable:
    """Interns strings as token ids so a decode is an exact inverse of an encode."""

    def __init__(self) -> None:
        self._pieces: list[str] = []

    def add(self, piece: str) -> int:
        """Return the id for ``piece``, assigning one if it is new."""
        if piece not in self._pieces:
            self._pieces.append(piece)
        return self._pieces.index(piece)

    def decode(self, ids: list[int]) -> str:
        """Join the pieces the ids stand for."""
        return "".join(self._pieces[int(index)] for index in ids)


class FakeTokenizer:
    """A tokenizer that only has to decode, plus a template mentioning the reasoning flag."""

    def __init__(self, table: _TokenTable) -> None:
        """Wrap a token table."""
        self._table = table
        self.chat_template = (
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
        )
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids back to text."""
        return self._table.decode(ids)


#: The empty reasoning block Qwen3's template emits. From the assistant branch,
#: ``'<think>\n' + ''.strip('\n') + '\n</think>\n\n'``, and from the generation-prompt branch
#: as the literal ``'<think>\n\n</think>\n\n'``. Byte-identical, which is what keeps the prefix
#: exact when the flag is set.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

#: A prefix-stable segmentation: special tokens, think delimiters, whitespace runs, words, then
#: any single character. Atomising a string prefix yields a prefix of the atoms, so the mimic
#: reproduces the real token-prefix property rather than approximating it.
_ATOM = re.compile(r"<\|[^|>]*\|>|</?think>|\s+|\w+|.")


def _render_qwen3(
    messages: list[dict[str, str]], *, add_generation_prompt: bool, enable_thinking: Any = None
) -> str:
    """Render messages the way Qwen3-4B's chat template does.

    Only the branches this project reaches are modelled: no tools, no tool calls, no
    ``reasoning_content``, and exactly one assistant turn which is last. Under those conditions
    ``loop.index0 > ns.last_query_index`` and ``loop.last`` both hold, so the assistant branch
    takes the think-block path unconditionally.
    """
    assert [m["role"] for m in messages].count("assistant") <= 1
    parts: list[str] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "assistant":
            parts.append(
                f"<|im_start|>assistant\n{EMPTY_THINK_BLOCK}{content}<|im_end|>\n"
            )
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
        # The one place enable_thinking is consulted, and only when explicitly false.
        if enable_thinking is False:
            parts.append(EMPTY_THINK_BLOCK)
    return "".join(parts)


def _tokenize_record(record: dict[str, Any], table: _TokenTable) -> dict[str, Any]:
    """Turn a conversational record into a tokenized row the way TRL v0.29.1 does.

    Reproduces the two ``apply_chat_template`` calls in ``SFTTrainer._prepare_dataset``, both
    of which receive ``**example.get("chat_template_kwargs", {})``, and then TRL's mask
    arithmetic verbatim.
    """
    template_kwargs = record.get(CHAT_TEMPLATE_KWARGS_COLUMN) or {}
    prompt_text = _render_qwen3(
        list(record["prompt"]),
        add_generation_prompt=True,
        enable_thinking=template_kwargs.get("enable_thinking"),
    )
    full_text = _render_qwen3(
        list(record["prompt"]) + list(record["completion"]),
        add_generation_prompt=False,
        enable_thinking=template_kwargs.get("enable_thinking"),
    )
    prompt_ids = [table.add(atom) for atom in _ATOM.findall(prompt_text)]
    full_ids = [table.add(atom) for atom in _ATOM.findall(full_text)]
    return {
        "input_ids": full_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids)),
        "example_id": record.get("example_id"),
    }


class FakeOptimizer:
    """Stands in for the optimiser transformers resolves from ``optim``."""


class FakeTrainer:
    """A trainer-shaped object that records whether it was asked to train."""

    def __init__(self, dataset: list[dict[str, Any]]) -> None:
        """Hold a prepared dataset and start with a zeroed state."""
        self.train_dataset = dataset
        self.train_calls = 0
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.optimizer: Any = None
        self.args = SimpleNamespace(optim="paged_adamw_8bit")

    def train(self) -> SimpleNamespace:
        """Record the call and return a plausible TrainOutput."""
        self.train_calls += 1
        self.state.global_step = 2
        self.state.log_history = [
            {"loss": 2.11, "step": 1, "epoch": 0.5, "learning_rate": 0.0, "grad_norm": 1.4},
            {"loss": 1.87, "step": 2, "epoch": 1.0, "learning_rate": 2e-4, "grad_norm": 1.1},
        ]
        self.optimizer = FakeOptimizer()
        return SimpleNamespace(
            global_step=2,
            training_loss=1.99,
            metrics={"train_loss": 1.99, "train_runtime": 4.25},
        )


class FakeParameter:
    """A parameter with a size and a gradient flag. Enough for counting."""

    def __init__(self, count: int, *, requires_grad: bool) -> None:
        """Store the element count and the gradient flag."""
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        """Return the parameter's element count."""
        return self.count


class FakeAdaptedModel:
    """An adapter-wrapped model that can save a checkpoint-shaped directory."""

    def __init__(self, *, base_weight_leak: bool = False) -> None:
        """Build a model with the measured parameter split."""
        self._parameters = [
            FakeParameter(4_022_468_096, requires_grad=False),
            FakeParameter(33_030_144, requires_grad=True),
        ]
        self.config = SimpleNamespace(use_cache=False)
        self.is_gradient_checkpointing = True
        self.saved_to: str | None = None
        self._base_weight_leak = base_weight_leak

    def parameters(self):
        """Yield the model's parameters."""
        return iter(self._parameters)

    def save_pretrained(self, path: str) -> None:
        """Write the files PEFT writes, and optionally one it never would."""
        self.saved_to = path
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"\x00" * 64)
        if self._base_weight_leak:
            (target / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 128)


@pytest.fixture
def stack(monkeypatch):
    """Replace the three expensive calls with recording stand-ins.

    Patches the names in :mod:`qa_gen_runtime.smoke`'s own namespace, which is the seam the
    module's ``from ... import`` style creates, and returns the objects so a test can assert
    what happened to them.
    """
    table = _TokenTable()
    tokenizer = FakeTokenizer(table)
    model = FakeAdaptedModel()
    built: dict[str, Any] = {}

    def fake_load_trainable_model(config):
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            precision=resolve_precision("fp32"),
            trainable_parameters=33_030_144,
            total_parameters=4_055_498_240,
            adapters_attached=True,
            notes=("stubbed for the test suite",),
        )

    def fake_build_hf_dataset(records):
        return list(records)

    def fake_build_trainer(config, *, model, tokenizer, train_dataset, output_dir, **kwargs):
        rows = [_tokenize_record(record, table) for record in train_dataset]
        trainer = FakeTrainer(rows)
        built["trainer"] = trainer
        built["output_dir"] = output_dir
        return trainer, plan_trainer_arguments(
            config, output_dir, train_examples=len(train_dataset)
        )

    monkeypatch.setattr(smoke_module, "load_trainable_model", fake_load_trainable_model)
    monkeypatch.setattr(smoke_module, "build_hf_dataset", fake_build_hf_dataset)
    monkeypatch.setattr(smoke_module, "build_trainer", fake_build_trainer)
    return SimpleNamespace(table=table, tokenizer=tokenizer, model=model, built=built)


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


class TestSmokeCorpus:
    """The six hand-written examples, and the fact that nothing fetched them."""

    def test_the_corpus_has_the_declared_size(self):
        assert len(build_smoke_examples()) == SMOKE_EXAMPLE_COUNT
        assert 4 <= SMOKE_EXAMPLE_COUNT <= 8

    def test_two_calls_return_the_same_corpus(self):
        """No shuffling, no sampling, no seed: determinism by construction."""
        assert build_smoke_examples() == build_smoke_examples()

    def test_every_example_carries_a_context_a_topic_and_one_target(self):
        for example in build_smoke_examples():
            assert example.id
            assert len(example.context) >= 64
            assert example.topic
            assert len(example.targets) == 1
            assert example.source == SMOKE_SOURCE

    def test_every_target_carries_a_type_a_difficulty_marks_a_question_and_an_answer(self):
        for example in build_smoke_examples():
            target = example.primary_target
            assert target is not None
            assert target.question_type is not None
            assert target.difficulty is not None
            assert isinstance(target.marks, int)
            assert target.marks > 0
            assert target.question.strip()
            assert target.answer.strip()

    def test_the_corpus_validates_against_the_phase_17a_rules(self):
        report = validate_smoke_examples()
        assert report.ok, [issue.message for issue in report.errors]
        assert not report.warnings, [issue.message for issue in report.warnings]

    def test_the_corpus_validates_against_the_shipped_dataset_configuration(self):
        """The context length bounds in qgen-smoke.yaml are stricter than the defaults."""
        report = validate_dataset(build_smoke_examples(), config=bounded_config().dataset)
        assert report.ok, [issue.message for issue in report.errors]

    def test_the_contexts_are_distinct(self):
        examples = build_smoke_examples()
        assert len({example.context for example in examples}) == len(examples)
        assert len({example.context_fingerprint for example in examples}) == len(examples)

    def test_the_corpus_covers_several_question_types_and_difficulties(self):
        examples = build_smoke_examples()
        types = {example.question_type for example in examples}
        difficulties = {example.difficulty for example in examples}
        marks = {example.marks for example in examples}
        assert len(types) >= 4
        assert len(difficulties) == 3
        assert len(marks) >= 3

    def test_the_corpus_covers_a_target_with_options_and_one_without(self):
        """Both branches of the compact JSON contract, so neither is inspected by accident."""
        targets = [example.primary_target for example in build_smoke_examples()]
        assert any(target.has_options for target in targets)
        assert any(not target.has_options for target in targets)
        assert any(target.explanation for target in targets)
        assert any(target.explanation is None for target in targets)

    def test_the_provenance_record_says_nothing_was_downloaded(self):
        described = describe_smoke_corpus()
        assert described["downloaded"] is False
        assert described["source"] == SMOKE_SOURCE
        assert described["example_count"] == SMOKE_EXAMPLE_COUNT

    def test_the_provenance_record_is_json_serializable(self):
        payload = json.loads(json.dumps(describe_smoke_corpus()))
        assert len(payload["examples"]) == SMOKE_EXAMPLE_COUNT


class TestExpectedJsonTarget:
    """The completion the model is trained to emit, and the prefix the harness checks for."""

    def test_the_expected_prefix_follows_the_canonical_field_order(self):
        """Derived rather than recalled: question_type is the first key by contract."""
        assert TARGET_JSON_FIELDS[0] == "question_type"
        assert EXPECTED_COMPLETION_PREFIX == '{"question_type":'

    def test_every_target_serializes_with_the_expected_prefix(self):
        for example in build_smoke_examples():
            for target in example.targets:
                assert target.to_json().startswith(EXPECTED_COMPLETION_PREFIX)

    def test_every_target_round_trips_through_the_canonical_parser(self):
        for example in build_smoke_examples():
            for target in example.targets:
                assert target_from_json(target.to_json()) == target

    def test_the_rendered_records_are_conversational_prompt_completion(self):
        config = bounded_config()
        assert resolve_record_format(config).value == "conversational"
        records = build_training_records(build_smoke_examples(), config)
        assert len(records) == SMOKE_EXAMPLE_COUNT
        for record in records:
            assert [message["role"] for message in record["prompt"]] == ["system", "user"]
            assert [message["role"] for message in record["completion"]] == ["assistant"]
            content = record["completion"][0]["content"]
            assert content.startswith(EXPECTED_COMPLETION_PREFIX)
            assert target_from_json(content) is not None

    def test_no_reasoning_markup_reaches_a_rendered_record(self):
        serialized = json.dumps(build_training_records(build_smoke_examples(), bounded_config()))
        for marker in THINKING_MARKERS:
            assert marker not in serialized


# ---------------------------------------------------------------------------
# The bounded overrides
# ---------------------------------------------------------------------------


class TestSmokeOverrides:
    """The first measured execution is defined here, not by the planning configuration."""

    def test_the_defaults_are_the_bounded_experiment(self):
        training = smoke_training_overrides()["training"]
        assert training["max_steps"] == SMOKE_MAX_STEPS == 2
        assert training["per_device_train_batch_size"] == SMOKE_BATCH_SIZE == 1
        assert training["gradient_accumulation_steps"] == SMOKE_GRADIENT_ACCUMULATION_STEPS == 1
        assert training["evaluation_strategy"] == "no"
        assert training["save_strategy"] == "no"
        assert training["load_best_model_at_end"] is False
        assert training["logging_steps"] == 1

    def test_the_overrides_touch_only_the_training_section(self):
        assert set(smoke_training_overrides()) == {"training"}

    def test_the_overrides_leave_the_measured_baseline_settings_alone(self):
        """Changing any of these would make the VRAM figure incomparable to the baseline."""
        training = smoke_training_overrides()["training"]
        for name in (
            "gradient_checkpointing",
            "completion_only_loss",
            "packing",
            "optimizer",
            "precision",
            "learning_rate",
            "seed",
        ):
            assert name not in training

    def test_the_shipped_configuration_file_is_not_modified(self):
        """The planning configuration keeps its 20 steps at accumulation 8."""
        plain = load_experiment_config(smoke_config_path())
        assert plain.training.max_steps == 20
        assert plain.training.gradient_accumulation_steps == 8

    def test_the_overrides_apply_over_the_shipped_configuration(self):
        config = bounded_config()
        assert config.training.max_steps == 2
        assert config.training.gradient_accumulation_steps == 1
        assert config.training.per_device_train_batch_size == 1
        assert config.training.effective_batch_size == 1

    def test_the_untouched_training_fields_survive_the_merge(self):
        config = bounded_config()
        assert config.training.optimizer == "paged_adamw_8bit"
        assert config.training.gradient_checkpointing is True
        assert config.training.completion_only_loss is True
        assert config.training.packing is False
        assert config.training.seed == 42

    def test_the_sequence_length_stays_at_the_measured_value(self):
        assert bounded_config().model.max_seq_length == 1024

    def test_the_warmup_converts_to_exactly_one_step(self):
        plan = plan_trainer_arguments(bounded_config(), "out", train_examples=SMOKE_EXAMPLE_COUNT)
        assert plan.total_steps == 2
        assert plan.warmup_steps == 1

    def test_the_bounded_configuration_is_still_the_verified_baseline(self):
        """The overrides are all schedule; none of them is a field the measurement covered."""
        from qa_gen import VERIFIED_QWEN3_4B_L4

        assert bounded_config().baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()


# ---------------------------------------------------------------------------
# CLI mode selection
# ---------------------------------------------------------------------------


class TestCliModeSelection:
    """Neither mode is the default, and only one of them can train."""

    def test_inspect_only_resolves_to_the_non_training_mode(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--inspect-only"])
        assert resolve_mode(args) is SmokeMode.INSPECT_ONLY

    def test_run_resolves_to_the_training_mode(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert resolve_mode(args) is SmokeMode.RUN

    def test_a_mode_is_required(self):
        """A command that does not say what it wants must not pick the expensive one."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "c.yaml"])

    def test_the_two_modes_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "c.yaml", "--inspect-only", "--run"])

    def test_a_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--run"])

    def test_the_bounded_values_are_the_parser_defaults(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.max_steps == SMOKE_MAX_STEPS
        assert args.batch_size == SMOKE_BATCH_SIZE
        assert args.gradient_accumulation_steps == SMOKE_GRADIENT_ACCUMULATION_STEPS
        assert args.output_dir is None
        assert args.json is False

    def test_anything_short_of_run_resolves_to_inspect_only(self):
        """Belt and braces: the mapping is written so the safe mode is the fallback."""
        assert resolve_mode(SimpleNamespace()) is SmokeMode.INSPECT_ONLY
        assert resolve_mode(SimpleNamespace(run=False)) is SmokeMode.INSPECT_ONLY
        assert resolve_mode(SimpleNamespace(run=None)) is SmokeMode.INSPECT_ONLY


# ---------------------------------------------------------------------------
# The training guard
# ---------------------------------------------------------------------------


class TestTrainingIsGuarded:
    """Where training can happen, asserted rather than assumed."""

    def test_the_train_call_refuses_without_the_explicit_flag(self):
        trainer = FakeTrainer([])
        with pytest.raises(SmokeHarnessError, match="--run"):
            smoke_module._call_trainer_train(trainer, execute=False)
        assert trainer.train_calls == 0

    def test_the_train_call_works_with_the_explicit_flag(self):
        trainer = FakeTrainer([])
        smoke_module._call_trainer_train(trainer, execute=True)
        assert trainer.train_calls == 1

    def test_exactly_one_function_in_the_harness_calls_train(self):
        """An AST scan, so a second call site anywhere in the module fails this test."""
        tree = ast.parse(inspect.getsource(smoke_module))
        callers = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "train"
                ):
                    callers.add(node.name)
        assert callers == {"_call_trainer_train"}

    def test_the_harness_never_calls_backward(self):
        tree = ast.parse(inspect.getsource(smoke_module))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "backward" not in called

    def test_the_corpus_module_neither_trains_nor_downloads(self):
        tree = ast.parse(inspect.getsource(smoke_data_module))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "load_dataset" not in names
        assert "from_pretrained" not in names
        assert "train" not in names

    def test_neither_module_downloads_a_dataset(self):
        for module in (smoke_module, smoke_data_module):
            tree = ast.parse(inspect.getsource(module))
            names = {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            } | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            assert "load_dataset" not in names, module.__name__
            assert "hf_hub_download" not in names, module.__name__
            assert "snapshot_download" not in names, module.__name__

    def test_the_corpus_module_imports_no_ml_stack(self):
        """The fixtures are plain data; they must not drag torch or TRL in."""
        tree = ast.parse(inspect.getsource(smoke_data_module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"torch", "transformers", "trl", "peft", "bitsandbytes", "datasets"}
        assert not imported & forbidden


# ---------------------------------------------------------------------------
# Inspect-only
# ---------------------------------------------------------------------------


class TestInspectOnly:
    """The mode that builds everything and steps nothing."""

    def test_inspect_only_never_trains(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(tmp_path / "runs"),
            check_git=False,
        )
        assert report.trained is False
        assert report.status == "inspected"
        assert stack.built["trainer"].train_calls == 0

    def test_inspect_only_creates_no_run_directory_of_its_own(self, stack, tmp_path):
        root = tmp_path / "runs"
        run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(root),
            check_git=False,
        )
        assert not root.exists()

    def test_inspect_only_writes_no_artifacts(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(tmp_path / "runs"),
            check_git=False,
        )
        assert report.artifacts == {}
        assert report.measurements is None
        assert report.output_audit is None
        assert list(tmp_path.iterdir()) == []

    def test_inspect_only_inspects_the_first_record_and_it_passes(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(tmp_path / "runs"),
            check_git=False,
        )
        inspection = report.inspection
        assert inspection is not None
        assert inspection.example_id == "qgen-smoke-001"
        assert inspection.mask_column == "completion_mask"
        assert inspection.ok, inspection.failed_checks + inspection.undetermined_checks
        assert inspection.decoded_completion.startswith(EXPECTED_COMPLETION_PREFIX)
        assert inspection.parsed_target["question_type"] == "short_answer"

    def test_inspect_only_still_reports_the_diagnostics_and_the_plan(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(tmp_path / "runs"),
            check_git=False,
        )
        assert report.diagnostics["model_id"] == "Qwen/Qwen3-4B"
        assert report.diagnostics["trainable_parameters"] == 33_030_144
        assert report.trainer_plan["arguments"]["max_length"] == 1024
        assert report.trainer_plan["arguments"]["max_steps"] == 2

    def test_inspect_only_records_that_nothing_was_downloaded(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.INSPECT_ONLY,
            output_dir=str(tmp_path / "runs"),
            check_git=False,
        )
        assert report.dataset["downloaded"] is False
        assert report.record_format == "conversational"


# ---------------------------------------------------------------------------
# Run mode
# ---------------------------------------------------------------------------


class TestRunMode:
    """The one mode that trains, and what it leaves behind."""

    def test_run_mode_trains_exactly_once(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(tmp_path / "runs"),
            run_id="smoke-test-run",
            check_git=False,
        )
        assert report.trained is True
        assert report.status == "trained"
        assert stack.built["trainer"].train_calls == 1

    def test_run_mode_saves_the_adapter_under_the_run_root(self, stack, tmp_path):
        root = tmp_path / "runs"
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(root),
            run_id="smoke-test-run",
            check_git=False,
        )
        adapter = root / "smoke-test-run" / "adapter"
        assert adapter.is_dir()
        assert (adapter / "adapter_config.json").is_file()
        assert (adapter / "adapter_model.safetensors").is_file()
        audit = report.output_audit
        assert audit is not None
        assert audit.checks["adapter_directory_exists"] is True
        assert audit.checks["adapter_checkpoint_is_complete"] is True
        assert audit.checks["adapter_is_under_the_run_root"] is True
        assert audit.checks["no_base_model_weights_were_copied"] is True
        assert audit.ok

    def test_run_mode_records_the_measurements(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(tmp_path / "runs"),
            run_id="smoke-test-run",
            check_git=False,
        )
        measured = report.measurements
        assert measured is not None
        assert measured.trainable_parameters == 33_030_144
        assert measured.total_parameters == 4_055_498_240
        assert measured.trainable_fraction is not None
        assert measured.optimizer_steps == 2
        assert measured.warmup_steps == 1
        assert measured.wall_clock_seconds is not None
        assert measured.seconds_per_optimizer_step is not None
        assert [entry["loss"] for entry in measured.losses] == [2.11, 1.87]
        assert measured.final_loss == 1.99
        assert measured.optimizer_requested == "paged_adamw_8bit"
        assert measured.optimizer_class.endswith("FakeOptimizer")
        assert measured.gradient_checkpointing_requested is True
        assert measured.gradient_checkpointing_active is True
        assert measured.use_cache is False
        assert measured.dropped_trainer_arguments == ()
        assert "total_vram_gib" in measured.memory_after

    def test_run_mode_writes_the_run_documents(self, stack, tmp_path):
        root = tmp_path / "runs"
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(root),
            run_id="smoke-test-run",
            check_git=False,
        )
        run_dir = root / "smoke-test-run"
        assert set(report.artifacts) == {
            "config",
            "diagnostics",
            "dataset",
            "record",
            "smoke",
        }
        for name in ("config.resolved.json", "diagnostics.json", "dataset.json", "run.json"):
            assert (run_dir / name).is_file(), name
        assert (run_dir / SMOKE_REPORT_FILENAME).is_file()

    def test_the_written_smoke_report_is_readable_json(self, stack, tmp_path):
        root = tmp_path / "runs"
        run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(root),
            run_id="smoke-test-run",
            check_git=False,
        )
        payload = json.loads(
            (root / "smoke-test-run" / SMOKE_REPORT_FILENAME).read_text(encoding="utf-8")
        )
        assert payload["trained"] is True
        assert payload["mode"] == "run"
        assert payload["inspection"]["ok"] is True
        assert payload["measurements"]["optimizer_steps"] == 2

    def test_the_run_record_states_it_was_a_smoke_execution(self, stack, tmp_path):
        root = tmp_path / "runs"
        run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(root),
            run_id="smoke-test-run",
            check_git=False,
        )
        record = json.loads((root / "smoke-test-run" / "run.json").read_text(encoding="utf-8"))
        assert record["trainable_parameters"] == 33_030_144
        assert record["training"]["smoke"]["overrides"]["training"]["max_steps"] == 2
        assert record["training"]["smoke"]["inspection_ok"] is True
        assert any("smoke execution" in note for note in record["notes"])

    def test_a_second_run_with_the_same_id_refuses(self, stack, tmp_path):
        from qa_gen_runtime.outputs import RunOutputError

        root = tmp_path / "runs"
        kwargs = {
            "mode": SmokeMode.RUN,
            "output_dir": str(root),
            "run_id": "smoke-test-run",
            "check_git": False,
        }
        run_smoke(smoke_config_path(), **kwargs)
        with pytest.raises(RunOutputError):
            run_smoke(smoke_config_path(), **kwargs)

    def test_a_base_weight_leak_is_caught(self, monkeypatch, stack, tmp_path):
        """The audit is the guard against an output directory holding the base checkpoint."""
        monkeypatch.setattr(stack.model, "_base_weight_leak", True)
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(tmp_path / "runs"),
            run_id="smoke-test-run",
            check_git=False,
        )
        audit = report.output_audit
        assert audit.checks["no_base_model_weights_were_copied"] is False
        assert audit.base_weight_files == ("adapter/model-00001-of-00002.safetensors",)
        assert audit.ok is False
        assert report.ok is False


# ---------------------------------------------------------------------------
# The mask invariant
# ---------------------------------------------------------------------------


def _built_record(*, with_reasoning_flag: bool = True) -> tuple[dict[str, Any], Any, Any]:
    """Return a tokenized first record, its tokenizer and the example it came from.

    ``with_reasoning_flag=False`` withholds the tokenizer from
    :func:`build_training_records`, so no ``chat_template_kwargs`` column is emitted and the
    record is the pre-fix one the Studio inspection reported.
    """
    table = _TokenTable()
    tokenizer = FakeTokenizer(table)
    examples = build_smoke_examples()
    records = build_training_records(
        examples,
        bounded_config(),
        tokenizer=tokenizer if with_reasoning_flag else None,
    )
    row = _tokenize_record(records[0], table)
    return row, tokenizer, examples[0]


class TestRecordInspection:
    """The completion mask must cover the target JSON and nothing else."""

    def inspect(self, row, tokenizer, example, **overrides):
        """Run the inspection with the harness's own arguments."""
        kwargs = {
            "max_length": 1024,
            "context_probe": example.context[:48],
            "source_target": example.primary_target,
        }
        kwargs.update(overrides)
        return inspect_tokenized_record(row, tokenizer, **kwargs)

    def test_a_correct_mask_passes_every_check(self):
        row, tokenizer, example = _built_record()
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.ok, inspection.failed_checks + inspection.undetermined_checks
        assert inspection.checks["mask_length_matches_input_ids"] is True
        assert inspection.checks["mask_region_is_contiguous"] is True
        assert inspection.checks["mask_excludes_the_first_token"] is True
        assert inspection.checks["mask_reaches_the_final_token"] is True
        assert inspection.checks["completion_starts_with_expected_json_prefix"] is True
        assert inspection.checks["parsed_target_matches_the_source_target"] is True

    def test_the_two_halves_are_reported_separately(self):
        row, tokenizer, example = _built_record()
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.input_ids_length == len(row["input_ids"])
        assert inspection.completion_mask_length == len(row["completion_mask"])
        assert inspection.prompt_token_count + inspection.completion_token_count == (
            inspection.input_ids_length
        )
        assert example.context[:48] in inspection.decoded_prompt
        assert example.context[:48] not in inspection.decoded_completion
        # With the flag set, the prompt absorbs the empty think block, so the boundary sits
        # after it rather than after the assistant-turn opener.
        assert inspection.decoded_prompt.endswith(
            f"<|im_start|>assistant\n{EMPTY_THINK_BLOCK}"
        )

    def test_the_completion_parses_back_through_the_canonical_parser(self):
        row, tokenizer, example = _built_record()
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.parse_error is None
        assert inspection.extracted_json == example.primary_target.to_json()
        assert target_from_json(inspection.extracted_json) == example.primary_target

    def test_a_mask_that_starts_too_late_is_caught(self):
        """The dangerous failure: the head of the target JSON never enters the loss."""
        row, tokenizer, example = _built_record()
        shifted = list(row["completion_mask"])
        first = shifted.index(1)
        shifted[first] = 0
        row = {**row, "completion_mask": shifted}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["completion_starts_with_expected_json_prefix"] is False
        assert inspection.checks["completion_json_parses"] is False
        assert inspection.ok is False

    def test_a_mask_that_starts_too_early_is_caught(self):
        r"""The other direction: prompt tokens would be trained on.

        The prompt's last atom is the ``\n\n`` closing the think block, so this is caught by
        the leading-whitespace check rather than the prefix check -- which is why the two are
        separate assertions in the harness.
        """
        row, tokenizer, example = _built_record()
        shifted = list(row["completion_mask"])
        first = shifted.index(1)
        shifted[first - 1] = 1
        row = {**row, "completion_mask": shifted}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["completion_begins_without_leading_whitespace"] is False
        assert inspection.decoded_completion.startswith("\n")
        assert inspection.ok is False

    def test_a_mask_that_swallows_the_whole_think_block_is_caught(self):
        """Shifted far enough that the prefix check fires too."""
        row, tokenizer, example = _built_record()
        shifted = list(row["completion_mask"])
        first = shifted.index(1)
        for index in range(max(0, first - 4), first):
            shifted[index] = 1
        row = {**row, "completion_mask": shifted}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["completion_starts_with_expected_json_prefix"] is False
        assert "</think>" in inspection.decoded_completion
        assert inspection.ok is False

    def test_a_mask_covering_the_whole_sequence_is_caught(self):
        row, tokenizer, example = _built_record()
        row = {**row, "completion_mask": [1] * len(row["input_ids"])}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["mask_excludes_the_first_token"] is False
        assert inspection.checks["completion_excludes_the_source_context"] is False
        assert inspection.ok is False

    def test_a_missing_mask_column_is_reported_rather_than_assumed(self):
        row, tokenizer, example = _built_record()
        row = {"input_ids": row["input_ids"], "example_id": row["example_id"]}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.mask_column is None
        assert inspection.checks["mask_column_present"] is False
        assert inspection.ok is False
        assert any("no completion mask column" in note for note in inspection.notes)

    def test_a_misaligned_mask_leaves_the_dependent_checks_undetermined(self):
        row, tokenizer, example = _built_record()
        row = {**row, "completion_mask": row["completion_mask"][:-1]}
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["mask_length_matches_input_ids"] is False
        assert inspection.checks["mask_covers_a_non_empty_region"] is None
        assert "mask_covers_a_non_empty_region" in inspection.undetermined_checks
        assert inspection.ok is False

    def test_an_assistant_mask_column_is_recognised_as_a_fallback(self):
        row, tokenizer, example = _built_record()
        row = {
            "input_ids": row["input_ids"],
            "assistant_masks": row["completion_mask"],
            "example_id": row["example_id"],
        }
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.mask_column == "assistant_masks"
        assert inspection.ok

    def test_a_record_without_input_ids_raises(self):
        _, tokenizer, example = _built_record()
        with pytest.raises(SmokeHarnessError, match="input_ids"):
            self.inspect({"prompt": [], "completion": []}, tokenizer, example)

    def test_a_sequence_at_the_truncation_limit_is_reported(self):
        row, tokenizer, example = _built_record()
        inspection = self.inspect(row, tokenizer, example, max_length=len(row["input_ids"]))
        assert inspection.checks["sequence_within_max_length"] is False

    def test_the_markers_are_counted_and_the_record_is_never_altered(self):
        """Qwen3 puts the block in the prompt once the flag is set; it is reported, not removed.

        The failure direction -- the block inside the completion -- is covered by
        :class:`TestQwen3ThinkingBlock`, which reproduces it from the real template branches.
        """
        row, tokenizer, example = _built_record()
        inspection = self.inspect(row, tokenizer, example)
        markers = inspection.thinking_markers
        assert markers["present_in_sequence"] is True
        assert markers["present_in_prompt"] is True
        assert markers["present_in_completion"] is False
        assert markers["counts"]["<think>"] == 1
        assert markers["counts"]["</think>"] == 1
        assert markers["record_modified_to_remove_them"] is False
        assert "<think>" in inspection.decoded_sequence
        assert "<think>" not in inspection.decoded_completion

    def test_no_marker_is_reported_for_a_template_that_emits_none(self):
        """The counting itself reports zero rather than defaulting to a finding."""
        table = _TokenTable()
        example = build_smoke_examples()[0]
        target_json = example.primary_target.to_json()
        prompt_atoms = _ATOM.findall(f"<|im_start|>user\n{example.context}<|im_end|>\n")
        completion_atoms = _ATOM.findall(f"{target_json}<|im_end|>\n")
        prompt_ids = [table.add(atom) for atom in prompt_atoms]
        completion_ids = [table.add(atom) for atom in completion_atoms]
        row = {
            "input_ids": prompt_ids + completion_ids,
            "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
            "example_id": example.id,
        }
        inspection = self.inspect(row, FakeTokenizer(table), example)
        markers = inspection.thinking_markers
        assert markers["present_in_sequence"] is False
        assert markers["counts"] == dict.fromkeys(THINKING_MARKERS, 0)
        assert inspection.ok, inspection.failed_checks + inspection.undetermined_checks

    def test_the_first_record_is_read_from_the_trainer_not_rebuilt(self):
        row, _, _ = _built_record()
        trainer = FakeTrainer([row, {"input_ids": [9]}])
        assert first_tokenized_record(trainer) == row

    def test_a_trainer_without_a_dataset_raises(self):
        with pytest.raises(SmokeHarnessError, match="train_dataset"):
            first_tokenized_record(SimpleNamespace(train_dataset=None))

    def test_an_empty_dataset_raises(self):
        with pytest.raises(SmokeHarnessError, match="first record"):
            first_tokenized_record(FakeTrainer([]))


class TestQwen3ThinkingBlock:
    r"""The defect a Studio inspect-only run found, and the mechanism that fixes it.

    The reported symptom was a decoded completion of
    ``<think>\n\n</think>\n\n{"question_type": ...}`` with
    ``completion_starts_with_expected_json_prefix`` as the single failed invariant. Nothing
    about the mask arithmetic was wrong; the boundary was in the wrong place, because Qwen3's
    assistant branch emits the block whatever ``enable_thinking`` says while the
    generation-prompt branch emits it only when the flag is explicitly false.
    """

    def inspect(self, row, tokenizer, example):
        """Run the inspection with the harness's own arguments."""
        return inspect_tokenized_record(
            row,
            tokenizer,
            max_length=4096,
            context_probe=example.context[:48],
            source_target=example.primary_target,
        )

    def test_the_template_mimic_matches_the_published_qwen3_branches(self):
        """Both branches produce the same block, byte for byte. That is what aligns them."""
        messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "C"}]
        full = _render_qwen3(messages, add_generation_prompt=False)
        assert full.endswith(f"<|im_start|>assistant\n{EMPTY_THINK_BLOCK}C<|im_end|>\n")

        without = _render_qwen3(messages[:1], add_generation_prompt=True)
        assert without.endswith("<|im_start|>assistant\n")
        assert EMPTY_THINK_BLOCK not in without

        with_flag = _render_qwen3(
            messages[:1], add_generation_prompt=True, enable_thinking=False
        )
        assert with_flag.endswith(f"<|im_start|>assistant\n{EMPTY_THINK_BLOCK}")

        # The fixed prompt is a byte-exact prefix of the full rendering; the unfixed one is too,
        # just a shorter one. Neither offsets the mask; they disagree about where it starts.
        assert full.startswith(with_flag)
        assert full.startswith(without)
        assert len(with_flag) > len(without)

    def test_without_the_flag_the_think_block_lands_in_the_completion(self):
        """The exact failure the Studio reported, reproduced."""
        row, tokenizer, example = _built_record(with_reasoning_flag=False)
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.decoded_completion.startswith(EMPTY_THINK_BLOCK)
        assert inspection.thinking_markers["present_in_completion"] is True
        assert inspection.thinking_markers["present_in_prompt"] is False
        assert inspection.checks["completion_starts_with_expected_json_prefix"] is False
        assert inspection.ok is False

    def test_without_the_flag_the_mask_is_still_aligned(self):
        """Distinguishes the two failure modes: the boundary moved, the arithmetic did not."""
        row, tokenizer, example = _built_record(with_reasoning_flag=False)
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.checks["mask_length_matches_input_ids"] is True
        assert inspection.checks["mask_region_is_contiguous"] is True
        assert inspection.checks["mask_excludes_the_first_token"] is True
        assert inspection.checks["mask_reaches_the_final_token"] is True
        assert inspection.failed_checks == ("completion_starts_with_expected_json_prefix",)

    def test_with_the_flag_the_completion_begins_at_the_json(self):
        """The requirement: no <think> block, and the JSON starts immediately."""
        row, tokenizer, example = _built_record(with_reasoning_flag=True)
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.decoded_completion.startswith('{"question_type":')
        assert not inspection.decoded_completion.startswith(EMPTY_THINK_BLOCK)
        assert "<think>" not in inspection.decoded_completion
        assert "</think>" not in inspection.decoded_completion
        assert inspection.checks["completion_starts_with_expected_json_prefix"] is True
        assert inspection.checks["completion_begins_without_leading_whitespace"] is True
        assert inspection.ok, inspection.failed_checks + inspection.undetermined_checks

    def test_with_the_flag_the_block_moves_into_the_masked_prompt(self):
        """It is not deleted, it is excluded from the loss. Reported, not hidden."""
        row, tokenizer, example = _built_record(with_reasoning_flag=True)
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.thinking_markers["present_in_sequence"] is True
        assert inspection.thinking_markers["present_in_prompt"] is True
        assert inspection.thinking_markers["present_in_completion"] is False
        assert inspection.thinking_markers["record_modified_to_remove_them"] is False
        assert EMPTY_THINK_BLOCK in inspection.decoded_prompt

    def test_the_flag_shifts_the_boundary_by_exactly_the_block(self):
        """Before and after, side by side, on the same corpus."""
        before_row, before_tok, example = _built_record(with_reasoning_flag=False)
        after_row, after_tok, _ = _built_record(with_reasoning_flag=True)
        before = self.inspect(before_row, before_tok, example)
        after = self.inspect(after_row, after_tok, example)

        assert before.input_ids_length == after.input_ids_length
        assert after.prompt_token_count > before.prompt_token_count
        assert after.completion_token_count < before.completion_token_count
        moved = after.prompt_token_count - before.prompt_token_count
        assert moved == len(_ATOM.findall(EMPTY_THINK_BLOCK))
        # The trained sequence is identical; only the loss boundary differs.
        assert before.decoded_sequence == after.decoded_sequence

    def test_the_target_json_still_round_trips_after_the_fix(self):
        row, tokenizer, example = _built_record(with_reasoning_flag=True)
        inspection = self.inspect(row, tokenizer, example)
        assert inspection.parse_error is None
        assert inspection.extracted_json == example.primary_target.to_json()
        assert target_from_json(inspection.extracted_json) == example.primary_target

    def test_the_column_is_what_the_records_carry(self):
        table = _TokenTable()
        tokenizer = FakeTokenizer(table)
        records = build_training_records(
            build_smoke_examples(), bounded_config(), tokenizer=tokenizer
        )
        for record in records:
            assert record[CHAT_TEMPLATE_KWARGS_COLUMN] == {REASONING_TEMPLATE_FLAG: False}

    def test_every_example_in_the_corpus_is_fixed_not_just_the_first(self):
        table = _TokenTable()
        tokenizer = FakeTokenizer(table)
        examples = build_smoke_examples()
        records = build_training_records(examples, bounded_config(), tokenizer=tokenizer)
        for record, example in zip(records, examples, strict=True):
            inspection = inspect_tokenized_record(
                _tokenize_record(record, table),
                tokenizer,
                max_length=4096,
                context_probe=example.context[:48],
                source_target=example.primary_target,
            )
            assert inspection.ok, (example.id, inspection.failed_checks)
            assert inspection.decoded_completion.startswith('{"question_type":')


# ---------------------------------------------------------------------------
# The output audit
# ---------------------------------------------------------------------------


class TestOutputAudit:
    """What a run is allowed to leave on disk."""

    def paths_with(self, tmp_path: Path, *files: str) -> tuple[RunPaths, Path]:
        """Build a run directory containing ``files`` under the adapter subdirectory."""
        root = tmp_path / "qgen-runs"
        paths = RunPaths.under(root / "run-1")
        paths.adapter.mkdir(parents=True, exist_ok=True)
        for name in files:
            (paths.adapter / name).write_bytes(b"\x00" * 16)
        return paths, root

    def test_a_complete_adapter_directory_passes(self, tmp_path):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.ok
        assert audit.adapter_exists is True
        assert audit.missing_expected_files == ()
        assert audit.adapter_bytes == 32

    def test_a_missing_adapter_fails(self, tmp_path):
        root = tmp_path / "qgen-runs"
        paths = RunPaths.under(root / "run-1")
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.checks["adapter_directory_exists"] is False
        assert audit.checks["adapter_checkpoint_is_complete"] is False
        assert audit.ok is False

    def test_an_incomplete_adapter_fails(self, tmp_path):
        paths, root = self.paths_with(tmp_path, "adapter_config.json")
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.missing_expected_files == ("adapter_model.safetensors",)
        assert audit.checks["adapter_checkpoint_is_complete"] is False

    @pytest.mark.parametrize(
        "leaked",
        [
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "pytorch_model.bin",
            "weights.gguf",
        ],
    )
    def test_copied_base_weights_are_detected(self, tmp_path, leaked):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors", leaked
        )
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.checks["no_base_model_weights_were_copied"] is False
        assert f"adapter/{leaked}" in audit.base_weight_files

    def test_the_adapter_checkpoint_itself_is_not_mistaken_for_base_weights(self, tmp_path):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.base_weight_files == ()

    def test_an_oversized_file_is_detected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(smoke_module, "_MAX_EXPECTED_FILE_BYTES", 8)
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.checks["no_unexpectedly_large_files"] is False
        assert len(audit.oversized_files) == 2
        assert audit.largest_file["bytes"] == 16

    def test_an_adapter_outside_the_run_root_fails(self, tmp_path):
        paths, _ = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(paths, run_root=tmp_path / "elsewhere", check_git=False)
        assert audit.checks["adapter_is_under_the_run_root"] is False
        assert audit.checks["run_directory_is_under_the_run_root"] is False

    def test_git_is_undetermined_rather_than_clean_when_not_consulted(self, tmp_path):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(paths, run_root=root, check_git=False)
        assert audit.checks["the_run_added_no_tracked_changes"] is None
        assert audit.git_after["clean"] is None
        assert audit.ok, "an unavailable git must not fail an otherwise good run"

    def test_an_unchanged_tree_passes_even_when_it_is_dirty(self, tmp_path, monkeypatch):
        """The invariant is 'the run added nothing tracked', not 'your checkout is clean'."""
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        snapshot = {
            "available": True,
            "clean": False,
            "entries": [" M src/x.py", "?? notes.md"],
            "entry_count": 2,
        }
        monkeypatch.setattr(smoke_module, "git_status", lambda repo_root=None: dict(snapshot))
        audit = audit_output(
            paths, run_root=root, git_before=dict(snapshot), check_git=True
        )
        assert audit.checks["the_run_added_no_tracked_changes"] is True
        assert audit.git_after["clean"] is False
        assert audit.ok
        assert any("added nothing tracked" in note for note in audit.notes)

    def test_a_run_in_a_clean_tree_that_stays_clean_passes(self, tmp_path, monkeypatch):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        clean = {"available": True, "clean": True, "entries": [], "entry_count": 0}
        monkeypatch.setattr(smoke_module, "git_status", lambda repo_root=None: dict(clean))
        audit = audit_output(paths, run_root=root, git_before=dict(clean), check_git=True)
        assert audit.checks["the_run_added_no_tracked_changes"] is True
        assert audit.ok

    def test_git_is_undetermined_when_only_one_side_is_available(self, tmp_path):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        audit = audit_output(
            paths,
            run_root=root,
            git_before={"available": False, "clean": None, "entries": []},
            repo_root=tmp_path,
            check_git=True,
        )
        assert audit.checks["the_run_added_no_tracked_changes"] is None
        assert audit.ok

    def test_a_changed_tree_fails(self, tmp_path, monkeypatch):
        paths, root = self.paths_with(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        monkeypatch.setattr(
            smoke_module,
            "git_status",
            lambda repo_root=None: {
                "available": True,
                "clean": False,
                "entries": [" M src/x.py", "?? leaked.safetensors"],
                "entry_count": 2,
            },
        )
        before = {
            "available": True,
            "clean": False,
            "entries": [" M src/x.py"],
            "entry_count": 1,
        }
        audit = audit_output(paths, run_root=root, git_before=before, check_git=True)
        assert audit.checks["the_run_added_no_tracked_changes"] is False
        assert audit.ok is False

    def test_git_status_reports_unavailability_rather_than_guessing(self, tmp_path):
        status = smoke_module.git_status(tmp_path / "not-a-checkout")
        assert status["clean"] in (None, True, False)
        if not status["available"]:
            assert status["clean"] is None


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


class TestReportSerialization:
    """A measurement that cannot be written down is not a measurement."""

    def test_an_empty_report_serializes(self):
        payload = json.loads(json.dumps(SmokeReport(mode="run", status="trained").as_dict()))
        assert payload["mode"] == "run"
        assert payload["trained"] is False
        assert payload["inspection"] is None

    def test_the_inspection_serializes_its_checks(self):
        row, tokenizer, example = _built_record()
        inspection = inspect_tokenized_record(
            row,
            tokenizer,
            max_length=1024,
            context_probe=example.context[:48],
            source_target=example.primary_target,
        )
        payload = json.loads(json.dumps(inspection.as_dict()))
        assert payload["ok"] is True
        assert payload["failed_checks"] == []
        assert payload["undetermined_checks"] == []
        assert payload["mask_column"] == "completion_mask"
        assert payload["checks"]["completion_json_parses"] is True

    def test_the_measurements_serialize(self):
        measured = SmokeMeasurements(
            trainable_parameters=1,
            total_parameters=2,
            trainable_fraction=0.5,
            losses=({"step": 1, "loss": 2.0},),
            dropped_trainer_arguments=("weird_arg",),
        )
        payload = json.loads(json.dumps(measured.as_dict()))
        assert payload["losses"] == [{"step": 1, "loss": 2.0}]
        assert payload["dropped_trainer_arguments"] == ["weird_arg"]

    def test_the_audit_serializes(self, tmp_path):
        paths = RunPaths.under(tmp_path / "run-1")
        payload = json.loads(
            json.dumps(audit_output(paths, run_root=tmp_path, check_git=False).as_dict())
        )
        assert payload["ok"] is False
        assert "adapter_directory_exists" in payload["failed_checks"]

    def test_a_full_report_serializes(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(tmp_path / "runs"),
            run_id="smoke-test-run",
            check_git=False,
        )
        payload = json.loads(json.dumps(report.as_dict()))
        assert payload["ok"] is True
        assert payload["records"][0]["prompt"][0]["role"] == "system"
        assert payload["diagnostics"]["lora"]["rank"] == 16
        assert payload["dataset_validation"]["ok"] is True

    def test_the_report_ok_flag_follows_the_checks(self):
        failing = RecordInspection(checks={"a": True, "b": False})
        assert SmokeReport(mode="run", status="trained", inspection=failing).ok is False
        passing = RecordInspection(checks={"a": True})
        assert SmokeReport(mode="run", status="trained", inspection=passing).ok is True
        assert (
            SmokeReport(
                mode="run",
                status="trained",
                inspection=passing,
                output_audit=OutputAudit(checks={"c": False}),
            ).ok
            is False
        )

    def test_the_text_report_names_the_mode_and_lists_the_checks(self, stack, tmp_path):
        report = run_smoke(
            smoke_config_path(),
            mode=SmokeMode.RUN,
            output_dir=str(tmp_path / "runs"),
            run_id="smoke-test-run",
            check_git=False,
        )
        text = format_report(report)
        assert "mode          : run" in text
        assert "trained       : True" in text
        assert "completion_starts_with_expected_json_prefix" in text
        assert "no_base_model_weights_were_copied" in text
        assert "per-step loss" in text
        assert "overall       : ok" in text


# ---------------------------------------------------------------------------
# The command line, end to end against the stubs
# ---------------------------------------------------------------------------


class TestCli:
    """Exit codes, because a CI check and a human both read them."""

    def test_inspect_only_exits_zero(self, stack, tmp_path, capsys):
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--inspect-only",
                "--output-dir",
                str(tmp_path / "runs"),
                "--log-level",
                "WARNING",
            ]
        )
        output = capsys.readouterr().out
        assert code == 0
        assert "mode          : inspect_only" in output
        assert "trained       : False" in output
        assert stack.built["trainer"].train_calls == 0

    def test_run_exits_zero_and_reports_the_adapter(self, stack, tmp_path, capsys):
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--run",
                "--output-dir",
                str(tmp_path / "runs"),
                "--run-id",
                "cli-run",
                "--log-level",
                "WARNING",
            ]
        )
        output = capsys.readouterr().out
        assert code == 0
        assert "adapter_model.safetensors" in output
        assert stack.built["trainer"].train_calls == 1

    def test_a_failed_invariant_exits_three(self, monkeypatch, stack, tmp_path, capsys):
        monkeypatch.setattr(stack.model, "_base_weight_leak", True)
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--run",
                "--output-dir",
                str(tmp_path / "runs"),
                "--run-id",
                "cli-run",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 3
        assert "no_base_model_weights_were_copied" in capsys.readouterr().err

    def test_a_missing_config_exits_one(self, stack, tmp_path, capsys):
        code = main(
            [
                "--config",
                str(tmp_path / "absent.yaml"),
                "--inspect-only",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_an_invalid_config_exits_one(self, stack, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text("name: bad\nlora:\n  rank: 0\n", encoding="utf-8")
        code = main(["--config", str(path), "--inspect-only", "--log-level", "WARNING"])
        assert code == 1
        assert "rank" in capsys.readouterr().err

    def test_json_output_is_parseable(self, stack, tmp_path, capsys):
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--inspect-only",
                "--output-dir",
                str(tmp_path / "runs"),
                "--json",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["experiment"] == "qgen-smoke"
        assert payload["trained"] is False

    def test_the_report_file_is_written_when_asked(self, stack, tmp_path):
        destination = tmp_path / "nested" / "report.json"
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--inspect-only",
                "--output-dir",
                str(tmp_path / "runs"),
                "--report",
                str(destination),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload["mode"] == "inspect_only"

    def test_the_step_count_can_be_overridden_from_the_command_line(self, stack, tmp_path):
        code = main(
            [
                "--config",
                smoke_config_path(),
                "--inspect-only",
                "--output-dir",
                str(tmp_path / "runs"),
                "--max-steps",
                "5",
                "--gradient-accumulation-steps",
                "2",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        plan = plan_trainer_arguments(
            load_experiment_config(
                smoke_config_path(),
                overrides=smoke_training_overrides(
                    max_steps=5, gradient_accumulation_steps=2
                ),
            ),
            "out",
        )
        assert plan.arguments["max_steps"] == 5
        assert plan.arguments["gradient_accumulation_steps"] == 2


# ---------------------------------------------------------------------------
# The phase boundary
# ---------------------------------------------------------------------------


class TestPhaseBoundary:
    """What this phase deliberately does not do."""

    def test_the_full_corpus_entry_point_is_a_separate_command(self):
        """Phase 18 wired `train --execute-training`; this harness is still the bounded one.

        The two must not converge. ``train --execute-training`` steps over a whole prepared
        corpus and needs one to exist; this harness builds six examples in memory and takes a
        handful of steps. Asserted because the cheap way to "reuse" the harness for production
        would be to widen its corpus, and then nothing would be bounded any more.
        """
        from qa_gen_runtime.train import build_parser as train_parser

        train_flags = {
            action.dest for action in train_parser()._actions  # noqa: SLF001 - parser introspection
        }
        assert "dataset_fingerprint" in train_flags
        assert "execute_training" in train_flags

        smoke_flags = {action.dest for action in build_parser()._actions}  # noqa: SLF001
        assert "dataset_fingerprint" not in smoke_flags
        assert smoke_flags & {"inspect_only", "run"} == {"inspect_only", "run"}

    def test_the_smoke_corpus_is_still_six_hand_written_examples(self):
        """The production path reads a prepared corpus; this one does not read anything."""
        assert len(build_smoke_examples()) == 6

    def test_no_answerability_verification_is_wired_up(self):
        """DeBERTa answerability checking is a later phase; nothing here pretends otherwise."""
        source = inspect.getsource(smoke_module)
        assert "deberta" not in source.lower()

    def test_the_optional_dependencies_are_still_absent_here(self):
        """These tests prove nothing about CUDA, TRL or PEFT, and this is why."""
        from qa_gen_runtime.deps import is_available

        assert not is_available("trl")
        assert not is_available("peft")

    def test_the_configuration_used_by_the_harness_is_the_shipped_one(self):
        """No private copy of qgen-smoke.yaml: the harness reads the reviewed file."""
        assert Path(smoke_config_path()).is_file()
        assert experiment_config_from_dict(
            load_experiment_config(smoke_config_path()).to_dict()
        ).name == "qgen-smoke"
