r"""Tests for the real-data QLoRA training benchmark and the prepared-dataset format.

What these tests prove, and what they cannot
--------------------------------------------
**Proved here.** The prepared-dataset format round-trips: what ``prepare --write-dataset`` writes
is exactly what the benchmark reads, through one implementation. Subset selection is
deterministic, seeded, reproducible and sized so the run consumes it exactly once. The overrides
bound the run and leave every cost-determining setting alone. ``--inspect-only`` never reaches
``trainer.train()``, and exactly one function in the module contains that call. The subset checks
catch an unrepresentative length distribution, a zero-length completion and truncation. The
output audit catches copied base weights and an incomplete adapter. The step projection is the
arithmetic transformers will do. Every report shape serializes.

**Not proved here.** Any timing figure. That Qwen3-4B loads in 4-bit NF4, that
``paged_adamw_8bit`` constructs, that a step takes any particular number of seconds, or that the
peak VRAM resembles the 4.775 GiB Phase 17B.3 measured. A benchmark is a measurement instrument;
these tests check the instrument, not the reading. The reading needs the L4.

No GPU, no corpus, no model
---------------------------
The three expensive calls -- ``load_trainable_model``, ``build_hf_dataset`` and ``build_trainer``
-- are replaced in the module's own namespace, and the prepared dataset is written to ``tmp_path``
by the real writer. So the orchestration, the selection, the checks and the reporting are
exercised end to end while nothing downloads and nothing touches CUDA.
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

from qa_gen import (
    QuestionGenerationExample,
    QuestionGenerationTarget,
    adapt_records,
    compute_dataset_fingerprint,
    experiment_config_from_dict,
)
from qa_gen.splitting import SplitName
from qa_gen_runtime import benchmark as benchmark_module
from qa_gen_runtime.benchmark import (
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_GRADIENT_ACCUMULATION_STEPS,
    BENCHMARK_REPORT_FILENAME,
    BENCHMARK_STEPS,
    BENCHMARK_SUBSET_SEED,
    BenchmarkError,
    BenchmarkFinding,
    BenchmarkMeasurements,
    BenchmarkMode,
    BenchmarkReport,
    BenchmarkSubset,
    audit_benchmark_output,
    benchmark_training_overrides,
    build_parser,
    check_subset,
    estimate_padded_tokens,
    format_report,
    main,
    project_production_run,
    resolve_mode,
    run_benchmark,
    select_benchmark_subset,
    subset_size_for,
)
from qa_gen_runtime.dataset import CHAT_TEMPLATE_KWARGS_COLUMN, build_training_records
from qa_gen_runtime.loader import LoadedModel
from qa_gen_runtime.precision import resolve_precision
from qa_gen_runtime.prepared import (
    DATASET_DOCUMENT,
    PreparedDatasetError,
    discover_prepared_datasets,
    read_prepared_split,
    resolve_prepared_directory,
    split_filename,
    verify_fingerprint,
    write_prepared_split,
)
from qa_gen_runtime.sizing import SplitSizing, TokenLengthSummary
from qa_gen_runtime.trainer import plan_trainer_arguments
from qa_paper import Difficulty, QuestionType

PASSAGE = (
    "Photosynthesis is the process by which green plants convert light energy into chemical "
    "energy. Chlorophyll in the leaves absorbs sunlight, and the plant combines carbon dioxide "
    "from the air with water drawn up through the roots to produce glucose and oxygen."
)

EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"
_ATOM = re.compile(r"<\|[^|>]*\|>|</?think>|\s+|\w+|.")

CONFIG_YAML = """
name: qgen-bench-test
phase: "17"
dataset:
  sources: [squad-qg]
  seed: 42
  train_ratio: 0.9
  validation_ratio: 0.05
  test_ratio: 0.05
  group_by: context
  min_context_chars: 64
  max_context_chars: 4000
model:
  model_id: Qwen/Qwen3-4B
  quantization: 4bit
  quantization_type: nf4
  double_quantization: true
  compute_dtype: bf16
  max_seq_length: 1024
  reasoning_mode: disabled
  chat_template: tokenizer
training:
  learning_rate: 0.0002
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  precision: auto
  optimizer: paged_adamw_8bit
  evaluation_strategy: "no"
  save_strategy: "no"
  load_best_model_at_end: false
  gradient_checkpointing: true
  completion_only_loss: true
  packing: false
  seed: 42
"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def squad_record(index: int = 0, *, padding: int = 0) -> dict[str, Any]:
    """Build a SQuAD-shaped record whose annotated offset verifies."""
    context = f"{PASSAGE} Paragraph {index}. " + ("Additional detail. " * padding)
    answer = "Chlorophyll"
    return {
        "id": f"sq-{index}",
        "title": f"Article {index % 7}",
        "context": context,
        "question": f"Which pigment absorbs sunlight, variant {index}?",
        "answers": {"text": [answer], "answer_start": [context.index(answer)]},
    }


def squad_examples(count: int, *, vary: bool = True) -> list[QuestionGenerationExample]:
    """Adapt realistic SQuAD rows through the real adapter."""
    rows = [
        squad_record(index, padding=(index % 5) if vary else 0) for index in range(count)
    ]
    return list(adapt_records("squad-qg", rows))


def target(**overrides: Any) -> QuestionGenerationTarget:
    """Build a valid short-answer target."""
    defaults = {
        "question_type": QuestionType.SHORT_ANSWER,
        "question": "Which pigment absorbs sunlight?",
        "answer": "Chlorophyll",
        "difficulty": Difficulty.EASY,
        "marks": 1,
    }
    return QuestionGenerationTarget(**{**defaults, **overrides})


def example(index: int = 0, **overrides: Any) -> QuestionGenerationExample:
    """Build a valid single-target example."""
    defaults = {
        "id": f"bench-{index:04d}",
        "context": f"{PASSAGE} Passage {index}.",
        "targets": (target(question=f"Question {index}?"),),
        "source": "squad-qg",
        "topic": "Photosynthesis",
    }
    return QuestionGenerationExample(**{**defaults, **overrides})


def experiment_config(**overrides: Any) -> Any:
    """Build a validated experiment configuration."""
    payload: dict[str, Any] = {"name": "bench-test"}
    payload.update(overrides)
    return experiment_config_from_dict(payload)


def length_summary(mean: float) -> TokenLengthSummary:
    """Build a summary with a chosen mean, for the representativeness check."""
    return TokenLengthSummary(
        count=10, minimum=int(mean), mean=mean, p50=int(mean), p95=int(mean),
        maximum=int(mean), total=int(mean) * 10,
    )


def sizing_with(*, mean: float = 445.0, completion_min: int = 40, truncated: int = 0):
    """Build a SplitSizing with chosen characteristics."""
    return SplitSizing(
        split="train",
        examples=10,
        prompt_tokens=length_summary(mean - 40),
        completion_tokens=TokenLengthSummary(
            count=10, minimum=completion_min, mean=float(completion_min),
            p50=completion_min, p95=completion_min, maximum=completion_min,
            total=completion_min * 10,
        ),
        total_tokens=length_summary(mean),
        truncated=truncated,
        max_seq_length=1024,
    )


class FakeTokenizer:
    """Renders both Qwen3 template branches; ``return_dict`` defaults to True as 5.x does."""

    def __init__(self) -> None:
        """Start with an empty vocabulary; atoms are interned as they are seen."""
        self.chat_template = (
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
            "{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}"
        )
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|im_end|>"
        self._atoms: list[str] = []

    def _identifier(self, atom: str) -> int:
        """Return a stable id for one atom."""
        if atom not in self._atoms:
            self._atoms.append(atom)
        return self._atoms.index(atom)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render, then tokenize when asked."""
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
            if kwargs.get("enable_thinking") is False:
                parts.append(EMPTY_THINK_BLOCK)
        text = "".join(parts)
        if not tokenize:
            return text
        ids = [self._identifier(atom) for atom in _ATOM.findall(text)]
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


class FakeParameter:
    """A parameter with a size and a gradient flag."""

    def __init__(self, count: int, *, requires_grad: bool) -> None:
        """Store the element count and the gradient flag."""
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        """Return the parameter's element count."""
        return self.count


class FakeAdaptedModel:
    """An adapter-wrapped model that can save a checkpoint-shaped directory."""

    def __init__(self, *, base_weight_leak: bool = False, incomplete: bool = False) -> None:
        """Build a model with the measured parameter split."""
        self._parameters = [
            FakeParameter(4_022_468_096, requires_grad=False),
            FakeParameter(33_030_144, requires_grad=True),
        ]
        self.config = SimpleNamespace(use_cache=False)
        self.is_gradient_checkpointing = True
        self.saved_to: str | None = None
        self._base_weight_leak = base_weight_leak
        self._incomplete = incomplete

    def parameters(self):
        """Yield the model's parameters."""
        return iter(self._parameters)

    def save_pretrained(self, path: str) -> None:
        """Write the files PEFT writes, and optionally ones it never would."""
        self.saved_to = path
        target_dir = Path(path)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        if not self._incomplete:
            (target_dir / "adapter_model.safetensors").write_bytes(b"\x00" * 4096)
        if self._base_weight_leak:
            (target_dir / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 128)


class FakeOptimizer:
    """Stands in for the optimiser transformers resolves from ``optim``."""


class FakeTrainer:
    """A trainer-shaped object that records whether it was asked to train."""

    def __init__(self, dataset: list[dict[str, Any]], *, steps: int = 50) -> None:
        """Hold a dataset and start with a zeroed state."""
        self.train_dataset = dataset
        self.train_calls = 0
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.optimizer: Any = None
        self.args = SimpleNamespace(optim="paged_adamw_8bit")
        self._steps = steps

    def train(self) -> SimpleNamespace:
        """Record the call and return a plausible TrainOutput."""
        self.train_calls += 1
        self.state.global_step = self._steps
        self.state.log_history = [
            {
                "loss": round(2.4 - index * 0.01, 4),
                "step": index + 1,
                "epoch": (index + 1) / self._steps,
                "learning_rate": 2e-4,
                "grad_norm": 1.1,
            }
            for index in range(self._steps)
        ]
        self.optimizer = FakeOptimizer()
        return SimpleNamespace(
            global_step=self._steps,
            training_loss=1.93,
            metrics={"train_loss": 1.93, "train_runtime": 62.5},
        )


@pytest.fixture
def prepared(tmp_path):
    """Write a prepared dataset and a config, and return the paths."""
    directory = tmp_path / "qgen-datasets" / "qgen-bench-test-abc123def456"
    examples = squad_examples(600)
    write_prepared_split(directory, SplitName.TRAIN, examples)
    write_prepared_split(directory, SplitName.VALIDATION, squad_examples(40))
    (directory / DATASET_DOCUMENT).write_text(
        json.dumps({"fingerprint": compute_dataset_fingerprint(examples)}), encoding="utf-8"
    )
    config_path = tmp_path / "bench.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    return SimpleNamespace(
        root=tmp_path,
        dataset_root=tmp_path / "qgen-datasets",
        directory=directory,
        config=config_path,
        examples=examples,
        output=tmp_path / "benchmarks",
    )


@pytest.fixture
def stack(monkeypatch, prepared):
    """Replace the three expensive calls with recording stand-ins."""
    tokenizer = FakeTokenizer()
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

    def fake_build_trainer(config, *, model, tokenizer, train_dataset, output_dir, **kwargs):
        trainer = FakeTrainer(list(train_dataset), steps=config.training.max_steps or 50)
        built["trainer"] = trainer
        built["output_dir"] = output_dir
        return trainer, plan_trainer_arguments(
            config, output_dir, train_examples=len(train_dataset)
        )

    monkeypatch.setattr(benchmark_module, "load_trainable_model", fake_load_trainable_model)
    monkeypatch.setattr(benchmark_module, "build_hf_dataset", list)
    monkeypatch.setattr(benchmark_module, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(benchmark_module, "_dataset_root", lambda: prepared.dataset_root)
    return SimpleNamespace(tokenizer=tokenizer, model=model, built=built, prepared=prepared)


# ---------------------------------------------------------------------------
# The prepared-dataset format
# ---------------------------------------------------------------------------


class TestPreparedFormat:
    """What prepare writes is what the benchmark reads, through one implementation."""

    def test_the_split_filenames_are_stable(self):
        assert split_filename(SplitName.TRAIN) == "train.jsonl"
        assert split_filename("validation") == "validation.jsonl"
        assert split_filename("test") == "test.jsonl"

    def test_an_unknown_split_is_refused(self):
        with pytest.raises(PreparedDatasetError, match="unknown split"):
            split_filename("holdout")

    def test_examples_round_trip(self, tmp_path):
        examples = squad_examples(25)
        write_prepared_split(tmp_path, SplitName.TRAIN, examples)
        assert read_prepared_split(tmp_path, SplitName.TRAIN) == tuple(examples)

    def test_the_order_is_preserved(self, tmp_path):
        examples = squad_examples(30)
        write_prepared_split(tmp_path, SplitName.TRAIN, examples)
        restored = read_prepared_split(tmp_path)
        assert [item.id for item in restored] == [item.id for item in examples]

    def test_the_fingerprint_survives_the_round_trip(self, tmp_path):
        examples = squad_examples(20)
        write_prepared_split(tmp_path, SplitName.TRAIN, examples)
        restored = read_prepared_split(tmp_path)
        assert compute_dataset_fingerprint(restored) == compute_dataset_fingerprint(examples)

    def test_grounding_survives_the_round_trip(self, tmp_path):
        """SQuAD examples carry verified offsets; losing them silently would matter."""
        examples = squad_examples(5)
        assert all(item.grounding is not None for item in examples)
        write_prepared_split(tmp_path, SplitName.TRAIN, examples)
        assert all(item.grounding is not None for item in read_prepared_split(tmp_path))

    def test_a_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(PreparedDatasetError, match="--write-dataset"):
            read_prepared_split(tmp_path / "absent")

    def test_a_missing_split_lists_what_is_present(self, tmp_path):
        write_prepared_split(tmp_path, SplitName.TRAIN, squad_examples(2))
        with pytest.raises(PreparedDatasetError, match="train.jsonl"):
            read_prepared_split(tmp_path, SplitName.TEST)

    def test_a_malformed_line_names_the_line(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text(
            json.dumps(squad_examples(1)[0].as_dict()) + "\nnot json\n", encoding="utf-8"
        )
        with pytest.raises(PreparedDatasetError, match="line 2"):
            read_prepared_split(tmp_path)

    def test_an_unreconstructable_example_is_reported(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text(json.dumps({"unexpected": True}) + "\n", encoding="utf-8")
        with pytest.raises(PreparedDatasetError, match="could not be read"):
            read_prepared_split(tmp_path)

    def test_an_empty_split_is_refused(self, tmp_path):
        (tmp_path / "train.jsonl").write_text("\n\n", encoding="utf-8")
        with pytest.raises(PreparedDatasetError, match="no examples"):
            read_prepared_split(tmp_path)

    def test_blank_lines_are_skipped(self, tmp_path):
        examples = squad_examples(3)
        path = tmp_path / "train.jsonl"
        path.write_text(
            "\n".join(json.dumps(item.as_dict()) for item in examples) + "\n\n",
            encoding="utf-8",
        )
        assert len(read_prepared_split(tmp_path)) == 3

    def test_discovery_finds_prepared_datasets(self, prepared):
        found = discover_prepared_datasets(prepared.dataset_root)
        assert len(found) == 1
        assert found[0].fingerprint == "abc123def456"
        assert found[0].experiment == "qgen-bench-test"
        assert found[0].has_train

    def test_discovery_of_nothing_is_not_an_error(self, tmp_path):
        assert discover_prepared_datasets(tmp_path / "absent") == ()

    def test_resolution_prefers_an_explicit_directory(self, prepared):
        info = resolve_prepared_directory(prepared.dataset_root, directory=prepared.directory)
        assert info.directory == prepared.directory

    def test_resolution_by_fingerprint(self, prepared):
        info = resolve_prepared_directory(prepared.dataset_root, fingerprint="abc123def456")
        assert info.directory == prepared.directory

    def test_an_unknown_fingerprint_lists_what_exists(self, prepared):
        with pytest.raises(PreparedDatasetError, match="qgen-bench-test-abc123def456"):
            resolve_prepared_directory(prepared.dataset_root, fingerprint="nope")

    def test_resolving_nothing_says_how_to_prepare(self, tmp_path):
        with pytest.raises(PreparedDatasetError, match="--write-dataset"):
            resolve_prepared_directory(tmp_path / "absent")

    def test_a_directory_without_splits_is_refused(self, tmp_path):
        empty = tmp_path / "qgen-x-000000"
        empty.mkdir(parents=True)
        with pytest.raises(PreparedDatasetError, match="no split files"):
            resolve_prepared_directory(tmp_path, directory=empty)

    def test_the_info_serializes(self, prepared):
        found = discover_prepared_datasets(prepared.dataset_root)[0]
        payload = json.loads(json.dumps(found.as_dict()))
        assert payload["has_train"] is True

    def test_fingerprint_verification_reports_rather_than_raises(self, prepared):
        examples = read_prepared_split(prepared.directory)
        report = verify_fingerprint(examples, compute_dataset_fingerprint(examples))
        assert report["matches_whole_corpus"] is True
        mismatch = verify_fingerprint(examples, "deadbeef")
        assert mismatch["matches_whole_corpus"] is False


# ---------------------------------------------------------------------------
# Sizing and overrides
# ---------------------------------------------------------------------------


class TestSubsetSizing:
    """How many examples a step count consumes."""

    def test_the_default_subset_matches_the_default_steps(self):
        assert subset_size_for(BENCHMARK_STEPS) == BENCHMARK_STEPS * 8
        assert subset_size_for(50) == 400

    def test_accumulation_multiplies_the_subset(self):
        assert subset_size_for(10, batch_size=1, gradient_accumulation_steps=8) == 80
        assert subset_size_for(10, batch_size=2, gradient_accumulation_steps=8) == 160

    @pytest.mark.parametrize(
        ("field", "value"),
        [("steps", 0), ("batch_size", 0), ("gradient_accumulation_steps", 0), ("steps", -1)],
    )
    def test_non_positive_values_are_refused(self, field, value):
        with pytest.raises(BenchmarkError, match="positive integer"):
            subset_size_for(**{"steps": 10, field: value})


class TestBenchmarkOverrides:
    """The measurement must describe the production configuration."""

    def test_the_defaults_match_the_proposed_production_configuration(self):
        training = benchmark_training_overrides()["training"]
        assert training["max_steps"] == BENCHMARK_STEPS == 50
        assert training["per_device_train_batch_size"] == BENCHMARK_BATCH_SIZE == 1
        assert training["gradient_accumulation_steps"] == (
            BENCHMARK_GRADIENT_ACCUMULATION_STEPS
        ) == 8

    def test_only_the_training_section_is_overridden(self):
        assert set(benchmark_training_overrides()) == {"training"}

    def test_every_cost_determining_setting_is_left_alone(self):
        """Changing any of these makes the timing describe a run nobody intends."""
        training = benchmark_training_overrides()["training"]
        for name in (
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "optimizer",
            "precision",
            "gradient_checkpointing",
            "completion_only_loss",
            "packing",
            "seed",
        ):
            assert name not in training, name

    def test_evaluation_and_checkpointing_are_off(self):
        training = benchmark_training_overrides()["training"]
        assert training["evaluation_strategy"] == "no"
        assert training["save_strategy"] == "no"
        assert training["load_best_model_at_end"] is False

    def test_every_step_is_logged(self):
        assert benchmark_training_overrides()["training"]["logging_steps"] == 1

    def test_the_overrides_preserve_the_shipped_settings(self, prepared):
        from qa_gen_runtime.config_io import load_experiment_config

        config = load_experiment_config(
            prepared.config, overrides=benchmark_training_overrides()
        )
        assert config.training.max_steps == 50
        assert config.training.gradient_accumulation_steps == 8
        assert config.training.optimizer == "paged_adamw_8bit"
        assert config.training.gradient_checkpointing is True
        assert config.training.completion_only_loss is True
        assert config.training.learning_rate == 0.0002
        assert config.training.lr_scheduler_type == "cosine"
        assert config.training.warmup_ratio == 0.03
        assert config.model.max_seq_length == 1024
        assert config.model.quantization_type == "nf4"
        assert config.model.double_quantization is True

    def test_the_configuration_stays_the_verified_baseline(self, prepared):
        from qa_gen import VERIFIED_QWEN3_4B_L4
        from qa_gen_runtime.config_io import load_experiment_config

        config = load_experiment_config(
            prepared.config, overrides=benchmark_training_overrides()
        )
        assert config.baseline_deviations(VERIFIED_QWEN3_4B_L4) == ()


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


class TestSubsetSelection:
    """Deterministic, seeded, reproducible, and drawn from the real corpus."""

    def test_the_subset_is_drawn_from_the_prepared_split(self, prepared):
        subset = select_benchmark_subset(prepared.dataset_root, size=80)
        assert len(subset) == 80
        assert subset.available == 600
        assert subset.source_split == "train"
        assert subset.source_directory == prepared.directory.as_posix()
        assert all(item.source == "squad-qg" for item in subset.examples)

    def test_the_subset_is_not_the_smoke_corpus(self, prepared):
        """The whole point: real adapted SQuAD ids, not hand-written ones."""
        subset = select_benchmark_subset(prepared.dataset_root, size=40)
        assert all(item.id.startswith("squad-qg-") for item in subset.examples)
        assert all("Chlorophyll" in item.targets[0].answer for item in subset.examples)

    def test_selection_is_deterministic(self, prepared):
        first = select_benchmark_subset(prepared.dataset_root, size=80, seed=7)
        second = select_benchmark_subset(prepared.dataset_root, size=80, seed=7)
        assert [i.id for i in first.examples] == [i.id for i in second.examples]
        assert first.fingerprint == second.fingerprint

    def test_a_different_seed_draws_a_different_subset(self, prepared):
        one = select_benchmark_subset(prepared.dataset_root, size=80, seed=1)
        two = select_benchmark_subset(prepared.dataset_root, size=80, seed=2)
        assert {i.id for i in one.examples} != {i.id for i in two.examples}

    def test_the_default_seed_is_recorded(self, prepared):
        subset = select_benchmark_subset(prepared.dataset_root, size=40)
        assert subset.seed == BENCHMARK_SUBSET_SEED

    def test_the_subset_records_its_fingerprint(self, prepared):
        subset = select_benchmark_subset(prepared.dataset_root, size=40)
        assert subset.fingerprint == compute_dataset_fingerprint(subset.examples)
        assert subset.source_fingerprint

    def test_requesting_more_than_exists_is_refused(self, prepared):
        with pytest.raises(BenchmarkError, match="were requested"):
            select_benchmark_subset(prepared.dataset_root, size=10_000)

    def test_a_split_can_be_chosen(self, prepared):
        subset = select_benchmark_subset(
            prepared.dataset_root, size=20, split=SplitName.VALIDATION
        )
        assert subset.source_split == "validation"
        assert subset.available == 40

    def test_the_corpus_mix_is_recorded(self, prepared):
        subset = select_benchmark_subset(prepared.dataset_root, size=40)
        assert subset.examples_by_source == {"squad-qg": 40}

    def test_the_subset_serializes_without_inlining_examples(self, prepared):
        payload = json.loads(
            json.dumps(select_benchmark_subset(prepared.dataset_root, size=40).as_dict())
        )
        assert payload["examples"] == 40
        assert len(payload["example_ids_head"]) == 5
        assert "targets" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class TestSubsetChecks:
    """A subset that cannot support a timing measurement must say so."""

    def subset(self, **overrides: Any) -> BenchmarkSubset:
        """Build a subset descriptor."""
        defaults = {"examples": tuple(squad_examples(10)), "requested": 10}
        return BenchmarkSubset(**{**defaults, **overrides})

    def test_a_healthy_subset_raises_nothing(self):
        findings = check_subset(self.subset(), sizing_with(), corpus_mean_tokens=445.0)
        assert findings == ()

    def test_a_size_mismatch_is_caught(self):
        findings = check_subset(self.subset(requested=400), sizing_with())
        assert {f.code for f in findings} == {"subset_size_mismatch"}

    def test_an_empty_completion_is_caught(self):
        findings = check_subset(self.subset(), sizing_with(completion_min=0))
        assert "empty_completion_tokens" in {f.code for f in findings}

    def test_high_truncation_is_caught(self):
        findings = check_subset(self.subset(), sizing_with(truncated=6))
        assert "subset_truncation_high" in {f.code for f in findings}

    def test_an_unrepresentative_length_distribution_is_caught(self):
        """Attention cost grows with length, so a short subset gives an optimistic figure."""
        findings = check_subset(self.subset(), sizing_with(mean=250.0), corpus_mean_tokens=445.0)
        codes = {f.code for f in findings}
        assert "subset_length_unrepresentative" in codes

    def test_a_close_length_distribution_passes(self):
        findings = check_subset(self.subset(), sizing_with(mean=430.0), corpus_mean_tokens=445.0)
        assert findings == ()

    def test_the_comparison_is_skipped_without_a_corpus_mean(self):
        """Not guessed at: an absent reference means the check is simply not made."""
        assert check_subset(self.subset(), sizing_with(mean=100.0)) == ()

    def test_no_sizing_means_only_structural_checks(self):
        assert check_subset(self.subset(), None) == ()

    def test_findings_serialize(self):
        payload = [f.as_dict() for f in check_subset(self.subset(requested=99), sizing_with())]
        assert json.loads(json.dumps(payload))[0]["blocking"] is True


class TestOutputAudit:
    """Only the adapter and the report belong in a benchmark output."""

    def build(self, root: Path, *files: str) -> tuple[Path, Path]:
        """Create a run directory with an adapter holding ``files``."""
        run_dir = root / "bench-run"
        adapter = run_dir / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        for name in files:
            (adapter / name).write_bytes(b"\x00" * 32)
        return adapter, run_dir

    def test_a_complete_adapter_passes(self, tmp_path):
        adapter, run_dir = self.build(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        findings, details = audit_benchmark_output(adapter, run_dir, expect_adapter=True)
        assert findings == ()
        assert details["adapter_exists"] is True
        assert details["adapter_bytes"] == 64

    def test_an_incomplete_adapter_is_caught(self, tmp_path):
        adapter, run_dir = self.build(tmp_path, "adapter_config.json")
        findings, _ = audit_benchmark_output(adapter, run_dir, expect_adapter=True)
        assert "adapter_incomplete" in {f.code for f in findings}

    def test_a_missing_adapter_is_tolerated_when_not_expected(self, tmp_path):
        run_dir = tmp_path / "bench-run"
        run_dir.mkdir()
        findings, _ = audit_benchmark_output(
            run_dir / "adapter", run_dir, expect_adapter=False
        )
        assert findings == ()

    @pytest.mark.parametrize(
        "leaked",
        ["model.safetensors", "model-00001-of-00002.safetensors", "pytorch_model.bin"],
    )
    def test_copied_base_weights_are_caught(self, tmp_path, leaked):
        adapter, run_dir = self.build(
            tmp_path, "adapter_config.json", "adapter_model.safetensors", leaked
        )
        findings, details = audit_benchmark_output(adapter, run_dir, expect_adapter=True)
        assert "base_weights_copied" in {f.code for f in findings}
        assert f"adapter/{leaked}" in details["base_weight_files"]

    def test_the_adapter_itself_is_not_flagged(self, tmp_path):
        adapter, run_dir = self.build(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        _, details = audit_benchmark_output(adapter, run_dir, expect_adapter=True)
        assert details["base_weight_files"] == []

    def test_an_oversized_file_is_caught(self, monkeypatch, tmp_path):
        monkeypatch.setattr(benchmark_module, "_MAX_EXPECTED_FILE_BYTES", 16)
        adapter, run_dir = self.build(
            tmp_path, "adapter_config.json", "adapter_model.safetensors"
        )
        findings, _ = audit_benchmark_output(adapter, run_dir, expect_adapter=True)
        assert "unexpectedly_large_files" in {f.code for f in findings}


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


class TestProjection:
    """The arithmetic that turns a measured step time into a decision."""

    def test_the_step_count_matches_the_stated_production_plan(self):
        """78,552 train examples at batch 1 / accum 8 / 2 epochs is 19,638 steps."""
        projection = project_production_run(78_552, seconds_per_step=None, epochs=2)
        assert projection["steps_per_epoch"] == 9819
        assert projection["total_steps"] == 19_638

    def test_wall_clock_scales_with_the_measured_rate(self):
        slow = project_production_run(78_552, seconds_per_step=2.0, epochs=2)
        fast = project_production_run(78_552, seconds_per_step=1.0, epochs=2)
        assert slow["estimated_hours"] == pytest.approx(2 * fast["estimated_hours"], rel=0.01)

    def test_no_rate_means_no_time_estimate(self):
        projection = project_production_run(78_552, seconds_per_step=None)
        assert projection["estimated_seconds"] is None
        assert projection["estimated_hours"] is None

    def test_the_assumption_is_stated(self):
        projection = project_production_run(78_552, seconds_per_step=1.5)
        assert "linear in the step count" in projection["assumption"]
        assert projection["measured_seconds_per_step"] == 1.5

    def test_the_warmup_ratio_is_applied(self):
        projection = project_production_run(78_552, seconds_per_step=None, warmup_ratio=0.03)
        assert projection["warmup_steps"] == 589

    def test_the_projection_serializes(self):
        json.dumps(project_production_run(78_552, seconds_per_step=1.2))


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCliArguments:
    """Neither mode is the default, and the defaults match the production plan."""

    def test_a_config_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_a_mode_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "c.yaml"])

    def test_the_modes_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "c.yaml", "--inspect-only", "--run"])

    def test_inspect_only_resolves_to_the_non_training_mode(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--inspect-only"])
        assert resolve_mode(args) is BenchmarkMode.INSPECT_ONLY

    def test_run_resolves_to_the_training_mode(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert resolve_mode(args) is BenchmarkMode.RUN

    def test_anything_short_of_run_resolves_to_inspect_only(self):
        assert resolve_mode(SimpleNamespace()) is BenchmarkMode.INSPECT_ONLY
        assert resolve_mode(SimpleNamespace(run=False)) is BenchmarkMode.INSPECT_ONLY
        assert resolve_mode(SimpleNamespace(run=None)) is BenchmarkMode.INSPECT_ONLY

    def test_the_step_count_is_configurable_and_defaults_to_fifty(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.steps == 50
        custom = build_parser().parse_args(
            ["--config", "c.yaml", "--run", "--steps", "200"]
        )
        assert custom.steps == 200

    def test_the_batch_defaults_match_production(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.batch_size == 1
        assert args.gradient_accumulation_steps == 8

    def test_the_subset_seed_is_fixed_by_default(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.subset_seed == BENCHMARK_SUBSET_SEED

    def test_the_split_choice_is_restricted(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.split == "train"
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--config", "c.yaml", "--run", "--split", "holdout"])

    def test_the_corpus_mean_is_optional(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.corpus_mean_tokens is None


# ---------------------------------------------------------------------------
# The training guard
# ---------------------------------------------------------------------------


class TestTrainingIsGuarded:
    """Where training can happen, asserted rather than assumed."""

    def test_the_train_call_refuses_without_the_explicit_flag(self):
        trainer = FakeTrainer([])
        with pytest.raises(BenchmarkError, match="--run"):
            benchmark_module._call_trainer_train(trainer, execute=False)
        assert trainer.train_calls == 0

    def test_the_train_call_works_with_the_explicit_flag(self):
        trainer = FakeTrainer([], steps=3)
        benchmark_module._call_trainer_train(trainer, execute=True)
        assert trainer.train_calls == 1

    def test_exactly_one_function_calls_train(self):
        tree = ast.parse(inspect.getsource(benchmark_module))
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

    def test_the_module_never_calls_backward(self):
        tree = ast.parse(inspect.getsource(benchmark_module))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "backward" not in called

    def test_the_module_downloads_no_dataset(self):
        tree = ast.parse(inspect.getsource(benchmark_module))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        for forbidden in ("load_dataset", "snapshot_download", "hf_hub_download"):
            assert forbidden not in names

    def test_no_credential_is_referenced(self):
        tree = ast.parse(inspect.getsource(benchmark_module))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        keywords = {
            keyword.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg
        }
        forbidden = {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "use_auth_token", "token"}
        assert not (names | keywords) & forbidden

    def test_the_prepared_format_module_neither_trains_nor_downloads(self):
        import qa_gen_runtime.prepared as prepared_module

        tree = ast.parse(inspect.getsource(prepared_module))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "train" not in called
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "load_dataset" not in names


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestInspectOnly:
    """Everything except the optimiser step."""

    def invoke(self, prepared, **overrides: Any):
        """Run the benchmark in inspect-only mode."""
        arguments = {
            "mode": BenchmarkMode.INSPECT_ONLY,
            "steps": 5,
            "output_dir": str(prepared.output),
        }
        arguments.update(overrides)
        return run_benchmark(str(prepared.config), **arguments)

    def test_inspect_only_never_trains(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.trained is False
        assert report.status == "inspected"
        assert stack.built["trainer"].train_calls == 0

    def test_inspect_only_creates_no_run_directory(self, stack, prepared):
        self.invoke(prepared)
        assert not prepared.output.exists()

    def test_inspect_only_measures_the_subset(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.subset is not None
        assert len(report.subset) == 40
        assert report.sizing is not None
        assert report.sizing.prompt_tokens.minimum > 100
        assert report.sizing.completion_tokens.minimum > 10

    def test_inspect_only_reports_a_projection_without_a_time(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.projection is not None
        assert report.projection["estimated_hours"] is None

    def test_inspect_only_writes_nothing(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.artifacts == {}
        assert report.measurements is None


class TestRunMode:
    """The one mode that trains, and what it measures."""

    def invoke(self, prepared, **overrides: Any):
        """Run the benchmark in run mode."""
        arguments = {
            "mode": BenchmarkMode.RUN,
            "steps": 5,
            "output_dir": str(prepared.output),
            "run_id": "bench-test",
        }
        arguments.update(overrides)
        return run_benchmark(str(prepared.config), **arguments)

    def test_run_mode_trains_exactly_once(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.trained is True
        assert report.status == "measured"
        assert stack.built["trainer"].train_calls == 1

    def test_the_measurements_are_recorded(self, stack, prepared):
        report = self.invoke(prepared)
        measured = report.measurements
        assert measured is not None
        assert measured.optimizer_steps == 5
        assert measured.requested_steps == 5
        assert measured.effective_batch_size == 8
        assert measured.examples_processed == 40
        assert measured.tokens_processed > 0
        assert measured.wall_clock_seconds is not None
        assert measured.seconds_per_optimizer_step is not None
        assert measured.examples_per_second is not None
        assert measured.tokens_per_second is not None
        assert measured.trainable_parameters == 33_030_144
        assert measured.total_parameters == 4_055_498_240
        assert measured.trainable_fraction is not None
        assert measured.optimizer_requested == "paged_adamw_8bit"
        assert measured.optimizer_class.endswith("FakeOptimizer")
        assert measured.gradient_checkpointing_requested is True
        assert measured.gradient_checkpointing_active is True
        assert measured.use_cache is False
        assert measured.final_loss == 1.93
        assert len(measured.losses) == 5
        assert measured.adapter_bytes == 4096 + len('{"r": 16}')
        assert "total_vram_gib" in measured.memory_after

    def test_the_adapter_is_saved_and_nothing_else(self, stack, prepared):
        report = self.invoke(prepared)
        adapter = prepared.output / "bench-test" / "adapter"
        assert (adapter / "adapter_config.json").is_file()
        assert (adapter / "adapter_model.safetensors").is_file()
        assert report.output["base_weight_files"] == []
        assert report.ok

    def test_the_report_is_written(self, stack, prepared):
        report = self.invoke(prepared)
        path = prepared.output / "bench-test" / BENCHMARK_REPORT_FILENAME
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["trained"] is True
        assert payload["measurements"]["optimizer_steps"] == 5
        assert report.artifacts["report"] == path.as_posix()

    def test_the_projection_uses_the_measured_rate(self, stack, prepared):
        report = self.invoke(prepared)
        assert report.projection["measured_seconds_per_step"] == (
            report.measurements.seconds_per_optimizer_step
        )
        assert report.projection["estimated_hours"] is not None

    def test_a_second_run_with_the_same_id_refuses(self, stack, prepared):
        self.invoke(prepared)
        with pytest.raises(BenchmarkError, match="already exists"):
            self.invoke(prepared)

    def test_a_base_weight_leak_blocks(self, monkeypatch, stack, prepared):
        monkeypatch.setattr(stack.model, "_base_weight_leak", True)
        report = self.invoke(prepared)
        assert "base_weights_copied" in {f.code for f in report.findings}
        assert report.ok is False

    def test_an_incomplete_adapter_blocks(self, monkeypatch, stack, prepared):
        monkeypatch.setattr(stack.model, "_incomplete", True)
        report = self.invoke(prepared)
        assert "adapter_incomplete" in {f.code for f in report.findings}
        assert report.ok is False

    def test_the_records_carry_the_reasoning_column(self, stack, prepared):
        """The Phase 17B.2 fix must be present on the real records too."""
        report = self.invoke(prepared)
        assert report.sizing.records_missing_template_kwargs == 0
        config = experiment_config()
        records = build_training_records(
            report.subset.examples[:2], config, tokenizer=stack.tokenizer
        )
        assert records[0][CHAT_TEMPLATE_KWARGS_COLUMN] == {"enable_thinking": False}

    def test_the_notes_warn_about_reading_the_loss(self, stack, prepared):
        report = self.invoke(prepared)
        assert any("loss curve does not" in note for note in report.notes)


class TestMicrobatchComparison:
    """A/B/C at a constant effective batch of 8: the same data, three memory layouts."""

    CONFIGURATIONS = (("A", 1, 8), ("B", 2, 4), ("C", 4, 2))

    def test_every_configuration_has_the_same_effective_batch(self):
        for label, batch, accum in self.CONFIGURATIONS:
            assert batch * accum == 8, label

    def test_every_configuration_consumes_the_same_subset_size(self):
        sizes = {
            label: subset_size_for(
                50, batch_size=batch, gradient_accumulation_steps=accum
            )
            for label, batch, accum in self.CONFIGURATIONS
        }
        assert set(sizes.values()) == {400}, sizes

    def test_every_configuration_selects_the_identical_subset(self, prepared):
        """The property the whole comparison rests on, asserted rather than assumed."""
        subsets = {}
        for label, batch, accum in self.CONFIGURATIONS:
            size = subset_size_for(50, batch_size=batch, gradient_accumulation_steps=accum)
            subsets[label] = select_benchmark_subset(prepared.dataset_root, size=size)
        fingerprints = {label: subset.fingerprint for label, subset in subsets.items()}
        assert len(set(fingerprints.values())) == 1, fingerprints
        ids = {label: [i.id for i in subset.examples] for label, subset in subsets.items()}
        assert ids["A"] == ids["B"] == ids["C"]

    def test_the_effective_batch_guard_accepts_every_configuration(self, stack, prepared):
        for label, batch, accum in self.CONFIGURATIONS:
            report = run_benchmark(
                str(prepared.config),
                mode=BenchmarkMode.INSPECT_ONLY,
                steps=5,
                batch_size=batch,
                gradient_accumulation_steps=accum,
                expect_effective_batch=batch * accum,
                output_dir=str(prepared.output),
                label=label,
            )
            assert report.measurements is None
            assert report.overrides["training"]["per_device_train_batch_size"] == batch
            assert report.overrides["training"]["gradient_accumulation_steps"] == accum

    def test_the_effective_batch_guard_refuses_a_mismatch(self, stack, prepared):
        """A mistyped accumulation would measure a different optimisation problem."""
        with pytest.raises(BenchmarkError, match="effective batch of 4"):
            run_benchmark(
                str(prepared.config),
                mode=BenchmarkMode.INSPECT_ONLY,
                steps=5,
                batch_size=2,
                gradient_accumulation_steps=2,
                expect_effective_batch=8,
                output_dir=str(prepared.output),
            )

    def test_the_subset_fingerprint_guard_accepts_a_match(self, stack, prepared):
        first = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.INSPECT_ONLY,
            steps=5,
            batch_size=1,
            gradient_accumulation_steps=8,
            output_dir=str(prepared.output),
        )
        second = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.INSPECT_ONLY,
            steps=5,
            batch_size=4,
            gradient_accumulation_steps=2,
            expect_subset_fingerprint=first.subset.fingerprint,
            output_dir=str(prepared.output),
        )
        assert second.subset.fingerprint == first.subset.fingerprint

    def test_the_subset_fingerprint_guard_refuses_a_mismatch(self, stack, prepared):
        with pytest.raises(BenchmarkError, match="different data"):
            run_benchmark(
                str(prepared.config),
                mode=BenchmarkMode.INSPECT_ONLY,
                steps=5,
                expect_subset_fingerprint="deadbeefdeadbeef",
                output_dir=str(prepared.output),
            )

    def test_a_changed_step_count_changes_the_subset_and_is_caught(self, stack, prepared):
        """Holding the effective batch constant is what keeps the subset identical."""
        first = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.INSPECT_ONLY,
            steps=5,
            output_dir=str(prepared.output),
        )
        with pytest.raises(BenchmarkError, match="different data"):
            run_benchmark(
                str(prepared.config),
                mode=BenchmarkMode.INSPECT_ONLY,
                steps=6,
                expect_subset_fingerprint=first.subset.fingerprint,
                output_dir=str(prepared.output),
            )

    def test_the_label_names_the_run_directory(self, stack, prepared):
        run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
            label="B",
        )
        directories = [item.name for item in prepared.output.iterdir()]
        assert len(directories) == 1
        assert directories[0].startswith("bench-B-")

    def test_the_notes_state_the_microbatch_arithmetic(self, stack, prepared):
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.INSPECT_ONLY,
            steps=5,
            batch_size=4,
            gradient_accumulation_steps=2,
            output_dir=str(prepared.output),
        )
        assert any(
            "micro-batch 4 x accumulation 2 = effective batch 8" in note
            for note in report.notes
        )

    def test_the_reported_effective_batch_matches(self, stack, prepared):
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            batch_size=2,
            gradient_accumulation_steps=4,
            output_dir=str(prepared.output),
            label="B",
        )
        assert report.measurements.effective_batch_size == 8
        assert report.measurements.examples_processed == 32


class TestPaddingEstimate:
    """Why a larger micro-batch is not automatically more efficient."""

    def test_batch_one_has_no_padding_overhead(self):
        estimate = estimate_padded_tokens([100, 500, 300], batch_size=1)
        assert estimate["unpadded_tokens"] == 900
        assert estimate["estimated_padded_tokens"] == 900
        assert estimate["estimated_padding_overhead"] == 0.0

    def test_a_larger_batch_pads_to_the_longest_member(self):
        estimate = estimate_padded_tokens([100, 500], batch_size=2)
        assert estimate["unpadded_tokens"] == 600
        assert estimate["estimated_padded_tokens"] == 1000
        assert estimate["estimated_padding_overhead"] == pytest.approx(0.6667, abs=1e-4)

    def test_a_ragged_final_batch_is_handled(self):
        estimate = estimate_padded_tokens([100, 200, 300], batch_size=2)
        assert estimate["estimated_padded_tokens"] == 400 + 300

    def test_uniform_lengths_pad_to_nothing(self):
        estimate = estimate_padded_tokens([400] * 8, batch_size=4)
        assert estimate["estimated_padding_overhead"] == 0.0

    def test_overhead_grows_with_the_micro_batch(self):
        lengths = [200, 400, 600, 800] * 8
        overheads = [
            estimate_padded_tokens(lengths, batch_size=size)["estimated_padding_overhead"]
            for size in (1, 2, 4)
        ]
        assert overheads[0] == 0.0
        assert overheads[0] < overheads[1] < overheads[2]

    def test_it_is_labelled_an_estimate(self):
        estimate = estimate_padded_tokens([100, 200], batch_size=2)
        assert estimate["measured"] is False
        assert "sampler shuffles" in estimate["note"]

    def test_no_lengths_yields_no_estimate(self):
        estimate = estimate_padded_tokens([], batch_size=4)
        assert estimate["estimated_padded_tokens"] == 0
        assert estimate["measured"] is False

    def test_a_non_positive_batch_is_refused(self):
        with pytest.raises(BenchmarkError, match="positive integer"):
            estimate_padded_tokens([100], batch_size=0)

    def test_the_lengths_are_retained_but_not_serialized(self, stack, prepared):
        """Needed in memory for the padding estimate; 78k integers must not reach a report."""
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.INSPECT_ONLY,
            steps=5,
            output_dir=str(prepared.output),
        )
        assert len(report.sizing.sequence_lengths) == 40
        assert "sequence_lengths" not in report.sizing.as_dict()

    def test_the_padding_estimate_reaches_the_report(self, stack, prepared):
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            batch_size=4,
            gradient_accumulation_steps=2,
            output_dir=str(prepared.output),
            label="C",
        )
        padding = report.measurements.padding
        assert padding["batch_size"] == 4
        assert padding["estimated_padded_tokens"] >= padding["unpadded_tokens"]
        assert padding["unpadded_tokens"] == report.measurements.tokens_processed


class TestOutOfMemoryHandling:
    """A micro-batch that does not fit is a result, not a crash."""

    def oom_stack(self, monkeypatch, prepared, *, exception: BaseException):
        """Patch the trainer so training raises ``exception``."""
        tokenizer = FakeTokenizer()
        model = FakeAdaptedModel()

        def fake_load(config):
            return LoadedModel(
                model=model,
                tokenizer=tokenizer,
                precision=resolve_precision("fp32"),
                trainable_parameters=33_030_144,
                total_parameters=4_055_498_240,
                adapters_attached=True,
            )

        class ExplodingTrainer(FakeTrainer):
            """Raises instead of training."""

            def train(self):
                self.train_calls += 1
                raise exception

        def fake_build_trainer(config, *, model, tokenizer, train_dataset, output_dir, **kw):
            return ExplodingTrainer(list(train_dataset)), plan_trainer_arguments(
                config, output_dir, train_examples=len(train_dataset)
            )

        monkeypatch.setattr(benchmark_module, "load_trainable_model", fake_load)
        monkeypatch.setattr(benchmark_module, "build_hf_dataset", list)
        monkeypatch.setattr(benchmark_module, "build_trainer", fake_build_trainer)
        monkeypatch.setattr(benchmark_module, "_dataset_root", lambda: prepared.dataset_root)
        return model

    def test_a_torch_oom_is_recognised(self):
        import torch

        assert benchmark_module.is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA oom"))

    def test_a_runtime_error_mentioning_oom_is_recognised(self):
        assert benchmark_module.is_out_of_memory(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        )

    def test_an_unrelated_error_is_not_recognised(self):
        assert not benchmark_module.is_out_of_memory(RuntimeError("shape mismatch"))
        assert not benchmark_module.is_out_of_memory(ValueError("out of memory"))

    def test_an_oom_is_recorded_rather_than_raised(self, monkeypatch, prepared):
        import torch

        self.oom_stack(
            monkeypatch, prepared, exception=torch.cuda.OutOfMemoryError("CUDA out of memory")
        )
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            batch_size=4,
            gradient_accumulation_steps=2,
            output_dir=str(prepared.output),
            label="C",
        )
        assert report.status == "out_of_memory"
        assert report.measurements.failed is True
        assert report.measurements.failure_type == "out_of_memory"
        assert "out of memory" in report.measurements.failure_message.lower()

    def test_no_timing_is_reported_for_a_failed_run(self, monkeypatch, prepared):
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
        )
        measured = report.measurements
        assert measured.seconds_per_optimizer_step is None
        assert measured.examples_per_second is None
        assert measured.tokens_per_second is None
        assert measured.final_loss is None

    def test_the_peak_memory_is_still_captured(self, monkeypatch, prepared):
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
        )
        assert "max_allocated_gib" in report.measurements.memory_after
        assert any("lower bound" in note for note in report.measurements.notes)

    def test_no_adapter_is_saved_after_a_failure(self, monkeypatch, prepared):
        model = self.oom_stack(
            monkeypatch, prepared, exception=RuntimeError("CUDA out of memory")
        )
        run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
            run_id="oom-run",
        )
        assert model.saved_to is None
        assert not (prepared.output / "oom-run" / "adapter").exists()

    def test_the_failure_is_a_blocking_finding(self, monkeypatch, prepared):
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
        )
        codes = {finding.code for finding in report.findings}
        assert "out_of_memory" in codes
        assert report.ok is False

    def test_the_report_is_still_written(self, monkeypatch, prepared):
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
            run_id="oom-report",
        )
        path = prepared.output / "oom-report" / BENCHMARK_REPORT_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "out_of_memory"
        assert payload["measurements"]["failed"] is True

    def test_an_unrelated_error_still_propagates(self, monkeypatch, prepared):
        """Only OOM is absorbed; a real bug must not be reported as a memory limit."""
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("shape mismatch"))
        with pytest.raises(RuntimeError, match="shape mismatch"):
            run_benchmark(
                str(prepared.config),
                mode=BenchmarkMode.RUN,
                steps=4,
                output_dir=str(prepared.output),
            )

    def test_the_cli_exits_four_on_oom(self, monkeypatch, prepared, capsys):
        """A distinct code so a sweep can tell 'does not fit' from 'is broken'."""
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        code = main(
            [
                "--config",
                str(prepared.config),
                "--run",
                "--steps",
                "4",
                "--output-dir",
                str(prepared.output),
                "--run-id",
                "oom-cli",
                "--log-level",
                "CRITICAL",
            ]
        )
        captured = capsys.readouterr()
        assert code == 4
        assert "does not fit" in captured.err
        assert "[ FAILED ]" in captured.out

    def test_the_text_report_omits_a_timing_figure_after_a_failure(
        self, monkeypatch, prepared
    ):
        self.oom_stack(monkeypatch, prepared, exception=RuntimeError("CUDA out of memory"))
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
        )
        text = format_report(report)
        assert "[ FAILED ]" in text
        assert "[ MEASURED ]" not in text
        assert "no adapter was saved" in text


class TestCliMicrobatchArguments:
    """The new flags, and that they default to the measured baseline."""

    def test_the_batch_flags_are_configurable(self):
        args = build_parser().parse_args(
            [
                "--config",
                "c.yaml",
                "--run",
                "--batch-size",
                "4",
                "--gradient-accumulation-steps",
                "2",
            ]
        )
        assert args.batch_size == 4
        assert args.gradient_accumulation_steps == 2

    def test_the_new_guards_default_to_absent(self):
        args = build_parser().parse_args(["--config", "c.yaml", "--run"])
        assert args.expect_effective_batch is None
        assert args.expect_subset_fingerprint is None
        assert args.label is None

    def test_the_guards_are_parsed(self):
        args = build_parser().parse_args(
            [
                "--config",
                "c.yaml",
                "--run",
                "--label",
                "C",
                "--expect-effective-batch",
                "8",
                "--expect-subset-fingerprint",
                "abc123",
            ]
        )
        assert args.label == "C"
        assert args.expect_effective_batch == 8
        assert args.expect_subset_fingerprint == "abc123"

    def test_a_guard_violation_exits_one(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "4",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--expect-effective-batch",
                "8",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "effective batch of 4" in capsys.readouterr().err

    @pytest.mark.parametrize(("label", "batch", "accum"), [("A", 1, 8), ("B", 2, 4), ("C", 4, 2)])
    def test_each_configuration_runs_through_the_cli(
        self, stack, prepared, capsys, label, batch, accum
    ):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--run",
                "--steps",
                "4",
                "--batch-size",
                str(batch),
                "--gradient-accumulation-steps",
                str(accum),
                "--expect-effective-batch",
                "8",
                "--label",
                label,
                "--json",
                "--output-dir",
                str(prepared.output),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["measurements"]["effective_batch_size"] == 8
        assert payload["measurements"]["padding"]["batch_size"] == batch
        assert payload["subset"]["examples"] == 4 * batch * accum


class TestReportSerialization:
    """A measurement that cannot be written down is not a measurement."""

    def test_an_empty_report_serializes(self):
        payload = json.loads(
            json.dumps(BenchmarkReport(mode="run", status="measured").as_dict())
        )
        assert payload["trained"] is False
        assert payload["subset"] is None

    def test_the_measurements_serialize(self):
        measured = BenchmarkMeasurements(
            optimizer_steps=50,
            requested_steps=50,
            losses=({"step": 1, "loss": 2.0},),
            dropped_trainer_arguments=("weird",),
        )
        payload = json.loads(json.dumps(measured.as_dict()))
        assert payload["losses"] == [{"step": 1, "loss": 2.0}]
        assert payload["dropped_trainer_arguments"] == ["weird"]

    def test_a_full_report_serializes(self, stack, prepared):
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
            run_id="bench-serialize",
        )
        payload = json.loads(json.dumps(report.as_dict(), default=str))
        assert payload["ok"] is True
        assert payload["subset"]["examples"] == 32
        assert payload["diagnostics"]["lora"]["rank"] == 16

    def test_the_ok_flag_follows_the_findings(self):
        blocking = BenchmarkFinding(code="x", message="m", blocking=True)
        advisory = BenchmarkFinding(code="y", message="m", blocking=False)
        assert BenchmarkReport(mode="run", status="measured", findings=(advisory,)).ok is True
        assert BenchmarkReport(mode="run", status="measured", findings=(blocking,)).ok is False

    def test_the_text_report_covers_every_section(self, stack, prepared):
        report = run_benchmark(
            str(prepared.config),
            mode=BenchmarkMode.RUN,
            steps=4,
            output_dir=str(prepared.output),
            run_id="bench-text",
        )
        text = format_report(report)
        for section in (
            "[ SUBSET ]",
            "[ SUBSET SEQUENCE LENGTHS ]",
            "[ MEASURED ]",
            "[ PROJECTED PRODUCTION RUN ]",
            "[ OUTPUT ]",
            "[ ARTIFACTS ]",
        ):
            assert section in text, section
        assert "mode          : run" in text
        assert "overall       : ok" in text


class TestCliBehaviour:
    """Exit codes, because a human and a CI check both read them."""

    def test_inspect_only_exits_zero(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "4",
                "--output-dir",
                str(prepared.output),
                "--log-level",
                "WARNING",
            ]
        )
        output = capsys.readouterr().out
        assert code == 0
        assert "mode          : inspect_only" in output
        assert "trained       : False" in output
        assert stack.built["trainer"].train_calls == 0

    def test_run_exits_zero(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--run",
                "--steps",
                "4",
                "--output-dir",
                str(prepared.output),
                "--run-id",
                "cli-run",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        assert "adapter_model.safetensors" in capsys.readouterr().out
        assert stack.built["trainer"].train_calls == 1

    def test_json_output_is_parseable(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--run",
                "--steps",
                "4",
                "--json",
                "--output-dir",
                str(prepared.output),
                "--run-id",
                "cli-json",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["experiment"] == "qgen-bench-test"
        assert payload["measurements"]["optimizer_steps"] == 4

    def test_a_blocking_finding_exits_three(self, monkeypatch, stack, prepared, capsys):
        monkeypatch.setattr(stack.model, "_base_weight_leak", True)
        code = main(
            [
                "--config",
                str(prepared.config),
                "--run",
                "--steps",
                "4",
                "--output-dir",
                str(prepared.output),
                "--run-id",
                "cli-block",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 3
        assert "base_weights_copied" in capsys.readouterr().err

    def test_a_missing_config_exits_one(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.root / "absent.yaml"),
                "--inspect-only",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "not found" in capsys.readouterr().err

    def test_a_missing_prepared_dataset_exits_one(self, monkeypatch, stack, prepared, capsys):
        monkeypatch.setattr(
            benchmark_module, "_dataset_root", lambda: prepared.root / "nowhere"
        )
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "4",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "--write-dataset" in capsys.readouterr().err

    def test_too_many_steps_for_the_corpus_exits_one(self, stack, prepared, capsys):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "5000",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 1
        assert "were requested" in capsys.readouterr().err

    def test_the_report_file_is_written_when_asked(self, stack, prepared):
        destination = prepared.root / "nested" / "bench.json"
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "4",
                "--report",
                str(destination),
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        assert json.loads(destination.read_text(encoding="utf-8"))["mode"] == "inspect_only"

    def test_the_corpus_mean_enables_the_representativeness_check(
        self, stack, prepared, capsys
    ):
        code = main(
            [
                "--config",
                str(prepared.config),
                "--inspect-only",
                "--steps",
                "4",
                "--corpus-mean-tokens",
                "10000",
                "--json",
                "--log-level",
                "WARNING",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 3
        codes = {finding["code"] for finding in payload["findings"]}
        assert "subset_length_unrepresentative" in codes


class TestPhaseBoundary:
    """What this step deliberately does not do."""

    def test_the_shipped_configs_are_untouched(self):
        from qa_ml.paths import find_repo_root

        smoke = (find_repo_root() / "ml/configs/qgen/qgen-smoke.yaml").read_text(
            encoding="utf-8"
        )
        assert "max_steps: 20" in smoke
        assert "gradient_accumulation_steps: 8" in smoke
        squad = (find_repo_root() / "ml/configs/qgen/qgen-squad.yaml").read_text(
            encoding="utf-8"
        )
        assert "sources: [squad-qg]" in squad
        assert "max_steps" not in squad

    def test_the_smoke_harness_is_unchanged_in_its_guard(self):
        from qa_gen_runtime.smoke import SmokeHarnessError, _call_trainer_train

        with pytest.raises(SmokeHarnessError, match="--run"):
            _call_trainer_train(SimpleNamespace(), execute=False)

    def test_the_benchmark_does_not_use_the_smoke_corpus(self):
        """The explicit requirement: real data, not the six hand-written examples."""
        source = inspect.getsource(benchmark_module)
        assert "smoke_data" not in source
        assert "build_smoke_examples" not in source

    def test_no_evaluation_is_implemented_yet(self):
        source = inspect.getsource(benchmark_module)
        assert "deberta" not in source.lower()
        assert "compute_metrics" not in source
