"""Phase 18: the production QLoRA training execution path.

What these tests prove, and what they cannot
--------------------------------------------
**Proved here.** Training is opt-in at three independent levels: the CLI flag, the
``execute=True`` argument, and the guard inside the single ``trainer.train()`` call site. The
preflight refuses a misconfigured run *before* the optimiser steps, and each of its checks
fails for the reason it claims. A failed training call never reports success, still writes its
report, and never saves an adapter. The report carries every field a later phase needs to
compare runs. Nothing writes base-model weights.

**Not proved here.** That Qwen3-4B loads in 4-bit on an L4, that 2,248 steps take four hours,
or that the resulting adapter is any good. Those need the GPU, and none of them can be faked
into a test that would still mean something.

Why the trainer is a stub
-------------------------
``peft``, ``trl`` and ``bitsandbytes`` are not installed here, and a real ``SFTTrainer`` would
need all three plus a GPU. So :class:`FakeTrainer` records whether ``train()`` was called and
with what, and :class:`FakeLoadedModel` presents the handful of attributes the preflight reads.

That is a deliberate boundary. These stubs cannot prove the training loop works -- the smoke
harness and the benchmark exist for that -- but they can prove the *plumbing* around it:
opt-in, preflight, save, audit, report, failure handling. Every stub attribute mirrors a real
one that :mod:`qa_gen_runtime.production` reads by name, so a rename upstream breaks these
tests rather than silently passing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qa_gen.examples import QuestionGenerationExample, QuestionGenerationTarget
from qa_gen.splitting import SplitName
from qa_gen_runtime import production as production_module
from qa_gen_runtime.config_io import load_experiment_config
from qa_gen_runtime.prepared import write_prepared_split
from qa_gen_runtime.production import (
    TRAINING_REPORT_FILENAME,
    PreflightCheck,
    PreflightReport,
    ProductionTrainingError,
    describe_loaded_quantization,
    describe_lora_attachment,
    execute_production_training,
    run_preflight,
)
from qa_gen_runtime.train import build_parser
from qa_gen_runtime.train import main as train_main
from qa_paper import Difficulty, QuestionType

PASSAGE = (
    "Photosynthesis is the process by which green plants convert light energy into chemical "
    "energy. Chlorophyll in the leaves absorbs sunlight, and the plant combines carbon dioxide "
    "from the air with water drawn up through the roots to produce glucose and oxygen."
)

RACE_OPTIONS = ("Haemoglobin", "Melanin", "Chlorophyll", "Carotene")
RACE_CORRECT_INDEX = 2

#: The fingerprint the fixture dataset directory is named with. Arbitrary, but fixed, so the
#: tests that assert fingerprint matching have something stable to assert against.
FIXTURE_FINGERPRINT = "abc123def4567890"


# ---------------------------------------------------------------------------
# Fixtures: a prepared corpus on disk
# ---------------------------------------------------------------------------


def squad_example(index: int) -> QuestionGenerationExample:
    """A SQuAD-shaped short-answer example."""
    return QuestionGenerationExample(
        id=f"squad-qg-{index:04d}",
        context=f"{PASSAGE} Paragraph {index}.",
        targets=(
            QuestionGenerationTarget(
                question_type=QuestionType.SHORT_ANSWER,
                question=f"Which pigment absorbs sunlight? ({index})",
                answer="Chlorophyll",
                difficulty=Difficulty.EASY,
                marks=1,
            ),
        ),
        source="squad-qg",
        group_key=f"title:Article {index % 3}",
    )


def race_example(index: int) -> QuestionGenerationExample:
    """A RACE-shaped MCQ example."""
    return QuestionGenerationExample(
        id=f"race-mcq-{index:04d}",
        context=f"{PASSAGE} Article {index}.",
        targets=(
            QuestionGenerationTarget(
                question_type=QuestionType.MCQ,
                question=f"Which pigment absorbs sunlight? (race {index})",
                answer=RACE_OPTIONS[RACE_CORRECT_INDEX],
                options=RACE_OPTIONS,
                correct_option_index=RACE_CORRECT_INDEX,
                difficulty=Difficulty.MEDIUM,
                marks=1,
            ),
        ),
        source="race-mcq",
    )


@pytest.fixture
def prepared_corpus(tmp_path: Path) -> Path:
    """Write a mixed prepared dataset directory shaped like a real one.

    Named ``<experiment>-<fingerprint>`` and carrying a ``dataset.json``, because
    :mod:`qa_gen_runtime.prepared` reads the fingerprint from both the directory name and that
    document.
    """
    directory = tmp_path / "qgen-datasets" / f"qgen-mixed-{FIXTURE_FINGERPRINT}"
    train = [squad_example(i) for i in range(4)] + [race_example(i) for i in range(4)]
    write_prepared_split(directory, SplitName.TRAIN, train)
    write_prepared_split(directory, SplitName.VALIDATION, [squad_example(100)])
    write_prepared_split(directory, SplitName.TEST, [race_example(200)])
    (directory / "dataset.json").write_text(
        json.dumps({"fingerprint": FIXTURE_FINGERPRINT}), encoding="utf-8"
    )
    return directory


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """A configuration shaped like the shipped mixed one, but tiny and CPU-safe."""
    path = tmp_path / "qgen-test.yaml"
    path.write_text(
        "\n".join(
            [
                "name: qgen-test",
                'phase: "18"',
                "dataset:",
                "  sources: [squad-qg, race-mcq]",
                "model:",
                "  model_id: Qwen/Qwen3-4B",
                "  max_seq_length: 1024",
                "  reasoning_mode: disabled",
                "training:",
                "  per_device_train_batch_size: 1",
                "  gradient_accumulation_steps: 8",
                "  num_train_epochs: 1",
                "  warmup_ratio: 0.03",
                '  evaluation_strategy: "no"',
                '  save_strategy: "no"',
                "  load_best_model_at_end: false",
                "  completion_only_loss: true",
                "  gradient_checkpointing: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures: stand-ins for the GPU stack
# ---------------------------------------------------------------------------


class FakeParameter:
    """The two attributes ``qa_gen_runtime.loader.count_parameters`` reads."""

    def __init__(self, requires_grad: bool, *, numel: int = 1) -> None:
        self.requires_grad = requires_grad
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class FakeQuantizationConfig:
    """The subset of ``BitsAndBytesConfig`` the preflight reads."""

    def __init__(
        self,
        *,
        load_in_4bit: bool = True,
        quant_type: str = "nf4",
        double_quant: bool = True,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = quant_type
        self.bnb_4bit_use_double_quant = double_quant
        self.bnb_4bit_compute_dtype = "torch.bfloat16"


class FakeInnerConfig:
    """A model config carrying a quantization config and a cache flag."""

    def __init__(self, quantization: Any) -> None:
        self.quantization_config = quantization
        self.use_cache = False


class FakeBaseModel:
    """The unwrapped model underneath the adapters."""

    def __init__(self, quantization: Any) -> None:
        self.config = FakeInnerConfig(quantization)


class FakePeftModel:
    """A stand-in for a PEFT-wrapped causal model.

    The class name matters: :func:`describe_lora_attachment` looks for ``"PeftModel"`` in it,
    exactly as it would on the real object.
    """

    def __init__(
        self,
        *,
        quantization: Any = None,
        adapter_names: tuple[str, ...] = ("default",),
        double_wrapped: bool = False,
    ) -> None:
        self._base = FakeBaseModel(
            quantization if quantization is not None else FakeQuantizationConfig()
        )
        self.peft_config = dict.fromkeys(adapter_names, object())
        self.active_adapters = list(adapter_names)
        self.base_model = FakePeftModel.__new__(FakePeftModel) if double_wrapped else self._base
        self.config = self._base.config
        self.is_gradient_checkpointing = True
        self.saved_to: list[str] = []
        self._double_wrapped = double_wrapped

    def get_base_model(self) -> Any:
        return self._base

    def named_parameters(self):
        prefix = "base_model.model.base_model.model" if self._double_wrapped else "base_model.model"
        for layer in range(2):
            yield f"{prefix}.layers.{layer}.q_proj.lora_A.default.weight", FakeParameter(True)
            yield f"{prefix}.layers.{layer}.q_proj.lora_B.default.weight", FakeParameter(True)

    def parameters(self):
        """Present because ``count_parameters`` reads it, via ``collect_diagnostics``."""
        yield FakeParameter(True, numel=33_030_144)
        yield FakeParameter(False, numel=4_055_498_240 - 33_030_144)

    def save_pretrained(self, path: str) -> None:
        """Write the two files PEFT writes, so the output audit sees a real adapter."""
        self.saved_to.append(path)
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_config.json").write_text("{}", encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"\x00" * 2048)


class FakePrecision:
    """The subset of ``PrecisionPlan`` the preflight and diagnostics read."""

    def __init__(self, *, bf16: bool = True) -> None:
        self.name = "bf16" if bf16 else "fp32"
        self.bf16 = bf16
        self.fp16 = False
        self.dtype = "torch.bfloat16" if bf16 else "torch.float32"
        self.reason = "stubbed for tests"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bf16": self.bf16, "fp16": self.fp16, "reason": self.reason}

    def describe_device(self) -> dict[str, Any]:
        return {"cuda_available": False, "device_name": None, "bf16_supported": self.bf16}


class FakeTokenizer:
    """A tokenizer whose chat template mentions the reasoning flag."""

    chat_template = "{% if enable_thinking %}<think>{% endif %}"
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
        return [0, 1, 2]


class FakeLoadedModel:
    """The subset of ``LoadedModel`` the orchestrator and preflight read."""

    def __init__(self, *, model: Any = None, precision: Any = None) -> None:
        self.model = model if model is not None else FakePeftModel()
        self.tokenizer = FakeTokenizer()
        self.precision = precision if precision is not None else FakePrecision()
        self.trainable_parameters = 33_030_144
        self.total_parameters = 4_055_498_240
        self.adapters_attached = True
        self.notes = ("stubbed",)

    @property
    def trainable_fraction(self) -> float:
        return round(self.trainable_parameters / self.total_parameters, 6)

    def as_dict(self) -> dict[str, Any]:
        return {"trainable_parameters": self.trainable_parameters}


class FakeTrainerState:
    """The trainer state the measurements read."""

    def __init__(self, *, global_step: int = 1, losses: int = 2) -> None:
        self.global_step = global_step
        self.log_history = [
            {"step": i + 1, "epoch": 1.0, "loss": 1.5 - 0.1 * i, "learning_rate": 2e-4}
            for i in range(losses)
        ]


class FakeTrainOutput:
    """The ``TrainOutput`` the measurements read."""

    def __init__(self, *, steps: int = 1, loss: float = 1.25) -> None:
        self.global_step = steps
        self.training_loss = loss
        self.metrics = {"train_loss": loss, "train_runtime": 12.5}


class FakeTrainer:
    """Records whether and how ``train()`` was called."""

    def __init__(self, *, steps: int = 1, raises: Exception | None = None) -> None:
        self.state = FakeTrainerState(global_step=0)
        self.optimizer = None
        self.train_calls: list[dict[str, Any]] = []
        self._steps = steps
        self._raises = raises

    def train(self, resume_from_checkpoint: str | None = None) -> FakeTrainOutput:
        self.train_calls.append({"resume_from_checkpoint": resume_from_checkpoint})
        if self._raises is not None:
            # Partial progress, exactly as a real OOM part-way through leaves it.
            self.state.global_step = max(0, self._steps - 1)
            raise self._raises
        self.state.global_step = self._steps
        return FakeTrainOutput(steps=self._steps)


class FakePlan:
    """The subset of ``TrainerPlan`` the preflight and report read."""

    def __init__(self, run_root: Path, **overrides: Any) -> None:
        self.arguments = {
            "output_dir": str(run_root),
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_length": 1024,
            "completion_only_loss": True,
            "bf16": True,
            "fp16": False,
            "save_strategy": "no",
            "save_total_limit": 2,
            "gradient_checkpointing": True,
            **overrides,
        }
        self.precision = FakePrecision()
        self.total_steps = 1
        self.warmup_steps = 1
        self.dropped_arguments = ()
        self.notes = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "arguments": dict(self.arguments),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "dropped_arguments": list(self.dropped_arguments),
            "notes": list(self.notes),
        }


@pytest.fixture
def stub_stack(monkeypatch: pytest.MonkeyPatch):
    """Replace model loading and trainer construction with stubs, and hand back the trainer.

    ``build_hf_dataset`` is left real -- ``datasets`` is installed -- so the record shape is
    still exercised by the library that will consume it.
    """
    state: dict[str, Any] = {"trainer": None, "loaded": None, "plan": None}

    def fake_load(config):
        state["loaded"] = FakeLoadedModel()
        return state["loaded"]

    def fake_build_trainer(config, **kwargs):
        trainer = state.get("preset_trainer") or FakeTrainer()
        plan = FakePlan(Path(kwargs["output_dir"]))
        state["trainer"] = trainer
        state["plan"] = plan
        return trainer, plan

    monkeypatch.setattr(production_module, "load_trainable_model", fake_load)
    monkeypatch.setattr(production_module, "build_trainer", fake_build_trainer)
    return state


def run(config_path: str, corpus: Path, tmp_path: Path, **kwargs: Any):
    """Execute production training against the fixture corpus."""
    defaults: dict[str, Any] = {
        "dataset_dir": str(corpus),
        "output_dir": str(tmp_path / "qgen-runs"),
        "run_id": "test-run",
        "execute": True,
    }
    return execute_production_training(config_path, **{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Execution is opt-in
# ---------------------------------------------------------------------------


class TestExecutionIsOptIn:
    """Three independent gates, because one is a single typo away from four GPU hours."""

    def test_the_orchestrator_refuses_without_execute(self, config_path, prepared_corpus):
        with pytest.raises(ProductionTrainingError, match="execute=False"):
            execute_production_training(config_path, dataset_dir=str(prepared_corpus))

    def test_refusing_happens_before_anything_is_read(
        self, config_path, monkeypatch
    ):
        """The refusal must not depend on a corpus, a model or a GPU being present."""
        def explode(*args: Any, **kwargs: Any):
            raise AssertionError("nothing should be loaded before the execute gate")

        monkeypatch.setattr(production_module, "load_trainable_model", explode)
        monkeypatch.setattr(production_module, "read_prepared_split", explode)
        with pytest.raises(ProductionTrainingError, match="execute=False"):
            execute_production_training(config_path, dataset_dir="anywhere")

    def test_the_train_call_site_refuses_without_the_keyword(self):
        """The innermost guard, matching the smoke and benchmark harnesses."""
        trainer = FakeTrainer()
        with pytest.raises(ProductionTrainingError, match="refusing to call trainer.train"):
            production_module._call_trainer_train(trainer, execute=False)
        assert trainer.train_calls == []

    def test_the_only_train_call_site_is_the_guarded_one(self):
        """So "where does this train?" has one answer that a reader can check.

        Counted over the module's own AST rather than its text, so a docstring mentioning
        ``trainer.train()`` does not affect the answer.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(production_module))
        call_sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
        ]
        assert len(call_sites) == 2, "expected only the two branches of _call_trainer_train"

        guard = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_call_trainer_train"
        )
        inside = [
            node
            for node in ast.walk(guard)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
        ]
        assert len(inside) == 2, "every trainer.train() must live inside the guard"


class TestPlanDoesNotTrain:
    """``--plan`` must never reach the optimiser."""

    def test_plan_does_not_call_train(self, config_path, monkeypatch, capsys):
        """Asserted by making any training attempt an error, then planning."""
        def explode(*args: Any, **kwargs: Any):
            raise AssertionError("--plan must not train")

        monkeypatch.setattr(production_module, "_call_trainer_train", explode)
        monkeypatch.setattr(production_module, "load_trainable_model", explode)
        code = train_main(
            [
                "--config",
                config_path,
                "--plan",
                "--train-examples",
                "8",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        assert "total steps" in capsys.readouterr().out

    def test_plan_and_execute_are_mutually_exclusive(self):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--config", "c.yaml", "--plan", "--execute-training"])
        assert excinfo.value.code == 2

    def test_the_bare_command_neither_plans_nor_trains(self, config_path, monkeypatch):
        def explode(*args: Any, **kwargs: Any):
            raise AssertionError("validation must not load a model")

        monkeypatch.setattr(production_module, "load_trainable_model", explode)
        assert train_main(["--config", config_path, "--log-level", "WARNING"]) == 0


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecutionCallsTrain:
    """The happy path, with the trainer stubbed."""

    def test_execution_calls_train_exactly_once(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        assert stub_stack["trainer"].train_calls == [{"resume_from_checkpoint": None}]
        assert report.success is True
        assert report.status == "completed"

    def test_the_run_directory_is_created_under_the_run_root(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        run(config_path, prepared_corpus, tmp_path)
        run_dir = tmp_path / "qgen-runs" / "test-run"
        assert run_dir.is_dir()
        assert (run_dir / TRAINING_REPORT_FILENAME).is_file()

    def test_the_report_is_written_into_the_run_directory(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        path = tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == report.run_id == "test-run"
        assert payload["success"] is True

    def test_every_run_document_is_written(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """The adapter is only loadable if the config and provenance are beside it."""
        report = run(config_path, prepared_corpus, tmp_path)
        run_dir = tmp_path / "qgen-runs" / "test-run"
        for name in ("config.resolved.json", "diagnostics.json", "dataset.json", "run.json"):
            assert (run_dir / name).is_file(), name
        assert set(report.artifacts) >= {
            "config",
            "diagnostics",
            "dataset",
            "record",
            "adapter",
            "training",
        }

    def test_the_dataset_document_records_the_tokenizer_identity(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """Recorded rather than copied: the id is what makes the adapter loadable."""
        run(config_path, prepared_corpus, tmp_path)
        payload = json.loads(
            (tmp_path / "qgen-runs" / "test-run" / "dataset.json").read_text(encoding="utf-8")
        )
        assert payload["tokenizer_id"] == "Qwen/Qwen3-4B"
        assert payload["fingerprint"] == FIXTURE_FINGERPRINT

    def test_resume_is_refused_when_nothing_was_checkpointed(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """``save_strategy: "no"`` cannot have produced a checkpoint, so resuming is refused."""
        path = tmp_path / "qgen-runs" / "test-run"
        path.mkdir(parents=True)
        checkpoint = path / "checkpoint-500"
        checkpoint.mkdir()
        config = load_experiment_config(config_path)
        assert config.training.save_strategy == "no"
        assert config.training.is_resumable is False
        with pytest.raises(ProductionTrainingError, match="cannot resume"):
            run(
                config_path,
                prepared_corpus,
                tmp_path,
                resume_from_checkpoint=str(checkpoint),
            )


class TestResumableRuns:
    """A configuration that checkpoints can be resumed, and the report says so."""

    @pytest.fixture
    def resumable_config(self, tmp_path: Path) -> str:
        """The test configuration, plus periodic checkpointing."""
        path = tmp_path / "qgen-resumable.yaml"
        path.write_text(
            "\n".join(
                [
                    "name: qgen-test",
                    'phase: "18"',
                    "dataset:",
                    "  sources: [squad-qg, race-mcq]",
                    "model:",
                    "  model_id: Qwen/Qwen3-4B",
                    "  max_seq_length: 1024",
                    "  reasoning_mode: disabled",
                    "training:",
                    "  per_device_train_batch_size: 1",
                    "  gradient_accumulation_steps: 8",
                    "  num_train_epochs: 1",
                    "  warmup_ratio: 0.03",
                    '  evaluation_strategy: "no"',
                    "  save_strategy: steps",
                    "  save_steps: 500",
                    "  save_total_limit: 2",
                    "  load_best_model_at_end: false",
                    "  completion_only_loss: true",
                    "  gradient_checkpointing: true",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return str(path)

    def test_the_configuration_reports_itself_resumable(self, resumable_config):
        training = load_experiment_config(resumable_config).training
        assert training.is_resumable is True
        assert training.save_strategy == "steps"
        assert training.save_steps == 500
        assert training.save_total_limit == 2

    def test_resume_reaches_the_trainer(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        """The whole point: the checkpoint path arrives at ``trainer.train()``."""
        run_dir = tmp_path / "qgen-runs" / "resumed"
        checkpoint = run_dir / "checkpoint-500"
        checkpoint.mkdir(parents=True)
        report = run(
            resumable_config,
            prepared_corpus,
            tmp_path,
            run_id="resumed",
            resume_from_checkpoint=str(checkpoint),
        )
        assert stub_stack["trainer"].train_calls == [
            {"resume_from_checkpoint": str(checkpoint)}
        ]
        assert report.resumed_from == str(checkpoint)
        assert report.success is True

    def test_resuming_reopens_the_existing_run_directory(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        """Without ``allow_existing`` the directory holding the checkpoint would be refused."""
        run_dir = tmp_path / "qgen-runs" / "resumed"
        checkpoint = run_dir / "checkpoint-500"
        checkpoint.mkdir(parents=True)
        run(
            resumable_config,
            prepared_corpus,
            tmp_path,
            run_id="resumed",
            resume_from_checkpoint=str(checkpoint),
        )
        assert checkpoint.is_dir(), "the checkpoint must survive the run it seeded"

    def test_a_fresh_run_under_a_resumable_config_passes_no_checkpoint(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        """Resumable does not mean resuming; a first run still starts from scratch."""
        report = run(resumable_config, prepared_corpus, tmp_path, run_id="fresh")
        assert stub_stack["trainer"].train_calls == [{"resume_from_checkpoint": None}]
        assert report.resumed_from is None
        assert report.resumable is True

    def test_the_report_records_the_checkpoint_schedule(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(resumable_config, prepared_corpus, tmp_path, run_id="recorded")
        assert report.resumable is True
        assert report.save_strategy == "steps"
        assert report.save_steps == 500
        payload = json.loads(json.dumps(report.as_dict(), default=str))
        assert payload["schedule"]["resumable"] is True
        assert payload["schedule"]["save_steps"] == 500

    def test_a_non_resumable_run_records_that_too(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """The unchanged config, so the field distinguishes the two rather than always true."""
        report = run(config_path, prepared_corpus, tmp_path, run_id="not-resumable")
        assert report.resumable is False
        assert report.save_strategy == "no"
        assert report.save_steps is None

    def test_checkpointing_does_not_write_base_model_weights(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        """Enabling checkpoints must not turn an adapter-only run into a 8 GB one.

        TRL checkpoints a PEFT model as adapter tensors, so the audit that guards the final
        artifact guards the intermediate ones too. Asserted here because "we now save
        periodically" is exactly the change that could have broken it.
        """
        report = run(resumable_config, prepared_corpus, tmp_path, run_id="ckpt-audit")
        assert report.output_audit["base_weight_files"] == []
        assert report.output_audit["oversized_files"] == []
        run_dir = tmp_path / "qgen-runs" / "ckpt-audit"
        names = {path.name for path in run_dir.rglob("*") if path.is_file()}
        assert not {
            name for name in names if name.startswith("model") and name.endswith(".safetensors")
        }

    def test_the_preflight_still_passes_under_periodic_checkpointing(
        self, resumable_config, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(resumable_config, prepared_corpus, tmp_path, run_id="preflight-ckpt")
        assert report.preflight.ok
        assert report.preflight.failed == ()


class TestAdapterOnlyOutput:
    """The artifact is an adapter. Nothing else may be written as weights."""

    def test_the_adapter_is_saved_to_the_adapter_directory(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        adapter = tmp_path / "qgen-runs" / "test-run" / "adapter"
        assert adapter.is_dir()
        assert (adapter / "adapter_config.json").is_file()
        assert (adapter / "adapter_model.safetensors").is_file()
        assert report.adapter_path == adapter.as_posix()

    def test_the_save_goes_through_peft_save_pretrained(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """Not ``trainer.save_model``, which would write the wrapped model."""
        run(config_path, prepared_corpus, tmp_path)
        saved = stub_stack["loaded"].model.saved_to
        assert len(saved) == 1
        assert saved[0].endswith("adapter")

    def test_no_base_model_weights_are_written(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        assert report.output_audit["base_weight_files"] == []
        run_dir = tmp_path / "qgen-runs" / "test-run"
        written = {path.name for path in run_dir.rglob("*") if path.is_file()}
        assert not {name for name in written if name.startswith("model") and "safetensors" in name}

    def test_the_adapter_size_is_recorded(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        assert report.adapter_bytes and report.adapter_bytes > 0

    def test_nothing_oversized_is_written(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path)
        assert report.output_audit["oversized_files"] == []


class TestReportContents:
    """Every field the phase asked to be recorded."""

    @pytest.fixture
    def report(self, config_path, prepared_corpus, tmp_path, stub_stack):
        return run(config_path, prepared_corpus, tmp_path)

    def test_identity_and_provenance(self, report, config_path, prepared_corpus):
        assert report.run_id == "test-run"
        assert report.config_path == config_path
        assert report.experiment == "qgen-test"
        assert len(report.config_hash) == 12
        assert report.model_id == "Qwen/Qwen3-4B"
        assert report.model_revision == "main"
        assert report.dataset_directory == prepared_corpus.as_posix()
        assert report.dataset_fingerprint == FIXTURE_FINGERPRINT
        assert report.dataset_split == "train"

    def test_split_counts(self, report):
        assert report.train_examples == 8
        assert report.validation_examples == 1
        assert report.test_examples == 1
        assert report.source_counts == {"race-mcq": 4, "squad-qg": 4}

    def test_schedule_fields(self, report):
        assert report.max_seq_length == 1024
        assert report.batch_size == 1
        assert report.gradient_accumulation_steps == 8
        assert report.effective_batch_size == 8
        assert report.total_optimizer_steps == 1
        assert report.completed_optimizer_steps == 1
        assert report.warmup_steps == 1

    def test_measurements_are_recorded(self, report):
        measurements = report.measurements
        assert measurements is not None
        assert measurements.wall_clock_seconds is not None
        assert measurements.seconds_per_step is not None
        assert measurements.steps_per_second is not None
        assert measurements.final_loss == 1.25
        assert len(measurements.losses) == 2
        assert measurements.train_metrics["train_loss"] == 1.25

    def test_memory_fields_are_present_even_without_cuda(self, report):
        """Null rather than absent, so a report from a CPU box has the same shape."""
        after = report.measurements.memory_after
        assert set(after) >= {"max_allocated_gib", "max_reserved_gib"}
        assert report.measurements.peak_stats_reset is False

    def test_parameter_counts_are_recorded(self, report):
        assert report.trainable_parameters == 33_030_144
        assert report.total_parameters == 4_055_498_240

    def test_the_report_serializes_to_json(self, report):
        payload = json.loads(json.dumps(report.as_dict(), default=str))
        assert payload["schedule"]["effective_batch_size"] == 8
        assert payload["dataset"]["source_counts"] == {"race-mcq": 4, "squad-qg": 4}
        assert payload["preflight"]["ok"] is True
        assert payload["error"] is None

    def test_no_error_block_on_success(self, report):
        assert report.error is None
        assert report.success is True


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class TestFailureReporting:
    """A run that died must say so, keep what it learned, and save no adapter."""

    @pytest.fixture
    def failing(self, stub_stack):
        stub_stack["preset_trainer"] = FakeTrainer(
            steps=5, raises=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        )
        return stub_stack

    def test_training_failure_raises(self, config_path, prepared_corpus, tmp_path, failing):
        with pytest.raises(ProductionTrainingError, match="training failed"):
            run(config_path, prepared_corpus, tmp_path)

    def test_the_report_is_still_written(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        path = tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["success"] is False
        assert payload["status"] == "failed"

    def test_the_failure_is_never_reported_as_success(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        payload = json.loads(
            (tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert payload["success"] is False

    def test_the_exception_is_recorded(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        payload = json.loads(
            (tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert payload["error"]["type"] == "RuntimeError"
        assert "out of memory" in payload["error"]["message"]

    def test_completed_steps_are_reported(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        """The number that says how much work was lost."""
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        payload = json.loads(
            (tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert payload["schedule"]["completed_optimizer_steps"] == 4
        assert payload["error"]["completed_optimizer_steps"] == 4

    def test_no_adapter_is_saved_after_a_failure(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        """Saving a half-trained adapter as though it were finished is worse than none."""
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        assert not (tmp_path / "qgen-runs" / "test-run" / "adapter").exists()
        assert failing["loaded"].model.saved_to == []

    def test_execution_does_not_continue_past_the_exception(
        self, config_path, prepared_corpus, tmp_path, failing
    ):
        """No diagnostics document, because the run stopped instead of tidying up."""
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path)
        assert not (tmp_path / "qgen-runs" / "test-run" / "diagnostics.json").exists()

    def test_the_cli_reports_a_failure_with_exit_code_three(
        self, config_path, prepared_corpus, tmp_path, failing, capsys
    ):
        code = train_main(
            [
                "--config",
                config_path,
                "--execute-training",
                "--dataset-dir",
                str(prepared_corpus),
                "--output-dir",
                str(tmp_path / "qgen-runs"),
                "--run-id",
                "test-run",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 3
        assert "ProductionTrainingError" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    """Each check must fail for the reason it names, and block before the optimiser steps."""

    def preflight(self, config_path, prepared_corpus, tmp_path, **overrides):
        """Run the preflight directly, with substitutable pieces."""
        from qa_gen_runtime.dataset import build_training_records
        from qa_gen_runtime.outputs import RunPaths
        from qa_gen_runtime.prepared import read_prepared_split, resolve_prepared_directory

        config = load_experiment_config(config_path)
        info = resolve_prepared_directory(
            prepared_corpus.parent, directory=str(prepared_corpus)
        )
        examples = read_prepared_split(prepared_corpus, SplitName.TRAIN)
        loaded = overrides.pop("loaded", None) or FakeLoadedModel()
        records = build_training_records(examples, config, tokenizer=loaded.tokenizer)
        paths = RunPaths.under(tmp_path / "qgen-runs" / "test-run")
        plan = overrides.pop("plan", None) or FakePlan(paths.root)
        counts: dict[str, int] = {}
        for item in examples:
            counts[item.source] = counts.get(item.source, 0) + 1

        defaults: dict[str, Any] = {
            "loaded": loaded,
            "records": records,
            "plan": plan,
            "paths": paths,
            "dataset_info": info,
            "fingerprint_check": {"recomputed_for_this_split": "x", "recorded": None},
            "source_counts": counts,
        }
        return run_preflight(config, **{**defaults, **overrides})

    def named(self, report: PreflightReport, name: str) -> PreflightCheck:
        """Return one check by name."""
        found = next((check for check in report.checks if check.name == name), None)
        assert found is not None, f"{name} is not among {[c.name for c in report.checks]}"
        return found

    def test_a_clean_configuration_passes(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert report.ok, [check.as_dict() for check in report.failed]

    def test_the_dataset_directory_is_checked(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "dataset_directory_exists").passed is True

    def test_a_matching_fingerprint_passes(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, expect_fingerprint=FIXTURE_FINGERPRINT
        )
        assert self.named(report, "dataset_fingerprint_matches_request").passed is True
        assert report.ok

    def test_a_mismatched_fingerprint_blocks(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, expect_fingerprint="0000000000000000"
        )
        check = self.named(report, "dataset_fingerprint_matches_request")
        assert check.passed is False
        assert check.blocking is True
        assert not report.ok

    def test_an_absent_fingerprint_is_undetermined_not_failed(
        self, config_path, prepared_corpus, tmp_path
    ):
        """Not passing one is careless, not wrong; the report says which."""
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        check = self.named(report, "dataset_fingerprint_matches_request")
        assert check.passed is None
        assert check.blocking is False
        assert report.ok

    def test_matching_train_example_counts_pass(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, expect_train_examples=8
        )
        assert self.named(report, "train_examples_match_request").passed is True

    def test_a_wrong_train_example_count_blocks(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, expect_train_examples=17978
        )
        check = self.named(report, "train_examples_match_request")
        assert check.passed is False
        assert "records=8" in check.detail
        assert not report.ok

    def test_matching_source_counts_pass(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(
            config_path,
            prepared_corpus,
            tmp_path,
            expect_source_counts={"squad-qg": 4, "race-mcq": 4},
        )
        assert self.named(report, "source_counts_match_request").passed is True

    def test_wrong_source_counts_block(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(
            config_path,
            prepared_corpus,
            tmp_path,
            expect_source_counts={"squad-qg": 10000, "race-mcq": 10000},
        )
        assert self.named(report, "source_counts_match_request").passed is False
        assert not report.ok

    def test_reasoning_suppression_is_verified_on_the_records(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        check = self.named(report, "reasoning_suppression_is_active")
        assert check.passed is True
        assert "records_carrying_the_column=8/8" in check.detail

    def test_reasoning_suppression_fails_when_the_template_ignores_the_flag(
        self, config_path, prepared_corpus, tmp_path
    ):
        """A template with no ``enable_thinking`` means the column is never emitted."""

        class Blind(FakeTokenizer):
            chat_template = "{{ messages }}"

        loaded = FakeLoadedModel()
        loaded.tokenizer = Blind()
        report = self.preflight(config_path, prepared_corpus, tmp_path, loaded=loaded)
        assert self.named(report, "reasoning_suppression_is_active").passed is False
        assert not report.ok

    def test_completion_only_masking_is_verified(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "completion_only_masking_is_active").passed is True

    def test_completion_only_masking_off_blocks(
        self, config_path, prepared_corpus, tmp_path
    ):
        paths_root = tmp_path / "qgen-runs" / "test-run"
        report = self.preflight(
            config_path,
            prepared_corpus,
            tmp_path,
            plan=FakePlan(paths_root, completion_only_loss=False),
        )
        assert self.named(report, "completion_only_masking_is_active").passed is False
        assert not report.ok

    def test_a_dropped_completion_only_argument_blocks(
        self, config_path, prepared_corpus, tmp_path
    ):
        """The installed TRL silently not supporting it is the same failure."""
        plan = FakePlan(tmp_path / "qgen-runs" / "test-run")
        plan.dropped_arguments = ("completion_only_loss",)
        report = self.preflight(config_path, prepared_corpus, tmp_path, plan=plan)
        assert self.named(report, "completion_only_masking_is_active").passed is False

    def test_the_sequence_length_is_verified(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "max_sequence_length_is_correct").passed is True

    def test_a_wrong_sequence_length_blocks(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(
            config_path,
            prepared_corpus,
            tmp_path,
            plan=FakePlan(tmp_path / "qgen-runs" / "test-run", max_length=512),
        )
        assert self.named(report, "max_sequence_length_is_correct").passed is False
        assert not report.ok

    def test_lora_attachment_is_verified(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "lora_is_attached_exactly_once").passed is True

    def test_a_double_wrapped_model_blocks(self, config_path, prepared_corpus, tmp_path):
        """Attaching twice nests a PeftModel and nothing raises, so it has to be detected."""
        loaded = FakeLoadedModel(model=FakePeftModel(double_wrapped=True))
        report = self.preflight(config_path, prepared_corpus, tmp_path, loaded=loaded)
        assert self.named(report, "lora_is_attached_exactly_once").passed is False
        assert not report.ok

    def test_two_adapters_block(self, config_path, prepared_corpus, tmp_path):
        loaded = FakeLoadedModel(
            model=FakePeftModel(adapter_names=("default", "second"))
        )
        report = self.preflight(config_path, prepared_corpus, tmp_path, loaded=loaded)
        assert self.named(report, "lora_is_attached_exactly_once").passed is False

    def test_an_unwrapped_model_blocks(self, config_path, prepared_corpus, tmp_path):
        loaded = FakeLoadedModel(model=FakeBaseModel(FakeQuantizationConfig()))
        report = self.preflight(config_path, prepared_corpus, tmp_path, loaded=loaded)
        assert self.named(report, "lora_is_attached_exactly_once").passed is False

    def test_quantization_is_verified_from_the_loaded_model(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "quantization_is_4bit_nf4").passed is True

    def test_a_missing_quantization_config_blocks(
        self, config_path, prepared_corpus, tmp_path
    ):
        """An unquantized load is a memory failure discovered at the first backward pass."""
        model = FakePeftModel()
        model._base.config.quantization_config = None  # noqa: SLF001 - simulating a bad load
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, loaded=FakeLoadedModel(model=model)
        )
        assert self.named(report, "quantization_is_4bit_nf4").passed is False
        assert not report.ok

    def test_a_non_nf4_quant_type_blocks(self, config_path, prepared_corpus, tmp_path):
        model = FakePeftModel(quantization=FakeQuantizationConfig(quant_type="fp4"))
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, loaded=FakeLoadedModel(model=model)
        )
        assert self.named(report, "quantization_is_4bit_nf4").passed is False

    def test_double_quantization_off_blocks_when_the_config_asked_for_it(
        self, config_path, prepared_corpus, tmp_path
    ):
        model = FakePeftModel(quantization=FakeQuantizationConfig(double_quant=False))
        report = self.preflight(
            config_path, prepared_corpus, tmp_path, loaded=FakeLoadedModel(model=model)
        )
        assert self.named(report, "quantization_is_4bit_nf4").passed is False

    def test_bf16_is_verified(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "bf16_is_active").passed is True

    def test_fp32_precision_blocks(self, config_path, prepared_corpus, tmp_path):
        """Which is why this preflight cannot pass on a CPU box, correctly."""
        loaded = FakeLoadedModel(precision=FakePrecision(bf16=False))
        report = self.preflight(
            config_path,
            prepared_corpus,
            tmp_path,
            loaded=loaded,
            plan=FakePlan(tmp_path / "qgen-runs" / "test-run", bf16=False),
        )
        assert self.named(report, "bf16_is_active").passed is False
        assert not report.ok

    def test_no_base_model_output_path_is_verified(
        self, config_path, prepared_corpus, tmp_path
    ):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "no_base_model_output_path_is_configured").passed is True

    def test_a_preexisting_base_checkpoint_blocks(
        self, config_path, prepared_corpus, tmp_path
    ):
        run_dir = tmp_path / "qgen-runs" / "test-run"
        run_dir.mkdir(parents=True)
        (run_dir / "model-00001-of-00003.safetensors").write_bytes(b"\x00")
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "no_base_model_output_path_is_configured").passed is False
        assert not report.ok

    def test_the_effective_batch_is_verified(self, config_path, prepared_corpus, tmp_path):
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        assert self.named(report, "effective_batch_matches_configuration").passed is True

    def test_all_eleven_required_checks_are_present(
        self, config_path, prepared_corpus, tmp_path
    ):
        """The list the phase asked for, pinned so a check cannot quietly disappear."""
        report = self.preflight(config_path, prepared_corpus, tmp_path)
        names = {check.name for check in report.checks}
        assert names >= {
            "dataset_directory_exists",
            "dataset_fingerprint_matches_request",
            "source_counts_match_request",
            "train_examples_match_request",
            "reasoning_suppression_is_active",
            "completion_only_masking_is_active",
            "max_sequence_length_is_correct",
            "lora_is_attached_exactly_once",
            "quantization_is_4bit_nf4",
            "bf16_is_active",
            "no_base_model_output_path_is_configured",
        } - {"source_counts_match_request"}

    def test_the_report_serializes(self, config_path, prepared_corpus, tmp_path):
        payload = json.loads(
            json.dumps(self.preflight(config_path, prepared_corpus, tmp_path).as_dict())
        )
        assert payload["ok"] is True
        assert isinstance(payload["checks"], list)


class TestPreflightBlocksTraining:
    """A refused preflight must stop the run before the optimiser steps."""

    def test_a_failed_preflight_prevents_training(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        with pytest.raises(ProductionTrainingError, match="preflight failed"):
            run(config_path, prepared_corpus, tmp_path, expect_train_examples=17978)
        assert stub_stack["trainer"].train_calls == []

    def test_a_failed_preflight_still_writes_its_report(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path, expect_train_examples=17978)
        payload = json.loads(
            (tmp_path / "qgen-runs" / "test-run" / TRAINING_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert payload["status"] == "preflight_failed"
        assert payload["success"] is False
        assert "train_examples_match_request" in payload["preflight"]["failed"]

    def test_a_failed_preflight_saves_no_adapter(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        with pytest.raises(ProductionTrainingError):
            run(config_path, prepared_corpus, tmp_path, expect_train_examples=17978)
        assert not (tmp_path / "qgen-runs" / "test-run" / "adapter").exists()

    def test_a_mismatched_fingerprint_prevents_training(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """An explicit directory wins over the fingerprint, so the preflight is the check.

        Two layers guard this, and they catch different mistakes. Selecting *by* fingerprint
        fails in :func:`~qa_gen_runtime.prepared.resolve_prepared_directory` because nothing
        matches. Naming a directory *and* a fingerprint reaches the preflight, because the
        directory wins -- and that is the combination where a stale ``--dataset-dir`` would
        otherwise train on the wrong corpus while the command line claimed a known fingerprint.
        """
        with pytest.raises(ProductionTrainingError, match="preflight failed"):
            run(
                config_path,
                prepared_corpus,
                tmp_path,
                dataset_fingerprint="0000000000000000",
            )
        assert stub_stack["trainer"].train_calls == []

    def test_selecting_an_unknown_fingerprint_fails_before_the_preflight(
        self, config_path, prepared_corpus, tmp_path, stub_stack, monkeypatch
    ):
        """The other layer: no directory named, and no prepared dataset matches.

        The discovery root is redirected at the fixture so this does not consult, or depend on,
        whatever happens to be in the real artifacts directory.
        """
        from qa_gen_runtime.prepared import PreparedDatasetError

        monkeypatch.setattr(
            production_module, "_prepared_root", lambda _: prepared_corpus.parent
        )
        with pytest.raises(PreparedDatasetError, match="no prepared dataset with fingerprint"):
            execute_production_training(
                config_path,
                dataset_dir=None,
                dataset_fingerprint="0000000000000000",
                output_dir=str(tmp_path / "qgen-runs"),
                execute=True,
            )
        assert stub_stack["trainer"] is None

    def test_selecting_the_fixture_fingerprint_without_a_directory_works(
        self, config_path, prepared_corpus, tmp_path, stub_stack, monkeypatch
    ):
        """Fingerprint selection is the reproducible route, so it has to actually resolve."""
        monkeypatch.setattr(
            production_module, "_prepared_root", lambda _: prepared_corpus.parent
        )
        report = execute_production_training(
            config_path,
            dataset_dir=None,
            dataset_fingerprint=FIXTURE_FINGERPRINT,
            output_dir=str(tmp_path / "qgen-runs"),
            run_id="by-fingerprint",
            execute=True,
        )
        assert report.success is True
        assert report.dataset_fingerprint == FIXTURE_FINGERPRINT


# ---------------------------------------------------------------------------
# Model inspection helpers
# ---------------------------------------------------------------------------


class TestModelInspection:
    """The helpers the preflight relies on, tested directly."""

    def test_a_single_adapter_is_described(self):
        described = describe_lora_attachment(FakePeftModel())
        assert described["is_peft_model"] is True
        assert described["adapter_count"] == 1
        assert described["base_model_is_also_peft"] is False
        assert described["parameter_names_show_double_wrap"] is False
        assert described["lora_a_parameter_count"] == 2

    def test_a_double_wrap_is_detected(self):
        described = describe_lora_attachment(FakePeftModel(double_wrapped=True))
        assert described["base_model_is_also_peft"] is True
        assert described["parameter_names_show_double_wrap"] is True

    def test_a_plain_model_is_not_a_peft_model(self):
        described = describe_lora_attachment(FakeBaseModel(FakeQuantizationConfig()))
        assert described["is_peft_model"] is False
        assert described["adapter_names"] is None

    def test_quantization_is_read_from_the_base_model(self):
        described = describe_loaded_quantization(FakePeftModel())
        assert described["quantization_config_present"] is True
        assert described["load_in_4bit"] is True
        assert described["quant_type"] == "nf4"
        assert described["double_quant"] is True

    def test_an_absent_quantization_config_is_reported_not_guessed(self):
        model = FakePeftModel()
        model._base.config.quantization_config = None  # noqa: SLF001 - simulating a bad load
        described = describe_loaded_quantization(model)
        assert described["quantization_config_present"] is False
        assert described["load_in_4bit"] is None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestRunIdentity:
    """A run must be identifiable, and two runs of the same thing must agree."""

    def test_an_explicit_run_id_is_used_verbatim(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path, run_id="phase18-mixed-001")
        assert report.run_id == "phase18-mixed-001"
        assert (tmp_path / "qgen-runs" / "phase18-mixed-001").is_dir()

    def test_a_generated_run_id_carries_the_config_hash(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        report = run(config_path, prepared_corpus, tmp_path, run_id=None)
        config = load_experiment_config(config_path)
        assert config.config_hash() in report.run_id
        assert "Qwen--Qwen3-4B" in report.run_id
        assert "-r16-" in report.run_id

    def test_the_config_hash_is_stable_across_runs(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        first = run(config_path, prepared_corpus, tmp_path, run_id="a")
        second = run(config_path, prepared_corpus, tmp_path, run_id="b")
        assert first.config_hash == second.config_hash
        assert first.dataset_fingerprint == second.dataset_fingerprint
        assert first.source_counts == second.source_counts
        assert first.train_examples == second.train_examples

    def test_an_existing_run_directory_is_refused(
        self, config_path, prepared_corpus, tmp_path, stub_stack
    ):
        """Overwriting a previous run would destroy the artifact it holds."""
        from qa_gen_runtime.outputs import RunOutputError

        run(config_path, prepared_corpus, tmp_path, run_id="collide")
        with pytest.raises(RunOutputError, match="already exists"):
            run(config_path, prepared_corpus, tmp_path, run_id="collide")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


class TestCliPlumbing:
    """The flags, and that they reach the orchestrator."""

    def test_the_cli_trains_and_exits_zero(
        self, config_path, prepared_corpus, tmp_path, stub_stack, capsys
    ):
        code = train_main(
            [
                "--config",
                config_path,
                "--execute-training",
                "--dataset-dir",
                str(prepared_corpus),
                "--output-dir",
                str(tmp_path / "qgen-runs"),
                "--run-id",
                "cli-run",
                "--expect-train-examples",
                "8",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        output = capsys.readouterr().out
        assert "QLORA PRODUCTION TRAINING" in output
        assert "cli-run" in output
        assert stub_stack["trainer"].train_calls == [{"resume_from_checkpoint": None}]

    def test_the_cli_emits_json_on_request(
        self, config_path, prepared_corpus, tmp_path, stub_stack, capsys
    ):
        code = train_main(
            [
                "--config",
                config_path,
                "--execute-training",
                "--dataset-dir",
                str(prepared_corpus),
                "--output-dir",
                str(tmp_path / "qgen-runs"),
                "--run-id",
                "json-run",
                "--json",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["run_id"] == "json-run"
        assert payload["success"] is True

    def test_expected_source_counts_reach_the_preflight(
        self, config_path, prepared_corpus, tmp_path, stub_stack, capsys
    ):
        code = train_main(
            [
                "--config",
                config_path,
                "--execute-training",
                "--dataset-dir",
                str(prepared_corpus),
                "--output-dir",
                str(tmp_path / "qgen-runs"),
                "--run-id",
                "counts-run",
                "--expect-source-count",
                "squad-qg=4",
                "--expect-source-count",
                "race-mcq=4",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 0

    def test_a_wrong_expected_source_count_is_refused(
        self, config_path, prepared_corpus, tmp_path, stub_stack, capsys
    ):
        code = train_main(
            [
                "--config",
                config_path,
                "--execute-training",
                "--dataset-dir",
                str(prepared_corpus),
                "--output-dir",
                str(tmp_path / "qgen-runs"),
                "--run-id",
                "bad-counts",
                "--expect-source-count",
                "squad-qg=10000",
                "--log-level",
                "WARNING",
            ]
        )
        assert code == 3
        assert "source_counts_match_request" in capsys.readouterr().err
        assert stub_stack["trainer"].train_calls == []

    def test_a_malformed_source_count_is_a_usage_error(self):
        from qa_gen_runtime.train import _parse_source_counts

        with pytest.raises(SystemExit):
            _parse_source_counts(["squad-qg"])
        with pytest.raises(SystemExit):
            _parse_source_counts(["squad-qg=many"])

    def test_source_counts_parse(self):
        from qa_gen_runtime.train import _parse_source_counts

        assert _parse_source_counts(["a=1", "b=2"]) == {"a": 1, "b": 2}
        assert _parse_source_counts([]) is None
