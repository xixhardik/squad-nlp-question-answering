"""GPU smoke test: does this machine actually train?

Deliberately cheap. It verifies the mechanics of a training step end to end -- CUDA
detection, moving a model to the device, forward pass, loss, backward pass, optimizer
step -- without touching the dataset or running an epoch. Run it on the Lightning L4
before committing GPU budget to a real run:

.. code-block:: bash

    pytest tests/test_gpu_smoke.py -v -m gpu

Everything is skip-guarded, so the same file passes on the CPU-only development
machine (where the CUDA tests skip and the CPU equivalents still execute). A test that
silently passed by doing nothing on a broken GPU would be worse than no test, so the
CUDA tests assert on measured device placement rather than merely completing.
"""

from __future__ import annotations

import pytest
import torch

from qa_torch.device import describe_device, resolve_device, resolve_precision

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available.")

# A tiny randomly-initialised BERT rather than a pretrained download: the training
# mechanics are what is under test, and this keeps the smoke test seconds long.
TINY_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 128,
    "max_position_embeddings": 128,
}


@pytest.fixture
def tiny_qa_model():
    """A small untrained BertForQuestionAnswering, built offline."""
    from transformers import BertConfig, BertForQuestionAnswering

    return BertForQuestionAnswering(BertConfig(**TINY_CONFIG))


@pytest.fixture
def tiny_batch():
    """One batch of QA inputs with start/end labels."""
    torch.manual_seed(0)
    batch_size, seq_len = 4, 32
    return {
        "input_ids": torch.randint(0, 512, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "start_positions": torch.randint(0, seq_len, (batch_size,)),
        "end_positions": torch.randint(0, seq_len, (batch_size,)),
    }


class TestDeviceDetection:
    def test_cuda_availability_is_reported_truthfully(self):
        assert torch.cuda.is_available() is CUDA_AVAILABLE

    def test_resolved_device_matches_availability(self):
        assert resolve_device().type == ("cuda" if CUDA_AVAILABLE else "cpu")

    @requires_cuda
    def test_gpu_is_identified(self):
        info = describe_device("cuda")
        assert info.is_cuda is True
        assert info.name, "the GPU reported no name"
        assert info.capability is not None
        assert info.total_memory_bytes and info.total_memory_bytes > 0

    @requires_cuda
    def test_gpu_has_usable_memory(self):
        info = describe_device("cuda")
        assert info.total_memory_gib is not None
        # The intended L4 has ~22 GiB; 4 GiB is a floor that only rejects a device far
        # too small for this workload rather than asserting one specific GPU.
        assert info.total_memory_gib > 4.0, (
            f"only {info.total_memory_gib:.1f} GiB of VRAM detected"
        )

    @requires_cuda
    def test_precision_plan_prefers_bf16_when_supported(self):
        """bf16 is preferred over fp16 because it needs no gradient loss scaling."""
        plan = resolve_precision("auto", "cuda")
        if torch.cuda.is_bf16_supported():
            assert plan.resolved == "bf16"
            assert plan.bf16 is True
            assert plan.fp16 is False
        else:
            assert plan.resolved == "fp16"

    def test_precision_plan_falls_back_to_fp32_on_cpu(self):
        plan = resolve_precision("auto", "cpu")
        assert plan.resolved == "fp32"
        assert plan.bf16 is False
        assert plan.fp16 is False


class TestTensorOperations:
    def test_matmul_agrees_with_cpu(self):
        from qa_ml.environment import run_tensor_test

        result = run_tensor_test(matrix_size=256)
        assert result["passed"] is True, result
        assert result["max_abs_deviation"] < result["tolerance"]

    @requires_cuda
    def test_tensors_actually_land_on_the_gpu(self):
        tensor = torch.ones(8, 8, device="cuda")
        assert tensor.is_cuda
        assert tensor.device.type == "cuda"
        torch.cuda.synchronize()

    @requires_cuda
    def test_host_to_device_round_trip_preserves_values(self):
        original = torch.randn(64, 64)
        assert torch.equal(original.cuda().cpu(), original)


class TestTrainingStep:
    """The actual smoke test: one complete optimizer step."""

    @staticmethod
    def _run_step(model, batch, device: torch.device) -> dict[str, object]:
        """Move everything to ``device`` and run forward, backward and step."""
        model = model.to(device)
        model.train()
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        before = model.qa_outputs.weight.detach().clone()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        grad = model.qa_outputs.weight.grad
        grad_norm = None if grad is None else float(grad.norm().item())

        optimizer.step()
        optimizer.zero_grad()

        after = model.qa_outputs.weight.detach().cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()

        return {
            "loss": float(loss.item()),
            "loss_device": loss.device.type,
            "start_logits_device": outputs.start_logits.device.type,
            "start_logits_shape": tuple(outputs.start_logits.shape),
            "grad_norm": grad_norm,
            "weights_changed": not torch.equal(before.cpu(), after),
        }

    def test_forward_backward_step_on_cpu(self, tiny_qa_model, tiny_batch):
        """Runs everywhere, so the mechanics are covered even without a GPU."""
        result = self._run_step(tiny_qa_model, tiny_batch, torch.device("cpu"))
        assert result["loss"] > 0
        assert result["grad_norm"] is not None and result["grad_norm"] > 0
        assert result["weights_changed"] is True

    @requires_cuda
    def test_model_moves_to_cuda(self, tiny_qa_model):
        model = tiny_qa_model.to("cuda")
        assert all(parameter.is_cuda for parameter in model.parameters())

    @requires_cuda
    def test_forward_pass_produces_logits_on_gpu(self, tiny_qa_model, tiny_batch):
        model = tiny_qa_model.to("cuda").eval()
        inputs = {
            key: value.to("cuda")
            for key, value in tiny_batch.items()
            if key in ("input_ids", "attention_mask")
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        assert outputs.start_logits.is_cuda
        assert outputs.end_logits.is_cuda
        assert outputs.start_logits.shape == tiny_batch["input_ids"].shape

    @requires_cuda
    def test_full_training_step_on_gpu(self, tiny_qa_model, tiny_batch):
        """Forward, loss, backward and optimizer step, all on the GPU."""
        result = self._run_step(tiny_qa_model, tiny_batch, torch.device("cuda"))
        assert result["loss_device"] == "cuda"
        assert result["start_logits_device"] == "cuda"
        assert result["loss"] > 0
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert result["grad_norm"] is not None and result["grad_norm"] > 0
        assert result["weights_changed"] is True, "the optimizer step changed nothing"

    @requires_cuda
    def test_loss_decreases_over_a_few_steps(self, tiny_qa_model, tiny_batch):
        """Overfitting one batch is the cheapest real check that learning works."""
        model = tiny_qa_model.to("cuda")
        model.train()
        batch = {key: value.to("cuda") for key, value in tiny_batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        torch.cuda.synchronize()
        assert losses[-1] < losses[0], f"loss did not fall: {losses[0]} -> {losses[-1]}"

    @requires_cuda
    def test_bf16_autocast_step(self, tiny_qa_model, tiny_batch):
        """The precision mode training will actually use on the L4."""
        if not torch.cuda.is_bf16_supported():
            pytest.skip("bf16 not supported on this GPU.")

        model = tiny_qa_model.to("cuda")
        model.train()
        batch = {key: value.to("cuda") for key, value in tiny_batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        assert torch.isfinite(loss), "bf16 loss was not finite"
        assert float(loss.item()) > 0

    @requires_cuda
    def test_peak_memory_is_measurable(self):
        """Peak GPU memory is recorded in every experiment, so it must be readable."""
        torch.cuda.reset_peak_memory_stats()
        tensor = torch.empty(256, 1024, 1024, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del tensor
        torch.cuda.empty_cache()
        assert peak > 0


class TestEngineOnDevice:
    """The batched logit collector must work on whichever device is resolved."""

    def test_collect_logits_matches_feature_count(self, tiny_qa_model):
        from qa_torch.engine import collect_qa_logits

        features = {
            "input_ids": [[1, 2, 3, 4] for _ in range(6)],
            "attention_mask": [[1, 1, 1, 1] for _ in range(6)],
        }
        start_logits, end_logits = collect_qa_logits(
            tiny_qa_model, features, batch_size=4
        )
        assert len(start_logits) == 6
        assert len(end_logits) == 6
        assert len(start_logits[0]) == 4

    def test_returns_python_floats_for_the_decoder(self, tiny_qa_model):
        """qa_core is torch-free, so logits must arrive as plain floats."""
        from qa_torch.engine import collect_qa_logits

        features = {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}
        start_logits, _ = collect_qa_logits(tiny_qa_model, features)
        assert all(isinstance(value, float) for value in start_logits[0])

    def test_rejects_a_non_positive_batch_size(self, tiny_qa_model):
        from qa_torch.engine import collect_qa_logits

        with pytest.raises(ValueError, match="batch_size must be positive"):
            collect_qa_logits(tiny_qa_model, {"input_ids": [[1]]}, batch_size=0)

    def test_rejects_features_without_input_ids(self, tiny_qa_model):
        from qa_torch.engine import collect_qa_logits

        with pytest.raises(ValueError, match="input_ids"):
            collect_qa_logits(tiny_qa_model, {"attention_mask": [[1]]})
