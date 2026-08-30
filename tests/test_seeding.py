"""Tests for reproducible seeding."""

from __future__ import annotations

import random

import pytest
import torch

from qa_ml.seeding import build_dataloader_generator, set_global_seed


class TestSetGlobalSeed:
    def test_returns_a_record_of_what_was_seeded(self):
        record = set_global_seed(123)
        assert record["seed"] == 123
        assert record["full_determinism"] is False
        assert record["torch"] is True

    def test_rejects_a_negative_seed(self):
        with pytest.raises(ValueError, match="non-negative"):
            set_global_seed(-1)

    def test_accepts_zero(self):
        assert set_global_seed(0)["seed"] == 0

    def test_python_random_is_reproducible(self):
        set_global_seed(42)
        first = [random.random() for _ in range(5)]
        set_global_seed(42)
        assert [random.random() for _ in range(5)] == first

    def test_torch_is_reproducible(self):
        set_global_seed(42)
        first = torch.randn(10)
        set_global_seed(42)
        assert torch.equal(torch.randn(10), first)

    def test_different_seeds_give_different_streams(self):
        set_global_seed(1)
        first = torch.randn(10)
        set_global_seed(2)
        assert not torch.equal(torch.randn(10), first)

    def test_numpy_is_reproducible(self):
        numpy = pytest.importorskip("numpy")
        set_global_seed(42)
        first = numpy.random.rand(5)
        set_global_seed(42)
        assert numpy.allclose(numpy.random.rand(5), first)

    def test_records_whether_cuda_was_seeded(self):
        record = set_global_seed(42)
        assert record["torch_cuda_seeded"] is torch.cuda.is_available()

    def test_model_head_initialisation_is_reproducible(self):
        """Seeding must happen before the model is built.

        AutoModelForQuestionAnswering initialises `qa_outputs` randomly, so the seed
        has to be set first for the starting weights to be reproducible. This checks
        the mechanism with a bare linear layer, no download required.
        """
        set_global_seed(7)
        first = torch.nn.Linear(8, 2).weight.detach().clone()
        set_global_seed(7)
        second = torch.nn.Linear(8, 2).weight.detach().clone()
        assert torch.equal(first, second)


class TestDataloaderGenerator:
    def test_is_seeded_and_reproducible(self):
        first = build_dataloader_generator(42)
        second = build_dataloader_generator(42)
        assert torch.equal(
            torch.randperm(10, generator=first), torch.randperm(10, generator=second)
        )

    def test_different_seeds_differ(self):
        first = torch.randperm(20, generator=build_dataloader_generator(1))
        second = torch.randperm(20, generator=build_dataloader_generator(2))
        assert not torch.equal(first, second)
