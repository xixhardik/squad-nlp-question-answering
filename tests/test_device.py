"""Tests for device resolution and CUDA diagnostics.

Written to pass on both target environments: the CPU-only local machine and the
Lightning AI L4. Assertions branch on measured availability rather than assuming
either case.
"""

from __future__ import annotations

import pytest
import torch

from qa_torch.device import (
    DeviceUnavailableError,
    collect_cuda_diagnostics,
    describe_device,
    require_cuda,
    resolve_device,
)

CUDA_AVAILABLE = torch.cuda.is_available()


class TestResolveDevice:
    """Behaviour of :func:`qa_torch.device.resolve_device`."""

    def test_explicit_cpu_is_honoured(self):
        assert resolve_device("cpu").type == "cpu"

    def test_case_and_whitespace_insensitive(self):
        assert resolve_device("  CPU  ").type == "cpu"

    def test_unknown_device_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown device type"):
            resolve_device("tpu")

    def test_automatic_resolution_matches_availability(self):
        device = resolve_device()
        assert device.type == ("cuda" if CUDA_AVAILABLE else "cpu")

    def test_automatic_cuda_selection_has_no_hard_coded_index(self):
        """A bare 'cuda' keeps the code correct under CUDA_VISIBLE_DEVICES."""
        if not CUDA_AVAILABLE:
            pytest.skip("No CUDA device on this machine.")
        assert resolve_device().index is None

    def test_env_var_override_is_respected(self, monkeypatch):
        monkeypatch.setenv("QAS_DEVICE", "cpu")
        assert resolve_device().type == "cpu"

    def test_empty_env_var_falls_back_to_automatic(self, monkeypatch):
        monkeypatch.setenv("QAS_DEVICE", "")
        device = resolve_device()
        assert device.type == ("cuda" if CUDA_AVAILABLE else "cpu")

    def test_explicit_argument_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("QAS_DEVICE", "cuda")
        assert resolve_device("cpu").type == "cpu"

    @pytest.mark.skipif(CUDA_AVAILABLE, reason="Requires a machine without CUDA.")
    def test_requesting_cuda_without_cuda_raises_rather_than_downgrading(self):
        """A silent CPU downgrade would turn a GPU run into a multi-day CPU run."""
        with pytest.raises(DeviceUnavailableError, match="CUDA was requested"):
            resolve_device("cuda")


class TestDescribeDevice:
    """Behaviour of :func:`qa_torch.device.describe_device`."""

    def test_describes_cpu_without_claiming_a_gpu(self):
        info = describe_device("cpu")
        assert info.type == "cpu"
        assert info.is_cuda is False
        assert info.index is None
        assert info.capability is None

    def test_accepts_a_torch_device_object(self):
        assert describe_device(torch.device("cpu")).type == "cpu"

    def test_defaults_to_the_resolved_device(self):
        info = describe_device()
        assert info.type == ("cuda" if CUDA_AVAILABLE else "cpu")

    def test_serializes_to_json_friendly_types(self):
        payload = describe_device("cpu").as_dict()
        assert payload["type"] == "cpu"
        assert payload["is_cuda"] is False
        assert "total_memory_gib" in payload

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA.")
    def test_reports_real_gpu_facts(self):
        info = describe_device("cuda")
        assert info.is_cuda is True
        assert info.name
        assert info.capability is not None
        assert info.total_memory_bytes and info.total_memory_bytes > 0
        assert info.total_memory_gib and info.total_memory_gib > 0


class TestCudaDiagnostics:
    """Diagnostics must be safe to call with no GPU present."""

    def test_does_not_raise_without_cuda(self):
        diagnostics = collect_cuda_diagnostics()
        assert isinstance(diagnostics, dict)

    def test_reports_availability_truthfully(self):
        diagnostics = collect_cuda_diagnostics()
        assert diagnostics["cuda_available"] is CUDA_AVAILABLE

    def test_contains_the_expected_keys(self):
        diagnostics = collect_cuda_diagnostics()
        for key in ("cuda_available", "cuda_compiled_version", "device_count", "devices"):
            assert key in diagnostics

    def test_device_count_is_consistent_with_availability(self):
        diagnostics = collect_cuda_diagnostics()
        if CUDA_AVAILABLE:
            assert diagnostics["device_count"] >= 1
            assert len(diagnostics["devices"]) == diagnostics["device_count"]
        else:
            assert diagnostics["device_count"] == 0
            assert diagnostics["devices"] == []


class TestRequireCuda:
    """The hard gate used before expensive training runs."""

    @pytest.mark.skipif(CUDA_AVAILABLE, reason="Requires a machine without CUDA.")
    def test_raises_without_cuda(self):
        with pytest.raises(DeviceUnavailableError, match="CUDA is required"):
            require_cuda()

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA.")
    def test_returns_cuda_device_when_available(self):
        assert require_cuda().type == "cuda"
