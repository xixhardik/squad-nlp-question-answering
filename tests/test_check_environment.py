"""Tests for the environment diagnostics script.

The script is the Phase 2 gate before GPU training, so its own behaviour needs to
be trustworthy: it must run on a CPU-only machine, report availability
truthfully, and fail loudly when CUDA is demanded but absent.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from qa_ml.paths import find_repo_root

REPO_ROOT = find_repo_root()
SCRIPT = REPO_ROOT / "ml" / "scripts" / "check_environment.py"
CUDA_AVAILABLE = torch.cuda.is_available()


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the diagnostics script as a subprocess from the repository root."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )


class TestHumanReadableReport:
    """Default table output."""

    def test_exits_successfully(self):
        result = _run_script()
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        "section",
        ["SYSTEM", "PACKAGES", "TORCH / CUDA", "NVIDIA DRIVER", "RESULT"],
    )
    def test_report_contains_section(self, section):
        assert section in _run_script().stdout

    def test_reports_the_required_versions(self):
        stdout = _run_script().stdout
        for label in ("python", "torch version", "transformers", "datasets", "tokenizers"):
            assert label in stdout

    def test_reports_cuda_availability_truthfully(self):
        stdout = _run_script().stdout
        expected = "YES" if CUDA_AVAILABLE else "NO"
        assert f"cuda available{' ' * 15}{expected}" in stdout

    @pytest.mark.skipif(CUDA_AVAILABLE, reason="Requires a machine without CUDA.")
    def test_cpu_only_machine_is_labelled_as_not_for_training(self):
        stdout = _run_script().stdout
        assert "CPU-only environment" in stdout
        assert "NOT for training" in stdout


class TestJsonReport:
    """Machine-readable output, used to populate experiment records."""

    def test_emits_valid_json(self):
        result = _run_script("--json")
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)

    @pytest.mark.parametrize("key", ["system", "packages", "torch", "nvidia_smi"])
    def test_top_level_key_present(self, key):
        assert key in json.loads(_run_script("--json").stdout)

    def test_records_every_version_needed_for_reproducibility(self):
        report = json.loads(_run_script("--json").stdout)
        system = report["system"]
        assert system["python_version"]
        assert system["os"]
        for package in ("torch", "transformers", "tokenizers", "datasets"):
            assert report["packages"][package], f"{package} version not recorded"

    def test_cuda_fields_are_consistent_with_availability(self):
        report = json.loads(_run_script("--json").stdout)
        torch_info = report["torch"]
        assert torch_info["available"] is True
        assert torch_info["cuda_available"] is CUDA_AVAILABLE
        if not CUDA_AVAILABLE:
            assert torch_info["device_count"] == 0
            assert torch_info["devices"] == []
            assert torch_info["resolved_device"] == "cpu"

    def test_does_not_claim_a_gpu_that_is_absent(self):
        """The central honesty requirement for this script."""
        report = json.loads(_run_script("--json").stdout)
        if not CUDA_AVAILABLE:
            assert report["torch"]["cuda_available"] is False
            assert not report["torch"]["devices"]
            assert report["nvidia_smi"]["present"] is False

    def test_writes_json_to_output_path(self, tmp_path):
        destination = tmp_path / "nested" / "env.json"
        result = _run_script("--json", "--output", str(destination))
        assert result.returncode == 0, result.stderr
        assert destination.is_file()
        assert json.loads(destination.read_text(encoding="utf-8"))["system"]


class TestTensorTest:
    """Numerical sanity check on the resolved device."""

    def test_passes_on_this_machine(self):
        result = _run_script("--json", "--tensor-test")
        assert result.returncode == 0, result.stderr
        test = json.loads(result.stdout)["tensor_test"]
        assert test["passed"] is True, test
        assert test["max_abs_deviation"] < test["tolerance"]

    def test_runs_on_the_resolved_device(self):
        report = json.loads(_run_script("--json", "--tensor-test").stdout)
        expected = "cuda" if CUDA_AVAILABLE else "cpu"
        assert report["tensor_test"]["device"].startswith(expected)

    def test_reported_in_table_output(self):
        stdout = _run_script("--tensor-test").stdout
        assert "TENSOR SANITY TEST" in stdout
        assert "PASSED" in stdout


class TestRequireCudaGate:
    """`--require-cuda` is the pre-training hard gate."""

    @pytest.mark.skipif(CUDA_AVAILABLE, reason="Requires a machine without CUDA.")
    def test_exits_nonzero_without_cuda(self):
        result = _run_script("--require-cuda")
        assert result.returncode == 1
        assert "CUDA is unavailable" in result.stderr

    @pytest.mark.skipif(CUDA_AVAILABLE, reason="Requires a machine without CUDA.")
    def test_still_prints_the_full_report_before_failing(self):
        """A gate that fails silently would be hard to diagnose."""
        result = _run_script("--require-cuda")
        assert "QAS-NLP ENVIRONMENT DIAGNOSTICS" in result.stdout

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="Requires CUDA.")
    def test_exits_zero_with_cuda(self):
        assert _run_script("--require-cuda").returncode == 0


class TestRunnableFromAnyDirectory:
    """The script must not depend on the caller's working directory."""

    def test_runs_from_a_subdirectory(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--json"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=REPO_ROOT / "ml" / "configs",
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["system"]["python_version"]
