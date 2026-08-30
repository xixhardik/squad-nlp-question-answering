"""Tests for repository layout, path handling and pinned dependency versions.

These are guard rails against the failure modes that silently break
reproducibility: a directory moving, a script becoming dependent on the working
directory, a git-ignore rule stopping short of a weight file, or the installed
environment drifting away from ``constraints.txt``.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pytest

from qa_ml.paths import find_repo_root, get_paths

REPO_ROOT = find_repo_root()


class TestRepositoryLayout:
    """Required files and directories must exist where the tooling expects them."""

    @pytest.mark.parametrize(
        "relative",
        [
            "README.md",
            ".gitignore",
            ".editorconfig",
            ".env.example",
            "pyproject.toml",
            "constraints.txt",
            "requirements-dev.txt",
            "ml/requirements.txt",
            "ml/requirements-cpu.txt",
            "ml/requirements-gpu.txt",
            "ml/configs/base.yaml",
            "ml/scripts/check_environment.py",
            "backend/requirements.txt",
            "backend/app/main.py",
            "src/qa_core/__init__.py",
            "src/qa_torch/__init__.py",
            "src/qa_ml/__init__.py",
            "docs/lightning-workflow.md",
            "docs/dataset-policy.md",
            "docs/dependencies.md",
            "docs/pipeline.md",
            "src/qa_core/alignment.py",
            "src/qa_core/postprocess.py",
            "src/qa_torch/features.py",
            "src/qa_torch/loader.py",
            "src/qa_torch/engine.py",
            "src/qa_torch/inference.py",
            "src/qa_ml/data.py",
            "src/qa_ml/preprocess.py",
            "src/qa_ml/train.py",
            "src/qa_ml/evaluate.py",
            "src/qa_ml/experiment.py",
            "src/qa_ml/seeding.py",
            "src/qa_ml/environment.py",
            "src/qa_ml/cli.py",
            "src/qa_ml/__main__.py",
        ],
    )
    def test_required_file_exists(self, relative):
        assert (REPO_ROOT / relative).is_file(), f"missing required file: {relative}"

    @pytest.mark.parametrize(
        "relative",
        ["src", "ml", "backend", "docs", "tests", "models", "reports", "artifacts", "data"],
    )
    def test_required_directory_exists(self, relative):
        assert (REPO_ROOT / relative).is_dir(), f"missing required directory: {relative}"

    def test_no_env_file_is_committed(self):
        """A real .env must never exist in the repository."""
        assert not (REPO_ROOT / ".env").exists(), (
            "A .env file exists in the repository root. It is git-ignored, but it "
            "must not be created by project tooling."
        )


class TestProjectPaths:
    """Path resolution must be robust and independent of the working directory."""

    def test_repo_root_contains_its_markers(self):
        assert (REPO_ROOT / "pyproject.toml").is_file()
        assert (REPO_ROOT / "constraints.txt").is_file()

    def test_all_paths_are_absolute(self):
        paths = get_paths()
        for field_name in paths.__slots__:
            value = getattr(paths, field_name)
            assert value.is_absolute(), f"{field_name} is not absolute: {value}"

    def test_paths_are_anchored_under_the_root(self):
        paths = get_paths()
        # artifacts and data are redirectable via environment variables, so they
        # are legitimately allowed to live outside the repository.
        for field_name in ("src", "ml", "configs", "scripts", "backend", "docs", "tests"):
            value = getattr(paths, field_name)
            assert value.is_relative_to(paths.root), f"{field_name} escaped the root"

    def test_resolution_is_independent_of_working_directory(self):
        """Resolve the root from a subprocess started in a different cwd."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from qa_ml.paths import find_repo_root; print(find_repo_root())",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT / "ml" / "configs",
            timeout=180,
            check=False,
            env={**_subprocess_env()},
        )
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()) == REPO_ROOT

    def test_run_dir_and_report_dir_are_derived_not_created(self):
        paths = get_paths()
        run_dir = paths.run_dir("some-run-id")
        assert run_dir.parent.name == "runs"
        assert not run_dir.exists()
        assert paths.report_dir("some-run-id").parent == paths.reports


def _subprocess_env() -> dict[str, str]:
    """Environment for subprocesses, with ``src`` on ``PYTHONPATH``."""
    import os

    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


class TestGitIgnore:
    """The ignore rules must actually cover the dangerous paths."""

    @pytest.mark.parametrize(
        "candidate",
        [
            "artifacts/runs/x/checkpoint/model.safetensors",
            "models/final/pytorch_model.bin",
            "models/final/model.pt",
            "data/squad/train.arrow",
            ".env",
            ".env.local",
            "frontend/node_modules/react/index.js",
            "frontend/.next/build-manifest.json",
            # create-next-app's template ignores `.env*`; a real local env file
            # must stay ignored even though .env.example is negated.
            "frontend/.env.local",
            "frontend/.env.production",
            "src/qa_core/__pycache__/normalize.cpython-312.pyc",
            ".venv/pyvenv.cfg",
            ".pytest_cache/CACHEDIR.TAG",
            ".ruff_cache/content",
            "wandb/run-x/files/output.log",
        ],
    )
    def test_dangerous_path_is_ignored(self, candidate):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", candidate],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (
            f"{candidate!r} is NOT git-ignored. Committing it would leak weights, "
            "caches or secrets into version control."
        )

    @pytest.mark.parametrize(
        "candidate",
        [
            "README.md",
            "pyproject.toml",
            "constraints.txt",
            ".env.example",
            "ml/configs/base.yaml",
            "ml/requirements-cpu.txt",
            "ml/scripts/check_environment.py",
            "src/qa_core/normalize.py",
            "backend/app/main.py",
            "backend/requirements.txt",
            "tests/test_metrics.py",
            "docs/lightning-workflow.md",
            "requirements.lock.txt",
            "frontend/package.json",
            "frontend/app/page.tsx",
            "frontend/lib/config.ts",
            # Holds variable NAMES only. create-next-app's blanket `.env*` rule
            # would hide it, so frontend/.gitignore negates it explicitly.
            "frontend/.env.example",
        ],
    )
    def test_source_and_config_are_not_ignored(self, candidate):
        """An over-broad rule that swallowed source or config would be worse."""
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", candidate],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 1, (
            f"{candidate!r} IS git-ignored but must be version-controlled."
        )


class TestPinnedDependencyVersions:
    """The installed environment must match ``constraints.txt`` exactly."""

    @staticmethod
    def _parse_constraints() -> dict[str, str]:
        """Parse ``name==version`` lines from constraints.txt."""
        text = (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8")
        pinned: dict[str, str] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9._-]+)==([A-Za-z0-9._+!-]+)", stripped)
            if match:
                pinned[match.group(1)] = match.group(2)
        return pinned

    def test_constraints_file_is_fully_pinned(self):
        """Every requirement line must use '==', never a range."""
        text = (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8")
        offenders = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and not line.strip().startswith("#")
            and "==" not in line
        ]
        assert not offenders, f"Unpinned constraint line(s): {offenders}"

    def test_constraints_are_parseable_and_non_empty(self):
        pinned = self._parse_constraints()
        assert pinned, "No pins were parsed from constraints.txt."
        for expected in ("torch", "transformers", "tokenizers", "datasets"):
            assert expected in pinned, f"{expected} is not pinned."

    def test_installed_versions_match_the_pins(self):
        """Guards against local/Lightning drift.

        A tokenizer version mismatch between environments would silently change
        offset mappings, so locally verified alignment tests would no longer say
        anything about the trained model.
        """
        mismatches = []
        for name, expected in self._parse_constraints().items():
            try:
                actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # Not every pin is installed in every environment (for example
                # the backend-only env has no torch). Absence is not drift.
                continue
            if actual != expected:
                mismatches.append(f"{name}: pinned {expected}, installed {actual}")
        assert not mismatches, "Installed versions drifted from constraints.txt:\n" + "\n".join(
            mismatches
        )

    @pytest.mark.parametrize(
        "requirements_file",
        [
            "requirements-dev.txt",
            "ml/requirements.txt",
            "ml/requirements-cpu.txt",
            "ml/requirements-gpu.txt",
            "backend/requirements.txt",
        ],
    )
    def test_every_requirements_file_references_constraints(self, requirements_file):
        """Otherwise two environments could resolve different shared versions."""
        text = (REPO_ROOT / requirements_file).read_text(encoding="utf-8")
        assert "-c " in text and "constraints.txt" in text, (
            f"{requirements_file} does not reference constraints.txt with '-c'."
        )
