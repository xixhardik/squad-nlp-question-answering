"""Tests for run directories and experiment records."""

from __future__ import annotations

import json

import pytest

from qa_ml.config import load_experiment_config
from qa_ml.experiment import (
    METRICS_FILENAME,
    RECORD_FILENAME,
    ExperimentExistsError,
    ExperimentRecord,
    create_run_directory,
    resolve_run_root,
    utc_timestamp,
    write_json,
)


@pytest.fixture
def config():
    return load_experiment_config("smoke.yaml")


class TestUtcTimestamp:
    def test_is_filesystem_safe(self):
        stamp = utc_timestamp()
        for character in '/\\:*?"<>| ':
            assert character not in stamp

    def test_has_the_expected_shape(self):
        stamp = utc_timestamp()
        assert len(stamp) == 16
        assert stamp.endswith("Z")
        assert stamp[8] == "T"


class TestCreateRunDirectory:
    def test_creates_the_directory(self, tmp_path):
        run_dir = create_run_directory(tmp_path / "runs", "run-001")
        assert run_dir.is_dir()
        assert run_dir.name == "run-001"

    def test_creates_missing_parents(self, tmp_path):
        run_dir = create_run_directory(tmp_path / "deep" / "nested" / "runs", "run-001")
        assert run_dir.is_dir()

    def test_refuses_to_overwrite_an_existing_run(self, tmp_path):
        """The core safety property: a completed experiment is never destroyed."""
        root = tmp_path / "runs"
        create_run_directory(root, "run-001")
        with pytest.raises(ExperimentExistsError, match="already exists"):
            create_run_directory(root, "run-001")

    def test_error_message_explains_the_options(self, tmp_path):
        root = tmp_path / "runs"
        create_run_directory(root, "run-001")
        with pytest.raises(ExperimentExistsError) as excinfo:
            create_run_directory(root, "run-001")
        message = str(excinfo.value)
        assert "--resume" in message
        assert "allow_existing" in message

    def test_allow_existing_permits_reuse(self, tmp_path):
        root = tmp_path / "runs"
        first = create_run_directory(root, "run-001")
        second = create_run_directory(root, "run-001", allow_existing=True)
        assert first == second

    def test_distinct_run_ids_coexist(self, tmp_path):
        root = tmp_path / "runs"
        create_run_directory(root, "run-001")
        create_run_directory(root, "run-002")
        assert {path.name for path in root.iterdir()} == {"run-001", "run-002"}


class TestResolveRunRoot:
    def test_defaults_under_artifacts(self, config):
        root = resolve_run_root(config)
        assert root.name == "runs"
        assert root.parent.name == "artifacts"

    def test_explicit_root_is_honoured(self, config, tmp_path):
        overridden = load_experiment_config(
            "smoke.yaml", overrides={"output": {"root": str(tmp_path / "custom")}}
        )
        assert resolve_run_root(overridden) == (tmp_path / "custom").resolve()


class TestWriteJson:
    def test_writes_and_creates_parents(self, tmp_path):
        path = write_json(tmp_path / "a" / "b" / "data.json", {"x": 1})
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}

    def test_non_serializable_values_do_not_lose_the_record(self, tmp_path):
        """A record must survive an unexpected type rather than raising."""
        path = write_json(tmp_path / "data.json", {"path": tmp_path})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload["path"], str)


class TestExperimentRecord:
    def test_create_captures_the_config_and_hash(self, config):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        assert record.run_id == "run-001"
        assert record.experiment_name == config.name
        assert record.config_hash == config.config_hash()
        assert record.config["model_name"] == config.model_name

    def test_starts_in_running_status(self, config):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        assert record.status == "running"
        assert record.finished_at is None

    def test_mark_completed(self, config):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.mark_completed()
        assert record.status == "completed"
        assert record.finished_at is not None

    def test_mark_failed_records_the_reason(self, config):
        """A failed experiment is still a result and must leave evidence."""
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.mark_failed(ValueError("something broke"))
        assert record.status == "failed"
        assert record.error == {"type": "ValueError", "message": "something broke"}
        assert "error" in record.as_dict()

    def test_reproducible_is_false_without_git_metadata(self, config):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        assert record.is_reproducible is False

    def test_reproducible_reflects_a_clean_tree(self, config):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.environment = {"git": {"available": True, "dirty": False}}
        assert record.is_reproducible is True

    def test_reproducible_is_false_for_a_dirty_tree(self, config):
        """A dirty tree means the recorded commit does not describe what ran."""
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.environment = {"git": {"available": True, "dirty": True}}
        assert record.is_reproducible is False

    def test_save_writes_record_and_metrics(self, config, tmp_path):
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.evaluation = {"exact_match": 12.5, "f1": 20.0, "total_examples": 8}
        record.model = {"model_name": "distilbert-base-uncased", "num_parameters": 66_000_000}
        record.precision = {"resolved": "fp32"}
        record.mark_completed()

        record.save(tmp_path)

        full = json.loads((tmp_path / RECORD_FILENAME).read_text(encoding="utf-8"))
        assert full["run_id"] == "run-001"
        assert full["status"] == "completed"
        assert full["config"]["model_name"] == config.model_name

        metrics = json.loads((tmp_path / METRICS_FILENAME).read_text(encoding="utf-8"))
        assert metrics["evaluation"]["exact_match"] == 12.5
        assert metrics["num_parameters"] == 66_000_000
        assert metrics["precision"] == "fp32"

    def test_record_contains_every_required_provenance_field(self, config, tmp_path):
        """The fields Phase 2 requires every experiment to record."""
        record = ExperimentRecord.create(config, "run-001", include_environment=False)
        record.environment = {"git": {"available": True, "dirty": False, "commit": "abc"}}
        record.seeding = {"seed": 42}
        payload = record.as_dict()
        for key in (
            "run_id",
            "started_at",
            "config",
            "config_hash",
            "environment",
            "seeding",
            "precision",
            "model",
            "dataset",
            "preprocessing",
            "training",
            "evaluation",
            "checkpoint_path",
            "is_reproducible",
        ):
            assert key in payload, f"missing required record field: {key}"
