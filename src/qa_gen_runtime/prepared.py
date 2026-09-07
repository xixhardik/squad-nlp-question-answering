r"""Reading and writing a prepared dataset on disk, in one place.

Why the format needs an owner
-----------------------------
Phase 17C's ``prepare`` command writes ``train.jsonl``, ``validation.jsonl`` and
``test.jsonl`` beside a ``dataset.json``. Everything downstream -- the benchmark, and later a
real training run and an evaluation pass -- has to read exactly what was written. A format with
a writer in one module and an ad-hoc reader in each consumer drifts, and the failure mode is
silent: a consumer that mis-parses a line drops an example and reports a slightly smaller
corpus that nobody notices.

So the filenames, the one-JSON-object-per-line convention and both directions live here.
:mod:`qa_gen_runtime.prepare` writes through :func:`write_prepared_split`, and every consumer
reads through :func:`read_prepared_split`.

Finding a prepared dataset
--------------------------
A prepared dataset is named ``<experiment>-<fingerprint>``, so several can coexist and the
fingerprint says which corpus each one is. :func:`resolve_prepared_directory` picks one by
explicit path, by fingerprint, or -- failing both -- the most recently written, and its error
message lists what is actually on disk. That last part matters more than it sounds: the usual
mistake is pointing a benchmark at a directory that was never written, and "no such file" three
frames deep does not say which directories do exist.

JSONL rather than a single JSON array, deliberately
---------------------------------------------------
78,552 training examples in one array must be parsed entirely before the first can be used, and
a truncated write leaves an unparseable file. One object per line streams, and a truncated write
loses only the final line -- which :func:`read_prepared_split` reports rather than absorbing,
because a corpus that is quietly one example short is a corpus whose fingerprint no longer
matches its metadata.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_gen.examples import QuestionGenerationExample
from qa_gen.serialization import example_from_dict
from qa_gen.splitting import SplitName, compute_dataset_fingerprint
from qa_paper.serialization import SerializationError

logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_DOCUMENT",
    "PreparedDatasetError",
    "PreparedSplitInfo",
    "discover_prepared_datasets",
    "read_prepared_split",
    "resolve_prepared_directory",
    "split_filename",
    "verify_fingerprint",
    "write_prepared_split",
]

#: The metadata document ``prepare`` writes beside the splits.
DATASET_DOCUMENT = "dataset.json"

#: Report at most this many malformed lines before giving up on a file.
_MAX_REPORTED_LINES = 5


class PreparedDatasetError(RuntimeError):
    """Raised when a prepared dataset cannot be found, read or trusted.

    Carries what to do about it. Every path that raises this knows whether the cause was a
    missing directory, a missing split, a malformed line or an unreadable example, and says so.
    """


def split_filename(split: SplitName | str) -> str:
    """Return the filename holding one split.

    Args:
        split: The split, as a :class:`~qa_gen.splitting.SplitName` or its string value.

    Returns:
        ``"train.jsonl"`` and so on.

    Raises:
        PreparedDatasetError: If the name is not one of the three splits.
    """
    try:
        name = SplitName(split)
    except ValueError as exc:
        valid = [member.value for member in SplitName]
        raise PreparedDatasetError(
            f"unknown split {split!r}; expected one of {valid}."
        ) from exc
    return f"{name.value}.jsonl"


def write_prepared_split(
    directory: Path, split: SplitName | str, examples: Sequence[QuestionGenerationExample]
) -> Path:
    """Write one split as JSON Lines.

    Args:
        directory: The prepared dataset directory. Created if absent.
        split: Which split.
        examples: The examples, in the order they should be read back.

    Returns:
        The path written.

    Raises:
        PreparedDatasetError: If the split name is unknown.
    """
    path = directory / split_filename(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(example.as_dict(), ensure_ascii=False, default=str) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )
    return path


def read_prepared_split(
    directory: Path | str, split: SplitName | str = SplitName.TRAIN
) -> tuple[QuestionGenerationExample, ...]:
    """Read one split back into canonical examples.

    Args:
        directory: The prepared dataset directory.
        split: Which split to read.

    Returns:
        The examples, in file order. File order is the order
        :func:`qa_gen.preparation.prepare_dataset` produced, which is the ``shuffle_seed``
        ordering -- so a consumer that reads a prefix gets a source-interleaved sample rather
        than one corpus at a time.

    Raises:
        PreparedDatasetError: If the directory or file is missing, a line is not JSON, a line
            is not an object, an example cannot be reconstructed, or the file is empty.
    """
    root = Path(directory).expanduser()
    path = root / split_filename(split)
    if not root.is_dir():
        raise PreparedDatasetError(
            f"prepared dataset directory not found: {root}\n"
            "Run python -m qa_gen_runtime.prepare --config <config> --allow-download "
            "--write-dataset first; without --write-dataset only the report is written."
        )
    if not path.is_file():
        present = sorted(item.name for item in root.iterdir() if item.is_file())
        raise PreparedDatasetError(
            f"split file not found: {path}\n"
            f"Files present in {root}: {', '.join(present) or '(none)'}.\n"
            "A dataset prepared without --write-dataset holds only sizing.json."
        )

    examples: list[QuestionGenerationExample] = []
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: not JSON ({exc})")
            continue
        if not isinstance(decoded, dict):
            problems.append(f"line {number}: {type(decoded).__name__}, not an object")
            continue
        try:
            examples.append(example_from_dict(decoded))
        except SerializationError as exc:
            problems.append(f"line {number}: {exc}")

    if problems:
        shown = problems[:_MAX_REPORTED_LINES]
        more = f" (and {len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
        raise PreparedDatasetError(
            f"{len(problems)} line(s) in {path} could not be read{more}:\n"
            + "\n".join(f"  - {problem}" for problem in shown)
            + "\nThe file was written by qa_gen_runtime.prepare; a mismatch means it was "
            "edited, truncated, or written by a different version of the schema."
        )
    if not examples:
        raise PreparedDatasetError(
            f"{path} holds no examples. An empty split cannot be benchmarked or trained on."
        )
    return tuple(examples)


@dataclass(frozen=True, slots=True)
class PreparedSplitInfo:
    """What is knowable about a prepared dataset without reading its examples.

    Attributes:
        directory: Where it lives.
        experiment: The experiment name, from the directory stem.
        fingerprint: The corpus fingerprint, from the directory stem.
        splits: Which split files are present, and how many bytes each holds.
        recorded_fingerprint: The fingerprint in ``dataset.json``, when readable. A mismatch
            against :attr:`fingerprint` means the directory was renamed.
        metadata: The parsed ``dataset.json``, when readable.
    """

    directory: Path
    experiment: str = ""
    fingerprint: str = ""
    splits: dict[str, int] = field(default_factory=dict)
    recorded_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_train(self) -> bool:
        """Whether a training split is present."""
        return SplitName.TRAIN.value in self.splits

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation, without the metadata document."""
        return {
            "directory": self.directory.as_posix(),
            "experiment": self.experiment,
            "fingerprint": self.fingerprint,
            "recorded_fingerprint": self.recorded_fingerprint,
            "splits": dict(sorted(self.splits.items())),
            "has_train": self.has_train,
        }


def _inspect(directory: Path) -> PreparedSplitInfo:
    """Describe one prepared dataset directory without reading its examples."""
    experiment, _, fingerprint = directory.name.rpartition("-")
    splits: dict[str, int] = {}
    for name in SplitName:
        path = directory / split_filename(name)
        if path.is_file():
            splits[name.value] = path.stat().st_size

    metadata: dict[str, Any] = {}
    recorded: str | None = None
    document = directory / DATASET_DOCUMENT
    if document.is_file():
        try:
            loaded = json.loads(document.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("could not parse %s; continuing without it", document)
        else:
            if isinstance(loaded, dict):
                metadata = loaded
                value = loaded.get("fingerprint")
                recorded = value if isinstance(value, str) else None

    return PreparedSplitInfo(
        directory=directory,
        experiment=experiment or directory.name,
        fingerprint=fingerprint,
        splits=splits,
        recorded_fingerprint=recorded,
        metadata=metadata,
    )


def discover_prepared_datasets(root: Path | str) -> tuple[PreparedSplitInfo, ...]:
    """Find every prepared dataset under a root, newest first.

    Args:
        root: The directory holding prepared datasets, normally
            ``<artifacts>/qgen-datasets``.

    Returns:
        One entry per subdirectory that holds at least one split file, ordered by modification
        time descending so the first is the most recent. Empty when the root does not exist,
        which is not an error -- it just means nothing has been prepared yet.
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        return ()
    found = [
        _inspect(item)
        for item in base.iterdir()
        if item.is_dir() and any((item / split_filename(n)).is_file() for n in SplitName)
    ]
    return tuple(
        sorted(found, key=lambda info: info.directory.stat().st_mtime, reverse=True)
    )


def resolve_prepared_directory(
    root: Path | str,
    *,
    directory: Path | str | None = None,
    fingerprint: str | None = None,
) -> PreparedSplitInfo:
    """Pick which prepared dataset to use.

    Args:
        root: Where prepared datasets live.
        directory: An explicit directory, which wins over everything else.
        fingerprint: Select by corpus fingerprint. Use this in anything whose numbers get
            reported: "the newest" is convenient and not reproducible, because it changes the
            moment somebody prepares another dataset.

    Returns:
        The chosen :class:`PreparedSplitInfo`.

    Raises:
        PreparedDatasetError: If the explicit directory holds no splits, the fingerprint
            matches nothing, or nothing has been prepared at all. The message lists what is
            available.
    """
    if directory is not None:
        chosen = Path(directory).expanduser()
        if not chosen.is_dir():
            raise PreparedDatasetError(f"prepared dataset directory not found: {chosen}")
        info = _inspect(chosen)
        if not info.splits:
            raise PreparedDatasetError(
                f"{chosen} holds no split files. Expected at least one of "
                f"{', '.join(split_filename(n) for n in SplitName)}; a dataset prepared "
                "without --write-dataset holds only sizing.json."
            )
        return info

    available = discover_prepared_datasets(root)
    if not available:
        raise PreparedDatasetError(
            f"no prepared dataset found under {Path(root).expanduser()}\n"
            "Run python -m qa_gen_runtime.prepare --config <config> --allow-download "
            "--write-dataset first."
        )

    if fingerprint is not None:
        matches = [info for info in available if info.fingerprint == fingerprint]
        if not matches:
            listed = ", ".join(
                f"{info.experiment}-{info.fingerprint}" for info in available
            )
            raise PreparedDatasetError(
                f"no prepared dataset with fingerprint {fingerprint!r}. Available: {listed}."
            )
        return matches[0]

    newest = available[0]
    if len(available) > 1:
        logger.warning(
            "%d prepared datasets found; using the most recent (%s). Pass --dataset-fingerprint "
            "to pin one, because 'newest' is not reproducible.",
            len(available),
            newest.directory.name,
        )
    return newest


def verify_fingerprint(
    examples: Iterable[QuestionGenerationExample], expected: str | None
) -> dict[str, Any]:
    """Recompute a corpus fingerprint and compare it with the recorded one.

    Args:
        examples: The examples that were read.
        expected: The fingerprint recorded in ``dataset.json``, or ``None``.

    Returns:
        A mapping stating both values and whether they agree. Reported rather than raised: a
        consumer reading a single split cannot match a whole-corpus fingerprint, so a
        disagreement is informative rather than fatal.
    """
    actual = compute_dataset_fingerprint(examples)
    return {
        "recorded": expected,
        "recomputed_for_this_split": actual,
        "matches_whole_corpus": expected == actual if expected else None,
        "note": (
            "the recorded fingerprint covers the whole prepared corpus; this one covers only "
            "the split that was read, so they agree only when the split is the corpus"
        ),
    }
