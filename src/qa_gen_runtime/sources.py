r"""Getting records out of a corpus, explicitly, with no download nobody asked for.

Why this is a separate module from the adapters
----------------------------------------------
:mod:`qa_gen.adapters` transforms one already-loaded record into one canonical example and does
no I/O at all -- a test enforces that it cannot even reference ``Path`` or ``open``. Something
still has to fetch the records, and that something needs ``datasets``, the filesystem and
possibly the network. This is that something, and keeping it here is what lets the whole mapping
layer stay testable on a laptop with no corpus present.

Nothing downloads unless asked
------------------------------
``allow_download`` defaults to ``False`` everywhere, and it is not advisory. When it is off,
:func:`load_source_records` sets the Hugging Face offline switches for the duration of the call,
so ``datasets`` resolves from the local cache or fails. A corpus that is not already cached
produces a :class:`SourceLoadError` naming the flag that would fetch it, rather than quietly
pulling several gigabytes because a report was requested.

That distinction matters more than it sounds. A sizing command is the kind of thing someone runs
to *decide* whether to commit to a corpus, and it should not commit them to it as a side effect.

No credentials, ever
--------------------
This module reads no token, accepts no token argument, writes no credential file and sends no
authentication of its own. It also does not clear whatever ambient Hugging Face configuration a
machine already has -- doing so would break a legitimately configured Studio for no security
gain. The rule is narrower and checkable: nothing here introduces a credential. Gated corpora
are therefore out of scope, which is stated in each source's notes rather than discovered as a
401 halfway through a run.

Local files are first-class, not a fallback
-------------------------------------------
Two of the four supported corpora have no canonical Hub mirror. LearningQ is distributed by its
authors, and ``edu-mcq`` is a shape rather than a dataset -- any MCQ bank matching the field map.
For both, a local JSON or JSONL path is the *only* route, so :func:`load_source_records` reads
those with the standard library and never involves ``datasets`` at all.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qa_gen.adapters import AdapterError, adapter_for, registered_sources
from qa_gen.examples import QuestionGenerationExample
from qa_gen.preparation import SourceIngestion
from qa_gen_runtime.deps import require_datasets

logger = logging.getLogger(__name__)

__all__ = [
    "SOURCE_CATALOGUE",
    "LoadedSource",
    "SourceLoadError",
    "SourceRequest",
    "adapt_source",
    "catalogue_entry",
    "describe_catalogue",
    "describe_requirements",
    "load_source_records",
    "request_is_readable",
    "resolve_requests",
]

#: Environment switches that make ``datasets`` and ``huggingface_hub`` refuse to reach the
#: network. Set only for the duration of a load, and only when ``allow_download`` is off.
_OFFLINE_SWITCHES = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")

#: Suffixes :func:`load_source_records` reads with the standard library.
_LOCAL_SUFFIXES = (".json", ".jsonl", ".ndjson")

#: Rejection messages kept per source, for diagnosis. Counts stay exact.
_REJECTION_SAMPLE_LIMIT = 5


class SourceLoadError(RuntimeError):
    """Raised when a corpus cannot be read.

    Carries what to do about it. Every path that raises this knows whether the cause was a
    missing flag, a missing file, a missing dependency or an unrecognised layout, and the
    message says which -- a bare "dataset not found" three frames deep costs an hour.
    """


@dataclass(frozen=True, slots=True)
class SourceRequest:
    """One corpus to read, and exactly where from.

    Attributes:
        source_id: The adapter source id, e.g. ``"squad-qg"``.
        dataset_id: Hub repository id. Ignored when :attr:`local_path` is set.
        split: Upstream split to read, e.g. ``"train"``.
        revision: Pinned dataset revision. ``"main"`` is fine for exploration; pin a commit
            before any figure that gets reported, for the same reason model revisions are
            pinned.
        config_name: Upstream configuration name, when the dataset has several.
        local_path: A JSON or JSONL file to read instead of the Hub. The only route for
            corpora with no Hub mirror.
        limit: Read at most this many records. Applied at read time, so it also bounds how
            much of a cached corpus is deserialized -- distinct from the dataset config's
            caps, which select from what was read.
    """

    source_id: str
    dataset_id: str = ""
    split: str = "train"
    revision: str = "main"
    config_name: str | None = None
    local_path: str | None = None
    limit: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "revision": self.revision,
            "config_name": self.config_name,
            "local_path": self.local_path,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """What is known about one supported corpus before anything is read.

    Attributes:
        source_id: The adapter source id.
        dataset_id: Default Hub repository, or a description when there is none.
        default_split: Split read unless overridden.
        default_config_name: Upstream configuration read unless overridden, for a repository
            that publishes several. ``None`` for a repository with a single default
            configuration. Recorded here rather than left to the caller because which
            configuration this project uses is a property of how the corpus was assessed --
            the counts and licence in :attr:`notes` describe one configuration, not all of
            them -- and because a repository with several and no default cannot be loaded at
            all without one.
        hub_available: Whether the corpus can be fetched from the Hub at all.
        requires_local_path: Whether a local file is the only route.
        gated: Whether access is restricted. ``True`` means this module cannot read it,
            because it introduces no credentials.
        notes: Anything a reader should know before choosing this corpus.
    """

    source_id: str
    dataset_id: str
    default_split: str = "train"
    default_config_name: str | None = None
    hub_available: bool = True
    requires_local_path: bool = False
    gated: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "notes", tuple(self.notes))

    def request(self, **overrides: Any) -> SourceRequest:
        """Build a :class:`SourceRequest` from this entry's defaults.

        Args:
            **overrides: Fields to override.

        Returns:
            The request.
        """
        base: dict[str, Any] = {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id if self.hub_available else "",
            "split": self.default_split,
            "config_name": self.default_config_name,
        }
        base.update(overrides)
        return SourceRequest(**base)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "default_split": self.default_split,
            "default_config_name": self.default_config_name,
            "hub_available": self.hub_available,
            "requires_local_path": self.requires_local_path,
            "gated": self.gated,
            "notes": list(self.notes),
        }


#: What this project knows about each supported corpus. Declarative on purpose: "can this be
#: fetched, and what will it cost me" is answerable without a network call, which is what makes
#: a dry-run plan possible.
SOURCE_CATALOGUE: dict[str, CatalogueEntry] = {
    "squad-qg": CatalogueEntry(
        source_id="squad-qg",
        dataset_id="rajpurkar/squad",
        default_split="train",
        hub_available=True,
        notes=(
            "public and ungated; roughly 87,599 train rows and 10,570 validation rows",
            "the only supported corpus carrying character offsets, so the only one whose "
            "examples can be genuinely grounded",
            "read the upstream 'validation' split for a held-out corpus rather than relying "
            "on this project's own split of 'train'",
        ),
    ),
    "lmqg-squad-qag": CatalogueEntry(
        source_id="lmqg-squad-qag",
        dataset_id="lmqg/qag_squad",
        default_split="train",
        hub_available=True,
        notes=(
            "public; SQuAD grouped by paragraph, one record per paragraph with every "
            "question-answer pair drawn from it",
            "overlaps squad-qg by construction: both derive from SQuAD, so mixing them "
            "without drop_duplicates will train on the same content twice",
            "carries no offsets, so its examples are not grounded",
        ),
    ),
    "learningq-qg": CatalogueEntry(
        source_id="learningq-qg",
        dataset_id="LearningQ (Chen et al.); no canonical Hub mirror",
        default_split="train",
        hub_available=False,
        requires_local_path=True,
        notes=(
            "distributed by the authors rather than the Hub, so a local JSON or JSONL path "
            "is the only route",
            "most items have no written answer and the adapter refuses those; expect a large "
            "rejection rate, which is a property of the corpus and not a fault",
            "research-use terms must be checked before publishing anything derived from it",
        ),
    ),
    "edu-mcq": CatalogueEntry(
        source_id="edu-mcq",
        dataset_id="configurable; any MCQ corpus matching the adapter's field map",
        default_split="train",
        hub_available=False,
        requires_local_path=True,
        notes=(
            "a record shape rather than a dataset: supply a local JSON or JSONL file whose "
            "rows carry a passage, a stem, options and the correct option",
            "licensing depends entirely on the corpus supplied and must be checked per source",
        ),
    ),
    "race-mcq": CatalogueEntry(
        source_id="race-mcq",
        dataset_id="ehovy/race",
        default_split="train",
        default_config_name="all",
        hub_available=True,
        notes=(
            "public and ungated English exam questions; the 'all' configuration holds 87,866 "
            "train, 4,887 validation and 4,934 test rows, and equals 'high' (62,445 train) "
            "plus 'middle' (25,421 train)",
            "the answer is an option label such as 'A', not the option's text, so the adapter "
            "reads it with answer_style='letter'",
            "NON-COMMERCIAL research use only, and the terms forbid redistributing any "
            "portion of the passages or of data derived from them: a prepared corpus, a "
            "tokenized cache and a trained adapter all count, so none may be published",
            "about 3.3 questions share each article, so a context-grouped split keeps them "
            "together; the article filename is deliberately not used as the example id, "
            "because it repeats and deduplication would then discard two thirds of the corpus",
            "passages are long: roughly 700 Qwen3 tokens per rendered record against SQuAD's "
            "445, and a 0.9% tail exceeds a 1,024-token window until max_context_chars is "
            "lowered to 3,000",
            "carries no offsets, so its examples are not grounded",
        ),
    ),
}


def catalogue_entry(source_id: str) -> CatalogueEntry:
    """Return the catalogue entry for a source id.

    Args:
        source_id: The adapter source id.

    Returns:
        The entry.

    Raises:
        SourceLoadError: If the source is not in the catalogue. The message lists what is,
            because a typo in a config should diagnose itself.
    """
    entry = SOURCE_CATALOGUE.get(source_id)
    if entry is None:
        raise SourceLoadError(
            f"no source catalogue entry for {source_id!r}. Known sources: "
            f"{', '.join(sorted(SOURCE_CATALOGUE))}. Registered adapters: "
            f"{', '.join(registered_sources())}."
        )
    return entry


def describe_catalogue() -> dict[str, Any]:
    """Return the whole catalogue, with each adapter's declared spec beside it.

    Written into the sizing report so the artifact states what was available, not just what was
    used -- "why is there no LearningQ in this dataset?" is then answerable from the report.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        source_id: {
            **entry.as_dict(),
            "adapter_spec": adapter_for(source_id).spec.as_dict(),
        }
        for source_id, entry in sorted(SOURCE_CATALOGUE.items())
    }


def resolve_requests(
    sources: Sequence[str],
    *,
    local_paths: dict[str, str] | None = None,
    splits: dict[str, str] | None = None,
    revisions: dict[str, str] | None = None,
    limit: int | None = None,
    require_readable: bool = True,
) -> tuple[SourceRequest, ...]:
    """Turn a list of source ids into concrete read requests.

    Args:
        sources: Source ids to read. Empty means every catalogue entry that can actually be
            read without extra input -- which excludes the two needing a local path, because
            defaulting to "everything" and then failing on a missing file would be worse than
            reading what is available.
        local_paths: Source id to local JSON/JSONL path.
        splits: Source id to upstream split override.
        revisions: Source id to dataset revision override.
        limit: Per-source read cap.
        require_readable: Refuse a source that could not actually be read. ``True`` for a real
            run. ``False`` for a plan, which exists precisely to *tell* someone that a corpus
            needs a local path -- raising there would mean you had to already know the answer
            to ask the question. An unreadable request comes back with neither
            :attr:`SourceRequest.local_path` nor :attr:`SourceRequest.dataset_id` set, which is
            how :func:`request_is_readable` identifies it.

    Returns:
        One request per source, in the order given (or catalogue order when defaulted).

    Raises:
        SourceLoadError: If a named source is unknown, or -- when ``require_readable`` is set --
            needs a local path that was not supplied, or is gated.
    """
    paths = dict(local_paths or {})
    chosen: list[str] = list(sources)
    if not chosen:
        chosen = [
            source_id
            for source_id, entry in sorted(SOURCE_CATALOGUE.items())
            if entry.hub_available or source_id in paths
        ]
        logger.info(
            "no sources named; reading %s (corpora needing a local path are skipped unless "
            "one was supplied)",
            ", ".join(chosen) or "(none)",
        )

    requests: list[SourceRequest] = []
    for source_id in chosen:
        entry = catalogue_entry(source_id)
        local = paths.get(source_id)
        if entry.gated and local is None and require_readable:
            raise SourceLoadError(
                f"{source_id}: access is gated and this pipeline introduces no credentials. "
                "Supply a local export instead, with --source-path "
                f"{source_id}=/path/to/file.jsonl."
            )
        if entry.requires_local_path and local is None and require_readable:
            raise SourceLoadError(
                f"{source_id}: {entry.dataset_id}. There is no Hub mirror to read, so a local "
                f"file is required: --source-path {source_id}=/path/to/file.jsonl "
                f"(accepted suffixes: {', '.join(_LOCAL_SUFFIXES)})."
            )
        requests.append(
            entry.request(
                local_path=local,
                split=(splits or {}).get(source_id, entry.default_split),
                revision=(revisions or {}).get(source_id, "main"),
                limit=limit,
            )
        )
    return tuple(requests)


def request_is_readable(request: SourceRequest) -> bool:
    """Whether a request names somewhere records could actually come from.

    Args:
        request: The request to check.

    Returns:
        ``False`` when neither a local path nor a dataset id is set, which is what
        :func:`resolve_requests` produces under ``require_readable=False`` for a corpus that
        needs input the caller has not supplied.
    """
    return bool(request.local_path or request.dataset_id)


def describe_requirements(requests: Sequence[SourceRequest]) -> tuple[str, ...]:
    """Return one line per request saying what it still needs, if anything.

    Args:
        requests: Resolved requests.

    Returns:
        Human-readable requirement lines, for a plan report.
    """
    lines: list[str] = []
    for request in requests:
        entry = SOURCE_CATALOGUE.get(request.source_id)
        if not request_is_readable(request):
            lines.append(
                f"{request.source_id}: needs --source-path {request.source_id}=<file>"
                f" ({entry.dataset_id if entry else 'no Hub mirror'})"
            )
        elif request.local_path:
            lines.append(f"{request.source_id}: reads the local file {request.local_path}")
        else:
            configuration = (
                f" configuration {request.config_name!r}," if request.config_name else ""
            )
            lines.append(
                f"{request.source_id}: reads {request.dataset_id}{configuration} split "
                f"{request.split!r} at revision {request.revision!r}; needs --allow-download "
                "unless already cached"
            )
    return tuple(lines)


@contextlib.contextmanager
def _offline(enabled: bool) -> Iterator[None]:
    """Force the Hugging Face libraries offline for the duration of the block.

    Args:
        enabled: Whether to apply the switches. ``False`` is a no-op, so the caller reads as a
            single path rather than branching around the ``with``.

    Yields:
        ``None``.
    """
    if not enabled:
        yield
        return

    previous = {name: os.environ.get(name) for name in _OFFLINE_SWITCHES}
    for name in _OFFLINE_SWITCHES:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _read_local(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Read records from a local JSON or JSONL file.

    Args:
        path: The file. A ``.json`` file must hold a list of objects, or an object with a
            ``"data"`` or ``"records"`` list; ``.jsonl``/``.ndjson`` is one object per line.
        limit: Stop after this many records.

    Returns:
        The records.

    Raises:
        SourceLoadError: If the file is missing, has an unsupported suffix, cannot be parsed,
            or does not hold a list of objects.
    """
    if not path.is_file():
        raise SourceLoadError(f"source file not found: {path}")
    if path.suffix.lower() not in _LOCAL_SUFFIXES:
        raise SourceLoadError(
            f"unsupported source file suffix {path.suffix!r} for {path.name}. Supported: "
            f"{', '.join(_LOCAL_SUFFIXES)}."
        )

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SourceLoadError(f"could not parse JSON in {path}: {exc}") from exc
        if isinstance(decoded, dict):
            for key in ("data", "records", "examples"):
                if isinstance(decoded.get(key), list):
                    decoded = decoded[key]
                    break
        if not isinstance(decoded, list):
            raise SourceLoadError(
                f"{path} must hold a list of records, or an object with a 'data', 'records' "
                f"or 'examples' list; got {type(decoded).__name__}."
            )
        rows: Sequence[Any] = decoded
    else:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    for position, row in enumerate(rows):
        if limit is not None and len(records) >= limit:
            break
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except json.JSONDecodeError as exc:
                raise SourceLoadError(
                    f"could not parse line {position + 1} of {path}: {exc}"
                ) from exc
        if not isinstance(row, dict):
            raise SourceLoadError(
                f"{path} record {position} is {type(row).__name__}, not an object."
            )
        records.append(row)
    return records


def _read_hub(request: SourceRequest, *, allow_download: bool) -> list[dict[str, Any]]:
    """Read records from the Hugging Face Hub or its local cache.

    Args:
        request: What to read.
        allow_download: Permit a network fetch. When ``False`` the libraries are forced
            offline, so a corpus that is not cached fails instead of downloading.

    Returns:
        The records, as plain dictionaries.

    Raises:
        SourceLoadError: If the dataset cannot be loaded. The offline case says which flag
            would fetch it.
        qa_gen_runtime.deps.RuntimeDependencyError: If ``datasets`` is not installed.
    """
    datasets = require_datasets()
    arguments: dict[str, Any] = {
        "path": request.dataset_id,
        "split": request.split,
        "revision": request.revision,
    }
    if request.config_name:
        arguments["name"] = request.config_name

    logger.info(
        "loading %s split=%s revision=%s (download %s)",
        request.dataset_id,
        request.split,
        request.revision,
        "allowed" if allow_download else "refused; cache only",
    )
    try:
        with _offline(not allow_download):
            dataset = datasets.load_dataset(**arguments)
    except Exception as exc:  # noqa: BLE001 - datasets raises a wide and unstable family
        hint = (
            "Pass --allow-download to fetch it, or point --source-path at a local export."
            if not allow_download
            else "Check the dataset id, the split name, the revision and disk space."
        )
        raise SourceLoadError(
            f"could not load {request.dataset_id!r} split {request.split!r} at revision "
            f"{request.revision!r}: {type(exc).__name__}: {exc}\n"
            f"Downloads were {'refused' if not allow_download else 'allowed'}. {hint}"
        ) from exc

    if request.limit is not None:
        dataset = dataset.select(range(min(request.limit, len(dataset))))
    return [dict(row) for row in dataset]


def load_source_records(
    request: SourceRequest, *, allow_download: bool = False
) -> list[dict[str, Any]]:
    """Read the raw records for one corpus.

    Args:
        request: What to read.
        allow_download: Permit a network fetch. Defaults to ``False``; see the module
            docstring for why that is the default rather than a convenience.

    Returns:
        The records, in upstream order.

    Raises:
        SourceLoadError: If the corpus cannot be read.
        qa_gen_runtime.deps.RuntimeDependencyError: If a Hub read is needed and ``datasets``
            is not installed.
    """
    if request.local_path:
        return _read_local(Path(request.local_path).expanduser(), request.limit)
    if not request.dataset_id:
        raise SourceLoadError(
            f"{request.source_id}: neither a local path nor a dataset id was given, so there "
            "is nothing to read."
        )
    return _read_hub(request, allow_download=allow_download)


@dataclass(frozen=True, slots=True)
class LoadedSource:
    """One corpus, read and mapped into canonical examples.

    Attributes:
        request: What was read.
        examples: The canonical examples, in record order.
        ingestion: Counts and rejection samples, ready for the preparation report.
        rejection_messages: A bounded sample of adapter refusals.
    """

    request: SourceRequest
    examples: tuple[QuestionGenerationExample, ...] = ()
    ingestion: SourceIngestion = field(
        default_factory=lambda: SourceIngestion(source_id="unknown")
    )
    rejection_messages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Coerce the sequence fields to tuples."""
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "rejection_messages", tuple(self.rejection_messages))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary. The examples are not inlined."""
        return {
            "request": self.request.as_dict(),
            "example_count": len(self.examples),
            "ingestion": self.ingestion.as_dict(),
        }


def adapt_source(
    request: SourceRequest,
    records: Sequence[dict[str, Any]],
    *,
    skip_invalid: bool = True,
) -> LoadedSource:
    """Map one corpus's records through its adapter, counting what it refused.

    ``qa_gen.adapters.adapt_records`` drops refusals silently when ``skip_invalid`` is on and
    reports nothing about them, so the loop is written out here instead: a corpus whose adapter
    rejected 80% of its records is the single most important thing a preparation report can
    say, and it cannot be recovered afterwards from the counts alone.

    Args:
        request: The corpus that was read, for provenance.
        records: The raw records.
        skip_invalid: Count and drop refusals rather than propagating them. On by default
            because LearningQ is *expected* to refuse most of its records. Turn it off to make
            the first schema mismatch fatal, which is what you want on a corpus you believe is
            clean.

    Returns:
        The :class:`LoadedSource`.

    Raises:
        qa_gen.adapters.AdapterError: If a record is refused and ``skip_invalid`` is ``False``.
        qa_gen.adapters.UnknownAdapterError: If no adapter is registered for the source.
    """
    adapter = adapter_for(request.source_id)
    examples: list[QuestionGenerationExample] = []
    messages: list[str] = []
    rejected = 0

    for record in records:
        try:
            examples.append(adapter.adapt(record))
        except AdapterError as exc:
            if not skip_invalid:
                raise
            rejected += 1
            if len(messages) < _REJECTION_SAMPLE_LIMIT:
                messages.append(str(exc))

    dataset_id = request.local_path or request.dataset_id
    ingestion = SourceIngestion(
        source_id=request.source_id,
        dataset_id=dataset_id,
        records_seen=len(records),
        examples_adapted=len(examples),
        rejected=rejected,
        rejection_samples=tuple(messages),
        notes=(
            f"split={request.split} revision={request.revision}"
            if not request.local_path
            else "read from a local file",
        ),
    )
    if rejected:
        logger.warning(
            "%s: adapter refused %d of %d records (%.1f%%)",
            request.source_id,
            rejected,
            len(records),
            100.0 * ingestion.rejection_rate,
        )
    return LoadedSource(
        request=request,
        examples=tuple(examples),
        ingestion=ingestion,
        rejection_messages=tuple(messages),
    )
