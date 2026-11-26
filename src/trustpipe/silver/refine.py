"""Silver refinement: turn one bronze part into documents, rejects and trust rows.

Every capture of the part ends up in exactly one of two places:

- it decodes to non-empty text: one silver.documents row (canonical text and script) and
  one silver.trust row (what the policy decides about this capture), in the same order;
- it does not: one silver.rejects row with a reason code, and the part carries on.

So rows(bronze) == rows(documents) + rows(rejects) for every part, and one bad row never
fails a whole part.

Plain text (text/plain, text/markdown) is decoded as it is; HTML pages are decoded and
reduced to their main text (html.py); any other media type, such as images or PDF, is
rejected as unsupported. Decoding (decoding.py) and the script guess (script.py) live in
their own modules.

Two entry points share one per-capture function, so they cannot disagree:

- ``refine`` takes records and returns records (single captures, tests);
- ``refine_part`` is the bulk path. It checks the bronze file against the SHA-256 in its
  receipt once, instead of re-hashing every payload and id while reading rows, and builds
  the silver tables column by column. That removes per-row validation that the write path
  already did, and keeps the cost per capture to decoding, hashing the text once, the
  script guess and the trust decision.

A policy change does not touch the text: ``refine_all`` re-judges parts that were refined
with today's text transform (``rejudge_part``), writing only a new trust part, and the
receipt it commits is byte for byte the one a full refine would commit.

The silver files are written first and the receipt last (lake/receipts.py). The policy that
decided the trust rows is stored under _policies/, so every row can be explained later.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc

from trustpipe import __version__
from trustpipe.bronze.ingest import STAGE as BRONZE_STAGE
from trustpipe.identity.ids import CANONICAL_FORM, canonical_text, sha256
from trustpipe.identity.uris import SourceUri, parse_uri, split_canonical
from trustpipe.lake.layout import (
    parse_bronze_captures_key,
    policy_key,
    silver_documents_key,
    silver_rejects_key,
    silver_trust_key,
)
from trustpipe.lake.receipts import (
    IntegrityError,
    Receipt,
    commit,
    committed_parts,
    is_committed,
    load_receipt,
    verify_file,
)
from trustpipe.lake.schema import (
    BRONZE_CAPTURES,
    SILVER_DOCUMENTS,
    SILVER_REJECTS,
    SILVER_TRUST,
    Acquisition,
    Document,
    RawCapture,
    Reject,
    TrustRecord,
)
from trustpipe.lake.storage import file_sha256, lake_path, read_table, write_bytes, write_table
from trustpipe.runtime.executor import PartReport, RunSummary, TaskEvent, run_stage
from trustpipe.silver.decoding import UnsupportedCharsetError, decode_bytes, media_type
from trustpipe.silver.html import HTML_TYPES, PLAIN_TYPES, extractor, main_text, meta_charset
from trustpipe.silver.script import detect_script
from trustpipe.trust.registry import Decision, Registry

STAGE = "silver.refine"
_BRONZE_COLUMNS = ["capture_id", "source_uri", "host", "fetched_at", "acquisition"]
_BRONZE_COLUMNS += ["content_type", "raw"]
_BRONZE_META = ["capture_id", "source_uri", "host", "fetched_at", "acquisition"]


class BronzeIntegrityError(ValueError):
    """A bronze file is not committed, or its bytes differ from its receipt."""


@dataclass(frozen=True, slots=True)
class Refined:
    documents: list[Document] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)
    trust: list[TrustRecord] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Text:
    text: str  # canonical
    doc_id: bytes
    script: str | None


@dataclass(frozen=True, slots=True)
class _Failure:
    reason: str
    detail: str


def _text_or_reason(raw: bytes, content_type: str | None) -> _Text | _Failure:
    media = media_type(content_type)
    if media not in PLAIN_TYPES and media not in HTML_TYPES:
        return _Failure("unsupported_media_type", media)
    try:
        if media in PLAIN_TYPES:
            decoded: str | None = decode_bytes(raw, content_type)
        else:
            html = decode_bytes(raw, content_type, fallback=meta_charset(raw) or "utf-8")
            decoded = main_text(html)
    except UnsupportedCharsetError as error:
        return _Failure("unsupported_charset", str(error))
    except UnicodeDecodeError as error:
        return _Failure("undecodable", f"{error.encoding}: {error.reason} at byte {error.start}")
    if decoded is None:
        return _Failure("no_main_text", "the page has no main text to extract")
    text = canonical_text(decoded)
    if not text.strip():
        return _Failure("empty_text", "the payload holds no visible text")
    return _Text(text, sha256(text.encode("utf-8")), detect_script(text))


def _judge(registry: Registry, uri: SourceUri, acquisition: str) -> Decision:
    return registry.judge(uri, Acquisition(acquisition))


def refine(captures: Iterable[RawCapture], registry: Registry) -> Refined:
    refined = Refined()
    for capture in captures:
        outcome = _text_or_reason(capture.raw, capture.content_type)
        if isinstance(outcome, _Failure):
            refined.rejects.append(
                Reject(capture.capture_id, STAGE, outcome.reason, outcome.detail)
            )
            continue
        decision = _judge(registry, parse_uri(capture.source_uri), capture.acquisition)
        refined.documents.append(
            Document(capture.capture_id, outcome.doc_id, outcome.text, outcome.script)
        )
        refined.trust.append(
            TrustRecord(
                capture_id=capture.capture_id,
                doc_id=outcome.doc_id,
                host=capture.host,
                fetched_at=capture.fetched_at,
                policy_id=registry.policy_id,
                rule_id=decision.rule.uri_prefix if decision.rule else None,
                tier=decision.tier,
                tier_rank=decision.tier.rank,
                license=decision.license,
                publisher=decision.rule.publisher if decision.rule else None,
                reason=decision.reason,
            )
        )
    return refined


def _committed_bronze(root: Path, bronze_key: str) -> None:
    """Refuse a bronze file that is not committed or whose bytes differ from its receipt."""
    part = parse_bronze_captures_key(bronze_key)[1]
    if not is_committed(root, BRONZE_STAGE, part):
        raise BronzeIntegrityError(f"{bronze_key} has no {BRONZE_STAGE} receipt")
    recorded = {o.key: o.sha256 for o in load_receipt(root, BRONZE_STAGE, part).outputs}
    if recorded.get(bronze_key) != file_sha256(lake_path(root, bronze_key)):
        raise BronzeIntegrityError(f"{bronze_key} does not match the SHA-256 in its receipt")


@dataclass(frozen=True, slots=True)
class SilverTables:
    documents: pa.Table
    rejects: pa.Table
    trust: pa.Table


def refine_table(captures: pa.Table, registry: Registry) -> SilverTables:
    """Bulk refinement of bronze rows (as read from a committed part) into silver tables."""
    ids = cast(list[bytes], captures.column("capture_id").to_pylist())
    uris = cast(list[str], captures.column("source_uri").to_pylist())
    methods = cast(list[str], captures.column("acquisition").to_pylist())
    content_types = cast(list[str | None], captures.column("content_type").to_pylist())
    raws = cast(list[bytes], captures.column("raw").to_pylist())
    kept: list[int] = []
    texts: list[str] = []
    doc_ids: list[bytes] = []
    scripts: list[str | None] = []
    rejected: list[int] = []
    reasons: list[str] = []
    details: list[str] = []
    for row, (raw, content_type) in enumerate(zip(raws, content_types, strict=True)):
        outcome = _text_or_reason(raw, content_type)
        if isinstance(outcome, _Failure):
            rejected.append(row)
            reasons.append(outcome.reason)
            details.append(outcome.detail)
            continue
        kept.append(row)
        texts.append(outcome.text)
        doc_ids.append(outcome.doc_id)
        scripts.append(outcome.script)
    kept_ids = pa.array([ids[row] for row in kept], SILVER_DOCUMENTS.field("capture_id").type)
    doc_id_array = pa.array(doc_ids, SILVER_DOCUMENTS.field("doc_id").type)
    documents = pa.Table.from_arrays(
        [
            kept_ids,
            doc_id_array,
            pa.array(texts, pa.large_string()),
            pa.array(scripts, pa.string()),
        ],
        schema=SILVER_DOCUMENTS,
    )
    take = pa.array(kept, pa.int64())
    trust = _trust_table(
        registry,
        kept_ids,
        doc_id_array,
        captures.column("host").take(take),
        captures.column("fetched_at").take(take),
        [uris[row] for row in kept],
        [methods[row] for row in kept],
    )
    rejects = pa.Table.from_arrays(
        [
            pa.array([ids[row] for row in rejected], SILVER_REJECTS.field("capture_id").type),
            pa.array([STAGE] * len(rejected), pa.string()),
            pa.array(reasons, pa.string()),
            pa.array(details, pa.string()),
        ],
        schema=SILVER_REJECTS,
    )
    return SilverTables(documents, rejects, trust)


def _trust_table(
    registry: Registry,
    capture_ids: "pa.Array[Any]",
    doc_ids: "pa.Array[Any]",
    hosts: "pa.ChunkedArray[Any]",
    fetched_at: "pa.ChunkedArray[Any]",
    uris: list[str],
    methods: list[str],
) -> pa.Table:
    """The silver.trust rows of the documents of a part, in document order."""
    decisions = [
        _judge(registry, split_canonical(uri), method)
        for uri, method in zip(uris, methods, strict=True)
    ]
    return pa.Table.from_arrays(
        [
            capture_ids,
            doc_ids,
            hosts,
            fetched_at,
            pa.array([registry.policy_id] * len(decisions), pa.string()),
            pa.array([d.rule.uri_prefix if d.rule else None for d in decisions], pa.string()),
            pa.array([d.tier.value for d in decisions], pa.string()),
            pa.array([d.tier.rank for d in decisions], pa.int8()),
            pa.array([d.license for d in decisions], pa.string()),
            pa.array([d.rule.publisher if d.rule else None for d in decisions], pa.string()),
            pa.array([d.reason for d in decisions], pa.string()),
        ],
        schema=SILVER_TRUST,
    )


def refine_part(root: Path, bronze_key: str, registry: Registry) -> Receipt:
    """Refine one committed bronze part into its silver parts and commit them."""
    day, part = parse_bronze_captures_key(bronze_key)
    _committed_bronze(root, bronze_key)
    captures = read_table(root, bronze_key, schema=BRONZE_CAPTURES, columns=_BRONZE_COLUMNS)
    tables = refine_table(captures, registry)
    write_bytes(root, policy_key(registry.policy_id), registry.canonical_bytes(), create_only=True)
    outputs = (
        write_table(tables.documents, root, silver_documents_key(day, part), create_only=False),
        write_table(tables.rejects, root, silver_rejects_key(day, part), create_only=False),
        write_table(
            tables.trust,
            root,
            silver_trust_key(registry.policy_id, day, part),
            create_only=False,
        ),
    )
    counters = Counter({"documents": tables.documents.num_rows, "rejects": tables.rejects.num_rows})
    counters.update(f"tier.{tier}" for tier in tables.trust.column("tier").to_pylist())
    counters.update(f"reject.{reason}" for reason in tables.rejects.column("reason").to_pylist())
    receipt = Receipt(
        stage=STAGE,
        part_id=part,
        inputs=(bronze_key,),
        outputs=outputs,
        transform=_transform(registry),
        counters=dict(sorted(counters.items())),
    )
    commit(root, receipt, create_only=False)
    return receipt


def _transform(registry: Registry) -> dict[str, str]:
    return {
        "trustpipe": __version__,
        "canonical_form": CANONICAL_FORM,
        "html_extractor": extractor(),
        "policy_id": registry.policy_id,
    }


class RejudgeError(ValueError):
    """A part cannot be re-judged in place; it has to be refined again from bronze."""


def can_rejudge(root: Path, part: str) -> bool:
    """Refined before with today's text transform, so at most its trust decision is stale."""
    if not is_committed(root, STAGE, part):
        return False
    transform = load_receipt(root, STAGE, part).transform
    return (
        transform.get("canonical_form") == CANONICAL_FORM
        and transform.get("html_extractor") == extractor()
    )


def rejudge_part(root: Path, bronze_key: str, registry: Registry) -> Receipt:
    """Judge a refined part again under a new policy without decoding its bytes again.

    Documents and rejects depend only on the bytes and the text transform, so when only the
    policy has changed they stay as they are and only the trust part is written, under the
    new policy's partition. The receipt is byte for byte the one a full ``refine_part``
    would commit. The previous trust part stays on disk until ``gc`` finds that no receipt
    names it any more.
    """
    day, part = parse_bronze_captures_key(bronze_key)
    if not can_rejudge(root, part):
        raise RejudgeError(f"{bronze_key} was not refined with the current text transform")
    _committed_bronze(root, bronze_key)
    previous = load_receipt(root, STAGE, part)
    documents_file, rejects_file = previous.outputs[0], previous.outputs[1]
    verify_file(root, documents_file)
    verify_file(root, rejects_file)
    documents = read_table(
        root, documents_file.key, schema=SILVER_DOCUMENTS, columns=["capture_id", "doc_id"]
    )
    captures = read_table(root, bronze_key, schema=BRONZE_CAPTURES, columns=_BRONZE_META)
    kept = captures.filter(
        pc.is_in(captures.column("capture_id"), value_set=documents.column("capture_id"))
    )
    if not kept.column("capture_id").equals(documents.column("capture_id")):
        raise RejudgeError(f"{documents_file.key} does not follow the row order of {bronze_key}")
    trust = _trust_table(
        registry,
        documents.column("capture_id").combine_chunks(),
        documents.column("doc_id").combine_chunks(),
        kept.column("host"),
        kept.column("fetched_at"),
        cast(list[str], kept.column("source_uri").to_pylist()),
        cast(list[str], kept.column("acquisition").to_pylist()),
    )
    write_bytes(root, policy_key(registry.policy_id), registry.canonical_bytes(), create_only=True)
    trust_file = write_table(
        trust, root, silver_trust_key(registry.policy_id, day, part), create_only=False
    )
    counters = Counter({k: v for k, v in previous.counters.items() if not k.startswith("tier.")})
    counters.update(f"tier.{tier}" for tier in trust.column("tier").to_pylist())
    receipt = Receipt(
        stage=STAGE,
        part_id=part,
        inputs=(bronze_key,),
        outputs=(documents_file, rejects_file, trust_file),
        transform=_transform(registry),
        counters=dict(sorted(counters.items())),
    )
    commit(root, receipt, create_only=False)
    return receipt


def _refine_task(root: Path, registry: Registry, bronze_key: str) -> PartReport:
    """Re-judge a part whose policy alone is stale; refine anything else from bronze."""
    rejudged = can_rejudge(root, parse_bronze_captures_key(bronze_key)[1])
    try:
        receipt = (rejudge_part if rejudged else refine_part)(root, bronze_key, registry)
    except (RejudgeError, IntegrityError):
        # Silver files that are damaged or out of step are rebuilt from bronze instead.
        rejudged = False
        receipt = refine_part(root, bronze_key, registry)
    documents, rejects = receipt.counters["documents"], receipt.counters["rejects"]
    return PartReport(
        tuple(o.key for o in receipt.outputs),
        documents + rejects,
        documents,
        lake_path(root, bronze_key).stat().st_size,
        sum(o.bytes for o in receipt.outputs),
        counters={**receipt.counters, "parts.rejudged": int(rejudged)},
    )


def _refined(root: Path, registry: Registry, bronze_key: str) -> bool:
    """Committed with this policy and this HTML extractor; anything else is refined again."""
    part = parse_bronze_captures_key(bronze_key)[1]
    if not is_committed(root, STAGE, part):
        return False
    transform = load_receipt(root, STAGE, part).transform
    return (
        transform["policy_id"] == registry.policy_id
        and transform.get("html_extractor") == extractor()
    )


def refine_all(
    root: Path,
    registry: Registry,
    *,
    workers: int = 1,
    on_event: Callable[[TaskEvent], None] | None = None,
) -> RunSummary:
    """Refine every committed bronze part that is not refined under this policy yet."""
    keys = [
        load_receipt(root, BRONZE_STAGE, part).outputs[0].key
        for part in committed_parts(root, BRONZE_STAGE)
    ]
    return run_stage(
        STAGE,
        partial(_refine_task, root, registry),
        keys,
        workers=workers,
        is_done=partial(_refined, root, registry),
        on_event=on_event,
    )
