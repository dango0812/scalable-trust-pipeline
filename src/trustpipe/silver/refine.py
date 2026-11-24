"""Silver refinement: turn one bronze part into documents, rejects and trust rows.

Every capture of the part ends up in exactly one of two places:

- it decodes to non-empty text: one silver.documents row (canonical text and script) and
  one silver.trust row (what the policy decides about this capture), in the same order;
- it does not: one silver.rejects row with a reason code, and the part carries on.

So rows(bronze) == rows(documents) + rows(rejects) for every part, and one bad row never
fails a whole part.

Decoding (decoding.py) and the script guess (script.py) live in their own modules.

Two entry points share one per-capture function, so they cannot disagree:

- ``refine`` takes records and returns records (single captures, tests);
- ``refine_part`` is the bulk path. It checks the bronze file against the SHA-256 in its
  receipt once, instead of re-hashing every payload and id while reading rows, and builds
  the silver tables column by column. That removes per-row validation that the write path
  already did, and keeps the cost per capture to decoding, hashing the text once, the
  script guess and the trust decision.

The silver files are written first and the receipt last (lake/receipts.py). The policy that
decided the trust rows is stored under _policies/, so every row can be explained later.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pyarrow as pa

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
from trustpipe.lake.receipts import Receipt, commit, is_committed, load_receipt
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
from trustpipe.silver.decoding import UnsupportedCharsetError, decode_bytes
from trustpipe.silver.script import detect_script
from trustpipe.trust.registry import Decision, Registry

STAGE = "silver.refine"
_BRONZE_COLUMNS = ["capture_id", "source_uri", "host", "fetched_at", "acquisition"]
_BRONZE_COLUMNS += ["content_type", "raw"]


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
    try:
        text = canonical_text(decode_bytes(raw, content_type))
    except UnsupportedCharsetError as error:
        return _Failure("unsupported_charset", str(error))
    except UnicodeDecodeError as error:
        return _Failure("undecodable", f"{error.encoding}: {error.reason} at byte {error.start}")
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
    decisions: list[Decision] = []
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
        decisions.append(_judge(registry, split_canonical(uris[row]), methods[row]))
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
    trust = pa.Table.from_arrays(
        [
            kept_ids,
            doc_id_array,
            captures.column("host").take(take),
            captures.column("fetched_at").take(take),
            pa.array([registry.policy_id] * len(kept), pa.string()),
            pa.array([d.rule.uri_prefix if d.rule else None for d in decisions], pa.string()),
            pa.array([d.tier.value for d in decisions], pa.string()),
            pa.array([d.tier.rank for d in decisions], pa.int8()),
            pa.array([d.license for d in decisions], pa.string()),
            pa.array([d.rule.publisher if d.rule else None for d in decisions], pa.string()),
            pa.array([d.reason for d in decisions], pa.string()),
        ],
        schema=SILVER_TRUST,
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
        transform={
            "trustpipe": __version__,
            "canonical_form": CANONICAL_FORM,
            "policy_id": registry.policy_id,
        },
        counters=dict(sorted(counters.items())),
    )
    commit(root, receipt, create_only=False)
    return receipt
