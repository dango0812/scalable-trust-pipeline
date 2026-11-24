"""Gold stages: signals per part, one deduplication, then documents and chunks per part.

    gold.signals  per silver part, in parallel: trust minimum, quality rules, MinHash bands
    gold.dedup    once over all signals: exact duplicates (same doc_id), then near
                  duplicates (shared MinHash band); decides which copy of each text is kept
    gold.build    per silver part, in parallel: kept documents with provenance, rejects with
                  the stage and reason, and retrieval chunks
    snapshot      a manifest of every gold part, written last

Which copy is kept is decided by trust, not by order: the copy with the highest tier, then
the earliest fetch, then the smallest capture id. A page copied by a scraper therefore
loses to the original publisher's page. ``corroboration`` counts the distinct hosts that
carry exactly the same text, a signal that the text is widely published.

The snapshot id is a hash of the gold policy and of every signals file (key and SHA-256),
so the same silver data and policy always give the same snapshot, and any change gives a
new one. Gold part ids are derived from the snapshot and the silver part.

Scale: signals and build are per part and run on the executor (runtime/executor.py), on
as many workers as there are. The deduplication holds the doc ids, hosts and band hashes of
all candidates (about 200 bytes each, so ~2 GB for ten million documents) on one machine.
Beyond that the same decisions are computed per bucket of doc ids and per band, as
datatrove does; the stage boundary stays the same.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from trustpipe import __version__
from trustpipe.gold.chunks import chunk_id, chunk_text
from trustpipe.gold.minhash import candidate_groups, document_bands
from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.quality import assess
from trustpipe.gold.tables import (
    GOLD_CHUNKS,
    GOLD_DEDUP,
    GOLD_DOCUMENTS,
    GOLD_REJECTS,
    GOLD_SIGNALS,
)
from trustpipe.identity.ids import to_text, typed_hash
from trustpipe.lake.layout import (
    gold_dedup_key,
    gold_signals_key,
    gold_table_key,
    parse_bronze_captures_key,
    snapshot_key,
)
from trustpipe.lake.receipts import (
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
    SILVER_TRUST,
    TrustTier,
)
from trustpipe.lake.storage import WrittenFile, read_table, write_json, write_table
from trustpipe.runtime.executor import PartReport, PermanentError, RunSummary, TaskEvent, run_stage
from trustpipe.silver.refine import STAGE as REFINE_STAGE

SIGNALS_STAGE = "gold.signals"
DEDUP_STAGE = "gold.dedup"
BUILD_STAGE = "gold.build"
CANDIDATE = "candidate"
KEEP = "keep"
SNAPSHOT_FORMAT = "trustpipe.gold-snapshot/1"


@dataclass(frozen=True, slots=True)
class SilverPart:
    """The committed silver files of one refined bronze part."""

    part: str
    day: date
    bronze: WrittenFile
    documents: WrittenFile
    trust: WrittenFile
    trust_policy_id: str


def silver_part(root: Path, part: str) -> SilverPart:
    receipt = load_receipt(root, REFINE_STAGE, part)
    documents, _, trust = receipt.outputs
    bronze_key = receipt.inputs[0]
    day = parse_bronze_captures_key(bronze_key)[0]
    bronze = next(
        o for o in load_receipt(root, "bronze.ingest", part).outputs if o.key == bronze_key
    )
    return SilverPart(part, day, bronze, documents, trust, receipt.transform["policy_id"])


def _read(root: Path, written: WrittenFile, schema: pa.Schema, columns: list[str]) -> pa.Table:
    verify_file(root, written)
    return read_table(root, written.key, schema=schema, columns=columns)


def _aligned(name: str, left: pa.Table, right: pa.Table) -> None:
    if not left.column("capture_id").equals(right.column("capture_id")):
        raise PermanentError(f"{name} rows are not aligned with silver.documents")


# --- gold.signals -------------------------------------------------------------------------


def _signals_table(silver: SilverPart, root: Path, policy: GoldPolicy) -> pa.Table:
    documents = _read(
        root, silver.documents, SILVER_DOCUMENTS, ["capture_id", "doc_id", "text", "script"]
    )
    trust = _read(
        root, silver.trust, SILVER_TRUST, ["capture_id", "host", "fetched_at", "tier_rank"]
    )
    _aligned("silver.trust", documents, trust)
    texts = cast(list[str], documents.column("text").to_pylist())
    scripts = cast(list[str | None], documents.column("script").to_pylist())
    ranks = cast(list[int], trust.column("tier_rank").to_pylist())
    minimum = policy.min_tier
    rules, dedup = policy.quality, policy.dedup
    verdicts: list[str] = []
    details: list[str | None] = []
    bands: list[list[int] | None] = []
    for text, script, rank in zip(texts, scripts, ranks, strict=True):
        if rank < minimum.rank:
            tier = next(t for t in TrustTier if t.rank == rank)
            verdicts.append("tier_below_minimum")
            details.append(f"{tier.value} < {minimum.value}")
            bands.append(None)
            continue
        verdict = assess(text, script, rules)
        if verdict is not None:
            verdicts.append(verdict.reason)
            details.append(verdict.detail)
            bands.append(None)
            continue
        verdicts.append(CANDIDATE)
        details.append(None)
        bands.append(
            document_bands(
                text, ngram=dedup.ngram, bands=dedup.bands, rows=dedup.rows, seed=dedup.seed
            )
        )
    return pa.Table.from_arrays(
        [
            documents.column("capture_id"),
            documents.column("doc_id"),
            trust.column("host"),
            trust.column("fetched_at"),
            trust.column("tier_rank"),
            pa.array(verdicts, pa.string()),
            pa.array(details, pa.string()),
            pa.array(bands, pa.list_(pa.uint64())),
        ],
        schema=GOLD_SIGNALS,
    )


def _silver_digest(silver: SilverPart) -> str:
    """One hash of the silver files a gold part is computed from."""
    fields = (silver.documents.sha256.encode(), silver.trust.sha256.encode())
    return typed_hash("silver-files", *fields).hex()


def signals_part(root: Path, part: str, policy: GoldPolicy) -> Receipt:
    silver = silver_part(root, part)
    table = _signals_table(silver, root, policy)
    written = write_table(
        table, root, gold_signals_key(policy.policy_id, silver.day, part), create_only=False
    )
    counters = Counter(table.column("verdict").to_pylist())
    receipt = Receipt(
        stage=SIGNALS_STAGE,
        part_id=part,
        inputs=(silver.documents.key, silver.trust.key),
        outputs=(written,),
        transform={
            "trustpipe": __version__,
            "gold_policy_id": policy.policy_id,
            "silver_sha256": _silver_digest(silver),
        },
        counters={f"verdict.{k}": v for k, v in sorted(counters.items())},
    )
    commit(root, receipt, create_only=False)
    return receipt


def _signals_current(root: Path, policy: GoldPolicy, part: str) -> bool:
    """A signals receipt exists for this policy and for the silver files as they are now."""
    if not is_committed(root, SIGNALS_STAGE, part):
        return False
    receipt = load_receipt(root, SIGNALS_STAGE, part)
    return receipt.transform.get("gold_policy_id") == policy.policy_id and receipt.transform.get(
        "silver_sha256"
    ) == _silver_digest(silver_part(root, part))


# --- gold.dedup ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Candidate:
    part: int
    capture_id: bytes
    doc_id: bytes
    host: str
    fetched_at: datetime
    tier_rank: int
    bands: list[int]

    @property
    def preference(self) -> tuple[int, datetime, bytes]:
        """Smaller is preferred: highest tier, then earliest fetch, then smallest id."""
        return (-self.tier_rank, self.fetched_at, self.capture_id)


def snapshot_id(policy: GoldPolicy, signals: Iterable[Receipt]) -> str:
    """Hash of the policy, every signals file and the silver files behind each of them."""
    fields = [policy.policy_id.encode()]
    for receipt in sorted(signals, key=lambda r: r.part_id):
        written = receipt.outputs[0]
        fields += [written.key.encode(), written.sha256.encode()]
        fields.append(receipt.transform["silver_sha256"].encode())
    return typed_hash("gold-snapshot", *fields).hex()[:16]


def _load_candidates(root: Path, receipts: list[Receipt]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    columns = ["capture_id", "doc_id", "host", "fetched_at", "tier_rank", "verdict", "bands"]
    for index, receipt in enumerate(receipts):
        table = _read(root, receipt.outputs[0], GOLD_SIGNALS, columns)
        rows = table.filter(pc.equal(table.column("verdict"), pa.scalar(CANDIDATE))).to_pylist()
        candidates += (
            _Candidate(
                index,
                row["capture_id"],
                row["doc_id"],
                row["host"],
                row["fetched_at"],
                row["tier_rank"],
                row["bands"],
            )
            for row in rows
        )
    return candidates


def _decide(candidates: list[_Candidate]) -> dict[bytes, tuple[str, bytes, int]]:
    """capture_id -> (verdict, kept capture_id, corroboration)."""
    decisions: dict[bytes, tuple[str, bytes, int]] = {}
    by_text: dict[bytes, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_text[candidate.doc_id].append(candidate)
    representatives: list[_Candidate] = []
    corroboration: dict[bytes, int] = {}
    for copies in by_text.values():
        kept = min(copies, key=lambda c: c.preference)
        representatives.append(kept)
        corroboration[kept.capture_id] = len({c.host for c in copies})
        for copy in copies:
            if copy is not kept:
                decisions[copy.capture_id] = ("exact_duplicate", kept.capture_id, 0)
    representatives.sort(key=lambda c: c.preference)
    if representatives:
        matrix = np.array([c.bands for c in representatives], dtype=np.uint64)
        groups = candidate_groups(matrix)
        clusters: dict[int, list[_Candidate]] = defaultdict(list)
        for index, candidate in enumerate(representatives):
            clusters[groups.find(index)].append(candidate)
        for members in clusters.values():
            kept = members[0]  # representatives are sorted by preference
            decisions[kept.capture_id] = (KEEP, kept.capture_id, corroboration[kept.capture_id])
            for member in members[1:]:
                decisions[member.capture_id] = ("near_duplicate", kept.capture_id, 0)
    return decisions


def dedup(root: Path, policy: GoldPolicy) -> Receipt:
    """Decide, across every signals part of this policy, which copy of each text is kept."""
    receipts = [
        load_receipt(root, SIGNALS_STAGE, part) for part in committed_parts(root, SIGNALS_STAGE)
    ]
    receipts = [r for r in receipts if r.transform["gold_policy_id"] == policy.policy_id]
    snapshot = snapshot_id(policy, receipts)
    if is_committed(root, DEDUP_STAGE, snapshot):
        return load_receipt(root, DEDUP_STAGE, snapshot)
    candidates = _load_candidates(root, receipts)
    decisions = _decide(candidates)
    by_part: dict[int, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_part[candidate.part].append(candidate)
    outputs: list[WrittenFile] = []
    for index, receipt in enumerate(receipts):
        rows = by_part.get(index, [])
        chosen = [decisions[c.capture_id] for c in rows]
        table = pa.Table.from_arrays(
            [
                pa.array([c.capture_id for c in rows], GOLD_DEDUP.field("capture_id").type),
                pa.array([d[0] for d in chosen], pa.string()),
                pa.array([d[1] for d in chosen], GOLD_DEDUP.field("kept_capture_id").type),
                pa.array([d[2] for d in chosen], pa.int32()),
            ],
            schema=GOLD_DEDUP,
        )
        key = gold_dedup_key(snapshot, receipt.part_id)
        outputs.append(write_table(table, root, key, create_only=False))
    verdicts = Counter(d[0] for d in decisions.values())
    dedup_receipt = Receipt(
        stage=DEDUP_STAGE,
        part_id=snapshot,
        inputs=tuple(r.outputs[0].key for r in receipts),
        outputs=tuple(outputs),
        transform={"trustpipe": __version__, "gold_policy_id": policy.policy_id},
        counters={
            "candidates": len(candidates),
            **{f"verdict.{k}": v for k, v in sorted(verdicts.items())},
        },
    )
    commit(root, dedup_receipt, create_only=False)
    return dedup_receipt


# --- gold.build ---------------------------------------------------------------------------


def gold_part_id(snapshot: str, silver: str) -> str:
    return typed_hash("gold-part", snapshot.encode(), silver.encode()).hex()[:16]


def _rejects(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=GOLD_REJECTS)


def build_part(root: Path, part: str, policy: GoldPolicy, dedup_receipt: Receipt) -> Receipt:
    """Write the gold documents, rejects and chunks of one silver part for one snapshot."""
    snapshot = dedup_receipt.part_id
    silver = silver_part(root, part)
    dedup_file = next(
        (o for o in dedup_receipt.outputs if o.key == gold_dedup_key(snapshot, part)), None
    )
    if dedup_file is None:
        raise PermanentError(f"silver part {part} is not in gold snapshot {snapshot}")
    signals_file = load_receipt(root, SIGNALS_STAGE, part).outputs[0]
    if signals_file.key not in dedup_receipt.inputs:
        raise PermanentError(f"the signals of part {part} changed after snapshot {snapshot}")
    documents = _read(root, silver.documents, SILVER_DOCUMENTS, SILVER_DOCUMENTS.names)
    trust = _read(root, silver.trust, SILVER_TRUST, SILVER_TRUST.names)
    signals = _read(root, signals_file, GOLD_SIGNALS, ["capture_id", "verdict", "detail"])
    _aligned("silver.trust", documents, trust)
    _aligned("gold.signals", documents, signals)
    decided = {
        row["capture_id"]: row
        for row in _read(root, dedup_file, GOLD_DEDUP, GOLD_DEDUP.names).to_pylist()
    }
    bronze = _read(root, silver.bronze, BRONZE_CAPTURES, ["capture_id", "source_uri"])
    uris = dict(
        zip(
            bronze.column("capture_id").to_pylist(),
            bronze.column("source_uri").to_pylist(),
            strict=True,
        )
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    rows = zip(documents.to_pylist(), trust.to_pylist(), signals.to_pylist(), strict=True)
    for document, judged, signal in rows:
        capture = document["capture_id"]
        reject = {"capture_id": capture, "doc_id": document["doc_id"]}
        if signal["verdict"] != CANDIDATE:
            reason = signal["verdict"]
            rejected.append(
                {**reject, "stage": SIGNALS_STAGE, "reason": reason, "detail": signal["detail"]}
            )
            continue
        decision = decided[capture]
        if decision["verdict"] != KEEP:
            detail = f"kept copy {to_text(decision['kept_capture_id'])}"
            rejected.append(
                {**reject, "stage": DEDUP_STAGE, "reason": decision["verdict"], "detail": detail}
            )
            continue
        source_uri = uris[capture]
        kept.append(
            {
                "doc_id": document["doc_id"],
                "capture_id": capture,
                "source_uri": source_uri,
                "host": judged["host"],
                "fetched_at": judged["fetched_at"],
                "trust_policy_id": judged["policy_id"],
                "tier": judged["tier"],
                "tier_rank": judged["tier_rank"],
                "license": judged["license"],
                "publisher": judged["publisher"],
                "corroboration": decision["corroboration"],
                "script": document["script"],
                "text": document["text"],
            }
        )
        chunks += (
            {
                "chunk_id": chunk_id(document["doc_id"], piece.ordinal),
                "doc_id": document["doc_id"],
                "ordinal": piece.ordinal,
                "char_start": piece.char_start,
                "char_end": piece.char_end,
                "source_uri": source_uri,
                "tier": judged["tier"],
                "tier_rank": judged["tier_rank"],
                "license": judged["license"],
                "text": piece.text,
            }
            for piece in chunk_text(document["text"], policy.chunks)
        )
    gold_part = gold_part_id(snapshot, part)
    outputs = tuple(
        write_table(
            table, root, gold_table_key(name, snapshot, silver.day, gold_part), create_only=False
        )
        for name, table in (
            ("documents", pa.Table.from_pylist(kept, schema=GOLD_DOCUMENTS)),
            ("rejects", _rejects(rejected)),
            ("chunks", pa.Table.from_pylist(chunks, schema=GOLD_CHUNKS)),
        )
    )
    reasons = Counter(f"reject.{r['reason']}" for r in rejected)
    receipt = Receipt(
        stage=BUILD_STAGE,
        part_id=gold_part,
        inputs=(
            silver.documents.key,
            silver.trust.key,
            signals_file.key,
            dedup_file.key,
            silver.bronze.key,
        ),
        outputs=outputs,
        transform={
            "trustpipe": __version__,
            "gold_policy_id": policy.policy_id,
            "trust_policy_id": silver.trust_policy_id,
            "snapshot": snapshot,
            "silver_part": part,
        },
        counters={
            "documents": len(kept),
            "rejects": len(rejected),
            "chunks": len(chunks),
            **dict(sorted(reasons.items())),
        },
    )
    commit(root, receipt, create_only=False)
    return receipt


# --- orchestration ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldSnapshot:
    snapshot: str
    manifest_key: str
    documents: int
    chunks: int
    rejects: int
    signals: RunSummary
    build: RunSummary


def _signals_task(root: Path, policy: GoldPolicy, part: str) -> PartReport:
    receipt = signals_part(root, part, policy)
    [written] = receipt.outputs
    rows = written.rows or 0
    return PartReport((written.key,), rows, rows, 0, written.bytes, counters=receipt.counters)


def _build_task(root: Path, policy: GoldPolicy, dedup_receipt: Receipt, part: str) -> PartReport:
    receipt = build_part(root, part, policy, dedup_receipt)
    counters = receipt.counters
    return PartReport(
        tuple(o.key for o in receipt.outputs),
        counters["documents"] + counters["rejects"],
        counters["documents"],
        0,
        sum(o.bytes for o in receipt.outputs),
        counters=counters,
    )


def _manifest(
    root: Path, policy: GoldPolicy, dedup_receipt: Receipt, parts: list[str]
) -> dict[str, Any]:
    snapshot = dedup_receipt.part_id
    entries = []
    totals: Counter[str] = Counter()
    trust_policies: set[str] = set()
    for part in parts:
        receipt = load_receipt(root, BUILD_STAGE, gold_part_id(snapshot, part))
        trust_policies.add(receipt.transform["trust_policy_id"])
        totals.update(
            {k: v for k, v in receipt.counters.items() if k in ("documents", "rejects", "chunks")}
        )
        entries.append(
            {
                "silver_part": part,
                "gold_part": receipt.part_id,
                "outputs": [
                    {"key": o.key, "rows": o.rows, "bytes": o.bytes, "sha256": o.sha256}
                    for o in receipt.outputs
                ],
            }
        )
    return {
        "format": SNAPSHOT_FORMAT,
        "snapshot": snapshot,
        "gold_policy_id": policy.policy_id,
        "trust_policy_ids": sorted(trust_policies),
        "dedup": dict(dedup_receipt.counters),
        "totals": dict(sorted(totals.items())),
        "parts": entries,
    }


def build_gold(
    root: Path,
    policy: GoldPolicy,
    *,
    workers: int = 1,
    on_event: Callable[[TaskEvent], None] | None = None,
) -> GoldSnapshot:
    """Run signals, dedup and build over every committed silver part; write the snapshot."""
    parts = committed_parts(root, REFINE_STAGE)
    signals = run_stage(
        SIGNALS_STAGE,
        partial(_signals_task, root, policy),
        parts,
        workers=workers,
        is_done=partial(_signals_current, root, policy),
        on_event=on_event,
    )
    if signals.failed:
        raise RuntimeError(f"gold.signals failed for parts {list(signals.failed_keys)}")
    dedup_receipt = dedup(root, policy)
    snapshot = dedup_receipt.part_id
    build = run_stage(
        BUILD_STAGE,
        partial(_build_task, root, policy, dedup_receipt),
        parts,
        workers=workers,
        is_done=lambda part: is_committed(root, BUILD_STAGE, gold_part_id(snapshot, part)),
        on_event=on_event,
    )
    if build.failed:
        raise RuntimeError(f"gold.build failed for parts {list(build.failed_keys)}")
    manifest = _manifest(root, policy, dedup_receipt, parts)
    write_json(root, snapshot_key(snapshot), manifest, create_only=False)
    totals = manifest["totals"]
    return GoldSnapshot(
        snapshot,
        snapshot_key(snapshot),
        totals.get("documents", 0),
        totals.get("chunks", 0),
        totals.get("rejects", 0),
        signals,
        build,
    )
