import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from corpus import english, korean

from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.stages import (
    BUILD_STAGE,
    DEDUP_STAGE,
    SIGNALS_STAGE,
    _aligned,
    build_gold,
    build_part,
    dedup,
    gold_part_id,
)
from trustpipe.gold.tables import GOLD_CHUNKS, GOLD_DOCUMENTS, GOLD_REJECTS, GOLD_SIGNALS
from trustpipe.identity.ids import to_text
from trustpipe.lake.receipts import IntegrityError, committed_parts, load_receipt, verify_file
from trustpipe.lake.schema import TrustTier
from trustpipe.lake.storage import lake_path, read_table
from trustpipe.runtime.executor import PermanentError

POLICY = GoldPolicy(version="test")
Lander = Callable[[list[tuple[str, str]]], Path]


def table(root: Path, key: str, schema: pa.Schema) -> list[dict[str, Any]]:
    return read_table(root, key, schema=schema).to_pylist()


def outputs(root: Path, snapshot: str, part: str) -> dict[str, str]:
    receipt = load_receipt(root, BUILD_STAGE, gold_part_id(snapshot, part))
    documents, rejects, chunks = (o.key for o in receipt.outputs)
    return {"documents": documents, "rejects": rejects, "chunks": chunks}


def test_trust_quality_and_duplicates_decide_what_is_kept(lake: Lander) -> None:
    original = english(1, 300)
    root = lake(
        [
            ("https://mirror.example.net/copy", original),  # reviewed copy, fetched first
            ("https://primary.example.org/original", original),  # primary: wins
            ("https://agency.example.org/edited", original.replace("trust", "faith", 2)),
            ("https://agency.example.org/korean", korean(1)),
            ("https://blog.example.com/post", english(2)),  # unverified
            ("https://primary.example.org/short", english(3, 20)),  # too short
        ]
    )
    snapshot = build_gold(root, POLICY)
    assert (snapshot.documents, snapshot.rejects) == (2, 4)
    [part] = committed_parts(root, "silver.refine")
    keys = outputs(root, snapshot.snapshot, part)

    documents = {d["source_uri"]: d for d in table(root, keys["documents"], GOLD_DOCUMENTS)}
    assert set(documents) == {
        "https://primary.example.org/original",
        "https://agency.example.org/korean",
    }
    kept = documents["https://primary.example.org/original"]
    assert (kept["tier"], kept["corroboration"], kept["license"]) == ("primary", 2, "CC-BY-4.0")
    assert documents["https://agency.example.org/korean"]["script"] == "Hang"

    rejects = {(r["stage"], r["reason"]) for r in table(root, keys["rejects"], GOLD_REJECTS)}
    assert rejects == {
        (DEDUP_STAGE, "exact_duplicate"),
        (DEDUP_STAGE, "near_duplicate"),
        (SIGNALS_STAGE, "tier_below_minimum"),
        (SIGNALS_STAGE, "short_doc"),
    }
    details = {r["reason"]: r["detail"] for r in table(root, keys["rejects"], GOLD_REJECTS)}
    assert details["exact_duplicate"] == f"kept copy {to_text(kept['capture_id'])}"
    assert details["tier_below_minimum"] == "unverified < reviewed"

    chunks = table(root, keys["chunks"], GOLD_CHUNKS)
    by_doc = [c for c in chunks if c["doc_id"] == kept["doc_id"]]
    assert "".join(c["text"] for c in by_doc).replace("\n", "") == kept["text"].replace("\n", "")
    assert {c["source_uri"] for c in by_doc} == {kept["source_uri"]}


def test_the_snapshot_is_reproducible_and_changes_with_its_inputs(lake: Lander) -> None:
    root = lake(
        [
            ("https://primary.example.org/a", english(1)),
            ("https://agency.example.org/b", english(2)),
        ]
    )
    first = build_gold(root, POLICY)
    again = build_gold(root, POLICY)
    assert first.snapshot == again.snapshot
    assert again.signals.skipped == again.build.skipped == 1
    manifest = json.loads(lake_path(root, first.manifest_key).read_text(encoding="utf-8"))
    assert manifest["snapshot"] == first.snapshot
    assert manifest["gold_policy_id"] == POLICY.policy_id
    assert manifest["totals"] == {"chunks": first.chunks, "documents": 2, "rejects": 0}

    lake([("https://agency.example.org/c", english(3))])
    grown = build_gold(root, POLICY)
    assert grown.snapshot != first.snapshot
    assert grown.documents == 3
    assert grown.signals.skipped == 1  # the first part's signals are still current

    stricter = GoldPolicy(version="test", min_tier=TrustTier.PRIMARY)
    strict = build_gold(root, stricter)
    assert strict.documents == 1
    assert strict.signals.skipped == 0


def test_a_duplicate_in_another_part_is_removed_across_parts(lake: Lander) -> None:
    text = english(7, 200)
    lake([("https://agency.example.org/first", text)])
    root = lake([("https://primary.example.org/later", text)])
    snapshot = build_gold(root, POLICY, workers=2)
    assert (snapshot.documents, snapshot.rejects) == (1, 1)
    receipt = load_receipt(root, DEDUP_STAGE, snapshot.snapshot)
    assert receipt.counters == {"candidates": 2, "verdict.exact_duplicate": 1, "verdict.keep": 1}


def test_nothing_to_keep_still_builds_an_empty_snapshot(lake: Lander) -> None:
    root = lake([("https://blog.example.com/a", english(1))])
    snapshot = build_gold(root, POLICY)
    assert (snapshot.documents, snapshot.chunks, snapshot.rejects) == (0, 0, 1)
    [part] = committed_parts(root, SIGNALS_STAGE)
    signals = read_table(
        root, load_receipt(root, SIGNALS_STAGE, part).outputs[0].key, schema=GOLD_SIGNALS
    )
    assert signals.column("bands").to_pylist() == [None]


def test_an_empty_lake_builds_an_empty_snapshot(tmp_path: Path) -> None:
    snapshot = build_gold(tmp_path, POLICY)
    assert (snapshot.documents, snapshot.chunks, snapshot.rejects) == (0, 0, 0)


def test_a_part_outside_the_snapshot_is_refused(lake: Lander) -> None:
    root = lake([("https://primary.example.org/a", english(1))])
    receipt = dedup(root, POLICY)  # no signals yet: an empty snapshot
    [part] = committed_parts(root, "silver.refine")
    with pytest.raises(PermanentError, match="is not in gold snapshot"):
        build_part(root, part, POLICY, receipt)


def test_changed_silver_files_are_refused(lake: Lander) -> None:
    root = lake([("https://primary.example.org/a", english(1))])
    [part] = committed_parts(root, "silver.refine")
    documents = load_receipt(root, "silver.refine", part).outputs[0]
    lake_path(root, documents.key).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match=r"gold\.signals failed"):
        build_gold(root, POLICY)
    lake_path(root, documents.key).unlink()
    with pytest.raises(IntegrityError, match="missing"):
        verify_file(root, documents)


def test_failed_builds_are_reported(lake: Lander, monkeypatch: pytest.MonkeyPatch) -> None:
    root = lake([("https://primary.example.org/a", english(1))])

    def broken(*args: object) -> None:
        raise PermanentError("boom")

    monkeypatch.setattr("trustpipe.gold.stages.build_part", broken)
    with pytest.raises(RuntimeError, match=r"gold\.build failed"):
        build_gold(root, POLICY)


def test_misaligned_rows_are_refused() -> None:
    left = pa.table({"capture_id": [b"a"]})
    right = pa.table({"capture_id": [b"b"]})
    with pytest.raises(PermanentError, match="not aligned"):
        _aligned("x", left, right)


def test_signals_recomputed_after_a_snapshot_are_refused(lake: Lander) -> None:
    root = lake([("https://primary.example.org/a", english(1))])
    first = build_gold(root, POLICY)
    old = load_receipt(root, DEDUP_STAGE, first.snapshot)
    build_gold(root, GoldPolicy(version="newer"))
    [part] = committed_parts(root, "silver.refine")
    with pytest.raises(PermanentError, match="changed after snapshot"):
        build_part(root, part, POLICY, old)
