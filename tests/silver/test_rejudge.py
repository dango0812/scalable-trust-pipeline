"""A policy change re-judges trust in place, and matches a full refine byte for byte."""

import shutil
from pathlib import Path

import pyarrow as pa
import pytest
from conftest import Lander
from corpus import english

from trustpipe.lake.receipts import Receipt, commit, committed_parts, load_receipt
from trustpipe.lake.schema import SILVER_DOCUMENTS
from trustpipe.lake.storage import file_sha256, lake_path, read_table, write_table
from trustpipe.silver.refine import (
    STAGE,
    RejudgeError,
    can_rejudge,
    refine_all,
    refine_part,
    rejudge_part,
)
from trustpipe.trust.registry import Registry, load_registry

PAGES = [
    ("https://primary.example.org/a", english(1, 60)),
    ("https://agency.example.org/b", english(2, 60)),
    ("https://unknown.example.net/c", english(3, 60)),
    ("https://primary.example.org/empty", "   "),
]
NEW_POLICY = """version = "next"

[[source]]
uri_prefix = "https://unknown.example.net"
license = "CC-BY-4.0"
tier = "reviewed"
reason = "reviewed after the policy change"
"""


def new_registry(tmp_path: Path) -> Registry:
    path = tmp_path / "next.toml"
    path.write_text(NEW_POLICY, encoding="utf-8")
    return load_registry(path)


def bronze_keys(root: Path) -> list[str]:
    return [
        load_receipt(root, "bronze.ingest", part).outputs[0].key
        for part in committed_parts(root, "bronze.ingest")
    ]


def snapshot(root: Path) -> dict[str, str]:
    """Every receipt and silver file of a lake, by key, with its SHA-256."""
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for tree in ("_receipts", "silver")
        for p in sorted((root / tree).rglob("*"))
        if p.is_file()
    }


def test_rejudging_gives_exactly_what_a_full_refine_gives(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    rejudged, refined = tmp_path / "rejudged", tmp_path / "refined"
    shutil.copytree(root, rejudged)
    shutil.copytree(root, refined)
    registry = new_registry(tmp_path)

    summary = refine_all(rejudged, registry)
    for key in bronze_keys(refined):
        refine_part(refined, key, registry)

    assert summary.ok == summary.counters["parts.rejudged"] == 1
    assert snapshot(rejudged) == snapshot(refined)
    receipt = load_receipt(rejudged, STAGE, committed_parts(rejudged, STAGE)[0])
    assert receipt.transform["policy_id"] == registry.policy_id
    assert receipt.counters["tier.reviewed"] == 1
    assert refine_all(rejudged, registry).skipped == 1


def test_documents_and_rejects_are_not_rewritten(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    part = committed_parts(root, STAGE)[0]
    before = load_receipt(root, STAGE, part)
    stamps = [lake_path(root, o.key).stat().st_mtime_ns for o in before.outputs[:2]]
    after = rejudge_part(root, bronze_keys(root)[0], new_registry(tmp_path))
    assert after.outputs[:2] == before.outputs[:2]
    assert [lake_path(root, o.key).stat().st_mtime_ns for o in after.outputs[:2]] == stamps
    assert after.outputs[2].key != before.outputs[2].key  # a new trust partition


def test_a_new_text_transform_needs_a_full_refine(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    part = committed_parts(root, STAGE)[0]
    receipt = load_receipt(root, STAGE, part)
    stale = Receipt(
        receipt.stage,
        receipt.part_id,
        receipt.inputs,
        receipt.outputs,
        {**receipt.transform, "html_extractor": "an older extractor"},
        receipt.counters,
    )
    commit(root, stale, create_only=False)
    assert not can_rejudge(root, part)
    assert not can_rejudge(root, "f" * 16)
    registry = new_registry(tmp_path)
    with pytest.raises(RejudgeError, match="current text transform"):
        rejudge_part(root, bronze_keys(root)[0], registry)
    summary = refine_all(root, registry)
    assert (summary.ok, summary.counters["parts.rejudged"]) == (1, 0)


def test_a_damaged_documents_file_is_rebuilt_from_bronze(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    receipt = load_receipt(root, STAGE, committed_parts(root, STAGE)[0])
    documents = lake_path(root, receipt.outputs[0].key)
    documents.write_bytes(documents.read_bytes() + b"damage")
    summary = refine_all(root, new_registry(tmp_path))
    assert (summary.ok, summary.counters["parts.rejudged"]) == (1, 0)
    assert file_sha256(documents) == receipt.outputs[0].sha256  # rebuilt, identical bytes


def test_documents_out_of_bronze_order_are_rebuilt(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    part = committed_parts(root, STAGE)[0]
    receipt = load_receipt(root, STAGE, part)
    key = receipt.outputs[0].key
    table = read_table(root, key, schema=SILVER_DOCUMENTS)
    reversed_rows = table.take(pa.array(range(table.num_rows - 1, -1, -1)))
    written = write_table(reversed_rows, root, key, create_only=False)
    commit(
        root,
        Receipt(
            receipt.stage,
            receipt.part_id,
            receipt.inputs,
            (written, *receipt.outputs[1:]),
            receipt.transform,
            receipt.counters,
        ),
        create_only=False,
    )
    registry = new_registry(tmp_path)
    with pytest.raises(RejudgeError, match="row order"):
        rejudge_part(root, bronze_keys(root)[0], registry)
    summary = refine_all(root, registry)
    assert (summary.ok, summary.counters["parts.rejudged"]) == (1, 0)
    assert file_sha256(lake_path(root, key)) != written.sha256  # back in bronze order
