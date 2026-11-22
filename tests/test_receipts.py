from pathlib import Path

import pytest

from trustpipe.receipts import (
    Receipt,
    commit,
    committed_parts,
    is_committed,
    load_receipt,
)
from trustpipe.storage import WriteConflictError, WrittenFile, lake_path

PART = "0123456789abcdef"


def receipt(part: str = PART, rows: int = 3) -> Receipt:
    return Receipt(
        stage="silver.refine",
        part_id=part,
        inputs=("bronze/captures/date=2026-11-22/part-0123456789abcdef.parquet",),
        outputs=(
            WrittenFile("silver/documents/date=2026-11-22/part-x.parquet", 10, "ab" * 32, rows),
        ),
        transform={"trustpipe": "0.1.0"},
        counters={"documents": rows},
    )


def test_a_committed_receipt_round_trips(tmp_path: Path) -> None:
    assert not is_committed(tmp_path, "silver.refine", PART)
    written = commit(tmp_path, receipt(), create_only=True)
    assert written.key == "_receipts/silver.refine/0123456789abcdef.json"
    assert is_committed(tmp_path, "silver.refine", PART)
    assert load_receipt(tmp_path, "silver.refine", PART) == receipt()


def test_receipts_hold_no_timestamps_so_a_retry_reproduces_them(tmp_path: Path) -> None:
    first = commit(tmp_path, receipt(), create_only=True)
    assert commit(tmp_path, receipt(), create_only=True) == first
    with pytest.raises(WriteConflictError):
        commit(tmp_path, receipt(rows=4), create_only=True)
    assert commit(tmp_path, receipt(rows=4), create_only=False).sha256 != first.sha256


def test_committed_parts_are_listed_in_order(tmp_path: Path) -> None:
    assert committed_parts(tmp_path, "silver.refine") == []
    for part in ("ffffffffffffffff", PART):
        commit(tmp_path, receipt(part), create_only=True)
    assert committed_parts(tmp_path, "silver.refine") == [PART, "ffffffffffffffff"]


def test_an_unknown_receipt_format_is_rejected(tmp_path: Path) -> None:
    commit(tmp_path, receipt(), create_only=True)
    path = lake_path(tmp_path, receipt().key)
    path.write_text(path.read_text(encoding="utf-8").replace("receipt/1", "receipt/9"), "utf-8")
    with pytest.raises(ValueError, match=r"not a trustpipe\.receipt/1 document"):
        load_receipt(tmp_path, "silver.refine", PART)
