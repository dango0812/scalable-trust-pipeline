from collections.abc import Callable
from datetime import date

import pytest

from trustpipe.lake.layout import (
    bronze_captures_key,
    parse_bronze_captures_key,
    part_id,
    policy_key,
    receipt_key,
    silver_documents_key,
    silver_rejects_key,
    silver_trust_key,
)

DAY = date(2026, 11, 22)
PART = "4fd1996c3d4b2944"
POLICY = "sha256:" + "7a9d" * 16


def test_keys_partition_by_day_only_and_name_parts_by_content() -> None:
    assert bronze_captures_key(DAY, PART) == (
        "bronze/captures/date=2026-11-22/part-4fd1996c3d4b2944.parquet"
    )
    assert silver_documents_key(DAY, PART) == (
        "silver/documents/date=2026-11-22/part-4fd1996c3d4b2944.parquet"
    )
    assert silver_rejects_key(DAY, PART) == (
        "silver/rejects/date=2026-11-22/part-4fd1996c3d4b2944.parquet"
    )
    assert silver_trust_key(POLICY, DAY, PART) == (
        "silver/trust/policy=7a9d7a9d7a9d/date=2026-11-22/part-4fd1996c3d4b2944.parquet"
    )


def test_silver_parts_carry_their_bronze_part_in_the_name() -> None:
    bronze = bronze_captures_key(DAY, PART).split("/")[2:]
    for key in (silver_documents_key(DAY, PART), silver_rejects_key(DAY, PART)):
        assert key.split("/")[2:] == bronze
    assert silver_trust_key(POLICY, DAY, PART).split("/")[3:] == bronze


def test_part_ids_are_content_derived_and_order_sensitive() -> None:
    a, b = b"\x01" * 32, b"\x02" * 32
    assert part_id([a, b]) == part_id([a, b])
    assert part_id([a, b]) != part_id([b, a])
    assert len(part_id([a])) == 16
    with pytest.raises(ValueError, match="at least one capture"):
        part_id([])


def test_bronze_keys_parse_back_into_their_parts() -> None:
    assert parse_bronze_captures_key(bronze_captures_key(DAY, PART)) == (DAY, PART)


@pytest.mark.parametrize(
    "key",
    [
        "bronze/captures/date=2026-11-22/part-4FD1996C3D4B2944.parquet",
        "bronze/captures/source=github.com/date=2026-11-22/part-00000.parquet",
        "silver/documents/date=2026-11-22/part-4fd1996c3d4b2944.parquet",
        "bronze/captures/date=2026-11-22/part-4fd1996c3d4b2944.parquet/x",
    ],
)
def test_anything_else_is_not_a_bronze_key(key: str) -> None:
    with pytest.raises(ValueError, match="not a bronze captures key"):
        parse_bronze_captures_key(key)


def test_control_files_have_their_own_keys() -> None:
    assert receipt_key("silver.refine", PART) == "_receipts/silver.refine/4fd1996c3d4b2944.json"
    assert policy_key(POLICY) == f"_policies/{'7a9d' * 16}.json"


@pytest.mark.parametrize(
    ("build", "problem"),
    [
        (lambda: bronze_captures_key(DAY, "00000"), "16 lower-case hex"),
        (lambda: silver_trust_key("v1", DAY, PART), "policy id must look like"),
        (lambda: receipt_key("Silver/Refine", PART), "dotted lower-case"),
        (lambda: receipt_key("silver.refine", "../../etc"), "16 lower-case hex"),
    ],
)
def test_invalid_key_parts_are_rejected(build: Callable[[], str], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        build()


def test_gold_keys_name_their_snapshot_and_refuse_bad_ids() -> None:
    from datetime import date

    from trustpipe.lake.layout import (
        gold_dedup_key,
        gold_table_key,
        gold_tokens_keys,
        snapshot_key,
    )

    snap = "0123456789abcdef"
    assert (
        gold_dedup_key(snap, "a" * 16)
        == f"gold/dedup/snapshot=0123456789ab/part-{'a' * 16}.parquet"
    )
    assert gold_table_key("chunks", snap, date(2026, 1, 2), "b" * 16).startswith(
        "gold/chunks/snapshot=0123456789ab/date=2026-01-02/"
    )
    assert gold_tokens_keys(snap, "c" * 64, "d" * 16)[1].endswith(".idx.parquet")
    assert snapshot_key(snap) == f"_snapshots/{snap}.json"
    with pytest.raises(ValueError, match="unknown gold table"):
        gold_table_key("signals", snap, date(2026, 1, 2), "b" * 16)
    with pytest.raises(ValueError, match="lower-case hex"):
        gold_dedup_key("SNAPSHOT", "a" * 16)
    with pytest.raises(ValueError, match="16 lower-case hex digits"):
        snapshot_key("0123")
