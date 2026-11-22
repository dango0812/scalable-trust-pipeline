from datetime import date

import pytest

from trustpipe.layout import (
    bronze_captures_key,
    parse_bronze_captures_key,
    part_name,
    silver_documents_key,
    silver_trust_key,
)

DAY = date(2026, 11, 22)


def test_keys_use_hive_partitions() -> None:
    assert bronze_captures_key("standards.example.org", DAY, 0) == (
        "bronze/captures/source=standards.example.org/date=2026-11-22/part-00000.parquet"
    )
    assert silver_documents_key("standards.example.org", DAY, 3) == (
        "silver/documents/source=standards.example.org/date=2026-11-22/part-00003.parquet"
    )


def test_silver_parts_mirror_their_bronze_part() -> None:
    bronze = bronze_captures_key("a.example", DAY, 7).split("/")
    for silver in (
        silver_documents_key("a.example", DAY, 7),
        silver_trust_key("a.example", DAY, 7),
    ):
        assert silver.split("/")[2:] == bronze[2:]


def test_bronze_keys_parse_back_into_their_parts() -> None:
    key = bronze_captures_key("github.com", DAY, 12)
    assert parse_bronze_captures_key(key) == ("github.com", DAY, 12)
    with pytest.raises(ValueError, match="not a bronze captures key"):
        parse_bronze_captures_key("silver/documents/source=github.com/part-00000.parquet")


def test_part_numbers_keep_growing_past_five_digits() -> None:
    assert part_name(123_456) == "part-123456.parquet"
    with pytest.raises(ValueError, match="shard must be >= 0"):
        part_name(-1)


@pytest.mark.parametrize("bad", ["Upper.example", "a/b", "../etc", "", "a..b", "-a"])
def test_unsafe_partition_values_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="partition value"):
        silver_documents_key(bad, DAY, 0)
