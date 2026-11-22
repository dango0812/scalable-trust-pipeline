from datetime import date

import pytest

from trustpipe.layout import bronze_captures_key, part_name, silver_documents_key, silver_trust_key


def test_keys_use_hive_partitions() -> None:
    assert bronze_captures_key("standards.example.org", date(2026, 11, 22), 0) == (
        "bronze/captures/source=standards.example.org/date=2026-11-22/part-00000.parquet"
    )
    assert silver_documents_key("standards.example.org", 3) == (
        "silver/documents/source=standards.example.org/part-00003.parquet"
    )


def test_documents_and_trust_parts_are_aligned() -> None:
    docs = silver_documents_key("a.example", 7).split("/")
    trust = silver_trust_key("a.example", 7).split("/")
    assert docs[2:] == trust[2:]


def test_part_numbers_keep_growing_past_five_digits() -> None:
    assert part_name(123_456) == "part-123456.parquet"
    with pytest.raises(ValueError):
        part_name(-1)


@pytest.mark.parametrize("bad", ["Upper.example", "a/b", "../etc", "", "a..b", "-a"])
def test_unsafe_partition_values_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="partition value"):
        silver_documents_key(bad, 0)
