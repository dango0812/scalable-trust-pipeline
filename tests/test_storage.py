import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trustpipe.lake.schema import SILVER_DOCUMENTS, SILVER_TRUST, Document, to_table
from trustpipe.lake.storage import (
    COMPRESSION,
    WriteConflictError,
    describe,
    file_sha256,
    lake_path,
    read_table,
    write_json,
    write_table,
)

KEY = "silver/documents/date=2026-11-22/part-0123456789abcdef.parquet"
CAPTURE = b"\x11" * 32


def docs_table(*texts: str) -> pa.Table:
    return to_table([Document.from_text(t, capture_id=CAPTURE) for t in texts], Document)


def files_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_a_create_only_write_never_overwrites(tmp_path: Path) -> None:
    first = write_table(docs_table("first"), tmp_path, KEY, create_only=True)
    again = write_table(docs_table("first"), tmp_path, KEY, create_only=True)
    assert again == first  # a retry with the same bytes counts as done
    with pytest.raises(WriteConflictError, match="different content"):
        write_table(docs_table("second"), tmp_path, KEY, create_only=True)
    texts = read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS).column("text").to_pylist()
    assert texts == ["first"]
    assert files_in(lake_path(tmp_path, KEY).parent) == ["part-0123456789abcdef.parquet"]


def test_a_replacing_write_is_atomic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    write_table(docs_table("first"), tmp_path, KEY, create_only=False)
    written = write_table(docs_table("second"), tmp_path, KEY, create_only=False)
    path = lake_path(tmp_path, KEY)
    assert files_in(path.parent) == ["part-0123456789abcdef.parquet"]
    assert written.rows == 1
    assert written.bytes == path.stat().st_size
    assert written.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert describe(tmp_path, KEY) == written


def test_a_failed_write_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        write_table(docs_table("x"), tmp_path, KEY, create_only=True)
    assert files_in(lake_path(tmp_path, KEY).parent) == []


def test_files_are_zstd_compressed_with_page_checksums(tmp_path: Path) -> None:
    write_table(docs_table("x"), tmp_path, KEY, create_only=True)
    column = pq.ParquetFile(lake_path(tmp_path, KEY)).metadata.row_group(0).column(0)
    assert column.compression == COMPRESSION.upper()
    rows = pq.read_table(lake_path(tmp_path, KEY), page_checksum_verification=True).num_rows
    assert rows == 1


def test_reading_a_column_subset(tmp_path: Path) -> None:
    write_table(docs_table("a", "b"), tmp_path, KEY, create_only=True)
    table = read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS, columns=["doc_id"])
    assert table.column_names == ["doc_id"]
    assert table.num_rows == 2


def write_raw(tmp_path: Path, table: pa.Table) -> None:
    path = lake_path(tmp_path, KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def with_version(schema: pa.Schema, version: bytes, table: bytes | None = None) -> pa.Schema:
    metadata = dict(schema.metadata)
    metadata[b"trustpipe.schema_version"] = version
    if table is not None:
        metadata[b"trustpipe.table"] = table
    return schema.with_metadata(metadata)


def test_an_older_version_without_a_new_nullable_column_reads_as_nulls(tmp_path: Path) -> None:
    current = docs_table("a")
    old_schema = with_version(SILVER_DOCUMENTS.remove(3), b"1")  # no "script" column yet
    write_raw(tmp_path, pa.Table.from_arrays(current.columns[:3], schema=old_schema))
    table = read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS)
    assert table.schema.equals(SILVER_DOCUMENTS)
    assert table.column("script").to_pylist() == [None]
    assert table.column("text").to_pylist() == ["a"]


@pytest.mark.parametrize(
    ("stored", "problem"),
    [
        (with_version(SILVER_DOCUMENTS, b"3"), "schema version 3"),
        (with_version(SILVER_DOCUMENTS, b"2", b"silver.trust"), "holds table 'silver.trust'"),
        (with_version(SILVER_DOCUMENTS.remove(3), b"2"), "'script' is missing"),
        (with_version(SILVER_DOCUMENTS.remove(2), b"1"), "'text' is missing"),
        (
            with_version(SILVER_DOCUMENTS.set(3, pa.field("script", pa.int8())), b"2"),
            "'script' is int8",
        ),
        (
            with_version(SILVER_DOCUMENTS.append(pa.field("extra", pa.int8())), b"2"),
            "unknown columns",
        ),
    ],
)
def test_incompatible_files_are_rejected(tmp_path: Path, stored: pa.Schema, problem: str) -> None:
    write_raw(tmp_path, stored.empty_table())
    with pytest.raises(ValueError, match=problem):
        read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS)


def test_reading_with_another_tables_schema_fails(tmp_path: Path) -> None:
    write_table(docs_table("a"), tmp_path, KEY, create_only=True)
    with pytest.raises(ValueError, match="holds table"):
        read_table(tmp_path, KEY, schema=SILVER_TRUST)


@pytest.mark.parametrize("key", ["", "a//b", "a/./b", "../x", "a/../b", "a\\b", "C:/x", "a b"])
def test_keys_that_could_escape_the_lake_are_rejected(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="invalid lake key"):
        lake_path(tmp_path, key)


def test_json_files_are_deterministic_and_described(tmp_path: Path) -> None:
    first = write_json(
        tmp_path, "_receipts/x/0123456789abcdef.json", {"b": 1, "a": "é"}, create_only=True
    )
    again = write_json(
        tmp_path, "_receipts/x/0123456789abcdef.json", {"a": "é", "b": 1}, create_only=True
    )
    assert again == first
    path = lake_path(tmp_path, first.key)
    assert path.read_text(encoding="utf-8") == '{\n  "a": "é",\n  "b": 1\n}\n'
    described = describe(tmp_path, first.key)
    assert (described.sha256, described.rows) == (file_sha256(path), None)
