from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trustpipe.ids import content_id
from trustpipe.schema import SILVER_DOCUMENTS, SILVER_TRUST, Document, to_table
from trustpipe.storage import COMPRESSION, read_table, write_table

KEY = "silver/documents/source=a.example/part-00000.parquet"


def docs_table(*texts: str) -> pa.Table:
    capture_id = content_id(b"raw")
    return to_table([Document.from_text(t, capture_id=capture_id) for t in texts], Document)


def test_write_is_atomic_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = write_table(docs_table("first"), tmp_path, KEY)
    write_table(docs_table("second"), tmp_path, KEY)  # replacing an existing shard
    assert sorted(p.name for p in path.parent.iterdir()) == ["part-00000.parquet"]
    assert read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS).column("text").to_pylist() == [
        "second"
    ]


def test_files_are_zstd_compressed(tmp_path: Path) -> None:
    path = write_table(docs_table("x"), tmp_path, KEY)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == COMPRESSION.upper()


def test_reading_a_column_subset(tmp_path: Path) -> None:
    write_table(docs_table("a", "b"), tmp_path, KEY)
    table = read_table(tmp_path, KEY, schema=SILVER_DOCUMENTS, columns=["doc_id"])
    assert table.column_names == ["doc_id"]
    assert table.num_rows == 2


def test_reading_with_the_wrong_schema_fails(tmp_path: Path) -> None:
    write_table(docs_table("a"), tmp_path, KEY)
    with pytest.raises(ValueError, match="does not match"):
        read_table(tmp_path, KEY, schema=SILVER_TRUST)
