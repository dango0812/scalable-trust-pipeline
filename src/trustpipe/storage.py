"""Parquet reads and writes for lake tables.

Write settings, chosen for large corpora rather than for one record:

- zstd compression: a strong ratio with fast decoding; the usual choice for text corpora.
- Column statistics stay on, so readers can skip whole row groups by min/max values.
- Row groups are bounded so a reader can stream a shard instead of loading it whole.
- Atomic replace: data goes to a temporary file in the same directory that is renamed into
  place. A crashed or retried task never leaves a half-written shard for readers to find.
  (On object stores the equivalent guarantee comes from single-PUT uploads.)
"""

import uuid
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION: Final = "zstd"
ROW_GROUP_ROWS: Final = 10_000


def lake_path(root: Path, key: str) -> Path:
    return root.joinpath(*key.split("/"))


def write_table(table: pa.Table, root: Path, key: str) -> Path:
    path = lake_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            tmp,
            compression=COMPRESSION,
            row_group_size=ROW_GROUP_ROWS,
            write_statistics=True,
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def read_table(
    root: Path, key: str, *, schema: pa.Schema, columns: list[str] | None = None
) -> pa.Table:
    """Read a lake table and check it was written with the expected schema."""
    path = lake_path(root, key)
    stored = pq.read_schema(path)
    if not stored.equals(schema, check_metadata=True):
        raise ValueError(f"{key}: stored schema does not match the expected schema")
    return pq.read_table(path, columns=columns, schema=schema if columns is None else None)
