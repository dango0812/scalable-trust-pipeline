"""Reads and writes of lake files.

- Write once. Bronze files are published create-only: data goes to a temporary file in the
  same directory that is hard-linked to its final name, which fails if the name exists. If
  the existing file holds the same bytes, the write is a retry and counts as done;
  otherwise it is a conflict and nothing is overwritten. (On an object store the same
  guarantee comes from a conditional PUT with ``If-None-Match: *``.)
- Derived files (silver) can be rebuilt from bronze, so a re-run may replace them; a part
  only counts once its receipt is written, last (see receipts.py).
- Parquet with zstd compression, page checksums and bounded row groups.
- Every write returns the file's size and SHA-256, which the receipt records.
- Reads check the table name and schema version stored in the file, and accept files of an
  older version of the same table whose missing columns are nullable (they read as nulls):
  columns may only be added, never changed or removed.
"""

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION: Final = "zstd"
ROW_GROUP_ROWS: Final = 10_000
_SEGMENT = re.compile(r"[A-Za-z0-9._=-]+")


class WriteConflictError(Exception):
    """A create-only key already exists with different content."""


@dataclass(frozen=True, slots=True)
class WrittenFile:
    key: str
    bytes: int
    sha256: str
    rows: int | None = None  # None for non-tabular files such as receipts


def lake_path(root: Path, key: str) -> Path:
    """Resolve a key under the lake root, refusing anything that could escape it."""
    segments = key.split("/")
    if not all(_SEGMENT.fullmatch(s) and s not in (".", "..") for s in segments):
        raise ValueError(f"invalid lake key {key!r}")
    return root.joinpath(*segments)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(root: Path, key: str, write: Callable[[Path], None], *, create_only: bool) -> Path:
    path = lake_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        if not create_only:
            tmp.replace(path)
            return path
        try:
            os.link(tmp, path)
        except FileExistsError:
            if file_sha256(path) != file_sha256(tmp):
                raise WriteConflictError(f"{key} already exists with different content") from None
        return path
    finally:
        tmp.unlink(missing_ok=True)


def write_table(table: pa.Table, root: Path, key: str, *, create_only: bool) -> WrittenFile:
    def write(tmp: Path) -> None:
        pq.write_table(
            table,
            tmp,
            compression=COMPRESSION,
            row_group_size=ROW_GROUP_ROWS,
            write_statistics=True,
            write_page_checksum=True,
        )

    path = _publish(root, key, write, create_only=create_only)
    return WrittenFile(key, path.stat().st_size, file_sha256(path), table.num_rows)


def describe(root: Path, key: str) -> WrittenFile:
    """Size, SHA-256 and (for Parquet) row count of a file that is already in the lake."""
    path = lake_path(root, key)
    rows = pq.ParquetFile(path).metadata.num_rows if key.endswith(".parquet") else None
    return WrittenFile(key, path.stat().st_size, file_sha256(path), rows)


def dumps(document: dict[str, Any]) -> bytes:
    """Deterministic JSON: the same document always gives the same bytes."""
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def write_bytes(root: Path, key: str, data: bytes, *, create_only: bool) -> WrittenFile:
    def write(tmp: Path) -> None:
        tmp.write_bytes(data)

    _publish(root, key, write, create_only=create_only)
    return WrittenFile(key, len(data), hashlib.sha256(data).hexdigest())


def write_json(root: Path, key: str, document: dict[str, Any], *, create_only: bool) -> WrittenFile:
    return write_bytes(root, key, dumps(document), create_only=create_only)


def _missing_columns(stored: pa.Schema, expected: pa.Schema, key: str) -> set[str]:
    """Check that a stored schema is readable as the expected one; return columns to fill."""
    stored_meta = stored.metadata or {}
    table = stored_meta.get(b"trustpipe.table", b"").decode()
    wanted = expected.metadata[b"trustpipe.table"].decode()
    if table != wanted:
        raise ValueError(f"{key}: holds table {table!r}, expected {wanted!r}")
    version = int(stored_meta.get(b"trustpipe.schema_version", b"0"))
    current = int(expected.metadata[b"trustpipe.schema_version"])
    if version > current:
        raise ValueError(
            f"{key}: written with schema version {version}, this code reads <= {current}"
        )
    fields = {field.name: field for field in stored}
    extra = sorted(set(fields) - set(expected.names))
    if extra:
        raise ValueError(f"{key}: unknown columns {extra}")
    missing: set[str] = set()
    for field in expected:
        found = fields.get(field.name)
        if found is None:
            if version == current or not field.nullable:
                raise ValueError(f"{key}: column {field.name!r} is missing")
            missing.add(field.name)
        elif found.type != field.type or found.nullable != field.nullable:
            raise ValueError(
                f"{key}: column {field.name!r} is {found.type} (nullable={found.nullable}), "
                f"expected {field.type} (nullable={field.nullable})"
            )
    return missing


def read_table(
    root: Path, key: str, *, schema: pa.Schema, columns: list[str] | None = None
) -> pa.Table:
    """Read a lake table as ``schema``, projecting older compatible versions."""
    parquet = pq.ParquetFile(lake_path(root, key))
    missing = _missing_columns(parquet.schema_arrow, schema, key)
    wanted = columns or schema.names
    table = parquet.read(columns=[c for c in wanted if c not in missing])
    arrays = [
        pa.chunked_array([pa.nulls(table.num_rows, schema.field(c).type)])
        if c in missing
        else table.column(c)
        for c in wanted
    ]
    metadata = cast(dict[bytes | str, bytes | str], schema.metadata)
    target = pa.schema([schema.field(c) for c in wanted], metadata=metadata)
    return pa.Table.from_arrays(arrays, schema=target)
