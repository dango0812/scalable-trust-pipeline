"""Record schemas for each lake layer.

One source document becomes rows in three tables:

  bronze.captures   the raw bytes exactly as received, and how and when they were fetched
  silver.documents  the canonical text training may use, linked to the capture it came from
  silver.trust      trust metadata about that text: provenance, license, trust tier

Raw content and trust metadata are stored apart on purpose. Trust is re-evaluated far more
often than text changes (a new policy, new evidence about a source), and with separate
tables re-scoring rewrites only the small trust table, never terabytes of text. The two
are joined by ``doc_id``.

The Arrow schemas are the storage contract; the frozen dataclasses mirror them field for
field so code works with typed records instead of loose dicts.
"""

# pa.Field is generic only in the type stubs; keep annotations unevaluated at runtime.
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

import pyarrow as pa

from trustpipe.ids import canonical_text, content_id, document_id, is_content_id

SCHEMA_VERSION = "1"

_UTC_MICROS = pa.timestamp("us", tz="UTC")


def _table_schema(name: str, fields: list[pa.Field[Any]]) -> pa.Schema:
    # The table name and version travel inside every Parquet file's key-value metadata,
    # so a file found on its own still says what it is.
    return pa.schema(
        fields, metadata={"trustpipe.table": name, "trustpipe.schema_version": SCHEMA_VERSION}
    )


BRONZE_CAPTURES = _table_schema(
    "bronze.captures",
    [
        pa.field("capture_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("fetched_at", _UTC_MICROS, nullable=False),
        pa.field("http_status", pa.int16()),
        pa.field("content_type", pa.string()),
        pa.field("raw", pa.large_binary(), nullable=False),
    ],
)

SILVER_DOCUMENTS = _table_schema(
    "silver.documents",
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("capture_id", pa.string(), nullable=False),
        pa.field("text", pa.large_string(), nullable=False),
        pa.field("lang", pa.string()),
    ],
)

SILVER_TRUST = _table_schema(
    "silver.trust",
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("source_domain", pa.string(), nullable=False),
        pa.field("publisher", pa.string()),
        pa.field("license", pa.string(), nullable=False),
        pa.field("retrieved_at", _UTC_MICROS, nullable=False),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("tier_reason", pa.string(), nullable=False),
        pa.field("policy_version", pa.string(), nullable=False),
    ],
)


class TrustTier(StrEnum):
    """How much a source can be trusted, decided by policy and never by the document."""

    PRIMARY = "primary"  # the authoritative original: statute text, standards, official stats
    INSTITUTIONAL = "institutional"  # an accountable institution: agency, university, intl org
    REVIEWED = "reviewed"  # independently reviewed secondary work: peer-reviewed, edited
    UNVERIFIED = "unverified"  # no accountable publisher or review: blogs, forums, posts


def _require_id(name: str, value: str) -> None:
    if not is_content_id(value):
        raise ValueError(f"{name} must look like sha256:<64 hex chars>, got {value!r}")


def _require_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")


@dataclass(frozen=True, slots=True)
class RawCapture:
    """Bronze: bytes exactly as fetched. Never modified after being written."""

    capture_id: str
    source_uri: str
    fetched_at: datetime
    http_status: int | None
    content_type: str | None
    raw: bytes

    def __post_init__(self) -> None:
        _require_id("capture_id", self.capture_id)
        _require_utc("fetched_at", self.fetched_at)
        if self.capture_id != content_id(self.raw):
            raise ValueError("capture_id does not match the raw bytes")

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        source_uri: str,
        fetched_at: datetime,
        http_status: int | None = None,
        content_type: str | None = None,
    ) -> Self:
        return cls(content_id(raw), source_uri, fetched_at, http_status, content_type, raw)


@dataclass(frozen=True, slots=True)
class Document:
    """Silver: canonical text plus the capture it was extracted from. No trust fields."""

    doc_id: str
    capture_id: str
    text: str
    lang: str | None

    def __post_init__(self) -> None:
        _require_id("doc_id", self.doc_id)
        _require_id("capture_id", self.capture_id)
        if self.text != canonical_text(self.text):
            raise ValueError("text must be canonical; build documents with Document.from_text")
        if self.doc_id != document_id(self.text):
            raise ValueError("doc_id does not match the text")

    @classmethod
    def from_text(cls, text: str, *, capture_id: str, lang: str | None = None) -> Self:
        canonical = canonical_text(text)
        return cls(document_id(canonical), capture_id, canonical, lang)


@dataclass(frozen=True, slots=True)
class TrustRecord:
    """Silver: what we know about a document's origin and how far it can be trusted.

    ``license`` holds an SPDX identifier, or ``LicenseRef-<name>`` for terms SPDX does not
    list. ``tier_reason`` and ``policy_version`` record why and under which policy the tier
    was assigned, so every training decision can be traced back and re-evaluated.
    """

    doc_id: str
    source_uri: str
    source_domain: str
    publisher: str | None
    license: str
    retrieved_at: datetime
    tier: TrustTier
    tier_reason: str
    policy_version: str

    def __post_init__(self) -> None:
        _require_id("doc_id", self.doc_id)
        _require_utc("retrieved_at", self.retrieved_at)
        # Accept the plain string read back from Parquet and normalize it to the enum.
        object.__setattr__(self, "tier", TrustTier(self.tier))


Record = RawCapture | Document | TrustRecord

TABLE_SCHEMAS: dict[type[Any], pa.Schema] = {
    RawCapture: BRONZE_CAPTURES,
    Document: SILVER_DOCUMENTS,
    TrustRecord: SILVER_TRUST,
}


def to_table[R: Record](records: Iterable[R], record_type: type[R]) -> pa.Table:
    """Build an Arrow table; types are checked against the schema while converting."""
    rows: list[dict[str, Any]] = [
        {f.name: getattr(r, f.name) for f in dataclasses.fields(record_type)} for r in records
    ]
    return pa.Table.from_pylist(rows, schema=TABLE_SCHEMAS[record_type])


def from_table[R: Record](table: pa.Table, record_type: type[R]) -> list[R]:
    """Read rows back into records; each one is re-validated on construction."""
    expected = TABLE_SCHEMAS[record_type]
    if not table.schema.equals(expected):
        raise ValueError(f"table schema does not match {expected.metadata[b'trustpipe.table']!r}")
    return [record_type(**row) for row in table.to_pylist()]
