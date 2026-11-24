"""Record schemas of the lake tables (contract version 2).

One source document becomes rows in four tables:

    bronze.captures   one row per capture event: the raw bytes exactly as received, where
                      and when they came from, and how they were acquired
    silver.documents  one row per capture that yielded text: the canonical text
    silver.rejects    one row per capture that yielded no text, and why
    silver.trust      one row per (capture, policy): the tier a versioned policy assigns

Grain and key of each table:

    bronze.captures   capture_id
    silver.documents  capture_id   (doc_id is a content attribute shared by equal texts)
    silver.rejects    capture_id
    silver.trust      (capture_id, policy_id)

Every bronze row ends up in exactly one of silver.documents and silver.rejects. Content and
trust are stored apart: a new policy writes new trust rows under its own policy id and never
rewrites text, and the decisions of older policies stay readable for audits.

The Arrow schemas are the storage contract. Every field carries a permanent Parquet field
id (fields are only ever appended, never reordered) and every table a version; within a
version, only nullable columns may be added (see ``storage.read_table``). The frozen
dataclasses mirror the schemas field for field and serve the record-at-a-time path (single
records, tests); bulk stages work on Arrow batches directly.
"""

# pa.Field is generic only in the type stubs; keep annotations unevaluated at runtime.
from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

import pyarrow as pa

from trustpipe.identity.ids import ID_BYTES, canonical_text
from trustpipe.identity.ids import capture_id as make_capture_id
from trustpipe.identity.ids import doc_id as make_doc_id
from trustpipe.identity.ids import payload_id as make_payload_id
from trustpipe.identity.uris import parse_uri

ID = pa.binary(ID_BYTES)
_UTC_MICROS = pa.timestamp("us", tz="UTC")
_POLICY_ID = re.compile(r"sha256:[0-9a-f]{64}")


def table_schema(name: str, version: int, fields: list[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    """An Arrow schema with permanent field ids, the table name and its version.

    The table name and version travel inside every Parquet file's key-value metadata, so a
    file found on its own still says what it is.
    """
    return pa.schema(
        [
            pa.field(column, kind, nullable=nullable, metadata={b"PARQUET:field_id": b"%d" % i})
            for i, (column, kind, nullable) in enumerate(fields, start=1)
        ],
        metadata={b"trustpipe.table": name.encode(), b"trustpipe.schema_version": b"%d" % version},
    )


BRONZE_CAPTURES = table_schema(
    "bronze.captures",
    2,
    [
        ("capture_id", ID, False),
        ("payload_id", ID, False),
        ("source_uri", pa.string(), False),
        ("host", pa.string(), False),
        ("fetched_at", _UTC_MICROS, False),
        ("acquisition", pa.string(), False),
        ("http_status", pa.int16(), True),
        ("content_type", pa.string(), True),
        ("raw", pa.large_binary(), False),
    ],
)

SILVER_DOCUMENTS = table_schema(
    "silver.documents",
    2,
    [
        ("capture_id", ID, False),
        ("doc_id", ID, False),
        ("text", pa.large_string(), False),
        ("script", pa.string(), True),
    ],
)

SILVER_REJECTS = table_schema(
    "silver.rejects",
    1,
    [
        ("capture_id", ID, False),
        ("stage", pa.string(), False),
        ("reason", pa.string(), False),
        ("detail", pa.string(), False),
    ],
)

SILVER_TRUST = table_schema(
    "silver.trust",
    2,
    [
        ("capture_id", ID, False),
        ("doc_id", ID, False),
        ("host", pa.string(), False),
        ("fetched_at", _UTC_MICROS, False),
        ("policy_id", pa.string(), False),
        ("rule_id", pa.string(), True),
        ("tier", pa.string(), False),
        ("tier_rank", pa.int8(), False),
        ("license", pa.string(), False),
        ("publisher", pa.string(), True),
        ("reason", pa.string(), False),
    ],
)


class TrustTier(StrEnum):
    """How far a source can be trusted, decided by policy and never by the document."""

    PRIMARY = "primary"  # the authoritative original: statute text, standards, official stats
    INSTITUTIONAL = "institutional"  # an accountable institution: agency, university, intl org
    REVIEWED = "reviewed"  # independently reviewed secondary work: peer-reviewed, edited
    UNVERIFIED = "unverified"  # no accountable publisher or review: blogs, forums, posts

    @property
    def rank(self) -> int:
        """Order of the tiers, stored as tier_rank so range filters and statistics work."""
        return _TIER_RANK[self]


_TIER_RANK = {
    TrustTier.UNVERIFIED: 0,
    TrustTier.REVIEWED: 1,
    TrustTier.INSTITUTIONAL: 2,
    TrustTier.PRIMARY: 3,
}


class Acquisition(StrEnum):
    """How the bytes of a capture were obtained; a trust rule states which it accepts."""

    SUBMISSION = "submission"  # handed in as a file; the stated source URI is a claim
    HTTP_FETCH = "http_fetch"  # fetched from source_uri by the operator's own crawler (WARC)
    ARCHIVE_IMPORT = "archive_import"  # read from a third-party web archive


def _require_id(name: str, value: object) -> None:
    if not isinstance(value, bytes) or len(value) != ID_BYTES:
        raise ValueError(f"{name} must be {ID_BYTES} raw bytes")


def _utc(name: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class RawCapture:
    """bronze.captures: one capture event. Written once and never modified."""

    capture_id: bytes
    payload_id: bytes
    source_uri: str
    host: str
    fetched_at: datetime
    acquisition: Acquisition
    http_status: int | None
    content_type: str | None
    raw: bytes

    def __post_init__(self) -> None:
        _require_id("capture_id", self.capture_id)
        _require_id("payload_id", self.payload_id)
        object.__setattr__(self, "fetched_at", _utc("fetched_at", self.fetched_at))
        object.__setattr__(self, "acquisition", Acquisition(self.acquisition))
        uri = parse_uri(self.source_uri)
        if str(uri) != self.source_uri:
            raise ValueError(f"source_uri must be canonical: {str(uri)!r}, got {self.source_uri!r}")
        if uri.host != self.host:
            raise ValueError(f"host must be the host of source_uri ({uri.host!r})")
        if self.payload_id != make_payload_id(self.raw):
            raise ValueError("payload_id does not match the raw bytes")
        if self.capture_id != make_capture_id(self.source_uri, self.fetched_at, self.payload_id):
            raise ValueError("capture_id does not match source_uri, fetched_at and payload_id")

    @classmethod
    def from_bytes(
        cls,
        raw: bytes,
        *,
        source_uri: str,
        fetched_at: datetime,
        acquisition: Acquisition,
        http_status: int | None = None,
        content_type: str | None = None,
    ) -> Self:
        """Build a capture, storing the source URI in canonical form."""
        uri = parse_uri(source_uri)
        canonical = str(uri)
        moment = _utc("fetched_at", fetched_at)
        payload = make_payload_id(raw)
        return cls(
            make_capture_id(canonical, moment, payload),
            payload,
            canonical,
            uri.host,
            moment,
            Acquisition(acquisition),
            http_status,
            content_type,
            raw,
        )


@dataclass(frozen=True, slots=True)
class Document:
    """silver.documents: the canonical text one capture yielded. No trust fields."""

    capture_id: bytes
    doc_id: bytes
    text: str
    script: str | None

    def __post_init__(self) -> None:
        _require_id("capture_id", self.capture_id)
        _require_id("doc_id", self.doc_id)
        if self.text != canonical_text(self.text):
            raise ValueError("text must be canonical; build documents with Document.from_text")
        if self.doc_id != make_doc_id(self.text):
            raise ValueError("doc_id does not match the text")

    @classmethod
    def from_text(cls, text: str, *, capture_id: bytes, script: str | None = None) -> Self:
        canonical = canonical_text(text)
        return cls(capture_id, make_doc_id(canonical), canonical, script)


@dataclass(frozen=True, slots=True)
class Reject:
    """silver.rejects: a capture that yielded no document, with a reason code and detail."""

    capture_id: bytes
    stage: str
    reason: str
    detail: str

    def __post_init__(self) -> None:
        _require_id("capture_id", self.capture_id)
        if not self.reason:
            raise ValueError("a reject needs a reason")


@dataclass(frozen=True, slots=True)
class TrustRecord:
    """silver.trust: what one policy decided about one capture, and why.

    ``rule_id`` is the canonical prefix of the registry rule that decided the tier (None
    when no rule applied); ``license`` is an SPDX identifier, ``NONE``, ``NOASSERTION`` or
    ``LicenseRef-<name>``. With ``policy_id`` every training decision can be traced back to
    the exact policy file that made it, which the lake keeps under ``_policies/``.
    """

    capture_id: bytes
    doc_id: bytes
    host: str
    fetched_at: datetime
    policy_id: str
    rule_id: str | None
    tier: TrustTier
    tier_rank: int
    license: str
    publisher: str | None
    reason: str

    def __post_init__(self) -> None:
        _require_id("capture_id", self.capture_id)
        _require_id("doc_id", self.doc_id)
        object.__setattr__(self, "fetched_at", _utc("fetched_at", self.fetched_at))
        # Accept the plain string read back from Parquet and normalize it to the enum.
        object.__setattr__(self, "tier", TrustTier(self.tier))
        if self.tier_rank != self.tier.rank:
            raise ValueError(
                f"tier_rank of {self.tier!r} is {self.tier.rank}, got {self.tier_rank}"
            )
        if not _POLICY_ID.fullmatch(self.policy_id):
            raise ValueError(
                f"policy_id must look like sha256:<64 hex digits>, got {self.policy_id!r}"
            )


Record = RawCapture | Document | Reject | TrustRecord

TABLE_SCHEMAS: dict[type[Any], pa.Schema] = {
    RawCapture: BRONZE_CAPTURES,
    Document: SILVER_DOCUMENTS,
    Reject: SILVER_REJECTS,
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
