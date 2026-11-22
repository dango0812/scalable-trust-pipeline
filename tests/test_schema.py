import dataclasses
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from trustpipe.ids import content_id
from trustpipe.schema import (
    SILVER_DOCUMENTS,
    SILVER_TRUST,
    TABLE_SCHEMAS,
    Document,
    RawCapture,
    TrustRecord,
    TrustTier,
    from_table,
    to_table,
)

FETCHED_AT = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


def make_trust(doc_id: str, tier: TrustTier = TrustTier.PRIMARY) -> TrustRecord:
    return TrustRecord(
        doc_id=doc_id,
        source_uri="https://standards.example.org/spec-1",
        source_domain="standards.example.org",
        publisher="Example Standards Body",
        license="CC-BY-4.0",
        retrieved_at=FETCHED_AT,
        tier=tier,
        tier_reason="registry: standards.example.org is a primary source",
        policy_version="test-policy-1",
    )


@pytest.mark.parametrize("record_type", list(TABLE_SCHEMAS))
def test_dataclass_mirrors_arrow_schema(record_type: type) -> None:
    fields = [f.name for f in dataclasses.fields(record_type)]
    assert fields == TABLE_SCHEMAS[record_type].names


def test_content_and_trust_metadata_live_in_separate_tables() -> None:
    trust_only = {"source_uri", "license", "tier", "tier_reason", "policy_version"}
    assert trust_only.isdisjoint(SILVER_DOCUMENTS.names)
    assert {"text", "raw"}.isdisjoint(SILVER_TRUST.names)
    assert "doc_id" in SILVER_DOCUMENTS.names and "doc_id" in SILVER_TRUST.names


def test_schemas_carry_table_name_and_version() -> None:
    assert SILVER_TRUST.metadata[b"trustpipe.table"] == b"silver.trust"
    assert SILVER_TRUST.metadata[b"trustpipe.schema_version"] == b"1"


def test_capture_id_is_derived_from_raw_bytes() -> None:
    capture = RawCapture.from_bytes(
        b"<html>x</html>", source_uri="https://a.example/x", fetched_at=FETCHED_AT
    )
    assert capture.capture_id == content_id(b"<html>x</html>")
    with pytest.raises(ValueError, match="does not match the raw bytes"):
        dataclasses.replace(capture, raw=b"tampered")


def test_document_text_is_canonicalized() -> None:
    doc = Document.from_text("line one\r\nline two", capture_id=content_id(b"raw"))
    assert doc.text == "line one\nline two"
    with pytest.raises(ValueError, match="must be canonical"):
        Document(doc.doc_id, doc.capture_id, "line one\r\nline two", None)


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RawCapture.from_bytes(
            b"x", source_uri="https://a.example/x", fetched_at=datetime(2026, 11, 22)
        )


def test_trust_tier_is_restored_from_its_stored_string() -> None:
    table = to_table([make_trust(content_id(b"d"), TrustTier.REVIEWED)], TrustRecord)
    assert table.column("tier").to_pylist() == ["reviewed"]
    assert from_table(table, TrustRecord)[0].tier is TrustTier.REVIEWED


def test_unknown_trust_tier_is_rejected() -> None:
    table = to_table([make_trust(content_id(b"d"))], TrustRecord)
    forged = table.set_column(
        table.schema.get_field_index("tier"),
        table.schema.field("tier"),
        pa.array(["trust-me"], pa.string()),
    )
    with pytest.raises(ValueError, match="trust-me"):
        from_table(forged, TrustRecord)


def test_records_round_trip_through_arrow() -> None:
    capture = RawCapture.from_bytes(
        b"payload",
        source_uri="https://a.example/x",
        fetched_at=FETCHED_AT,
        http_status=200,
        content_type="text/plain; charset=utf-8",
    )
    doc = Document.from_text("payload", capture_id=capture.capture_id, lang="en")
    trust = make_trust(doc.doc_id)
    for record in (capture, doc, trust):
        table = to_table([record], type(record))
        assert from_table(table, type(record)) == [record]


def test_from_table_rejects_a_foreign_schema() -> None:
    table = pa.table({"doc_id": ["x"]})
    with pytest.raises(ValueError, match="does not match"):
        from_table(table, Document)
