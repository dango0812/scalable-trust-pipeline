import dataclasses
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from trustpipe.identity.ids import capture_id, payload_id
from trustpipe.lake.schema import (
    BRONZE_CAPTURES,
    SILVER_DOCUMENTS,
    SILVER_TRUST,
    TABLE_SCHEMAS,
    Acquisition,
    Document,
    RawCapture,
    Reject,
    TrustRecord,
    TrustTier,
    from_table,
    to_table,
)

AT = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
POLICY = "sha256:" + "ab" * 32


def make_capture(raw: bytes = b"payload", uri: str = "https://a.example/x") -> RawCapture:
    return RawCapture.from_bytes(
        raw,
        source_uri=uri,
        fetched_at=AT,
        acquisition=Acquisition.HTTP_FETCH,
        http_status=200,
        content_type="text/plain; charset=utf-8",
    )


def make_trust(capture: RawCapture, doc: Document, tier: TrustTier) -> TrustRecord:
    return TrustRecord(
        capture_id=capture.capture_id,
        doc_id=doc.doc_id,
        host=capture.host,
        fetched_at=capture.fetched_at,
        policy_id=POLICY,
        rule_id="https://a.example/x",
        tier=tier,
        tier_rank=tier.rank,
        license="CC-BY-4.0",
        publisher="Example",
        reason="registry rule",
    )


@pytest.mark.parametrize("record_type", list(TABLE_SCHEMAS))
def test_dataclass_mirrors_arrow_schema(record_type: type) -> None:
    fields = [f.name for f in dataclasses.fields(record_type) if f.init]
    assert fields == TABLE_SCHEMAS[record_type].names


@pytest.mark.parametrize("schema", list(TABLE_SCHEMAS.values()))
def test_fields_carry_permanent_sequential_parquet_field_ids(schema: pa.Schema) -> None:
    ids = [(field.metadata or {}).get(b"PARQUET:field_id") for field in schema]
    assert ids == [str(n).encode() for n in range(1, len(schema) + 1)]


def test_schemas_carry_table_name_and_version() -> None:
    assert BRONZE_CAPTURES.metadata[b"trustpipe.table"] == b"bronze.captures"
    assert BRONZE_CAPTURES.metadata[b"trustpipe.schema_version"] == b"2"
    assert SILVER_TRUST.metadata[b"trustpipe.table"] == b"silver.trust"


def test_content_and_trust_live_in_separate_tables() -> None:
    assert {"tier", "license", "policy_id", "reason"}.isdisjoint(SILVER_DOCUMENTS.names)
    assert {"text", "raw"}.isdisjoint(SILVER_TRUST.names)
    assert "capture_id" in SILVER_DOCUMENTS.names
    assert "capture_id" in SILVER_TRUST.names


def test_a_capture_derives_its_ids_and_canonical_uri() -> None:
    capture = make_capture(uri="HTTPS://A.example:443/x#frag")
    assert capture.source_uri == "https://a.example/x"
    assert capture.host == "a.example"
    assert capture.payload_id == payload_id(b"payload")
    assert capture.capture_id == capture_id("https://a.example/x", AT, capture.payload_id)
    assert capture.acquisition is Acquisition.HTTP_FETCH


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"raw": b"tampered"}, "payload_id does not match"),
        ({"capture_id": b"\x00" * 32}, "capture_id does not match"),
        ({"source_uri": "https://A.example/x"}, "must be canonical"),
        ({"host": "b.example"}, "host must be the host"),
        ({"payload_id": b"short"}, "32 raw bytes"),
        ({"fetched_at": datetime(2026, 11, 22)}, "timezone-aware"),  # noqa: DTZ001
        ({"acquisition": "telepathy"}, "telepathy"),
    ],
)
def test_inconsistent_captures_are_rejected(change: dict[str, object], problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        dataclasses.replace(make_capture(), **change)  # type: ignore[arg-type]


def test_document_text_is_canonical_and_matches_its_id() -> None:
    capture = make_capture()
    doc = Document.from_text("line one\r\nline two", capture_id=capture.capture_id)
    assert doc.text == "line one\nline two"
    with pytest.raises(ValueError, match="must be canonical"):
        Document(capture.capture_id, doc.doc_id, "line one\r\nline two", None)
    with pytest.raises(ValueError, match="doc_id does not match"):
        Document(capture.capture_id, payload_id(b"other"), doc.text, None)


def test_a_reject_needs_a_reason() -> None:
    with pytest.raises(ValueError, match="needs a reason"):
        Reject(make_capture().capture_id, "silver.refine", "", "detail")


def test_tiers_are_ordered_by_rank() -> None:
    ranked = sorted(TrustTier, key=lambda tier: tier.rank)
    assert ranked == [
        TrustTier.UNVERIFIED,
        TrustTier.REVIEWED,
        TrustTier.INSTITUTIONAL,
        TrustTier.PRIMARY,
    ]


def test_trust_records_validate_rank_and_policy_id() -> None:
    capture = make_capture()
    doc = Document.from_text("payload", capture_id=capture.capture_id)
    trust = make_trust(capture, doc, TrustTier.REVIEWED)
    with pytest.raises(ValueError, match="tier_rank of"):
        dataclasses.replace(trust, tier_rank=3)
    with pytest.raises(ValueError, match="policy_id must look like"):
        dataclasses.replace(trust, policy_id="v1")


def test_records_round_trip_through_arrow() -> None:
    capture = make_capture()
    doc = Document.from_text("payload", capture_id=capture.capture_id, script="Latn")
    reject = Reject(capture.capture_id, "silver.refine", "undecodable", "utf-8 at byte 0")
    trust = make_trust(capture, doc, TrustTier.PRIMARY)
    for record in (capture, doc, reject, trust):
        table = to_table([record], type(record))
        assert from_table(table, type(record)) == [record]
    assert to_table([trust], TrustRecord).column("tier").to_pylist() == ["primary"]


def test_from_table_rejects_a_foreign_schema() -> None:
    with pytest.raises(ValueError, match="does not match"):
        from_table(pa.table({"capture_id": [b"x"]}), Document)
