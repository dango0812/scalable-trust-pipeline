"""One source document, followed through every table it lands in."""

from datetime import UTC, datetime
from pathlib import Path

from trustpipe.layout import bronze_captures_key, silver_documents_key, silver_trust_key
from trustpipe.schema import (
    BRONZE_CAPTURES,
    SILVER_DOCUMENTS,
    SILVER_TRUST,
    Document,
    RawCapture,
    TrustRecord,
    TrustTier,
    from_table,
    to_table,
)
from trustpipe.storage import read_table, write_table

SOURCE = "standards.example.org"
FETCHED_AT = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
RAW = b"Spec 1\r\n\r\nA conforming parser MUST reject invalid input.\r\n"


def test_one_document_is_stored_as_capture_document_and_trust(tmp_path: Path) -> None:
    capture = RawCapture.from_bytes(
        RAW,
        source_uri=f"https://{SOURCE}/spec-1.txt",
        fetched_at=FETCHED_AT,
        http_status=200,
        content_type="text/plain; charset=utf-8",
    )
    doc = Document.from_text(RAW.decode("utf-8"), capture_id=capture.capture_id, lang="en")
    trust = TrustRecord(
        doc_id=doc.doc_id,
        source_uri=capture.source_uri,
        source_domain=SOURCE,
        publisher="Example Standards Body",
        license="CC-BY-4.0",
        retrieved_at=capture.fetched_at,
        tier=TrustTier.PRIMARY,
        tier_reason="registry: standards.example.org publishes the normative text",
        policy_version="test-policy-1",
    )

    capture_key = bronze_captures_key(SOURCE, FETCHED_AT.date(), 0)
    doc_key = silver_documents_key(SOURCE, FETCHED_AT.date(), 0)
    trust_key = silver_trust_key(SOURCE, FETCHED_AT.date(), 0)
    write_table(to_table([capture], RawCapture), tmp_path, capture_key)
    write_table(to_table([doc], Document), tmp_path, doc_key)
    write_table(to_table([trust], TrustRecord), tmp_path, trust_key)

    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
    assert written == sorted([capture_key, doc_key, trust_key])

    # Bronze keeps the bytes exactly as received; silver holds the canonical text.
    bronze = from_table(read_table(tmp_path, capture_key, schema=BRONZE_CAPTURES), RawCapture)
    silver = from_table(read_table(tmp_path, doc_key, schema=SILVER_DOCUMENTS), Document)
    trusted = from_table(read_table(tmp_path, trust_key, schema=SILVER_TRUST), TrustRecord)
    assert bronze == [capture] and bronze[0].raw == RAW
    assert silver == [doc] and silver[0].text == (
        "Spec 1\n\nA conforming parser MUST reject invalid input.\n"
    )
    assert trusted == [trust]
    assert silver[0].doc_id == trusted[0].doc_id and silver[0].capture_id == capture.capture_id
