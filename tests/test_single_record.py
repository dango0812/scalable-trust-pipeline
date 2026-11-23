"""The one sample record, followed through every file the lake keeps for it."""

from datetime import UTC, datetime
from pathlib import Path

from trustpipe.bronze.ingest import capture_file, land
from trustpipe.lake.schema import (
    BRONZE_CAPTURES,
    SILVER_DOCUMENTS,
    SILVER_REJECTS,
    SILVER_TRUST,
    Document,
    RawCapture,
    TrustRecord,
    from_table,
)
from trustpipe.lake.storage import read_table
from trustpipe.silver.refine import refine_part
from trustpipe.trust.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "dango0812-github-nickname.txt"


def test_one_record_lands_in_bronze_and_splits_into_text_and_trust(tmp_path: Path) -> None:
    registry = load_registry(ROOT / "config" / "sources.toml")
    capture = capture_file(
        SAMPLE,
        source_uri="https://github.com/dango0812",
        captured_at=datetime(2026, 11, 22, 13, 13, tzinfo=UTC),
    )
    bronze = land([capture], tmp_path)
    silver = refine_part(tmp_path, bronze.outputs[0].key, registry)
    documents_key, rejects_key, trust_key = (o.key for o in silver.outputs)

    policy = registry.policy_id.removeprefix("sha256:")
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert written == sorted(
        [
            bronze.outputs[0].key,
            documents_key,
            rejects_key,
            trust_key,
            f"_receipts/bronze.ingest/{bronze.part_id}.json",
            f"_receipts/silver.refine/{silver.part_id}.json",
            f"_policies/{policy}.json",
        ]
    )

    # Bronze keeps the bytes exactly as received; silver holds the canonical text, and the
    # trust row says what the policy decided about this capture.
    stored = from_table(
        read_table(tmp_path, bronze.outputs[0].key, schema=BRONZE_CAPTURES), RawCapture
    )
    [document] = from_table(read_table(tmp_path, documents_key, schema=SILVER_DOCUMENTS), Document)
    [trust] = from_table(read_table(tmp_path, trust_key, schema=SILVER_TRUST), TrustRecord)
    assert stored == [capture]
    assert document.capture_id == trust.capture_id == capture.capture_id
    assert document.doc_id == trust.doc_id
    assert read_table(tmp_path, rejects_key, schema=SILVER_REJECTS).num_rows == 0
