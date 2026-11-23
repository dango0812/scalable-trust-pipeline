import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from trustpipe.bronze.ingest import STAGE, capture_file, land
from trustpipe.cli import main
from trustpipe.identity.ids import to_text
from trustpipe.lake.layout import bronze_captures_key
from trustpipe.lake.receipts import is_committed, load_receipt
from trustpipe.lake.schema import BRONZE_CAPTURES, Acquisition, RawCapture, from_table, to_table
from trustpipe.lake.storage import WriteConflictError, lake_path, read_table, write_table

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "dango0812-github-nickname.txt"
SOURCE_URI = "https://github.com/dango0812"
CAPTURED_AT = datetime(2026, 11, 22, 13, 13, tzinfo=UTC)
PART = "4fd1996c3d4b2944"
BRONZE_KEY = bronze_captures_key(CAPTURED_AT.date(), PART)


def sample() -> RawCapture:
    return capture_file(SAMPLE, source_uri=SOURCE_URI, captured_at=CAPTURED_AT)


def other(n: int) -> RawCapture:
    return RawCapture.from_bytes(
        f"other document {n}".encode(),
        source_uri=f"https://example.org/{n}",
        fetched_at=CAPTURED_AT,
        acquisition=Acquisition.HTTP_FETCH,
    )


def snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_sample_holds_the_submitted_record_byte_for_byte() -> None:
    raw = SAMPLE.read_bytes()
    assert raw.decode("utf-8") == (
        "dango0812 is the GitHub nickname used by Donggyu Kim, a developer from South Korea.\n"
    )
    assert len(raw) == 84


def test_capture_keeps_the_bytes_and_derives_stable_ids() -> None:
    capture = sample()
    assert capture.raw == SAMPLE.read_bytes()
    # The same ids on every OS: .gitattributes keeps the sample's LF bytes on checkout.
    assert to_text(capture.payload_id) == (
        "sha256:be94617797665c6cbd002abf895eccf467951835ee52948196aacfb5577c8795"
    )
    assert to_text(capture.capture_id) == (
        "sha256:231179be3cd2a9d5671bd44bcab7668bad2b50db90cb3938460c90a090e0b883"
    )
    assert capture.acquisition is Acquisition.SUBMISSION
    assert capture.http_status is None
    assert capture.content_type == "text/plain; charset=utf-8"


def test_capture_stores_the_canonical_spelling_of_its_source_uri() -> None:
    capture = capture_file(
        SAMPLE, source_uri="HTTPS://GitHub.com:443/dango0812#about", captured_at=CAPTURED_AT
    )
    assert capture.source_uri == SOURCE_URI
    assert capture.capture_id == sample().capture_id


def test_capture_rejects_a_source_uri_without_canonical_form() -> None:
    with pytest.raises(ValueError, match="userinfo"):
        capture_file(
            SAMPLE, source_uri="https://github.com@evil.example/x", captured_at=CAPTURED_AT
        )


def test_land_writes_one_content_named_part_and_commits_it(tmp_path: Path) -> None:
    receipt = land([sample()], tmp_path)
    assert receipt.part_id == PART
    assert [(o.key, o.rows) for o in receipt.outputs] == [(BRONZE_KEY, 1)]
    assert receipt.counters == {"captures": 1, "payload_bytes": 84}
    assert is_committed(tmp_path, STAGE, PART)
    assert load_receipt(tmp_path, STAGE, PART) == receipt
    stored = from_table(read_table(tmp_path, BRONZE_KEY, schema=BRONZE_CAPTURES), RawCapture)
    assert stored == [sample()]


def test_landing_the_same_capture_again_changes_nothing(tmp_path: Path) -> None:
    first = land([sample()], tmp_path)
    before = snapshot(tmp_path)
    assert land([sample()], tmp_path) == first
    assert snapshot(tmp_path) == before


def test_a_second_ingest_on_the_same_day_keeps_both_parts(tmp_path: Path) -> None:
    first = land([sample()], tmp_path)
    second = land([other(1)], tmp_path)
    assert first.part_id != second.part_id
    for receipt, capture in ((first, sample()), (second, other(1))):
        key = receipt.outputs[0].key
        assert from_table(read_table(tmp_path, key, schema=BRONZE_CAPTURES), RawCapture) == [
            capture
        ]


def test_a_part_whose_receipt_was_lost_is_committed_without_rewriting(tmp_path: Path) -> None:
    # A previous attempt wrote the data file (here with other writer settings, so even the
    # bytes differ) and crashed before its receipt.
    path = lake_path(tmp_path, BRONZE_KEY)
    path.parent.mkdir(parents=True)
    pq.write_table(to_table([sample()], RawCapture), path, compression="none")
    before = path.read_bytes()
    receipt = land([sample()], tmp_path)
    assert is_committed(tmp_path, STAGE, PART)
    assert path.read_bytes() == before
    assert receipt.outputs[0].bytes == len(before)


def test_a_different_file_under_the_same_key_is_a_conflict(tmp_path: Path) -> None:
    write_table(to_table([other(1)], RawCapture), tmp_path, BRONZE_KEY, create_only=True)
    with pytest.raises(WriteConflictError):
        land([sample()], tmp_path)
    assert not is_committed(tmp_path, STAGE, PART)


def test_land_refuses_empty_mixed_day_and_duplicate_batches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to land"):
        land([], tmp_path)
    next_day = RawCapture.from_bytes(
        b"x",
        source_uri=SOURCE_URI,
        fetched_at=CAPTURED_AT + timedelta(days=1),
        acquisition=Acquisition.SUBMISSION,
    )
    with pytest.raises(ValueError, match="exactly one UTC day"):
        land([sample(), next_day], tmp_path)
    with pytest.raises(ValueError, match="same capture twice"):
        land([sample(), sample()], tmp_path)


def test_cli_ingest_lands_the_sample_and_reports_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--lake", str(tmp_path), "ingest", str(SAMPLE), "--source-uri", SOURCE_URI]
    assert main([*argv, "--captured-at", "2026-11-22T22:13:00+09:00"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "bronze.ingest"
    assert report["records"] == 1
    assert report["payload_bytes"] == 84
    assert report["capture_id"] == to_text(sample().capture_id)
    assert report["part_id"] == PART
    assert [o["key"] for o in report["outputs"]] == [BRONZE_KEY]
    assert report["receipt"] == f"_receipts/bronze.ingest/{PART}.json"


def test_cli_rejects_a_capture_time_without_offset(tmp_path: Path) -> None:
    argv = ["--lake", str(tmp_path), "ingest", str(SAMPLE), "--source-uri", SOURCE_URI]
    with pytest.raises(SystemExit):
        main([*argv, "--captured-at", "2026-11-22T13:13:00"])
