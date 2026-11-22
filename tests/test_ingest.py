import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trustpipe.cli import main
from trustpipe.ingest import capture_file, land
from trustpipe.schema import BRONZE_CAPTURES, RawCapture, from_table
from trustpipe.storage import read_table

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "dango0812-github-nickname.txt"
SOURCE_URI = "https://github.com/dango0812"
CAPTURED_AT = datetime(2026, 11, 22, 13, 13, tzinfo=UTC)
BRONZE_KEY = "bronze/captures/source=github.com/date=2026-11-22/part-00000.parquet"


def test_sample_holds_the_submitted_record_byte_for_byte() -> None:
    raw = SAMPLE.read_bytes()
    assert raw.decode("utf-8") == (
        "dango0812, donggyu, 김동규 개발자\n"
        "-> dango0812는 대한민국의 개발자 김동규가 사용하는 GitHub 닉네임입니다.\n"
    )
    assert len(raw) == 135


def test_capture_keeps_the_bytes_and_derives_a_stable_id() -> None:
    capture = capture_file(SAMPLE, source_uri=SOURCE_URI, captured_at=CAPTURED_AT)
    assert capture.raw == SAMPLE.read_bytes()
    # Same id on every OS: .gitattributes keeps the sample's LF bytes on checkout.
    assert capture.capture_id == (
        "sha256:e5371e225b0465531181d8cb97e1d572fc1ca3a1d2a9963eae34623dc1d724d2"
    )
    assert capture.http_status is None
    assert capture.content_type == "text/plain; charset=utf-8"


def test_land_writes_one_bronze_part_for_the_capture_day(tmp_path: Path) -> None:
    capture = capture_file(SAMPLE, source_uri=SOURCE_URI, captured_at=CAPTURED_AT)
    key = land([capture], tmp_path, source="github.com")
    assert key == BRONZE_KEY
    assert from_table(read_table(tmp_path, key, schema=BRONZE_CAPTURES), RawCapture) == [capture]


def test_land_refuses_captures_from_different_days(tmp_path: Path) -> None:
    first = capture_file(SAMPLE, source_uri=SOURCE_URI, captured_at=CAPTURED_AT)
    second = capture_file(
        SAMPLE, source_uri=SOURCE_URI, captured_at=CAPTURED_AT + timedelta(days=1)
    )
    with pytest.raises(ValueError, match="exactly one UTC day"):
        land([first, second], tmp_path, source="github.com")


def test_cli_ingest_lands_the_sample_and_reports_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "--lake",
            str(tmp_path),
            "ingest",
            str(SAMPLE),
            "--source",
            "github.com",
            "--source-uri",
            SOURCE_URI,
            "--captured-at",
            "2026-11-22T22:13:00+09:00",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "bronze.ingest"
    assert report["records"] == 1
    assert report["bytes"] == 135
    assert report["outputs"] == [BRONZE_KEY]
    assert (tmp_path / BRONZE_KEY).is_file()


def test_cli_rejects_a_capture_time_without_offset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--lake",
                str(tmp_path),
                "ingest",
                str(SAMPLE),
                "--source",
                "github.com",
                "--source-uri",
                SOURCE_URI,
                "--captured-at",
                "2026-11-22T13:13:00",
            ]
        )
