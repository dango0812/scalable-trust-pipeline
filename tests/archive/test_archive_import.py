import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trustpipe.archive.datasets import DatasetFields, iter_dataset
from trustpipe.archive.importer import ImportReport, PartBuffer, import_dataset, import_warc
from trustpipe.archive.warc import Skip, WarcFormatError, response_record
from trustpipe.bronze.ingest import STAGE
from trustpipe.cli import main
from trustpipe.lake.receipts import committed_parts, load_receipt
from trustpipe.lake.schema import BRONZE_CAPTURES, Acquisition, RawCapture
from trustpipe.lake.storage import read_table

AT = datetime(2026, 11, 24, 23, 59, tzinfo=UTC)
PAGE = b"<html><body><p>page</p></body></html>"


def crawl(path: Path, count: int, *, start: datetime = AT) -> Path:
    records = [
        response_record(f"https://example.org/{n}", start + timedelta(seconds=n), PAGE + b"%d" % n)
        for n in range(count)
    ]
    path.write_bytes(b"".join(records))
    return path


def test_a_crawl_lands_one_part_per_day_with_its_source_in_the_receipt(tmp_path: Path) -> None:
    warc = crawl(tmp_path / "crawl.warc.gz", 3)  # 23:59:00, :01, :02 on one day
    lake = tmp_path / "lake"
    report = import_warc(warc, lake, acquisition=Acquisition.HTTP_FETCH)
    assert (report.records, report.landed, report.skipped) == (3, 3, {})
    [receipt] = report.receipts
    assert receipt.transform["source"] == "crawl.warc.gz"
    assert receipt.transform["source_sha256"] == report.source_sha256
    table = read_table(lake, receipt.outputs[0].key, schema=BRONZE_CAPTURES)
    assert table.column("acquisition").to_pylist() == ["http_fetch"] * 3
    assert table.column("http_status").to_pylist() == [200] * 3


def test_importing_the_same_file_again_lands_nothing_new(tmp_path: Path) -> None:
    warc = crawl(tmp_path / "crawl.warc.gz", 2)
    first = import_warc(warc, tmp_path / "lake")
    again = import_warc(warc, tmp_path / "lake")
    assert first.receipts == again.receipts
    assert len(committed_parts(tmp_path / "lake", STAGE)) == 1


def test_records_that_are_not_captures_are_counted_by_reason(tmp_path: Path) -> None:
    warc = tmp_path / "mixed.warc.gz"
    ok = response_record("https://example.org/ok", AT, PAGE)
    missing = response_record("https://example.org/missing", AT, b"gone", status=404)
    info = gzip.compress(b"WARC/1.1\r\nWARC-Type: warcinfo\r\nContent-Length: 0\r\n\r\n\r\n\r\n")
    warc.write_bytes(info + ok + missing + ok)
    report = import_warc(warc, tmp_path / "lake")
    assert report.records == 4
    assert report.landed == 1
    assert report.skipped == {"record_warcinfo": 1, "http_status": 1, "duplicate_capture": 1}
    assert report.document()["skipped"] == {
        "duplicate_capture": 1,
        "http_status": 1,
        "record_warcinfo": 1,
    }


def test_a_cut_off_archive_is_refused(tmp_path: Path) -> None:
    warc = crawl(tmp_path / "crawl.warc.gz", 2)
    warc.write_bytes(warc.read_bytes()[:-10])
    with pytest.raises(WarcFormatError, match="truncated or corrupt"):
        import_warc(warc, tmp_path / "lake")


def test_parts_are_cut_by_rows_and_by_bytes_and_by_day(tmp_path: Path) -> None:
    warc = crawl(tmp_path / "crawl.warc.gz", 90)  # crosses midnight after 60 records
    by_rows = import_warc(warc, tmp_path / "rows", max_part_rows=25)
    assert [r.counters["captures"] for r in by_rows.receipts] == [25, 25, 25, 10, 5]
    by_bytes = import_warc(warc, tmp_path / "bytes", max_part_bytes=len(PAGE) * 30)
    assert sum(r.counters["captures"] for r in by_bytes.receipts) == 90
    assert all(r.counters["payload_bytes"] <= len(PAGE) * 32 for r in by_bytes.receipts)


def test_part_limits_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        PartBuffer(tmp_path, ImportReport("x", "y"), max_rows=0)


def write_jsonl(path: Path, rows: list[dict[str, object]], *, compress: bool = False) -> Path:
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(gzip.compress(data) if compress else data)
    return path


ROWS: list[dict[str, object]] = [
    {"text": "첫 문서", "url": "https://example.org/1", "date": "2024-05-01T10:00:00Z", "x": 1},
    {"text": "naive date", "url": "https://example.org/2", "date": "2024-05-01 11:00:00"},
    {"text": "no date", "url": "https://example.org/3"},
    {"text": None, "url": "https://example.org/4"},
    {"text": "no url"},
    {"text": "bad url", "url": "mailto:someone@example.org"},
    {"text": "bad date", "url": "https://example.org/7", "date": "May 1st"},
]


def test_jsonl_rows_become_archive_captures(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl.gz", ROWS, compress=True)
    fields = DatasetFields(date="date")
    default = datetime(2024, 1, 1, tzinfo=UTC)
    items = list(iter_dataset(path, fields, default_date=default))
    captures = [i for i in items if isinstance(i, RawCapture)]
    assert [c.raw.decode() for c in captures] == ["첫 문서", "naive date", "no date"]
    assert [c.fetched_at for c in captures] == [
        datetime(2024, 5, 1, 10, tzinfo=UTC),
        datetime(2024, 5, 1, 11, tzinfo=UTC),
        default,
    ]
    assert {c.acquisition for c in captures} == {Acquisition.ARCHIVE_IMPORT}
    assert [i.reason for i in items if isinstance(i, Skip)] == [
        "missing_text",
        "missing_url",
        "invalid_uri",
        "invalid_date",
    ]


def test_rows_without_any_date_need_a_default(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "data.jsonl", [{"text": "t", "url": "https://example.org/"}])
    [item] = list(iter_dataset(path, DatasetFields(date="date")))
    assert isinstance(item, Skip)
    assert item.reason == "missing_date"
    [capture] = list(iter_dataset(path, DatasetFields(), default_date=AT))
    assert isinstance(capture, RawCapture)


def test_parquet_datasets_with_timestamp_columns(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    when = datetime(2023, 3, 4, 5, 6, tzinfo=UTC)
    table = pa.table(
        {
            "content": ["a parquet document"],
            "link": ["https://example.org/p"],
            "crawled": pa.array([when], pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(table, path)
    report = import_dataset(path, tmp_path / "lake", DatasetFields("content", "link", "crawled"))
    assert (report.records, report.landed) == (1, 1)
    stored = read_table(
        tmp_path / "lake", report.receipts[0].outputs[0].key, schema=BRONZE_CAPTURES
    )
    assert stored.column("fetched_at").to_pylist() == [when]
    assert stored.column("content_type").to_pylist() == ["text/plain; charset=utf-8"]


def test_cli_imports_warc_and_dataset_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    warc = crawl(tmp_path / "crawl.warc.gz", 2)
    lake = str(tmp_path / "lake")
    assert main(["--lake", lake, "import-warc", str(warc), "--acquisition", "http_fetch"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["stage"], report["records"], report["landed"]) == ("bronze.ingest", 2, 2)
    part = report["parts"][0]
    assert load_receipt(tmp_path / "lake", STAGE, part).transform["source"] == "crawl.warc.gz"

    data = write_jsonl(tmp_path / "data.jsonl", [{"body": "t", "link": "https://example.org/"}])
    args = ["--text-field", "body", "--url-field", "link", "--default-date", "2024-01-01T00:00Z"]
    assert main(["--lake", lake, "import-dataset", str(data), *args]) == 0
    report = json.loads(capsys.readouterr().out)
    assert (report["records"], report["landed"], report["skipped"]) == (1, 1, {})
