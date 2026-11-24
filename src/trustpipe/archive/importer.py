"""Land the captures of an archive file in bronze, part by part.

An archive file can hold millions of captures from several UTC days, more than one part
should hold and more than memory should. ``PartBuffer`` keeps one buffer per day and lands
a part (``bronze.ingest.land``) as soon as a buffer reaches its row or byte limit, so
memory stays bounded by the limits, whatever the file size.

Reading is sequential and parts are content-addressed, so importing the same file again
produces the same parts and lands nothing new; importing an overlapping file lands only
the parts that differ. A capture that repeats inside one buffer (same URI, moment and
bytes) is counted as a duplicate and landed once.

Every receipt names the archive it came from (file name and SHA-256) in its transform, and
the import report counts every record that did not become a capture, by reason.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from trustpipe.archive.datasets import DatasetFields, iter_dataset
from trustpipe.archive.warc import Skip, WarcFormatError, capture_from_record, iter_records
from trustpipe.archive.warc import open_archive as open_warc
from trustpipe.bronze.ingest import land
from trustpipe.lake.receipts import Receipt
from trustpipe.lake.schema import Acquisition, RawCapture
from trustpipe.lake.storage import file_sha256

MAX_PART_ROWS = 50_000
MAX_PART_BYTES = 256 * 1024 * 1024


@dataclass(slots=True)
class ImportReport:
    source: str
    source_sha256: str
    records: int = 0
    landed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    receipts: list[Receipt] = field(default_factory=list)

    def document(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_sha256": self.source_sha256,
            "records": self.records,
            "landed": self.landed,
            "skipped": dict(sorted(self.skipped.items())),
            "parts": [receipt.part_id for receipt in self.receipts],
        }


class PartBuffer:
    """Per-day buffers that land a bronze part whenever one is full."""

    def __init__(
        self,
        root: Path,
        report: ImportReport,
        *,
        max_rows: int = MAX_PART_ROWS,
        max_bytes: int = MAX_PART_BYTES,
    ) -> None:
        if max_rows < 1 or max_bytes < 1:
            raise ValueError("part limits must be positive")
        self.root = root
        self.report = report
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.source = {"source": report.source, "source_sha256": report.source_sha256}
        self._days: dict[date, list[RawCapture]] = {}
        self._sizes: dict[date, int] = {}
        self._seen: dict[date, set[bytes]] = {}

    def add(self, capture: RawCapture) -> None:
        day = capture.fetched_at.date()
        seen = self._seen.setdefault(day, set())
        if capture.capture_id in seen:
            self.report.skipped["duplicate_capture"] += 1
            return
        seen.add(capture.capture_id)
        self._days.setdefault(day, []).append(capture)
        self._sizes[day] = self._sizes.get(day, 0) + len(capture.raw)
        if len(self._days[day]) >= self.max_rows or self._sizes[day] >= self.max_bytes:
            self._land(day)

    def _land(self, day: date) -> None:
        captures = self._days.pop(day)
        del self._sizes[day], self._seen[day]
        self.report.receipts.append(land(captures, self.root, source=self.source))
        self.report.landed += len(captures)

    def close(self) -> None:
        for day in sorted(self._days):
            self._land(day)


def _consume(items: Iterable[RawCapture | Skip], buffer: PartBuffer) -> None:
    for item in items:
        buffer.report.records += 1
        if isinstance(item, Skip):
            buffer.report.skipped[item.reason] += 1
        else:
            buffer.add(item)
    buffer.close()


def _warc_captures(path: Path, acquisition: Acquisition) -> Iterable[RawCapture | Skip]:
    with open_warc(path) as stream:
        for record in iter_records(stream):
            try:
                yield capture_from_record(record, acquisition)
            except Skip as skip:
                yield skip


def import_warc(
    path: Path,
    root: Path,
    *,
    acquisition: Acquisition = Acquisition.ARCHIVE_IMPORT,
    max_part_rows: int = MAX_PART_ROWS,
    max_part_bytes: int = MAX_PART_BYTES,
) -> ImportReport:
    """Land every 2xx response of a WARC file; raise WarcFormatError on a corrupt file.

    ``acquisition`` is ``http_fetch`` for files written by the operator's own crawler and
    ``archive_import`` (the default) for third-party archives such as Common Crawl.
    """
    report = ImportReport(path.name, file_sha256(path))
    buffer = PartBuffer(root, report, max_rows=max_part_rows, max_bytes=max_part_bytes)
    try:
        _consume(_warc_captures(path, acquisition), buffer)
    except (EOFError, OSError) as error:  # gzip reports a cut-off file this way
        raise WarcFormatError(f"{path.name} is truncated or corrupt: {error}") from None
    return report


def import_dataset(
    path: Path,
    root: Path,
    fields: DatasetFields,
    *,
    default_date: datetime | None = None,
    acquisition: Acquisition = Acquisition.ARCHIVE_IMPORT,
    max_part_rows: int = MAX_PART_ROWS,
    max_part_bytes: int = MAX_PART_BYTES,
) -> ImportReport:
    """Land every usable row of a JSON Lines or Parquet dataset file (see datasets.py)."""
    report = ImportReport(path.name, file_sha256(path))
    buffer = PartBuffer(root, report, max_rows=max_part_rows, max_bytes=max_part_bytes)
    rows = iter_dataset(path, fields, default_date=default_date, acquisition=acquisition)
    _consume(rows, buffer)
    return report
