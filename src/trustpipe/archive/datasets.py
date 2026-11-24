"""Read published datasets (JSON Lines or Parquet) as bronze captures.

Public corpora ship as files of already extracted text with the page URL and, usually, the
crawl date: Hugging Face datasets as Parquet (FineWeb), others as compressed JSON Lines
(Dolma ``.json.gz``, DCLM ``.jsonl.zst``). Each row becomes one capture whose bytes are the
row's text in UTF-8, acquired by ``archive_import``: the pipeline did not fetch the page,
so a trust rule has to accept archive imports explicitly before the URI earns a tier.

Only the three mapped fields are read (``DatasetFields``); other fields are ignored, and
JSON Lines files are parsed by Arrow's multithreaded reader in blocks, so a file of any
size streams through in bounded memory. Compression is detected from the file name
(``.gz``, ``.bz2``, ``.zst``, ``.lz4``, ``.br``).

A row is skipped, with a counted reason, when its text or URL is missing, its URL has no
canonical form, or its date does not parse. Rows without a date field take the import's
``default_date``: the snapshot date the publisher documents for the dataset.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, cast

import pyarrow as pa
import pyarrow.json as pj
import pyarrow.parquet as pq

from trustpipe.archive.warc import Skip
from trustpipe.lake.schema import Acquisition, RawCapture

TEXT_UTF8 = "text/plain; charset=utf-8"
_BLOCK_BYTES = 16 * 1024 * 1024
_PARQUET = (".parquet", ".pq")


@dataclass(frozen=True, slots=True)
class DatasetFields:
    """Which fields of a dataset row hold the text, the page URL and the crawl date."""

    text: str = "text"
    url: str = "url"
    date: str | None = None


def _batches(path: Path, fields: DatasetFields) -> Iterator[pa.RecordBatch]:
    names = [fields.text, fields.url] + ([fields.date] if fields.date else [])
    if path.name.lower().endswith(_PARQUET):
        yield from pq.ParquetFile(path).iter_batches(batch_size=10_000, columns=names)
        return
    schema = pa.schema([(fields.text, pa.string()), (fields.url, pa.string())])
    if fields.date:
        # JSON has no date type: read the field as text and parse it per row.
        schema = schema.append(pa.field(fields.date, pa.string()))
    options = pj.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore")
    with pa.input_stream(str(path), compression="detect") as stream:
        yield from pj.open_json(
            cast(IO[bytes], stream),
            read_options=pj.ReadOptions(block_size=_BLOCK_BYTES),
            parse_options=options,
        )


def _date(value: Any, default: datetime | None) -> datetime:
    if value is None:
        if default is None:
            raise Skip("missing_date")
        return default
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value))
        except ValueError:
            raise Skip("invalid_date", str(value)[:40]) from None
    if moment.tzinfo is None:  # a dataset date without offset is taken as UTC
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def iter_dataset(
    path: Path,
    fields: DatasetFields,
    *,
    default_date: datetime | None = None,
    acquisition: Acquisition = Acquisition.ARCHIVE_IMPORT,
) -> Iterator[RawCapture | Skip]:
    """Yield one capture, or the reason it was skipped, per dataset row, in file order."""
    for batch in _batches(path, fields):
        texts = batch.column(fields.text).to_pylist()
        urls = batch.column(fields.url).to_pylist()
        dates = batch.column(fields.date).to_pylist() if fields.date else [None] * len(texts)
        for text, url, date in zip(texts, urls, dates, strict=True):
            if text is None:
                yield Skip("missing_text")
            elif url is None:
                yield Skip("missing_url")
            else:
                try:
                    fetched_at = _date(date, default_date)
                    yield RawCapture.from_bytes(
                        str(text).encode("utf-8"),
                        source_uri=str(url),
                        fetched_at=fetched_at,
                        acquisition=acquisition,
                        content_type=TEXT_UTF8,
                    )
                except Skip as skip:
                    yield skip
                except ValueError as error:
                    yield Skip("invalid_uri", str(error)[:200])
