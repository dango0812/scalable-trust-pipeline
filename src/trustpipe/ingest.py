"""Bronze ingestion: land source bytes exactly as they arrived.

Ingestion never parses, cleans or judges. It records what arrived, from where and when,
so every later stage can be re-run from bronze without contacting the source again.
"""

from datetime import UTC, datetime
from pathlib import Path

from trustpipe.layout import bronze_captures_key
from trustpipe.schema import RawCapture, to_table
from trustpipe.storage import write_table
from trustpipe.uris import canonical_uri

TEXT_UTF8 = "text/plain; charset=utf-8"


def capture_file(
    path: Path, *, source_uri: str, captured_at: datetime, content_type: str = TEXT_UTF8
) -> RawCapture:
    """Capture a submitted local file byte for byte (no HTTP fetch, so no status code).

    The source URI is stored in canonical form, so every later comparison (registry
    rules, deduplication by source) sees one spelling per location.
    """
    return RawCapture.from_bytes(
        path.read_bytes(),
        source_uri=canonical_uri(source_uri),
        fetched_at=captured_at,
        content_type=content_type,
    )


def land(captures: list[RawCapture], root: Path, *, source: str, shard: int = 0) -> str:
    """Write captures as one bronze part and return its key.

    A part holds captures from a single UTC day, matching its ``date=`` partition.
    """
    days = {c.fetched_at.astimezone(UTC).date() for c in captures}
    if len(days) != 1:
        raise ValueError(f"a bronze part holds captures from exactly one UTC day, got {days}")
    key = bronze_captures_key(source, days.pop(), shard)
    write_table(to_table(captures, RawCapture), root, key)
    return key
