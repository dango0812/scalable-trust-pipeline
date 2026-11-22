"""Bronze ingestion: land source bytes exactly as they arrived, exactly once.

Ingestion never parses, cleans or judges. It records what arrived, from where, when and how
it was acquired, so every later stage can be re-run from bronze without contacting the
source again.

A bronze part is written create-only under a key derived from its content and then
committed by its receipt. Landing the same batch again is a no-op that returns the existing
receipt; landing a different batch can never touch an existing part. If a previous attempt
wrote the data file but crashed before its receipt, the next attempt checks that the file
holds exactly the same captures and only adds the receipt.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from trustpipe import __version__
from trustpipe.layout import bronze_captures_key, part_id
from trustpipe.receipts import Receipt, commit, is_committed, load_receipt
from trustpipe.schema import BRONZE_CAPTURES, Acquisition, RawCapture, to_table
from trustpipe.storage import WriteConflictError, WrittenFile, describe, read_table, write_table

STAGE = "bronze.ingest"
TEXT_UTF8 = "text/plain; charset=utf-8"


def capture_file(
    path: Path, *, source_uri: str, captured_at: datetime, content_type: str = TEXT_UTF8
) -> RawCapture:
    """Capture a handed-in file byte for byte.

    Its source URI is recorded as a claim: the acquisition method is "submission", and only
    registry rules that accept submissions give it more than the "unverified" tier.
    """
    return RawCapture.from_bytes(
        path.read_bytes(),
        source_uri=source_uri,
        fetched_at=captured_at,
        acquisition=Acquisition.SUBMISSION,
        content_type=content_type,
    )


def _write_once(root: Path, key: str, captures: Sequence[RawCapture]) -> WrittenFile:
    table = to_table(captures, RawCapture)
    try:
        return write_table(table, root, key, create_only=True)
    except WriteConflictError:
        # The key is derived from the ordered capture ids, so an existing file at the same
        # key normally holds the same captures (an attempt that crashed before its receipt,
        # or bytes written by another library version). Accept it only if it really does.
        existing = read_table(root, key, schema=BRONZE_CAPTURES, columns=["capture_id"])
        if not existing.column("capture_id").equals(table.column("capture_id")):
            raise
        return describe(root, key)


def land(captures: Sequence[RawCapture], root: Path) -> Receipt:
    """Write captures as one bronze part, commit it and return its receipt."""
    if not captures:
        raise ValueError("there is nothing to land")
    days = {capture.fetched_at.date() for capture in captures}
    if len(days) != 1:
        raise ValueError(f"a bronze part holds captures from exactly one UTC day, got {days}")
    ids = [capture.capture_id for capture in captures]
    if len(set(ids)) != len(ids):
        raise ValueError("a bronze part cannot hold the same capture twice")
    part = part_id(ids)
    if is_committed(root, STAGE, part):
        return load_receipt(root, STAGE, part)
    written = _write_once(root, bronze_captures_key(days.pop(), part), captures)
    receipt = Receipt(
        stage=STAGE,
        part_id=part,
        inputs=(),
        outputs=(written,),
        transform={"trustpipe": __version__},
        counters={"captures": len(captures), "payload_bytes": sum(len(c.raw) for c in captures)},
    )
    commit(root, receipt, create_only=True)
    return receipt
