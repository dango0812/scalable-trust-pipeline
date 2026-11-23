"""Where each table's files live in the lake.

Keys are POSIX paths relative to the lake root, so the same keys work on a local disk now
and on an object store (``s3://bucket/lake/<key>``) later. Partitions use the Hive
``key=value`` convention that pyarrow, Spark, DuckDB and Trino read natively.

    bronze/captures/date=<yyyy-mm-dd>/part-<id>.parquet
    silver/documents/date=<yyyy-mm-dd>/part-<id>.parquet
    silver/rejects/date=<yyyy-mm-dd>/part-<id>.parquet
    silver/trust/policy=<policy>/date=<yyyy-mm-dd>/part-<id>.parquet
    _receipts/<stage>/<id>.json        commit record of one part (receipts.py)
    _policies/<sha256 hex>.json        every policy that ever produced trust rows
    _runs/<run id>/                    run logs (runlog.py)

The only data partition is the UTC day a capture was fetched: a bounded number of
directories, however many sources there are. Sources are a column (``host``), not a
directory, so a web-scale long tail of hosts never turns into millions of tiny files.

Scaling is by adding parts, never by growing a file: one record is one row in one part, a
trillion records are many parts read and written in parallel. A bronze part's id is derived
from the ordered capture ids it holds, so re-running the same batch produces the same key
and two different batches can never collide. Every silver part carries the id of the bronze
part it was refined from, so the lineage of each file is in its name.

Names starting with "_" are skipped by engines that read Hive-partitioned tables.
"""

import re
from collections.abc import Sequence
from datetime import date
from enum import StrEnum

from trustpipe.identity.ids import typed_hash

_PART_ID = re.compile(r"[0-9a-f]{16}")
_STAGE = re.compile(r"[a-z]+(?:\.[a-z]+)*")
_POLICY_ID = re.compile(r"sha256:([0-9a-f]{64})")
_BRONZE_KEY = re.compile(
    r"bronze/captures/date=(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2})/part-(?P<part>[0-9a-f]{16})\.parquet"
)


class Layer(StrEnum):
    BRONZE = "bronze"  # as received, written once
    SILVER = "silver"  # canonical text, rejects and trust decisions
    GOLD = "gold"  # training-ready shards


def part_id(capture_ids: Sequence[bytes]) -> str:
    """Content-derived id of a part: 64 bits of a typed hash over its ordered capture ids."""
    if not capture_ids:
        raise ValueError("a part holds at least one capture")
    return typed_hash("part", *capture_ids).hex()[:16]


def _part(part: str) -> str:
    if not _PART_ID.fullmatch(part):
        raise ValueError(f"part id must be 16 lower-case hex digits, got {part!r}")
    return f"part-{part}.parquet"


def _policy_hex(policy_id: str) -> str:
    match = _POLICY_ID.fullmatch(policy_id)
    if match is None:
        raise ValueError(f"policy id must look like sha256:<64 hex digits>, got {policy_id!r}")
    return match[1]


def bronze_captures_key(day: date, part: str) -> str:
    return f"{Layer.BRONZE}/captures/date={day.isoformat()}/{_part(part)}"


def silver_documents_key(day: date, part: str) -> str:
    return f"{Layer.SILVER}/documents/date={day.isoformat()}/{_part(part)}"


def silver_rejects_key(day: date, part: str) -> str:
    return f"{Layer.SILVER}/rejects/date={day.isoformat()}/{_part(part)}"


def silver_trust_key(policy_id: str, day: date, part: str) -> str:
    # 12 hex digits (48 bits) keep paths short; the full id is in every row and receipt.
    policy = _policy_hex(policy_id)[:12]
    return f"{Layer.SILVER}/trust/policy={policy}/date={day.isoformat()}/{_part(part)}"


def receipt_key(stage: str, part: str) -> str:
    if not _STAGE.fullmatch(stage):
        raise ValueError(f"stage must be dotted lower-case words, got {stage!r}")
    _part(part)
    return f"_receipts/{stage}/{part}.json"


def policy_key(policy_id: str) -> str:
    return f"_policies/{_policy_hex(policy_id)}.json"


def parse_bronze_captures_key(key: str) -> tuple[date, str]:
    """Split a bronze captures key into (day, part id); the exact inverse of the builder."""
    match = _BRONZE_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"not a bronze captures key: {key!r}")
    # The pattern admits exactly the spelling bronze_captures_key produces.
    return date.fromisoformat(match["day"]), match["part"]
