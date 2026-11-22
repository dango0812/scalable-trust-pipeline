"""Where each layer's files live in the lake.

Keys are POSIX paths relative to the lake root, so the same keys work on a local disk now
and on an object store (``s3://bucket/lake/<key>``) later. Partitions use the Hive
``key=value`` convention that pyarrow, Spark, DuckDB and Trino read natively.

    bronze/captures/source=<domain>/date=<yyyy-mm-dd>/part-<n>.parquet
    silver/documents/source=<domain>/date=<yyyy-mm-dd>/part-<n>.parquet
    silver/trust/source=<domain>/date=<yyyy-mm-dd>/part-<n>.parquet
    gold/...                                   training shards (added by a later layer)

Scaling is by adding parts, never by growing a file: one record is one row in part-00000,
a trillion records are many parts across many partitions, read and written in parallel.
Each silver part mirrors exactly one bronze part (same source, day and part number), so
refining a bronze part is idempotent: a re-run rewrites only that part's silver files.
A documents part and its trust part hold the same documents in the same row order, so the
two join without a shuffle.
"""

import re
from datetime import date
from enum import StrEnum

_PARTITION_VALUE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_BRONZE_KEY = re.compile(
    r"bronze/captures/source=(?P<source>[^/]+)/date=(?P<day>\d{4}-\d{2}-\d{2})"
    r"/part-(?P<shard>\d{5,})\.parquet"
)


class Layer(StrEnum):
    BRONZE = "bronze"  # as received, immutable
    SILVER = "silver"  # canonical text and trust metadata
    GOLD = "gold"  # training-ready shards


def part_name(shard: int) -> str:
    if shard < 0:
        raise ValueError(f"shard must be >= 0, got {shard}")
    return f"part-{shard:05d}.parquet"


def _partition(key: str, value: str) -> str:
    if not _PARTITION_VALUE.fullmatch(value):
        raise ValueError(f"partition value {value!r} must be lower-case [a-z0-9.-]")
    return f"{key}={value}"


def _key(layer: Layer, table: str, source: str, day: date, shard: int) -> str:
    partitions = [_partition("source", source), _partition("date", day.isoformat())]
    return "/".join([layer, table, *partitions, part_name(shard)])


def bronze_captures_key(source: str, day: date, shard: int) -> str:
    return _key(Layer.BRONZE, "captures", source, day, shard)


def silver_documents_key(source: str, day: date, shard: int) -> str:
    return _key(Layer.SILVER, "documents", source, day, shard)


def silver_trust_key(source: str, day: date, shard: int) -> str:
    return _key(Layer.SILVER, "trust", source, day, shard)


def parse_bronze_captures_key(key: str) -> tuple[str, date, int]:
    """Split a bronze captures key into (source, day, shard)."""
    match = _BRONZE_KEY.fullmatch(key)
    if match is None:
        raise ValueError(f"not a bronze captures key: {key!r}")
    return match["source"], date.fromisoformat(match["day"]), int(match["shard"])
