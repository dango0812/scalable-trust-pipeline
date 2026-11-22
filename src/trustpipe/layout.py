"""Where each layer's files live in the lake.

Keys are POSIX paths relative to the lake root, so the same keys work on a local disk now
and on an object store (``s3://bucket/lake/<key>``) later. Partitions use the Hive
``key=value`` convention that pyarrow, Spark, DuckDB and Trino read natively.

    bronze/captures/source=<domain>/date=<yyyy-mm-dd>/part-<n>.parquet
    silver/documents/source=<domain>/part-<n>.parquet
    silver/trust/source=<domain>/part-<n>.parquet
    gold/...                                   training shards (added by a later layer)

Scaling is by adding shards, never by growing a file: one record is one row in part-00000,
a trillion records are many parts across many partitions, read and written in parallel.
A silver documents part and its trust part share partition and part number, so the two
files hold the same documents in the same row order and join without a shuffle.
"""

import re
from datetime import date
from enum import StrEnum

_PARTITION_VALUE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


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


def bronze_captures_key(source: str, day: date, shard: int) -> str:
    return "/".join(
        [
            Layer.BRONZE,
            "captures",
            _partition("source", source),
            _partition("date", day.isoformat()),
            part_name(shard),
        ]
    )


def silver_documents_key(source: str, shard: int) -> str:
    return "/".join([Layer.SILVER, "documents", _partition("source", source), part_name(shard)])


def silver_trust_key(source: str, shard: int) -> str:
    return "/".join([Layer.SILVER, "trust", _partition("source", source), part_name(shard)])
