"""Command line entry point: ``trustpipe <command> ...``.

Every command prints one JSON line describing the run (stage, records, inputs, outputs,
duration and stage-specific counts), so runs can be compared and later collected as metrics.
"""

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustpipe.ingest import capture_file, land
from trustpipe.refine import refine_part
from trustpipe.registry import load_registry


def _utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"timestamp needs a UTC offset, got {value!r}")
    return parsed.astimezone(UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trustpipe")
    parser.add_argument("--lake", type=Path, default=Path("lake"), help="lake root directory")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="land a local file in bronze exactly as received")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", required=True, help="source partition, e.g. github.com")
    ingest.add_argument("--source-uri", required=True, help="where the bytes came from")
    ingest.add_argument(
        "--captured-at",
        type=_utc_timestamp,
        default=None,
        help="capture time with offset (default: now); fix it to make runs reproducible",
    )

    refine = commands.add_parser(
        "refine", help="turn one bronze part into its silver documents and trust parts"
    )
    refine.add_argument(
        "bronze_key", help="e.g. bronze/captures/source=.../date=.../part-00000.parquet"
    )
    refine.add_argument(
        "--registry", type=Path, default=Path("config/sources.toml"), help="source registry"
    )
    return parser


def _ingest(args: argparse.Namespace) -> dict[str, Any]:
    capture = capture_file(
        args.path,
        source_uri=args.source_uri,
        captured_at=args.captured_at or datetime.now(UTC),
    )
    key = land([capture], args.lake, source=args.source)
    return {
        "stage": "bronze.ingest",
        "records": 1,
        "bytes": len(capture.raw),
        "capture_id": capture.capture_id,
        "outputs": [key],
    }


def _refine(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    trust, documents_key, trust_key = refine_part(args.lake, args.bronze_key, registry)
    return {
        "stage": "silver.refine",
        "records": len(trust),
        "inputs": [args.bronze_key],
        "outputs": [documents_key, trust_key],
        "tiers": dict(Counter(record.tier.value for record in trust)),
        "policy_version": registry.version,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    report = _ingest(args) if args.command == "ingest" else _refine(args)
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
