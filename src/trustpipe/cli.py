"""Command line entry point: ``trustpipe <command> ...``.

Every command prints one JSON line describing the run (stage, records, bytes, output
keys, duration), so runs can be compared and later collected as metrics.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from trustpipe.ingest import capture_file, land


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()

    capture = capture_file(
        args.path,
        source_uri=args.source_uri,
        captured_at=args.captured_at or datetime.now(UTC),
    )
    key = land([capture], args.lake, source=args.source)

    report = {
        "stage": "bronze.ingest",
        "records": 1,
        "bytes": len(capture.raw),
        "capture_id": capture.capture_id,
        "outputs": [key],
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
