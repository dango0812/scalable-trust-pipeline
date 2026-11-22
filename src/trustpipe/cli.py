"""Command line entry point: ``trustpipe <command> ...``.

Every command prints one JSON line describing the run (stage, records, outputs, counts,
duration), so runs can be compared and later collected as metrics. Ids are printed in
their text form ("sha256:<hex>").
"""

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustpipe.ids import to_text
from trustpipe.ingest import capture_file, land
from trustpipe.receipts import Receipt
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

    ingest = commands.add_parser(
        "ingest", help="land a handed-in file in bronze exactly as received"
    )
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source-uri", required=True, help="where the bytes say they came from")
    ingest.add_argument(
        "--captured-at",
        type=_utc_timestamp,
        default=None,
        help="capture time with offset (default: now); fix it to make runs reproducible",
    )

    refine = commands.add_parser(
        "refine", help="turn one bronze part into its silver documents, rejects and trust parts"
    )
    refine.add_argument("bronze_key", help="e.g. bronze/captures/date=.../part-<id>.parquet")
    refine.add_argument(
        "--registry", type=Path, default=Path("config/sources.toml"), help="source registry"
    )
    return parser


def _outputs(receipt: Receipt) -> list[dict[str, Any]]:
    return [{"key": f.key, "rows": f.rows, "bytes": f.bytes} for f in receipt.outputs]


def _ingest(args: argparse.Namespace) -> dict[str, Any]:
    capture = capture_file(
        args.path,
        source_uri=args.source_uri,
        captured_at=args.captured_at or datetime.now(UTC),
    )
    receipt = land([capture], args.lake)
    return {
        "stage": receipt.stage,
        "records": receipt.counters["captures"],
        "payload_bytes": receipt.counters["payload_bytes"],
        "capture_id": to_text(capture.capture_id),
        "part_id": receipt.part_id,
        "outputs": _outputs(receipt),
        "receipt": receipt.key,
    }


def _refine(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    receipt = refine_part(args.lake, args.bronze_key, registry)
    return {
        "stage": receipt.stage,
        "records": receipt.counters["documents"] + receipt.counters["rejects"],
        "counters": receipt.counters,
        "policy_id": registry.policy_id,
        "part_id": receipt.part_id,
        "outputs": _outputs(receipt),
        "receipt": receipt.key,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    report = _ingest(args) if args.command == "ingest" else _refine(args)
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(report, ensure_ascii=False))
    return 0
