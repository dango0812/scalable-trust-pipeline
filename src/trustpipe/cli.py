"""Command line entry point: ``trustpipe <command> ...``.

Every command prints one JSON line describing the run (stage, records, outputs, counts,
duration), so runs can be compared and later collected as metrics. Ids are printed in
their text form ("sha256:<hex>").
"""

import argparse
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trustpipe.archive.datasets import DatasetFields
from trustpipe.archive.importer import import_dataset, import_warc
from trustpipe.bronze.ingest import capture_file, land
from trustpipe.identity.ids import to_text
from trustpipe.lake.receipts import Receipt
from trustpipe.lake.schema import Acquisition
from trustpipe.silver.refine import refine_part
from trustpipe.trust.registry import load_registry


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

    warc = commands.add_parser("import-warc", help="land the 2xx responses of a WARC file")
    warc.add_argument("path", type=Path)
    warc.add_argument(
        "--acquisition",
        choices=[Acquisition.ARCHIVE_IMPORT.value, Acquisition.HTTP_FETCH.value],
        default=Acquisition.ARCHIVE_IMPORT.value,
        help="http_fetch for your own crawler's files, archive_import for third-party archives",
    )

    dataset = commands.add_parser(
        "import-dataset", help="land the rows of a JSON Lines or Parquet dataset file"
    )
    dataset.add_argument("path", type=Path)
    dataset.add_argument("--text-field", default="text")
    dataset.add_argument("--url-field", default="url")
    dataset.add_argument("--date-field", default=None, help="crawl date field, if any")
    dataset.add_argument(
        "--default-date",
        type=_utc_timestamp,
        default=None,
        help="snapshot date with offset for rows without a date",
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


def _import_warc(args: argparse.Namespace) -> dict[str, Any]:
    report = import_warc(args.path, args.lake, acquisition=Acquisition(args.acquisition))
    return {"stage": "bronze.ingest", **report.document()}


def _import_dataset(args: argparse.Namespace) -> dict[str, Any]:
    fields = DatasetFields(args.text_field, args.url_field, args.date_field)
    report = import_dataset(args.path, args.lake, fields, default_date=args.default_date)
    return {"stage": "bronze.ingest", **report.document()}


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


COMMANDS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "ingest": _ingest,
    "import-warc": _import_warc,
    "import-dataset": _import_dataset,
    "refine": _refine,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    report = COMMANDS[args.command](args)
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(report, ensure_ascii=False))
    return 0
