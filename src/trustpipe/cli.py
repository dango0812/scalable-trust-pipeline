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
from trustpipe.gold.policy import load_gold_policy
from trustpipe.gold.stages import build_gold
from trustpipe.gold.tokens import tokenize_snapshot
from trustpipe.identity.ids import to_text
from trustpipe.lake.receipts import Receipt
from trustpipe.lake.schema import Acquisition
from trustpipe.runtime.executor import RunSummary
from trustpipe.runtime.runlog import RunLog
from trustpipe.silver.refine import refine_all, refine_part
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
    everything = commands.add_parser(
        "refine-all", help="refine every bronze part not yet refined under the registry"
    )
    everything.add_argument("--registry", type=Path, default=Path("config/sources.toml"))
    everything.add_argument("--workers", type=int, default=1)

    gold = commands.add_parser(
        "gold", help="build a gold snapshot: trust and quality filters, dedup, chunks"
    )
    gold.add_argument("--policy", type=Path, default=Path("config/gold.toml"))
    gold.add_argument("--workers", type=int, default=1)

    tokens = commands.add_parser("tokens", help="tokenize a gold snapshot into training files")
    tokens.add_argument("snapshot", help="snapshot id printed by the gold command")
    tokens.add_argument("--tokenizer", type=Path, required=True, help="a tokenizer.json file")
    tokens.add_argument("--eos", required=True, help="end-of-text token, e.g. <|endoftext|>")
    tokens.add_argument("--workers", type=int, default=1)
    return parser


def _summary(summary: RunSummary) -> dict[str, Any]:
    return {
        "parts": summary.parts,
        "ok": summary.ok,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "rows_in": summary.rows_in,
        "rows_out": summary.rows_out,
        "counters": summary.counters,
    }


def _refine_all(args: argparse.Namespace) -> dict[str, Any]:
    registry = load_registry(args.registry)
    with RunLog(args.lake, "silver.refine") as log:
        summary = refine_all(args.lake, registry, workers=args.workers, on_event=log.record)
        log.finish(summary)
    return {"stage": "silver.refine", "policy_id": registry.policy_id, **_summary(summary)}


def _gold(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_gold_policy(args.policy)
    with RunLog(args.lake, "gold") as log:
        snapshot = build_gold(args.lake, policy, workers=args.workers, on_event=log.record)
        log.finish(snapshot.build)
    return {
        "stage": "gold",
        "snapshot": snapshot.snapshot,
        "manifest": snapshot.manifest_key,
        "gold_policy_id": policy.policy_id,
        "documents": snapshot.documents,
        "chunks": snapshot.chunks,
        "rejects": snapshot.rejects,
    }


def _tokens(args: argparse.Namespace) -> dict[str, Any]:
    with RunLog(args.lake, "gold.tokens") as log:
        result = tokenize_snapshot(
            args.lake,
            args.snapshot,
            args.tokenizer,
            eos_token=args.eos,
            workers=args.workers,
            on_event=log.record,
        )
        log.finish(result.run)
    return {
        "stage": "gold.tokens",
        "manifest": result.manifest_key,
        "documents": result.documents,
        "tokens": result.tokens,
    }


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
    "refine-all": _refine_all,
    "gold": _gold,
    "tokens": _tokens,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    report = COMMANDS[args.command](args)
    report["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(report, ensure_ascii=False))
    return 0
