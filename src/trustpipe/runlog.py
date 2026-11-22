"""Append-only record of stage runs, kept inside the lake under ``_runs/``.

One directory per run:

    _runs/<run_id>/events.jsonl   one JSON line per attempt at a part, written as it happens
    _runs/<run_id>/summary.json   totals of the run, written once at the end

The underscore prefix hides the directory from engines that read Hive-partitioned tables
(Spark, Trino, DuckDB and pyarrow datasets skip names starting with "_").

The log is evidence, not state: no stage reads it to decide what to do, so a lost or
truncated log can never corrupt data or block a resume. It answers operational questions:
which parts failed and why, how long each phase took, how much memory workers needed.
The field names follow the shape of trace spans (run = trace, attempt = span), so the
events can later be exported to an OpenTelemetry collector without changing stages.
"""

import dataclasses
import json
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from trustpipe import __version__
from trustpipe.executor import RunSummary, TaskEvent

RUNS_DIR = "_runs"


def new_run_id(stage: str, now: datetime | None = None) -> str:
    """Sortable by start time, readable, and unique across concurrent runs."""
    started = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{started:%Y%m%dT%H%M%SZ}-{stage.replace('.', '-')}-{uuid.uuid4().hex[:6]}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class RunLog:
    """Writes the event stream and the summary of one stage run."""

    def __init__(
        self,
        root: Path,
        stage: str,
        *,
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.run_id = run_id or new_run_id(stage)
        self.directory = root / RUNS_DIR / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.context: dict[str, Any] = {
            "trustpipe_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            **(context or {}),
        }
        self._events = (self.directory / "events.jsonl").open("a", encoding="utf-8", newline="\n")
        self._write({"event": "run.start", **self.context})

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._events.close()

    def _write(self, fields: dict[str, Any]) -> None:
        line = {"ts": _timestamp(), "run_id": self.run_id, "stage": self.stage, **fields}
        self._events.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._events.flush()

    def record(self, event: TaskEvent) -> None:
        """Append one attempt; pass this method as ``on_event`` to ``run_stage``."""
        report = event.report
        self._write(
            {
                "event": "part",
                "key": event.key,
                "status": event.status.value,
                "attempt": event.attempt,
                "seconds": round(event.seconds, 6),
                "peak_rss_bytes": event.peak_rss_bytes,
                "pid": event.pid,
                "rows_in": report.rows_in if report else None,
                "rows_out": report.rows_out if report else None,
                "bytes_in": report.bytes_in if report else None,
                "bytes_out": report.bytes_out if report else None,
                "timings": report.timings if report else None,
                "counters": report.counters if report else None,
                "outputs": list(report.outputs) if report else None,
                "error": event.error,
            }
        )

    def finish(self, summary: RunSummary) -> Path:
        """Write the summary file and return its path."""
        wall = summary.wall_seconds
        totals: dict[str, Any] = {
            "run_id": self.run_id,
            **self.context,
            **dataclasses.asdict(summary),
            "rows_per_second": round(summary.rows_in / wall, 1) if wall > 0 else None,
            "megabytes_per_second": round(summary.bytes_in / wall / 1e6, 2) if wall > 0 else None,
        }
        self._write({"event": "run.end", "ok": summary.ok, "failed": summary.failed})
        path = self.directory / "summary.json"
        path.write_text(json.dumps(totals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
