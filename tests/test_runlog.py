import dataclasses
import json
import re
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from trustpipe.executor import PartReport, run_stage
from trustpipe.runlog import RunLog, new_run_id


def ok_stage(key: str) -> PartReport:
    return PartReport((f"out/{key}",), 10, 9, 2_000_000, 500, {"read": 0.5}, {"kept": 9})


def failing_stage(key: str) -> PartReport:
    raise ValueError("bad part")


def read_events(log: RunLog) -> list[dict[str, object]]:
    lines = (log.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_run_ids_sort_by_start_time_and_are_unique() -> None:
    kst = timezone(timedelta(hours=9))
    run_id = new_run_id("silver.refine", datetime(2026, 11, 22, 22, 13, 5, tzinfo=kst))
    assert re.fullmatch(r"20261122T131305Z-silver-refine-[0-9a-f]{6}", run_id)
    assert new_run_id("gold.build") != new_run_id("gold.build")
    assert new_run_id("gold.build").startswith(f"{datetime.now(UTC):%Y%m%d}")


def test_run_log_records_every_attempt_and_the_summary(tmp_path: Path) -> None:
    with RunLog(tmp_path, "silver.refine", context={"workers": 1}) as log:
        summary = run_stage("silver.refine", ok_stage, ["p0", "p1"], on_event=log.record)
        summary_path = log.finish(summary)

    assert log.directory == tmp_path / "_runs" / log.run_id
    events = read_events(log)
    assert [event["event"] for event in events] == ["run.start", "part", "part", "run.end"]
    assert all(event["run_id"] == log.run_id for event in events)
    assert all(event["stage"] == "silver.refine" for event in events)
    assert events[0]["workers"] == 1
    assert events[0]["trustpipe_version"] == "0.1.0"
    part = events[1]
    assert part["key"] == "p0"
    assert part["status"] == "ok"
    assert part["attempt"] == 1
    assert part["rows_in"] == 10
    assert part["timings"] == {"read": 0.5}
    assert part["outputs"] == ["out/p0"]
    assert part["error"] is None
    assert events[-1]["ok"] == 2

    totals = json.loads(summary_path.read_text(encoding="utf-8"))
    assert totals["run_id"] == log.run_id
    assert totals["ok"] == 2
    assert totals["rows_in"] == 20
    assert totals["timings"] == {"read": 1.0}
    assert totals["workers"] == 1
    assert totals["rows_per_second"] > 0
    assert totals["megabytes_per_second"] > 0


def test_failed_attempts_are_logged_with_their_error(tmp_path: Path) -> None:
    with RunLog(tmp_path, "silver.refine", run_id="fixed-run") as log:
        summary = run_stage(
            "silver.refine", failing_stage, ["p0"], max_attempts=1, on_event=log.record
        )
        log.finish(summary)
    part = read_events(log)[1]
    assert part["status"] == "failed"
    assert part["error"] == "ValueError: bad part"
    assert part["rows_in"] is None
    assert part["outputs"] is None


def test_a_run_id_is_never_reused(tmp_path: Path) -> None:
    with RunLog(tmp_path, "gold.build", run_id="same"):
        pass
    with pytest.raises(FileExistsError):
        RunLog(tmp_path, "gold.build", run_id="same")


def test_summary_of_an_instant_run_has_no_throughput(tmp_path: Path) -> None:
    with RunLog(tmp_path, "gold.build") as log:
        summary = run_stage("gold.build", ok_stage, [])
        instant = dataclasses.replace(summary, wall_seconds=0.0)
        totals = json.loads(log.finish(instant).read_text(encoding="utf-8"))
    assert totals["rows_per_second"] is None
    assert totals["megabytes_per_second"] is None
