import os
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from trustpipe.executor import (
    WORKER_DIED,
    PartReport,
    PermanentError,
    RunSummary,
    Status,
    TaskEvent,
    run_stage,
)

# Stage functions live at module level so worker processes can import them by name.


def report_for(key: str) -> PartReport:
    size = len(key)
    return PartReport(
        outputs=(f"out/{key}",),
        rows_in=size,
        rows_out=size - 1,
        bytes_in=size * 10,
        bytes_out=size * 5,
        timings={"read": 0.25, "write": 0.5},
        counters={"kept": size - 1},
    )


def always_ok(key: str) -> PartReport:
    return report_for(key)


def always_fails(key: str) -> PartReport:
    raise ValueError(f"cannot process {key}")


def _first_visit(scratch: Path, key: str) -> bool:
    """True exactly once per key, remembered across processes through a marker file."""
    marker = scratch / f"{key}.seen"
    if marker.exists():
        return False
    marker.write_text("seen", encoding="utf-8")
    return True


def fails_once(scratch: Path, key: str) -> PartReport:
    if _first_visit(scratch, key):
        raise OSError("transient")
    return report_for(key)


def rejects_its_input(key: str) -> PartReport:
    raise PermanentError(f"{key} is malformed")


def reports_its_pid(key: str) -> PartReport:
    return PartReport((key,), 1, 1, 1, 1, counters={str(os.getpid()): 1})


def fails_for_bad_keys(key: str) -> PartReport:
    if key.startswith("bad"):
        raise ValueError("poison part")
    return report_for(key)


def dies_once(scratch: Path, key: str) -> PartReport:
    if key.startswith("crash") and _first_visit(scratch, key):
        os._exit(13)  # simulate an out-of-memory kill: no exception, no cleanup
    return report_for(key)


def dies_always_for_bad_keys(key: str) -> PartReport:
    if key.startswith("bad"):
        os._exit(13)
    return report_for(key)


def dies_then_fails(scratch: Path, key: str) -> PartReport:
    if _first_visit(scratch, key):
        os._exit(13)
    raise ValueError("fails normally after the crash")


def statuses(events: list[TaskEvent]) -> list[tuple[str, str, int]]:
    return sorted((event.key, event.status.value, event.attempt) for event in events)


def test_serial_run_reports_every_part_and_sums_the_totals() -> None:
    events: list[TaskEvent] = []
    summary = run_stage("demo", always_ok, ["a", "bb", "ccc"], on_event=events.append)
    assert statuses(events) == [("a", "ok", 1), ("bb", "ok", 1), ("ccc", "ok", 1)]
    assert summary == RunSummary(
        stage="demo",
        parts=3,
        ok=3,
        failed=0,
        skipped=0,
        retries=0,
        rows_in=6,
        rows_out=3,
        bytes_in=60,
        bytes_out=30,
        wall_seconds=summary.wall_seconds,
        busy_seconds=summary.busy_seconds,
        max_peak_rss_bytes=summary.max_peak_rss_bytes,
        timings={"read": 0.75, "write": 1.5},
        counters={"kept": 3},
        failed_keys=(),
    )
    assert summary.wall_seconds >= summary.busy_seconds > 0
    assert summary.max_peak_rss_bytes > 0
    assert all(event.pid == os.getpid() for event in events)
    assert events[0].report == report_for("a")


def test_a_transient_failure_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo", partial(fails_once, tmp_path), ["a"], backoff_seconds=0, on_event=events.append
    )
    assert statuses(events) == [("a", "ok", 2), ("a", "retry", 1)]
    assert events[0].error == "OSError: transient"
    assert (summary.ok, summary.retries, summary.failed) == (1, 1, 0)


def test_a_poison_part_is_quarantined_and_the_rest_still_runs() -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo",
        fails_for_bad_keys,
        ["good-1", "bad-1", "good-2"],
        max_attempts=2,
        backoff_seconds=0,
        on_event=events.append,
    )
    assert statuses(events) == [
        ("bad-1", "failed", 2),
        ("bad-1", "retry", 1),
        ("good-1", "ok", 1),
        ("good-2", "ok", 1),
    ]
    assert summary.failed_keys == ("bad-1",)
    assert (summary.ok, summary.failed, summary.retries) == (2, 1, 1)
    failed = next(event for event in events if event.status is Status.FAILED)
    assert failed.error == "ValueError: poison part"
    assert failed.report is None


def test_a_permanent_error_fails_at_once_without_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    events: list[TaskEvent] = []
    summary = run_stage("demo", rejects_its_input, ["a"], max_attempts=5, on_event=events.append)
    assert statuses(events) == [("a", "failed", 1)]
    assert events[0].error == "PermanentError: a is malformed"
    assert (summary.failed, summary.retries) == (1, 0)
    assert slept == []


def test_parts_that_are_already_done_are_skipped() -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo", always_fails, ["done-1", "done-2"], is_done=lambda key: True, on_event=events.append
    )
    assert statuses(events) == [("done-1", "skipped", 0), ("done-2", "skipped", 0)]
    assert (summary.parts, summary.skipped, summary.ok, summary.failed) == (2, 2, 0, 0)


def test_backoff_doubles_per_attempt_and_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    run_stage("demo", always_fails, ["a"], max_attempts=4, backoff_seconds=1.0)
    assert slept == [1.0, 2.0, 4.0]
    slept.clear()
    run_stage("demo", always_fails, ["a"], max_attempts=3, backoff_seconds=100.0)
    assert slept == [30.0, 30.0]


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        run_stage("demo", always_ok, ["a"], max_attempts=0)


def test_a_part_given_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="'a' was given twice"):
        run_stage("demo", always_ok, ["a", "b", "a"])


def test_an_empty_run_has_an_empty_summary() -> None:
    summary = run_stage("demo", always_ok, [], workers=4)
    assert (summary.parts, summary.ok, summary.max_peak_rss_bytes) == (0, 0, 0)


def test_pool_run_processes_every_part_in_worker_processes() -> None:
    events: list[TaskEvent] = []
    keys = [f"part-{n}" for n in range(6)]
    summary = run_stage("demo", always_ok, keys, workers=2, on_event=events.append)
    assert statuses(events) == [(key, "ok", 1) for key in keys]
    assert summary.ok == 6
    assert summary.rows_in == sum(len(key) for key in keys)
    assert all(event.pid != os.getpid() for event in events)


def test_workers_are_replaced_after_max_tasks_per_child() -> None:
    keys = [f"p{n}" for n in range(4)]
    summary = run_stage("demo", reports_its_pid, keys, workers=2, max_tasks_per_child=1)
    # Every part ran in a fresh worker process, so four distinct pids reported.
    assert summary.ok == 4
    assert len(summary.counters) == 4


def test_pool_run_retries_failures_like_the_serial_run(tmp_path: Path) -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo",
        partial(fails_once, tmp_path),
        ["a", "b"],
        workers=2,
        backoff_seconds=0,
        on_event=events.append,
    )
    assert statuses(events) == [
        ("a", "ok", 2),
        ("a", "retry", 1),
        ("b", "ok", 2),
        ("b", "retry", 1),
    ]
    assert (summary.ok, summary.retries) == (2, 2)


def test_a_dead_worker_is_replaced_and_innocent_parts_lose_no_attempt(tmp_path: Path) -> None:
    events: list[TaskEvent] = []
    keys = ["crash-1", "ok-1", "ok-2", "ok-3"]
    summary = run_stage(
        "demo",
        partial(dies_once, tmp_path),
        keys,
        workers=2,
        max_attempts=2,
        backoff_seconds=0,
        on_event=events.append,
    )
    assert (summary.ok, summary.failed) == (4, 0)
    # Parts that shared the pool with the crash are re-run alone at the same attempt number,
    # so every part finishes on its first attempt: nobody is charged for the shared crash.
    assert statuses(events) == [(key, "ok", 1) for key in sorted(keys)]


def test_a_part_that_always_kills_its_worker_is_quarantined() -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo",
        dies_always_for_bad_keys,
        ["bad-1", "ok-1", "ok-2"],
        workers=2,
        max_attempts=2,
        backoff_seconds=0,
        on_event=events.append,
    )
    assert summary.failed_keys == ("bad-1",)
    assert summary.ok == 2
    assert statuses(events) == [
        ("bad-1", "failed", 2),
        ("bad-1", "retry", 1),
        ("ok-1", "ok", 1),
        ("ok-2", "ok", 1),
    ]
    assert {event.error for event in events if event.key == "bad-1"} == {WORKER_DIED}


def test_a_suspect_that_fails_normally_goes_back_to_the_retry_queue(tmp_path: Path) -> None:
    events: list[TaskEvent] = []
    summary = run_stage(
        "demo",
        partial(dies_then_fails, tmp_path),
        ["a"],
        workers=2,
        max_attempts=2,
        backoff_seconds=0,
        on_event=events.append,
    )
    assert statuses(events) == [("a", "failed", 2), ("a", "retry", 1)]
    assert summary.failed_keys == ("a",)
    assert events[-1].error == "ValueError: fails normally after the crash"


def test_a_pool_found_broken_at_submit_time_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[ProcessPoolExecutor] = []

    class FirstPoolIsBroken(ProcessPoolExecutor):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created.append(self)

        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            if created[0] is self:
                raise BrokenProcessPool("a worker died while the pool was idle")
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr("trustpipe.executor.ProcessPoolExecutor", FirstPoolIsBroken)
    events: list[TaskEvent] = []
    summary = run_stage("demo", always_ok, ["a", "b"], workers=2, on_event=events.append)
    assert statuses(events) == [("a", "ok", 1), ("b", "ok", 1)]
    assert summary.ok == 2
    assert len(created) == 2
