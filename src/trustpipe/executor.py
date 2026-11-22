"""Run one stage over many parts: parallel, retried, crash-tolerant and observable.

The unit of work is one part. A stage is a function ``fn(key) -> PartReport`` that reads
the part named by ``key`` and writes its outputs atomically, so running it again is safe.
Because parts are independent, the same contract runs in the calling process, on a local
process pool, or on a cluster scheduler (Ray tasks, Spark partitions, Slurm array jobs,
Kubernetes indexed jobs): only this module would change, never the stages or the data.

What the executor adds around the stage function:

- Parallelism: a process pool when ``workers`` > 1, the calling process otherwise.
- Retries: a failed part is retried with exponential backoff, up to ``max_attempts``.
  A stage raises ``PermanentError`` for failures a retry cannot fix (malformed input,
  bad configuration); those fail at once instead of burning attempts.
- Failover: when a worker process dies (out of memory, killed), the pool is rebuilt and
  the parts that were in flight run again one at a time, so the part that kills workers
  is identified and charged while innocent parts lose nothing.
- Quarantine: a part that keeps failing is reported as failed and the run continues;
  one poison part never blocks the others.
- Resume: parts for which ``is_done(key)`` is true are skipped.
- Observability: every attempt becomes one ``TaskEvent`` (duration, rows, bytes, peak
  memory, per-phase timings, error) handed to ``on_event``; the number of parts not yet
  started is the natural autoscaling signal for a cluster deployment.

Workers are always started with the "spawn" method, the only one available on every
platform: Linux, macOS and Windows then run the same code path, and no worker inherits
locks or threads from a parent that has already started Arrow's thread pool.

Not handled here, by design: a hung part (no timeout, because a running task cannot be
stopped portably before Python 3.14) and a run-wide failure budget. On a cluster both
belong to the scheduler; locally, ``max_tasks_per_child`` bounds memory growth.
"""

import multiprocessing
import os
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from enum import StrEnum

from trustpipe.metrics import peak_rss_bytes

MAX_BACKOFF_SECONDS = 30.0
WORKER_DIED = "worker process died while running this part"
_SPAWN = multiprocessing.get_context("spawn")


class PermanentError(Exception):
    """Raised by a stage for a failure that another attempt cannot fix."""


@dataclass(frozen=True, slots=True)
class PartReport:
    """What a stage function returns for one part."""

    outputs: tuple[str, ...]
    rows_in: int
    rows_out: int
    bytes_in: int
    bytes_out: int
    timings: dict[str, float] = field(default_factory=dict)  # phase -> seconds
    counters: dict[str, int] = field(default_factory=dict)  # stage-specific counts


class Status(StrEnum):
    OK = "ok"
    RETRY = "retry"  # this attempt failed and the part will run again
    FAILED = "failed"  # attempts exhausted: the part is quarantined
    SKIPPED = "skipped"  # already done, nothing ran


@dataclass(frozen=True, slots=True)
class TaskEvent:
    """One attempt at one part."""

    stage: str
    key: str
    status: Status
    attempt: int
    seconds: float
    peak_rss_bytes: int
    pid: int
    report: PartReport | None
    error: str | None


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Totals of one stage run; ``timings`` shows where the time went."""

    stage: str
    parts: int
    ok: int
    failed: int
    skipped: int
    retries: int
    rows_in: int
    rows_out: int
    bytes_in: int
    bytes_out: int
    wall_seconds: float
    busy_seconds: float  # summed duration of successful attempts across workers
    max_peak_rss_bytes: int
    timings: dict[str, float]
    counters: dict[str, int]
    failed_keys: tuple[str, ...]


StageFn = Callable[[str], PartReport]


@dataclass(frozen=True, slots=True)
class _Outcome:
    seconds: float
    peak_rss_bytes: int
    pid: int
    report: PartReport | None
    error: str | None
    retryable: bool = True


def _attempt(fn: StageFn, key: str, delay_seconds: float) -> _Outcome:
    """Run one attempt (in a worker process or in the caller) and never raise."""
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    started = time.perf_counter()
    report: PartReport | None = None
    error: str | None = None
    retryable = True
    try:
        report = fn(key)
    except Exception as exc:  # a stage may fail in any way; the failure is reported
        error = f"{type(exc).__name__}: {exc}"
        retryable = not isinstance(exc, PermanentError)
    seconds = time.perf_counter() - started
    return _Outcome(seconds, peak_rss_bytes(), os.getpid(), report, error, retryable)


def _pool(workers: int, max_tasks_per_child: int | None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=_SPAWN, max_tasks_per_child=max_tasks_per_child
    )


def _run_alone(fn: StageFn, key: str, delay_seconds: float) -> _Outcome | None:
    """Run one attempt in a pool of its own; None means the worker died."""
    with _pool(1, None) as solo:
        try:
            return solo.submit(_attempt, fn, key, delay_seconds).result()
        except BrokenProcessPool:
            return None


class _Totals:
    """Running totals of a run, so memory stays flat however many parts it has."""

    def __init__(self) -> None:
        self.by_status: dict[Status, int] = dict.fromkeys(Status, 0)
        self.rows_in = self.rows_out = self.bytes_in = self.bytes_out = 0
        self.busy_seconds = 0.0
        self.max_peak_rss_bytes = 0
        self.timings: dict[str, float] = {}
        self.counters: dict[str, int] = {}
        self.failed_keys: list[str] = []

    def add(self, event: TaskEvent) -> None:
        self.by_status[event.status] += 1
        self.max_peak_rss_bytes = max(self.max_peak_rss_bytes, event.peak_rss_bytes)
        if event.status is Status.FAILED:
            self.failed_keys.append(event.key)
        report = event.report
        if report is None:  # only successful attempts carry a report
            return
        self.busy_seconds += event.seconds
        self.rows_in += report.rows_in
        self.rows_out += report.rows_out
        self.bytes_in += report.bytes_in
        self.bytes_out += report.bytes_out
        for name, seconds in report.timings.items():
            self.timings[name] = self.timings.get(name, 0.0) + seconds
        for name, count in report.counters.items():
            self.counters[name] = self.counters.get(name, 0) + count

    def summary(self, stage: str, parts: int, wall_seconds: float) -> RunSummary:
        return RunSummary(
            stage=stage,
            parts=parts,
            ok=self.by_status[Status.OK],
            failed=self.by_status[Status.FAILED],
            skipped=self.by_status[Status.SKIPPED],
            retries=self.by_status[Status.RETRY],
            rows_in=self.rows_in,
            rows_out=self.rows_out,
            bytes_in=self.bytes_in,
            bytes_out=self.bytes_out,
            wall_seconds=wall_seconds,
            busy_seconds=self.busy_seconds,
            max_peak_rss_bytes=self.max_peak_rss_bytes,
            timings=dict(sorted(self.timings.items())),
            counters=dict(sorted(self.counters.items())),
            failed_keys=tuple(sorted(self.failed_keys)),
        )


class _Run:
    """Mutable state of one ``run_stage`` call."""

    def __init__(
        self,
        stage: str,
        fn: StageFn,
        max_attempts: int,
        backoff_seconds: float,
        on_event: Callable[[TaskEvent], None] | None,
        max_tasks_per_child: int | None,
    ) -> None:
        self.stage = stage
        self.fn = fn
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.on_event = on_event
        self.max_tasks_per_child = max_tasks_per_child
        self.queue: deque[tuple[str, int]] = deque()
        self.totals = _Totals()

    def delay(self, attempt: int) -> float:
        if attempt <= 1 or self.backoff_seconds <= 0:
            return 0.0
        return min(self.backoff_seconds * 2.0 ** (attempt - 2), MAX_BACKOFF_SECONDS)

    def emit(
        self, key: str, status: Status, attempt: int, outcome: _Outcome | None, error: str | None
    ) -> None:
        event = TaskEvent(
            stage=self.stage,
            key=key,
            status=status,
            attempt=attempt,
            seconds=outcome.seconds if outcome else 0.0,
            peak_rss_bytes=outcome.peak_rss_bytes if outcome else 0,
            pid=outcome.pid if outcome else 0,
            report=outcome.report if outcome else None,
            error=error,
        )
        self.totals.add(event)
        if self.on_event is not None:
            self.on_event(event)

    def settle(self, key: str, attempt: int, outcome: _Outcome | None) -> None:
        """Record the result of an attempt and queue a retry when one is due."""
        error = WORKER_DIED if outcome is None else outcome.error
        retryable = outcome is None or outcome.retryable
        if error is None:
            self.emit(key, Status.OK, attempt, outcome, None)
        elif retryable and attempt < self.max_attempts:
            self.emit(key, Status.RETRY, attempt, outcome, error)
            self.queue.append((key, attempt + 1))
        else:
            self.emit(key, Status.FAILED, attempt, outcome, error)

    def run_serial(self) -> None:
        while self.queue:
            key, attempt = self.queue.popleft()
            self.settle(key, attempt, _attempt(self.fn, key, self.delay(attempt)))

    def run_pool(self, workers: int) -> None:
        pool = _pool(workers, self.max_tasks_per_child)
        in_flight: dict[Future[_Outcome], tuple[str, int]] = {}
        try:
            while self.queue or in_flight:
                # Keep at most `workers` parts in flight, so a dead worker can only
                # take that many parts with it and the queue length stays observable.
                broken = False
                while self.queue and len(in_flight) < workers and not broken:
                    key, attempt = self.queue.popleft()
                    try:
                        future = pool.submit(_attempt, self.fn, key, self.delay(attempt))
                    except BrokenProcessPool:  # a worker died while the pool was idle
                        self.queue.appendleft((key, attempt))
                        broken = True
                    else:
                        in_flight[future] = (key, attempt)
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                suspects = self._collect(done, in_flight)
                if broken or suspects:
                    # A broken pool fails every in-flight part the same way: collect them
                    # all, replace the pool, then find the culprit by running each alone.
                    remaining, _ = wait(in_flight)
                    suspects += self._collect(remaining, in_flight)
                    pool.shutdown(wait=False)
                    pool = _pool(workers, self.max_tasks_per_child)
                    for key, attempt in suspects:
                        self._isolate(key, attempt)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _collect(
        self,
        futures: Iterable[Future[_Outcome]],
        in_flight: dict[Future[_Outcome], tuple[str, int]],
    ) -> list[tuple[str, int]]:
        """Settle finished futures; return the parts whose worker died."""
        suspects: list[tuple[str, int]] = []
        for future in futures:
            key, attempt = in_flight.pop(future)
            try:
                outcome = future.result()
            except BrokenProcessPool:
                suspects.append((key, attempt))
            else:
                self.settle(key, attempt, outcome)
        return suspects

    def _isolate(self, key: str, attempt: int) -> None:
        """Re-run a suspect alone until it finishes, fails normally or runs out of attempts."""
        while True:
            outcome = _run_alone(self.fn, key, self.delay(attempt))
            if outcome is not None:
                self.settle(key, attempt, outcome)
                return
            if attempt >= self.max_attempts:
                self.emit(key, Status.FAILED, attempt, None, WORKER_DIED)
                return
            self.emit(key, Status.RETRY, attempt, None, WORKER_DIED)
            attempt += 1


def run_stage(
    stage: str,
    fn: StageFn,
    keys: Iterable[str],
    *,
    workers: int = 1,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    is_done: Callable[[str], bool] | None = None,
    on_event: Callable[[TaskEvent], None] | None = None,
    max_tasks_per_child: int | None = None,
) -> RunSummary:
    """Run ``fn`` once per key and return the totals.

    With ``workers`` > 1, ``fn`` must be picklable: a module-level function or a
    ``functools.partial`` of one. ``max_tasks_per_child`` replaces a worker process after
    that many parts, which bounds memory a stage leaks or fragments over a long run.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    started = time.perf_counter()
    run = _Run(stage, fn, max_attempts, backoff_seconds, on_event, max_tasks_per_child)
    seen: set[str] = set()
    for key in keys:
        # Two workers on one part would race to replace the same output file.
        if key in seen:
            raise ValueError(f"part {key!r} was given twice")
        seen.add(key)
        if is_done is not None and is_done(key):
            run.emit(key, Status.SKIPPED, 0, None, None)
        else:
            run.queue.append((key, 1))
    if workers > 1:
        run.run_pool(workers)
    else:
        run.run_serial()
    return run.totals.summary(stage, len(seen), time.perf_counter() - started)
