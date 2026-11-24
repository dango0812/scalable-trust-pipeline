import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

from trustpipe.runtime.metrics import Stopwatch, peak_rss_bytes


def test_stopwatch_accumulates_seconds_per_phase() -> None:
    watch = Stopwatch()
    with watch.phase("read"):
        time.sleep(0.01)
    with watch.phase("read"):
        time.sleep(0.01)
    with watch.phase("write"):
        pass
    assert set(watch.seconds) == {"read", "write"}
    assert watch.seconds["read"] >= 0.02
    assert watch.seconds["read"] > watch.seconds["write"] >= 0.0


def test_stopwatch_records_a_phase_that_raises() -> None:
    watch = Stopwatch()
    with pytest.raises(RuntimeError), watch.phase("decode"):
        raise RuntimeError("boom")
    assert "decode" in watch.seconds


def measure_peak_growth() -> tuple[int, int, int]:
    before = peak_rss_bytes()
    block = bytearray(64 * 1024 * 1024)
    block[::4096] = b"x" * len(block[::4096])  # touch every page so it becomes resident
    after = peak_rss_bytes()
    del block
    return before, after, peak_rss_bytes()


def test_peak_rss_reports_a_plausible_number_and_never_shrinks() -> None:
    # Measured in a fresh process: in this one, earlier tests (loading models, building
    # tables) may already have set a peak that a 64 MiB block does not exceed.
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        before, after, final = pool.submit(measure_peak_growth).result()
    assert before > 5 * 1024 * 1024  # a Python process needs more than 5 MiB
    assert after >= before + 32 * 1024 * 1024
    assert final >= after
