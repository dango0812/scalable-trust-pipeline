import time

import pytest

from trustpipe.metrics import Stopwatch, peak_rss_bytes


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


def test_peak_rss_reports_a_plausible_number_and_never_shrinks() -> None:
    before = peak_rss_bytes()
    assert before > 5 * 1024 * 1024  # a Python process needs more than 5 MiB
    block = bytearray(64 * 1024 * 1024)
    block[::4096] = b"x" * len(block[::4096])  # touch every page so it becomes resident
    after = peak_rss_bytes()
    assert after >= before + 32 * 1024 * 1024
    del block
    assert peak_rss_bytes() >= after
