"""Measurements a stage reports about itself: phase timings and peak memory.

These are the raw signals for bottleneck analysis. A stage wraps each phase of its work
(read, decode, hash, write, ...) in ``Stopwatch.phase``; the executor adds wall time and
the worker's peak memory. Summed over all parts they show where the time of a run went.
"""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager


class Stopwatch:
    """Accumulates wall-clock seconds per named phase."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started


if sys.platform == "win32":  # pragma: no cover - only reachable on Windows

    def peak_rss_bytes() -> int:
        """Peak resident memory of this process (peak working set), in bytes."""
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        ok = kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else 0

else:  # pragma: no cover - only reachable on POSIX

    def peak_rss_bytes() -> int:
        """Peak resident memory of this process (ru_maxrss), in bytes."""
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kibibytes, macOS reports bytes.
        return peak if sys.platform == "darwin" else peak * 1024
