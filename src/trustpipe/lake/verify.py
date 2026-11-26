"""Check every committed file against the receipt that names it.

A receipt lists each output of a part with its size and SHA-256, and each input by key
(receipts.py). ``verify_lake`` re-reads the lake and reports, without changing anything:

- missing files: outputs or inputs that a receipt names but the lake no longer holds;
- corrupt files: outputs whose size or SHA-256 differ from the receipt (bit rot, a partial
  copy, a hand edit).

Hashing dominates the cost and releases the GIL, so files are hashed on a thread pool. A
file named by several receipts is read once and compared with each of them.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from trustpipe.lake.receipts import all_receipts
from trustpipe.lake.storage import file_sha256, lake_path


@dataclass(frozen=True, slots=True)
class VerifyReport:
    receipts: int
    files: int
    bytes: int
    missing: tuple[str, ...]
    corrupt: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.corrupt


def _measure(root: Path, key: str) -> tuple[int, str] | None:
    """Size and SHA-256 of a lake file, or None when it does not exist."""
    path = lake_path(root, key)
    if not path.is_file():
        return None
    return path.stat().st_size, file_sha256(path)


def verify_lake(root: Path, *, workers: int = 4) -> VerifyReport:
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    receipts = list(all_receipts(root))
    expected: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for receipt in receipts:
        for output in receipt.outputs:
            expected[output.key].add((output.bytes, output.sha256))
    inputs = {key for receipt in receipts for key in receipt.inputs} - expected.keys()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        measured = dict(zip(expected, pool.map(lambda k: _measure(root, k), expected), strict=True))
    missing = {key for key, found in measured.items() if found is None}
    missing |= {key for key in inputs if not lake_path(root, key).is_file()}
    corrupt = {
        key
        for key, found in measured.items()
        if found is not None and any(found != want for want in expected[key])
    }
    return VerifyReport(
        receipts=len(receipts),
        files=len(expected) + len(inputs),
        bytes=sum(found[0] for found in measured.values() if found is not None),
        missing=tuple(sorted(missing)),
        corrupt=tuple(sorted(corrupt)),
    )
