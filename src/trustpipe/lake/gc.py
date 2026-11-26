"""Find, and on request remove, lake files that no receipt names.

A file is live when some receipt lists it as an output or as an input. Everything else
under the data trees (bronze/, silver/, gold/, index/) is an orphan:

- outputs of a part that crashed before its receipt was written (a retry rewrites them);
- temporary files of interrupted writes (``.<name>.<uuid>.tmp``, storage.py);
- trust parts of an old policy once every part was re-judged and no gold snapshot still
  reads them (a gold receipt lists the trust part it read as an input, so it stays live).

A running stage writes its outputs before its receipt, so orphans younger than
``min_age`` are kept. Nothing is removed unless ``delete`` is set; then the orphans go,
and so do the directories they leave empty. Control files (``_receipts/``,
``_policies/``, ``_snapshots/``, ``_runs/``) are never touched.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trustpipe.lake.receipts import all_receipts

DATA_TREES = ("bronze", "silver", "gold", "index")


@dataclass(frozen=True, slots=True)
class GcReport:
    live: int
    orphans: tuple[str, ...]
    orphan_bytes: int
    kept_young: int
    deleted: bool


def collect_garbage(
    root: Path,
    *,
    delete: bool = False,
    min_age: timedelta = timedelta(hours=1),
    now: datetime | None = None,
) -> GcReport:
    live = {
        key
        for receipt in all_receipts(root)
        for key in (*receipt.inputs, *(output.key for output in receipt.outputs))
    }
    cutoff = (now or datetime.now(UTC)).timestamp() - min_age.total_seconds()
    orphans: list[str] = []
    orphan_bytes = 0
    kept_young = 0
    for tree in DATA_TREES:
        for path in sorted(p for p in (root / tree).rglob("*") if p.is_file()):
            key = path.relative_to(root).as_posix()
            if key in live:
                continue
            stat = path.stat()
            if stat.st_mtime > cutoff:
                kept_young += 1
                continue
            orphans.append(key)
            orphan_bytes += stat.st_size
    if delete:
        for key in orphans:
            (root / key).unlink()
        _remove_empty_directories(root)
    return GcReport(len(live), tuple(orphans), orphan_bytes, kept_young, delete)


def _remove_empty_directories(root: Path) -> None:
    for tree in DATA_TREES:
        directories = sorted((p for p in (root / tree).rglob("*") if p.is_dir()), reverse=True)
        for directory in directories:
            if not any(directory.iterdir()):
                directory.rmdir()
