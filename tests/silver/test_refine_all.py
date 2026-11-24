from datetime import UTC, datetime
from pathlib import Path

from trustpipe.bronze.ingest import land
from trustpipe.lake.schema import Acquisition, RawCapture
from trustpipe.silver.refine import refine_all
from trustpipe.trust.registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
AT = datetime(2026, 11, 24, tzinfo=UTC)


def capture(n: int) -> RawCapture:
    return RawCapture.from_bytes(
        f"document {n}".encode(),
        source_uri=f"https://example.org/{n}",
        fetched_at=AT.replace(hour=n),
        acquisition=Acquisition.HTTP_FETCH,
    )


def test_refine_all_refines_each_part_once_per_policy(tmp_path: Path) -> None:
    land([capture(1)], tmp_path)
    land([capture(2)], tmp_path)
    registry = load_registry(ROOT / "config" / "sources.toml")
    first = refine_all(tmp_path, registry)
    assert (first.parts, first.ok, first.skipped, first.rows_out) == (2, 2, 0, 2)
    assert refine_all(tmp_path, registry).skipped == 2
    other = tmp_path / "other.toml"
    other.write_text('version = "other"\n', encoding="utf-8")
    assert refine_all(tmp_path, load_registry(other)).ok == 2
