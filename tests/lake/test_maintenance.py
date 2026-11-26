"""verify finds missing and corrupt files; gc removes only files that no receipt names."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import Lander
from corpus import english

from trustpipe.cli import main
from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.stages import build_gold
from trustpipe.lake.gc import collect_garbage
from trustpipe.lake.receipts import Receipt, commit, committed_parts, load_receipt
from trustpipe.lake.storage import lake_path
from trustpipe.lake.verify import verify_lake
from trustpipe.silver.refine import STAGE, refine_all
from trustpipe.trust.registry import load_registry

PAGES = [
    ("https://primary.example.org/a", english(1, 120)),
    ("https://agency.example.org/b", english(2, 120)),
    ("https://unknown.example.net/c", english(3, 120)),
]
NOW = datetime(2026, 12, 1, tzinfo=UTC)


def silver_outputs(root: Path) -> tuple[str, str, str]:
    receipt = load_receipt(root, STAGE, committed_parts(root, STAGE)[0])
    documents, rejects, trust = (o.key for o in receipt.outputs)
    return documents, rejects, trust


def age(path: Path, days: int = 2) -> None:
    stamp = (NOW - timedelta(days=days)).timestamp()
    os.utime(path, (stamp, stamp))


def test_a_clean_lake_verifies(lake: Lander) -> None:
    report = verify_lake(lake(PAGES), workers=2)
    assert report.ok
    assert report.receipts == 2
    assert report.files == 4  # one bronze part, three silver parts
    assert report.bytes > 0


def test_an_empty_lake_has_nothing_to_verify_or_collect(tmp_path: Path) -> None:
    report = verify_lake(tmp_path)
    assert report.ok
    assert (report.receipts, report.files, report.bytes) == (0, 0, 0)
    assert collect_garbage(tmp_path, delete=True).orphans == ()


def test_missing_and_corrupt_files_are_reported(lake: Lander) -> None:
    root = lake(PAGES)
    documents, _, trust = silver_outputs(root)
    lake_path(root, trust).unlink()
    path = lake_path(root, documents)
    path.write_bytes(path.read_bytes() + b"x")
    commit(
        root,
        Receipt("test.stage", "0" * 16, ("bronze/nowhere.parquet",), (), {}, {}),
        create_only=False,
    )
    report = verify_lake(root)
    assert not report.ok
    assert report.missing == ("bronze/nowhere.parquet", trust)
    assert report.corrupt == (documents,)
    with pytest.raises(ValueError, match="workers"):
        verify_lake(root, workers=0)


def test_gc_finds_nothing_in_a_committed_lake(lake: Lander) -> None:
    report = collect_garbage(lake(PAGES), min_age=timedelta(0))
    assert report.orphans == ()
    assert (report.orphan_bytes, report.kept_young, report.deleted) == (0, 0, False)
    assert report.live == 4


def test_gc_lists_old_orphans_keeps_young_ones_and_deletes_on_request(lake: Lander) -> None:
    root = lake(PAGES)
    documents, _, _ = silver_outputs(root)
    stray = lake_path(root, "silver/documents/date=2026-01-01/part-0000000000000000.parquet")
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"from a part that crashed before its receipt")
    tmp = lake_path(root, documents).with_name(".part.parquet.123.tmp")
    tmp.write_bytes(b"interrupted write")
    young = lake_path(root, "gold/chunks/snapshot=000000000000/part-1111111111111111.parquet")
    young.parent.mkdir(parents=True)
    young.write_bytes(b"a stage is still writing this part")
    for path in (stray, tmp):
        age(path)
    age(young, days=0)

    listed = collect_garbage(root, now=NOW)
    assert set(listed.orphans) == {
        stray.relative_to(root).as_posix(),
        tmp.relative_to(root).as_posix(),
    }
    assert listed.kept_young == 1
    assert stray.exists()
    assert tmp.exists()

    deleted = collect_garbage(root, delete=True, now=NOW)
    assert deleted.deleted
    assert deleted.orphans == listed.orphans
    assert not stray.exists()
    assert not stray.parent.exists()
    assert not tmp.exists()
    assert young.exists()
    assert verify_lake(root).ok


def write_policy(tmp_path: Path) -> Path:
    path = tmp_path / "next.toml"
    path.write_text(
        'version = "next"\n[[source]]\nuri_prefix = "https://unknown.example.net"\n'
        'license = "CC-BY-4.0"\ntier = "reviewed"\nreason = "reviewed later"\n',
        encoding="utf-8",
    )
    return path


def test_an_old_trust_part_is_garbage_only_once_nothing_reads_it(
    lake: Lander, tmp_path: Path
) -> None:
    root = lake(PAGES)
    _, _, old_trust = silver_outputs(root)
    refine_all(root, load_registry(write_policy(tmp_path)))
    report = collect_garbage(root, min_age=timedelta(0))
    assert report.orphans == (old_trust,)
    collect_garbage(root, delete=True, min_age=timedelta(0))
    assert not lake_path(root, old_trust).exists()
    assert verify_lake(root).ok


def test_a_gold_snapshot_keeps_the_trust_part_it_read(lake: Lander, tmp_path: Path) -> None:
    root = lake(PAGES)
    _, _, old_trust = silver_outputs(root)
    build_gold(root, GoldPolicy(version="test"))
    refine_all(root, load_registry(write_policy(tmp_path)))
    assert old_trust not in collect_garbage(root, min_age=timedelta(0)).orphans


def test_cli_verifies_and_collects_garbage(
    lake: Lander, capsys: pytest.CaptureFixture[str]
) -> None:
    root = lake(PAGES)
    assert main(["--lake", str(root), "verify", "--workers", "2"]) == 0
    clean = json.loads(capsys.readouterr().out)
    assert (clean["stage"], clean["intact"]) == ("verify", True)
    assert (clean["missing"], clean["corrupt"]) == (0, 0)

    documents, _, _ = silver_outputs(root)
    lake_path(root, documents).write_bytes(b"corrupt")
    assert main(["--lake", str(root), "verify"]) == 1
    broken = json.loads(capsys.readouterr().out)
    assert (broken["intact"], broken["corrupt_keys"]) == (False, [documents])

    stray = lake_path(root, "silver/rejects/date=2026-01-01/part-0000000000000000.parquet")
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"orphan")
    assert main(["--lake", str(root), "gc", "--min-age-hours", "0"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert (listed["orphans"], listed["deleted"]) == (1, False)
    assert main(["--lake", str(root), "gc", "--delete", "--min-age-hours", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] is True
    assert not stray.exists()
