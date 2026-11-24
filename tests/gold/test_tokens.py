import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from corpus import REGISTRY_TOML, english, korean
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from trustpipe.cli import main
from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.stages import build_gold
from trustpipe.gold.tables import GOLD_TOKEN_INDEX
from trustpipe.gold.tokens import TokenWindows, _dtype, tokenize_snapshot
from trustpipe.lake.storage import lake_path, read_table

Lander = Callable[[list[tuple[str, str]]], Path]
ROOT = Path(__file__).resolve().parents[2]


def word_tokenizer(path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "<eos>"])
    tokenizer.train_from_iterator([english(1, 400), korean(1, 400)], trainer)
    tokenizer.save(str(path))
    return path


@pytest.fixture
def snapshot(lake: Lander) -> tuple[Path, str]:
    root = lake(
        [
            ("https://primary.example.org/a", english(1)),
            ("https://agency.example.org/b", korean(2)),
        ]
    )
    return root, build_gold(root, GoldPolicy(version="test")).snapshot


def test_documents_become_token_files_ending_in_end_of_text(
    tmp_path: Path, snapshot: tuple[Path, str]
) -> None:
    root, snap = snapshot
    tokenizer = word_tokenizer(tmp_path / "tokenizer.json")
    tokens = tokenize_snapshot(root, snap, tokenizer, eos_token="<eos>")
    manifest = json.loads(lake_path(root, tokens.manifest_key).read_text(encoding="utf-8"))
    assert manifest["dtype"] == "<u2"
    assert (manifest["documents"], tokens.documents) == (2, 2)
    [part] = manifest["parts"]
    ids = np.fromfile(lake_path(root, part["tokens"]), dtype="<u2")
    assert len(ids) == tokens.tokens == part["count"]
    index = read_table(root, part["index"], schema=GOLD_TOKEN_INDEX).to_pylist()
    eos = Tokenizer.from_file(str(tokenizer)).token_to_id("<eos>")
    for row in index:
        assert ids[row["offset"] + row["length"] - 1] == eos
    assert sum(row["length"] for row in index) == len(ids)
    again = tokenize_snapshot(root, snap, tokenizer, eos_token="<eos>")
    assert again.run.skipped == 1


def test_windows_slice_the_token_stream(tmp_path: Path, snapshot: tuple[Path, str]) -> None:
    root, snap = snapshot
    tokens = tokenize_snapshot(root, snap, word_tokenizer(tmp_path / "t.json"), eos_token="<eos>")
    windows = TokenWindows(root, tokens.manifest_key, 16)
    assert len(windows) == (tokens.tokens - 1) // 16
    first, second = windows[0], windows[1]
    assert first.shape == (17,)
    assert first.dtype == np.int64
    assert first[-1] == second[0]
    with pytest.raises(IndexError):
        windows[len(windows)]
    assert len(TokenWindows(root, tokens.manifest_key, 100_000)) == 0
    with pytest.raises(ValueError, match="positive"):
        TokenWindows(root, tokens.manifest_key, 0)


def test_a_tokenizer_without_the_end_token_fails_the_part(
    tmp_path: Path, snapshot: tuple[Path, str]
) -> None:
    root, snap = snapshot
    with pytest.raises(RuntimeError, match=r"gold\.tokens failed"):
        tokenize_snapshot(root, snap, word_tokenizer(tmp_path / "t.json"), eos_token="</s>")


def test_large_vocabularies_use_32_bit_ids() -> None:
    assert _dtype(65_536).str == "<u2"
    assert _dtype(65_537).str == "<u4"


def test_cli_runs_refine_gold_and_tokens(
    tmp_path: Path, lake: Lander, capsys: pytest.CaptureFixture[str]
) -> None:
    root = lake([("https://primary.example.org/a", english(1))])
    registry = tmp_path / "sources.toml"
    registry.write_text(REGISTRY_TOML, encoding="utf-8")
    base = ["--lake", str(root)]
    assert main([*base, "refine-all", "--registry", str(registry)]) == 0
    refined = json.loads(capsys.readouterr().out)
    assert (refined["parts"], refined["skipped"]) == (1, 1)

    assert main([*base, "gold", "--policy", str(ROOT / "config" / "gold.toml")]) == 0
    gold = json.loads(capsys.readouterr().out)
    assert (gold["documents"], gold["rejects"]) == (1, 0)

    tokenizer = str(word_tokenizer(tmp_path / "t.json"))
    assert (
        main([*base, "tokens", gold["snapshot"], "--tokenizer", tokenizer, "--eos", "<eos>"]) == 0
    )
    tokens = json.loads(capsys.readouterr().out)
    assert tokens["documents"] == 1
    assert tokens["tokens"] > 100
    assert len(list((root / "_runs").iterdir())) == 3
