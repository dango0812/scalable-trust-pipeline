import dataclasses
from pathlib import Path

import numpy as np
import pytest
from corpus import english, korean

from trustpipe.gold.chunks import chunk_id, chunk_text
from trustpipe.gold.minhash import (
    UnionFind,
    band_hashes,
    candidate_groups,
    document_bands,
    normalise,
    shingle_hashes,
    signature,
)
from trustpipe.gold.policy import (
    ChunkRules,
    DedupRules,
    GoldPolicy,
    GoldPolicyError,
    QualityRules,
    load_gold_policy,
)
from trustpipe.lake.schema import TrustTier

ROOT = Path(__file__).resolve().parents[2]
BANDS = {"ngram": 5, "bands": 14, "rows": 8, "seed": 1}


def shared_bands(a: str, b: str) -> int:
    return sum(
        x == y for x, y in zip(document_bands(a, **BANDS), document_bands(b, **BANDS), strict=True)
    )


def test_normalisation_ignores_case_punctuation_and_width() -> None:
    assert normalise("Hello, WORLD! \uff46\uff55\uff4c\uff4c-width_text") == [
        "hello",
        "world",
        "full",
        "width",
        "text",
    ]


def test_near_copies_share_bands_and_different_texts_do_not() -> None:
    text = english(1, 300)
    edited = text.replace("source", "origin", 1)
    assert shared_bands(text, text) == 14
    assert shared_bands(text, edited) >= 1
    assert shared_bands(text, english(2, 300)) == 0
    assert shared_bands(korean(1, 300), korean(1, 300).replace("신뢰", "믿음", 1)) >= 1


def test_signatures_are_deterministic_and_bounded() -> None:
    shingles = shingle_hashes(normalise(english(3, 50)), 5)
    assert len(shingles) == 46
    first, second = signature(shingles, 112, 1), signature(shingles, 112, 1)
    assert np.array_equal(first, second)
    assert int(first.max()) < (1 << 61) - 1
    assert len(band_hashes(first, 14, 8)) == 14


def test_short_and_empty_texts_still_get_a_signature() -> None:
    assert len(shingle_hashes(["only", "three", "words"], 5)) == 1
    assert len(shingle_hashes([], 5)) == 0
    empty = signature(shingle_hashes([], 5), 8, 1)
    assert (empty == np.iinfo(np.uint64).max).all()


def test_large_documents_are_signed_in_blocks() -> None:
    words = [f"w{i}" for i in range(10_000)]
    shingles = shingle_hashes(words, 5)
    whole = signature(shingles, 16, 1)
    halves = np.minimum(signature(shingles[:5000], 16, 1), signature(shingles[5000:], 16, 1))
    assert np.array_equal(whole, halves)


def test_union_find_joins_documents_that_share_any_band() -> None:
    bands = np.array([[1, 2], [1, 3], [4, 3], [5, 6], [7, 6], [8, 9]], dtype=np.uint64)
    groups = candidate_groups(bands)
    roots = [groups.find(i) for i in range(6)]
    assert roots[0] == roots[1] == roots[2]
    assert roots[3] == roots[4] != roots[0]
    assert roots[5] not in (roots[0], roots[3])
    sets = UnionFind(3)
    sets.union(0, 1)
    sets.union(1, 0)
    sets.union(2, 0)
    assert sets.find(2) == sets.find(1) == 0
    assert sets.size[0] == 3


def test_chunks_pack_lines_and_keep_exact_offsets() -> None:
    text = "\n".join(f"line {n} " + "word " * 30 for n in range(20))
    chunks = chunk_text(text, ChunkRules(max_chars=500, min_chars=100))
    assert len(chunks) > 1
    assert all(c.text == text[c.char_start : c.char_end] for c in chunks)
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_long_lines_are_cut_at_spaces_or_at_the_limit() -> None:
    spaced = " ".join(["word"] * 100)
    pieces = chunk_text(spaced, ChunkRules(max_chars=60, min_chars=10))
    assert all(len(c.text) <= 60 and not c.text.startswith(" ") for c in pieces)
    unspaced = "가" * 250
    hard = chunk_text(unspaced, ChunkRules(max_chars=100, min_chars=10))
    assert [len(c.text) for c in hard] == [100, 100, 50]


def test_a_short_last_piece_joins_the_one_before() -> None:
    text = "a" * 90 + "\n" + "b" * 20
    [chunk] = chunk_text(text, ChunkRules(max_chars=100, min_chars=30))
    assert chunk.text == text
    assert chunk_text("   \n\n  ", ChunkRules()) == []


def test_chunk_ids_depend_on_document_and_position() -> None:
    doc = bytes(32)
    assert chunk_id(doc, 0) != chunk_id(doc, 1) != chunk_id(b"\x01" * 32, 1)


def test_the_policy_file_holds_the_defaults() -> None:
    policy = load_gold_policy(ROOT / "config" / "gold.toml")
    assert policy == GoldPolicy(version=policy.version)
    assert policy.min_tier is TrustTier.REVIEWED
    assert policy.policy_id == GoldPolicy(version=policy.version).policy_id
    assert policy.policy_id != dataclasses.replace(policy, min_tier=TrustTier.PRIMARY).policy_id


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('version = "1"\nextra = 1', "unknown top-level keys: extra"),
        ("min_tier = 'primary'", "missing 'version'"),
        ('version = "1"\nmin_tier = "gold"', "unknown min_tier"),
        ('version = "1"\n[quality]\nmin_word = 5', r"\[quality\] has unknown keys: min_word"),
        ('version = "1"\nquality = 5', r"\[quality\] must be a table"),
        ('version = "1"\n[dedup]\nbands = 0', "must be positive"),
        ('version = "1"\n[chunks]\nmin_chars = 2000', "0 < min_chars < max_chars"),
        ('version = " "', "version must not be empty"),
    ],
)
def test_the_policy_loader_is_strict(tmp_path: Path, toml: str, message: str) -> None:
    path = tmp_path / "gold.toml"
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(GoldPolicyError, match=message):
        load_gold_policy(path)


def test_policy_sections_convert_lists_to_tuples(tmp_path: Path) -> None:
    path = tmp_path / "gold.toml"
    path.write_text('version = "1"\n[quality]\ntop_ngrams = [[2, 0.5]]\n', encoding="utf-8")
    policy = load_gold_policy(path)
    assert policy.quality == QualityRules(top_ngrams=((2, 0.5),))
    assert policy.dedup == DedupRules()
