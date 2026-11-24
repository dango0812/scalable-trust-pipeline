import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from corpus import english, korean
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from trustpipe.cli import main
from trustpipe.gold.policy import GoldPolicy
from trustpipe.gold.stages import build_gold
from trustpipe.index.build import build_index
from trustpipe.index.dense import DenseIndex
from trustpipe.index.embed import HashingEmbedder, TransformerEmbedder, embedder_from_spec
from trustpipe.index.lexical import LexicalIndex, terms
from trustpipe.index.search import Retriever, evaluate
from trustpipe.lake.schema import TrustTier

Lander = Callable[[list[tuple[str, str]]], Path]


def test_hashing_vectors_are_unit_length_and_reflect_overlap() -> None:
    embedder = HashingEmbedder(64)
    a, b, c = embedder.embed_passages(["trust registry policy", "trust registry rules", "고양이"])
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert float(a @ b) > float(a @ c)
    assert np.array_equal(embedder.embed_queries(["x"]), embedder.embed_passages(["x"]))
    assert embedder.embed_passages([]).shape == (0, 64)
    assert embedder_from_spec(embedder.spec).spec == embedder.spec
    with pytest.raises(ValueError, match="at least 8"):
        HashingEmbedder(4)
    with pytest.raises(ValueError, match="unknown embedder kind"):
        embedder_from_spec('{"kind": "other"}')


def tiny_encoder(directory: Path) -> Path:
    import torch
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tokenizer.train_from_iterator(
        [english(1, 300)], trainers.WordLevelTrainer(special_tokens=special)
    )
    fast = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    fast.save_pretrained(directory)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(fast),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(config).save_pretrained(directory)
    return directory


def test_a_transformer_encoder_is_pooled_and_rebuilt_from_its_spec(tmp_path: Path) -> None:
    model = tiny_encoder(tmp_path / "encoder")
    embedder = TransformerEmbedder(str(model), batch_size=2, max_length=32)
    passages = embedder.embed_passages(["trust policy source", "data model", "record capture"])
    assert passages.shape == (3, embedder.dimensions)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
    assert embedder.embed_queries([]).shape == (0, 32)
    rebuilt = embedder_from_spec(embedder.spec, model_path=model)
    assert np.allclose(
        rebuilt.embed_queries(["trust"]), embedder.embed_queries(["trust"]), atol=1e-5
    )
    assert json.loads(embedder.spec)["query_prefix"] == "query: "


def test_terms_split_latin_words_and_cjk_bigrams() -> None:
    assert terms("Trust, REGISTRY!") == ["trust", "registry"]
    assert terms("학습데이터를 검색") == ["학습", "습데", "데이", "이터", "터를", "검색"]
    assert terms("가 a") == ["가", "a"]


def test_bm25_ranks_rare_matches_first_and_round_trips(tmp_path: Path) -> None:
    texts = ["the trust registry decides", "the model reads text", "registry registry of trust"]
    index = LexicalIndex.build(texts)
    ranked = index.search("registry trust", 3)
    assert ranked[0][0] == 2
    assert {doc for doc, _ in ranked} == {0, 2}
    assert index.search("unknown words", 3) == []
    index.save(tmp_path)
    assert LexicalIndex.load(tmp_path).search("registry trust", 3) == ranked
    assert LexicalIndex.build([]).search("anything", 3) == []


def test_dense_indexes_switch_to_hnsw_above_the_exact_limit() -> None:
    vectors = HashingEmbedder(16).embed_passages([f"text {i}" for i in range(20)])
    assert DenseIndex.build(vectors).kind == "IndexFlatIP"
    graph = DenseIndex.build(vectors, exact_limit=5)
    assert graph.kind == "IndexHNSWFlat"
    assert graph.search(vectors[7], 1)[0][0] == 7


@pytest.fixture
def snapshot(lake: Lander) -> tuple[Path, str]:
    root = lake(
        [
            ("https://primary.example.org/registry", english(11, 200)),
            ("https://agency.example.org/korean", korean(12, 200)),
            ("https://mirror.example.net/review", english(13, 200)),
        ]
    )
    return root, build_gold(root, GoldPolicy(version="test")).snapshot


def test_hybrid_search_returns_passages_with_provenance(snapshot: tuple[Path, str]) -> None:
    root, snap = snapshot
    built = build_index(root, snap, HashingEmbedder())
    assert built.chunks >= 3
    retriever = Retriever(built.directory)
    assert len(retriever) == built.chunks
    query = korean(12, 200).split("\n")[0]
    [best, *_] = retriever.search(query, 3)
    assert best.source_uri == "https://agency.example.org/korean"
    assert best.tier == "institutional"
    assert best.cosine is not None
    assert best.bm25 is not None
    assert best.char_end > best.char_start
    trusted = retriever.search(query, 5, min_tier=TrustTier.PRIMARY)
    assert {hit.source_uri for hit in trusted} == {"https://primary.example.org/registry"}
    assert retriever.search(query, 0) == []
    assert build_index(root, snap, HashingEmbedder()).receipt == built.receipt


def test_retrieval_is_evaluated_by_recall_and_reciprocal_rank(snapshot: tuple[Path, str]) -> None:
    root, snap = snapshot
    retriever = Retriever(build_index(root, snap, HashingEmbedder()).directory)
    cases = [
        (english(11, 200).split("\n")[2], "https://primary.example.org/registry"),
        (korean(12, 200).split("\n")[1], "https://agency.example.org/korean"),
        ("question about nothing indexed", "https://nowhere.example.org/"),
    ]
    result = evaluate(retriever, cases, k=3)
    assert (result.questions, result.k) == (3, 3)
    assert result.recall == pytest.approx(2 / 3)
    assert 0 < result.mrr <= result.recall
    assert evaluate(retriever, [], k=3).questions == 0


def test_an_index_must_be_queried_with_its_own_embedder(snapshot: tuple[Path, str]) -> None:
    root, snap = snapshot
    built = build_index(root, snap, HashingEmbedder())
    with pytest.raises(ValueError, match="differs"):
        Retriever(built.directory, embedder=HashingEmbedder(128))


def test_an_empty_snapshot_gives_an_empty_index(lake: Lander) -> None:
    root = lake([("https://blog.example.com/a", english(1))])
    built = build_index(
        root, build_gold(root, GoldPolicy(version="test")).snapshot, HashingEmbedder()
    )
    assert (built.chunks, built.dense_kind) == (0, "IndexFlatIP")
    assert Retriever(built.directory).search("anything") == []
    empty = np.zeros((0, 8), dtype=np.float32)
    assert DenseIndex.build(empty).search(np.zeros(8, dtype=np.float32), 3) == []


def test_a_snapshot_without_parts_gives_an_empty_index(tmp_path: Path) -> None:
    snap = build_gold(tmp_path, GoldPolicy(version="test")).snapshot
    assert build_index(tmp_path, snap, HashingEmbedder()).chunks == 0


def test_cli_builds_and_searches_an_index(
    snapshot: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, snap = snapshot
    assert main(["--lake", str(root), "index", snap]) == 0
    built = json.loads(capsys.readouterr().out)
    query = english(13, 200).split("\n")[0]
    assert (
        main(["--lake", str(root), "search", built["directory"], query, "--min-tier", "reviewed"])
        == 0
    )
    hits = json.loads(capsys.readouterr().out)["hits"]
    assert hits[0]["source_uri"] == "https://mirror.example.net/review"
