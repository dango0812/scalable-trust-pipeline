import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trustpipe.bronze.ingest import land
from trustpipe.identity.uris import parse_uri, split_canonical
from trustpipe.lake.schema import (
    BRONZE_CAPTURES,
    Acquisition,
    Document,
    RawCapture,
    Reject,
    TrustRecord,
    from_table,
)
from trustpipe.lake.storage import read_table, write_table
from trustpipe.silver.decoding import codec_for, declared_charset, media_type
from trustpipe.silver.refine import BronzeIntegrityError, refine, refine_part, refine_table
from trustpipe.silver.script import ASCII_LETTER, HANGUL, LETTER, class_table, detect_script
from trustpipe.trust.registry import load_registry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = load_registry(ROOT / "config" / "sources.toml")
AT = datetime(2026, 11, 24, 9, 30, tzinfo=UTC)

_ASCII = re.compile(r"[A-Za-z]")
_LETTERS = re.compile(r"[^\W\d_]")
_HANGULS = re.compile("[가-힣ᄀ-ᇿ㄰-㆏]")


def reference_script(text: str) -> str | None:
    """The regex-counting implementation the class table replaced, kept as the oracle."""
    if text.isascii():
        return "Latn" if _ASCII.search(text) else None
    letters = len(_LETTERS.findall(text))
    if letters == 0:
        return None
    if len(_HANGULS.findall(text)) / letters >= 0.3:
        return "Hang"
    if len(_ASCII.findall(text)) / letters >= 0.5:
        return "Latn"
    return None


@given(st.text(alphabet=st.characters(codec=None), max_size=200))
def test_the_class_table_gives_the_same_guess_as_counting_matches(text: str) -> None:
    assert detect_script(text) == reference_script(text)


@pytest.mark.parametrize(
    "text",
    [
        "한국어 text mixed 데이터",
        "über café ünd",
        "日本語のテキスト",
        "١٢٣ ٤٥",
        "ㄱㄴㄷ abc",
        "\ud800 a",
    ],
)
def test_mixed_scripts_match_the_reference(text: str) -> None:
    assert detect_script(text) == reference_script(text)


def test_the_class_table_flags() -> None:
    table = class_table()
    assert table[ord("가")] == LETTER | HANGUL
    assert table[ord("a")] == LETTER | ASCII_LETTER
    assert table[ord("é")] == LETTER
    assert table[ord("1")] == table[ord("_")] == table[ord(" ")] == 0


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.org/",
        "https://example.org/a/b/",
        "http://example.org:8080/a?x=1/2?y",
        "https://example.org/%2Fa?",
        "HTTPS://Example.ORG:443/a/./b/../c?q",
    ],
)
def test_split_canonical_inverts_the_canonical_spelling(uri: str) -> None:
    parsed = parse_uri(uri)
    assert split_canonical(str(parsed)) == parsed


@given(
    st.sampled_from(["http", "https"]),
    st.from_regex(r"[a-z0-9]{1,8}(\.[a-z0-9]{1,8}){0,2}", fullmatch=True),
    st.one_of(st.none(), st.integers(1, 65535)),
    st.lists(st.from_regex(r"[A-Za-z0-9._~-]{0,6}", fullmatch=True), max_size=4),
    st.one_of(st.none(), st.from_regex(r"[A-Za-z0-9=&/?]{1,8}", fullmatch=True)),
)
def test_split_canonical_agrees_with_parse_uri(
    scheme: str, host: str, port: int | None, segments: list[str], query: str | None
) -> None:
    authority = host if port is None else f"{host}:{port}"
    uri = f"{scheme}://{authority}/{'/'.join(segments)}" + (f"?{query}" if query else "")
    parsed = parse_uri(uri)
    assert split_canonical(str(parsed)) == parsed


@pytest.mark.parametrize(
    ("content_type", "media", "charset"),
    [
        (None, "text/plain", None),
        ("text/HTML; Charset=EUC-KR", "text/html", "euc-kr"),
        ("application/xhtml+xml", "application/xhtml+xml", None),
    ],
)
def test_content_type_parsing(content_type: str | None, media: str, charset: str | None) -> None:
    assert media_type(content_type) == media
    assert declared_charset(content_type) == charset


def test_codec_labels_are_trimmed_and_case_insensitive() -> None:
    assert codec_for(" UTF-8 ") == "utf-8"


def mixed_captures() -> list[RawCapture]:
    fetch, archive = Acquisition.HTTP_FETCH, Acquisition.ARCHIVE_IMPORT
    submission = Acquisition.SUBMISSION
    utf8, euc_kr = "text/plain; charset=utf-8", "text/plain; charset=euc-kr"
    rows = [
        (b"first text", "https://example.org/1", None, fetch),
        (b"\xff", "https://example.org/2", utf8, fetch),
        ("둘째".encode("cp949"), "https://example.org/3", euc_kr, archive),
        (b"  ", "https://example.org/4", None, fetch),
        (b"owner text", "https://github.com/dango0812", None, submission),
    ]
    return [
        RawCapture.from_bytes(
            raw, source_uri=uri, fetched_at=AT, acquisition=method, content_type=content_type
        )
        for raw, uri, content_type, method in rows
    ]


def test_the_bulk_path_gives_the_same_rows_as_the_record_path(tmp_path: Path) -> None:
    captures = mixed_captures()
    key = land(captures, tmp_path).outputs[0].key
    bulk = refine_table(read_table(tmp_path, key, schema=BRONZE_CAPTURES), REGISTRY)
    records = refine(captures, REGISTRY)
    assert from_table(bulk.documents, Document) == records.documents
    assert from_table(bulk.rejects, Reject) == records.rejects
    assert from_table(bulk.trust, TrustRecord) == records.trust
    assert [r.reason for r in records.rejects] == ["undecodable", "empty_text"]
    assert [t.tier.value for t in records.trust] == ["unverified", "unverified", "primary"]


def test_refine_part_refuses_a_bronze_file_without_a_receipt(tmp_path: Path) -> None:
    key = land(mixed_captures(), tmp_path).outputs[0].key
    for path in (tmp_path / "_receipts").rglob("*.json"):
        path.unlink()
    with pytest.raises(BronzeIntegrityError, match=r"has no bronze\.ingest receipt"):
        refine_part(tmp_path, key, REGISTRY)


def test_refine_part_refuses_a_bronze_file_that_changed_after_its_receipt(
    tmp_path: Path,
) -> None:
    key = land(mixed_captures(), tmp_path).outputs[0].key
    table = read_table(tmp_path, key, schema=BRONZE_CAPTURES).slice(0, 2)
    write_table(table, tmp_path, key, create_only=False)
    with pytest.raises(BronzeIntegrityError, match="does not match the SHA-256"):
        refine_part(tmp_path, key, REGISTRY)
