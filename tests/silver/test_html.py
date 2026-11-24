from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustpipe.lake.schema import Acquisition, RawCapture
from trustpipe.silver.html import extractor, main_text, meta_charset
from trustpipe.silver.refine import refine
from trustpipe.trust.registry import load_registry

REGISTRY = load_registry(Path(__file__).resolve().parents[2] / "config" / "sources.toml")
AT = datetime(2026, 11, 24, tzinfo=UTC)

ARTICLE = """<html><head><meta charset="{charset}"><title>Registry</title></head><body>
<nav><a href="/">Home</a> | <a href="/about">About</a> | <a href="/login">Login</a></nav>
<article><h1>신뢰 레지스트리</h1>
<p>신뢰 레지스트리는 출처를 얼마나 믿을 수 있는지 정책으로 정합니다. 문서가 스스로
공식이라고 말해도 등급은 바뀌지 않습니다.</p>
<p>모든 판정은 그 판정을 내린 정책과 함께 기록되므로, 모델이 배운 내용을 출처까지
거슬러 올라가 확인할 수 있습니다.</p></article>
<footer>Copyright 2026 Example</footer>
<section class="comments"><p>first!!! great post</p></section>
</body></html>"""


def page(raw: bytes, content_type: str | None) -> RawCapture:
    return RawCapture.from_bytes(
        raw,
        source_uri="https://example.org/registry",
        fetched_at=AT,
        acquisition=Acquisition.HTTP_FETCH,
        content_type=content_type,
    )


def test_the_main_text_keeps_the_article_and_drops_the_chrome() -> None:
    text = main_text(ARTICLE.format(charset="utf-8"))
    assert text is not None
    assert text.startswith("신뢰 레지스트리\n신뢰 레지스트리는 출처를")
    for chrome in ("Login", "Copyright", "first!!!"):
        assert chrome not in text


def test_meta_charset_is_found_near_the_start_only() -> None:
    assert meta_charset(b'<meta http-equiv="Content-Type" content="text/html; charset=EUC-KR">')
    assert meta_charset(b"<META CHARSET='Shift_JIS'>") == "shift_jis"
    assert meta_charset(b" " * 2000 + b'<meta charset="utf-8">') is None
    assert meta_charset(b"<p>no meta</p>") is None


def test_an_euc_kr_page_declared_only_in_meta_is_decoded() -> None:
    raw = ARTICLE.format(charset="euc-kr").encode("cp949")
    [document] = refine([page(raw, "text/html")], REGISTRY).documents
    assert document.text.startswith("신뢰 레지스트리")
    assert document.script == "Hang"


def test_the_http_charset_wins_over_meta() -> None:
    raw = ARTICLE.format(charset="euc-kr").encode("utf-8")
    [document] = refine([page(raw, "text/html; charset=utf-8")], REGISTRY).documents
    assert "출처를" in document.text


@pytest.mark.parametrize(
    ("raw", "content_type", "reason"),
    [
        (b"<html><body><script>var x = 1;</script></body></html>", "text/html", "no_main_text"),
        (b"%PDF-1.7", "application/pdf", "unsupported_media_type"),
        (b"\x89PNG", "image/png", "unsupported_media_type"),
        (ARTICLE.format(charset="koi8-r").encode(), "application/xhtml+xml", "unsupported_charset"),
    ],
)
def test_pages_without_usable_text_become_rejects(
    raw: bytes, content_type: str, reason: str
) -> None:
    [reject] = refine([page(raw, content_type)], REGISTRY).rejects
    assert reject.reason == reason


def test_markdown_is_plain_text_and_the_extractor_is_named() -> None:
    [document] = refine([page(b"# Title\n\nbody", "text/markdown")], REGISTRY).documents
    assert document.text == "# Title\n\nbody"
    assert extractor().startswith("trafilatura/")
