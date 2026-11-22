import unicodedata

from trustpipe.ids import canonical_text, content_id, document_id, is_content_id


def test_content_id_is_prefixed_sha256() -> None:
    assert content_id(b"hello") == (
        "sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_document_id_ignores_unicode_composition() -> None:
    composed = "한국어 café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert document_id(composed) == document_id(decomposed)


def test_document_id_ignores_newline_style() -> None:
    assert document_id("a\r\nb\rc") == document_id("a\nb\nc")


def test_compatibility_characters_are_not_folded() -> None:
    # NFKC would turn these into ASCII and silently change the content.
    for compat, plain in [("ｆｕｌｌ", "full"), ("ﬁle", "file")]:  # noqa: RUF001
        assert canonical_text(compat) == compat
        assert document_id(compat) != document_id(plain)


def test_different_text_gets_a_different_id() -> None:
    assert document_id("a") != document_id("b")


def test_is_content_id() -> None:
    assert is_content_id(content_id(b""))
    assert not is_content_id("sha256:XYZ")
    assert not is_content_id("md5:" + "0" * 32)
