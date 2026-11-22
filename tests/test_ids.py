import unicodedata
from datetime import UTC, datetime, timedelta, timezone

import pytest

from trustpipe.ids import (
    CANONICAL_FORM,
    ID_BYTES,
    canonical_text,
    capture_id,
    doc_id,
    epoch_micros,
    from_text,
    payload_id,
    sha256,
    to_text,
    typed_hash,
)

AT = datetime(2026, 11, 22, 13, 13, tzinfo=UTC)


def test_payload_id_is_plain_sha256_of_the_bytes() -> None:
    assert payload_id(b"hello").hex() == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert len(payload_id(b"")) == ID_BYTES


def test_doc_id_ignores_unicode_composition() -> None:
    composed = "한국어 café"
    decomposed = unicodedata.normalize("NFD", composed)
    assert composed != decomposed
    assert doc_id(composed) == doc_id(decomposed)


def test_doc_id_ignores_newline_style() -> None:
    assert doc_id("a\r\nb\rc") == doc_id("a\nb\nc")


def test_compatibility_characters_are_not_folded() -> None:
    # NFKC would turn these into ASCII and silently change the content.
    for compat, plain in [("ｆｕｌｌ", "full"), ("ﬁle", "file")]:  # noqa: RUF001
        assert canonical_text(compat) == compat
        assert doc_id(compat) != doc_id(plain)


def test_typed_hash_separates_field_boundaries_and_kinds() -> None:
    assert typed_hash("k", b"ab", b"c") != typed_hash("k", b"a", b"bc")
    assert typed_hash("k", b"x") != typed_hash("j", b"x")
    assert typed_hash("k") != sha256(b"k")


def test_capture_id_identifies_the_event_not_the_bytes() -> None:
    payload = payload_id(b"same bytes")
    base = capture_id("https://a.example/x", AT, payload)
    kst = AT.astimezone(timezone(timedelta(hours=9)))
    assert base == capture_id("https://a.example/x", kst, payload)  # same instant
    assert base != capture_id("https://b.example/x", AT, payload)
    assert base != capture_id("https://a.example/x", AT + timedelta(microseconds=1), payload)
    assert base != capture_id("https://a.example/x", AT, payload_id(b"other bytes"))
    assert base != payload


def test_epoch_micros_is_exact_on_both_sides_of_the_epoch() -> None:
    assert epoch_micros(datetime(1970, 1, 1, tzinfo=UTC)) == 0
    assert epoch_micros(datetime(1969, 12, 31, 23, 59, 59, 999_999, tzinfo=UTC)) == -1
    moment = datetime(2026, 11, 22, 13, 13, 0, 123_456, tzinfo=UTC)
    assert epoch_micros(moment) == int(moment.replace(microsecond=0).timestamp()) * 10**6 + 123_456


def test_text_form_round_trips_and_is_validated() -> None:
    identifier = payload_id(b"x")
    assert to_text(identifier).startswith("sha256:")
    assert from_text(to_text(identifier)) == identifier
    with pytest.raises(ValueError, match="32 bytes"):
        to_text(b"short")
    with pytest.raises(ValueError, match="sha256:<64"):
        from_text("sha256:ABC")


def test_canonical_form_names_the_unicode_version_it_depends_on() -> None:
    assert f"nfc-lf/ucd-{unicodedata.unidata_version}" == CANONICAL_FORM
