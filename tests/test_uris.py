import pytest
from hypothesis import given
from hypothesis import strategies as st

from trustpipe.uris import SourceUri, canonical_uri, parse_uri


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("https://github.com/dango0812", "https://github.com/dango0812"),
        ("HTTPS://GitHub.COM/dango0812", "https://github.com/dango0812"),
        ("https://github.com:443/dango0812", "https://github.com/dango0812"),
        ("http://example.org:80/", "http://example.org/"),
        ("https://example.org:8443/a", "https://example.org:8443/a"),
        ("https://example.org:/a", "https://example.org/a"),
        ("https://example.org.", "https://example.org/"),
        ("https://example.org", "https://example.org/"),
        ("https://example.org/a#section", "https://example.org/a"),
        ("https://example.org/%7Euser/%41", "https://example.org/~user/A"),
        ("https://example.org/a%2fb", "https://example.org/a%2Fb"),
        ("https://example.org/a/./b/../c", "https://example.org/a/c"),
        ("https://example.org/a/%2e%2E/b", "https://example.org/b"),
        ("https://example.org/a/..", "https://example.org/"),
        ("https://example.org/../../x", "https://example.org/x"),
        ("https://example.org/a/b/.", "https://example.org/a/b/"),
        ("https://example.org/a?q=%7e%2a", "https://example.org/a?q=~%2A"),
        ("https://example.org/a?", "https://example.org/a"),
        ("https://xn--3e0b707e.kr/", "https://xn--3e0b707e.kr/"),
    ],
)
def test_equivalent_spellings_share_one_canonical_form(spelling: str, canonical: str) -> None:
    assert canonical_uri(spelling) == canonical


@pytest.mark.parametrize(
    ("uri", "problem"),
    [
        ("ftp://example.org/file", "http or https"),
        ("example.org/a", "http or https"),
        ("https://user@example.org/a", "userinfo"),
        ("https://github.com@evil.example/dango0812", "userinfo"),
        ("https://[2001:db8::1]/a", "IP literal"),
        ("https://exa_mple.org/a", "invalid host"),
        ("https:///a", "invalid host"),
        ("https://example..org/a", "invalid host"),
        ("https://example.org:99999/a", "invalid port"),
        ("https://example.org:8o/a", "invalid port"),
        ("https://example.org/a b", "not allowed in a URI"),
        ("https://example.org/a\\b", "not allowed in a URI"),
        ("https://example.org/한국", "not allowed in a URI"),
        ("https://example.org/100%", "two hex digits"),
        ("https://example.org/a?x=%zz", "two hex digits"),
    ],
)
def test_uris_without_a_canonical_form_are_rejected(uri: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_uri(uri)


def test_parsed_parts_and_segments() -> None:
    uri = parse_uri("https://Example.org:8443/a/b/?x=1")
    assert uri == SourceUri("https", "example.org", 8443, "/a/b/", "x=1")
    assert uri.segments == ("a", "b", "")
    assert parse_uri("https://example.org").segments == ()
    assert parse_uri("https://example.org/a").segments == ("a",)


UNRESERVED = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
segment = st.text(alphabet=UNRESERVED + "%!$&'()*+,;=:@", min_size=0, max_size=8).filter(
    lambda s: "%" not in s
)
host = st.lists(
    st.from_regex(r"[a-z0-9](?:[a-z0-9-]{0,8}[a-z0-9])?", fullmatch=True), min_size=1, max_size=3
).map(".".join)
uri = st.builds(
    lambda scheme, h, segments, dots: (
        f"{scheme}://{h}/" + "/".join(s for pair in zip(segments, dots, strict=False) for s in pair)
    ),
    st.sampled_from(["http", "https"]),
    host,
    st.lists(segment, max_size=6),
    st.lists(st.sampled_from([".", "..", "x"]), max_size=6),
)


@given(uri)
def test_canonicalisation_is_idempotent(value: str) -> None:
    once = canonical_uri(value)
    assert canonical_uri(once) == once


@given(uri)
def test_canonical_paths_never_keep_dot_segments(value: str) -> None:
    segments = parse_uri(value).segments
    assert "." not in segments
    assert ".." not in segments


@given(uri, st.data())
def test_percent_encoding_unreserved_characters_changes_nothing(
    value: str, data: st.DataObject
) -> None:
    parsed = parse_uri(value)
    if parsed.path == "/":
        return
    index = data.draw(st.integers(min_value=1, max_value=len(parsed.path) - 1))
    character = parsed.path[index]
    if character not in UNRESERVED:
        return
    encoded = f"%{ord(character):02x}"
    respelled = str(parsed).replace(
        parsed.path, parsed.path[:index] + encoded + parsed.path[index + 1 :], 1
    )
    assert canonical_uri(respelled) == str(parsed)
