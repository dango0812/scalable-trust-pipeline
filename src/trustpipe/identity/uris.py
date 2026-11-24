"""Canonical form of source URIs.

Trust is decided from where a capture came from. Two spellings of one location must
therefore compare equal, and a URI must never look as if it lies under a trusted prefix
while it resolves somewhere else (``/trusted/../other``, ``trusted@evil.example``).

Every source URI is reduced to one canonical form following the syntax-based
normalisation of RFC 3986 section 6.2.2, restricted to what is unambiguous:

- scheme and host in lower case; only http and https
- no userinfo, no fragment, no default port, no trailing dot on the host
- percent-escapes in upper case; escapes of unreserved characters decoded
- dot segments removed, including their percent-encoded spellings
- an empty path written as "/"

Anything that cannot be put in this form is rejected instead of guessed at: a URI that
two parsers could read differently is exactly the kind an attacker would choose.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}

# RFC 3986 section 2: the only characters a URI may contain. Spaces, control characters,
# non-ASCII characters, backslashes, quotes and the like must arrive percent-encoded.
_URI_CHARACTERS = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+")
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_BROKEN_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_PORT = re.compile(r"[0-9]{1,5}")


@dataclass(frozen=True, slots=True)
class SourceUri:
    """A source URI in canonical form, split into the parts trust rules compare."""

    scheme: str
    host: str
    port: int | None  # None means the default port of the scheme
    path: str  # always starts with "/"
    query: str | None

    @property
    def segments(self) -> tuple[str, ...]:
        """Path segments: "/" gives (), "/a" gives ("a",), "/a/b/" gives ("a", "b", "")."""
        return () if self.path == "/" else tuple(self.path[1:].split("/"))

    def __str__(self) -> str:
        authority = self.host if self.port is None else f"{self.host}:{self.port}"
        query = "" if self.query is None else f"?{self.query}"
        return f"{self.scheme}://{authority}{self.path}{query}"


def _normalise_escapes(component: str, what: str, uri: str) -> str:
    if _BROKEN_ESCAPE.search(component):
        raise ValueError(f"{what} of {uri!r} has a '%' that is not followed by two hex digits")

    def replace(match: re.Match[str]) -> str:
        character = chr(int(match[1], 16))
        return character if character in _UNRESERVED else f"%{match[1].upper()}"

    return _ESCAPE.sub(replace, component)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 section 5.2.4 for an absolute path."""
    kept: list[str] = []
    segments = path.split("/")[1:]
    for segment in segments:
        if segment == "..":
            if kept:
                kept.pop()
        elif segment != ".":
            kept.append(segment)
    # "/a/." and "/a/.." name a directory, so the result keeps its trailing slash.
    if segments[-1] in (".", ".."):
        kept.append("")
    return "/" + "/".join(kept)


def _host(netloc: str, uri: str) -> tuple[str, str | None]:
    if "@" in netloc:
        raise ValueError(f"{uri!r} carries userinfo before the host, which is not allowed")
    if "[" in netloc or "]" in netloc:
        raise ValueError(f"{uri!r} uses an IP literal; the host must be a DNS name")
    host, colon, port = netloc.partition(":")
    host = host.lower().removesuffix(".")
    labels = host.split(".")
    if len(host) > 253 or not all(_HOST_LABEL.fullmatch(label) for label in labels):
        raise ValueError(
            f"{uri!r} has an invalid host; use lower-case ASCII labels "
            "(internationalised names in their xn-- form)"
        )
    # RFC 3986 section 6.2.3: an empty port ("host:") is the same as no port.
    return host, (port if colon and port else None)


def parse_uri(uri: str) -> SourceUri:
    """Parse and canonicalise a source URI; raise ValueError if it has no canonical form."""
    if not _URI_CHARACTERS.fullmatch(uri):
        raise ValueError(
            f"{uri!r} contains characters that are not allowed in a URI; percent-encode them"
        )
    parts = urlsplit(uri)
    if parts.scheme not in DEFAULT_PORTS:
        raise ValueError(f"{uri!r} must be an absolute http or https URI")
    host, port_text = _host(parts.netloc, uri)
    port: int | None = None
    if port_text is not None:
        if not _PORT.fullmatch(port_text) or not 0 < int(port_text) < 65536:
            raise ValueError(f"{uri!r} has an invalid port")
        port = int(port_text)
        if port == DEFAULT_PORTS[parts.scheme]:
            port = None
    path = _remove_dot_segments(_normalise_escapes(parts.path or "/", "path", uri))
    query = _normalise_escapes(parts.query, "query", uri) if parts.query else None
    return SourceUri(parts.scheme, host, port, path, query)


def split_canonical(uri: str) -> SourceUri:
    """Split a URI that is already canonical (as stored in bronze) without re-validating it.

    About ten times cheaper than ``parse_uri``; the bulk refine path uses it for URIs whose
    canonical form was checked when the capture was written and whose file was then
    checked against its receipt. ``split_canonical(str(u)) == u`` for every parsed ``u``.
    """
    scheme, _, rest = uri.partition("://")
    authority, slash, path_and_query = rest.partition("/")
    path, question, query = ("/" + path_and_query if slash else "/").partition("?")
    host, colon, port = authority.partition(":")
    return SourceUri(scheme, host, int(port) if colon else None, path, query if question else None)


def canonical_uri(uri: str) -> str:
    """The one spelling under which a source URI is stored and compared."""
    return str(parse_uri(uri))
