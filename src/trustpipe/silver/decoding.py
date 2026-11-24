"""Decode captured bytes into text, deterministically.

Decoding follows the WHATWG Encoding Standard where it matters for determinism: a byte
order mark overrides the declared charset and is removed, and only an allow-list of labels
is accepted, each mapped to the codec the standard prescribes (for example "iso-8859-1" to
windows-1252). The same bytes therefore decode the same way on every platform; any other
label is rejected as unsupported.
"""

from email.message import Message
from functools import lru_cache

from trustpipe.lake.schema import RawCapture

# WHATWG Encoding Standard labels this pipeline decodes, with the codec the standard maps
# each one to. https://encoding.spec.whatwg.org/#names-and-labels
_CODECS = {
    "utf-8": "utf-8",
    "utf8": "utf-8",
    "unicode-1-1-utf-8": "utf-8",
    "us-ascii": "cp1252",
    "ascii": "cp1252",
    "iso-8859-1": "cp1252",
    "iso8859-1": "cp1252",
    "latin1": "cp1252",
    "l1": "cp1252",
    "windows-1252": "cp1252",
    "cp1252": "cp1252",
    "euc-kr": "cp949",
    "ks_c_5601-1987": "cp949",
    "windows-949": "cp949",
    "utf-16": "utf-16-le",
    "utf-16le": "utf-16-le",
    "utf-16be": "utf-16-be",
}
_BOMS = ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))


class UnsupportedCharsetError(ValueError):
    """The declared charset is not on the allow-list."""


def media_type(content_type: str | None) -> str:
    """The lower-case media type of a Content-Type value; text/plain when there is none."""
    return _parsed(content_type)[0]


def declared_charset(content_type: str | None) -> str | None:
    """The charset label of a Content-Type value, lower-cased, if it declares one."""
    return _parsed(content_type)[1]


# Content-Type values repeat across a part (one per server configuration), so each one is
# parsed once: the email parser costs ~20 µs per call and dominated decoding before.
@lru_cache(maxsize=4096)
def _parsed(content_type: str | None) -> tuple[str, str | None]:
    header = Message()
    header["content-type"] = content_type or "text/plain"
    return header.get_content_type(), header.get_content_charset()


def codec_for(label: str) -> str:
    """The Python codec the WHATWG standard prescribes for a charset label."""
    codec = _CODECS.get(label.strip().lower())
    if codec is None:
        raise UnsupportedCharsetError(f"charset {label!r} is not supported")
    return codec


def decode_bytes(raw: bytes, content_type: str | None, *, fallback: str = "utf-8") -> str:
    """Decode strictly: a byte order mark first, then the declared charset, then fallback."""
    for bom, bom_codec in _BOMS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(bom_codec)
    return raw.decode(codec_for(declared_charset(content_type) or fallback))


def decode(capture: RawCapture) -> str:
    """Decode a capture's payload (see ``decode_bytes``)."""
    return decode_bytes(capture.raw, capture.content_type)
