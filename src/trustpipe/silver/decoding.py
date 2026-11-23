"""Decode captured bytes into text, deterministically.

Decoding follows the WHATWG Encoding Standard where it matters for determinism: a byte
order mark overrides the declared charset and is removed, and only an allow-list of labels
is accepted, each mapped to the codec the standard prescribes (for example "iso-8859-1" to
windows-1252). The same bytes therefore decode the same way on every platform; any other
label is rejected as unsupported.
"""

from email.message import Message

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


def decode(capture: RawCapture) -> str:
    """Decode the payload strictly: a byte order mark first, then the declared charset."""
    raw = capture.raw
    for bom, bom_codec in _BOMS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(bom_codec)
    header = Message()
    header["content-type"] = capture.content_type or "text/plain"
    label = header.get_content_charset() or "utf-8"
    codec = _CODECS.get(label)
    if codec is None:
        raise UnsupportedCharsetError(f"charset {label!r} is not supported")
    return raw.decode(codec)
