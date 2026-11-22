"""Content-addressed identifiers.

A record's id is the SHA-256 of its canonical bytes, so the same content gets the same id
on every machine and in every run. Two properties follow that matter at scale:

- Idempotent writes: re-running a failed shard reproduces the same ids, so retries never
  create duplicates.
- Free exact deduplication: equal ids mean equal content, no text comparison needed.

Ids carry their algorithm as a prefix ("sha256:<hex>"), like OCI image digests, so a
later switch to another hash can coexist with old ids.
"""

import hashlib
import re
import unicodedata

ID_PREFIX = "sha256:"
_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def canonical_text(text: str) -> str:
    """Return the canonical form a document id is computed from.

    Newlines are unified to LF and the text is NFC-normalized, so the same characters
    typed on Windows or macOS (which may produce decomposed Hangul or accents) hash alike.
    NFKC is deliberately not used: it rewrites compatibility characters such as
    full-width letters, ligatures and Hangul compatibility jamo, which changes the content.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", unified)


def content_id(data: bytes) -> str:
    """Id of raw bytes, e.g. a captured HTTP payload."""
    return ID_PREFIX + hashlib.sha256(data).hexdigest()


def document_id(text: str) -> str:
    """Id of a document's text, computed over its canonical UTF-8 bytes."""
    return content_id(canonical_text(text).encode("utf-8"))


def is_content_id(value: str) -> bool:
    return _ID_PATTERN.fullmatch(value) is not None
