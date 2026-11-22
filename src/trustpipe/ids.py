"""Content-addressed identifiers.

Three ids with three different meanings:

    payload_id  SHA-256 of the raw bytes                       which bytes arrived
    capture_id  typed hash of (source URI, fetch time, payload) which capture event happened
    doc_id      SHA-256 of the canonical text                  which text a model may see

The same bytes fetched from two places, or twice from one place, are two captures with one
payload_id; their texts share one doc_id. Keeping the three apart is what lets the lake
count corroborating sources, deduplicate text and still trace every document back to the
exact event that produced it.

payload_id and doc_id are plain SHA-256 digests, so anyone can check them with standard
tools. capture_id and every other id built from several fields use ``typed_hash``: each
field is length-prefixed and the record kind is hashed first, so two different records
can never share an encoding and ids of different kinds can never collide.

Ids are stored as 32 raw bytes (Arrow ``fixed_size_binary(32)``); "sha256:<hex>" is their
text form at the boundaries (CLI output, JSON files, logs).

Because every id is a function of content, a retried task reproduces the same ids, and
equal ids mean equal content without comparing text.
"""

import hashlib
import re
import unicodedata
from datetime import UTC, datetime, timedelta

ID_BYTES = 32
TEXT_PREFIX = "sha256:"
_TEXT_ID = re.compile(r"sha256:[0-9a-f]{64}")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Name of the canonical text form. NFC is only guaranteed stable for characters that are
# assigned in the Unicode version used, so the version is part of the name and every file
# holding doc_ids records it.
CANONICAL_FORM = f"nfc-lf/ucd-{unicodedata.unidata_version}"


def canonical_text(text: str) -> str:
    """Return the canonical form a doc_id is computed from.

    Newlines are unified to LF and the text is NFC-normalized, so the same characters typed
    on Windows or macOS (which may produce decomposed Hangul or accents) hash alike. NFKC is
    deliberately not used: it rewrites compatibility characters such as full-width letters,
    ligatures and Hangul compatibility jamo, which changes the content.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", unified)


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def typed_hash(kind: str, *fields: bytes) -> bytes:
    """Hash a record of several byte fields without ambiguity between records or kinds."""
    digest = hashlib.sha256(kind.encode("ascii") + b"\x00")
    for value in fields:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.digest()


def payload_id(raw: bytes) -> bytes:
    return sha256(raw)


def doc_id(text: str) -> bytes:
    """Id of a document's text, computed over its canonical UTF-8 bytes."""
    return sha256(canonical_text(text).encode("utf-8"))


def epoch_micros(moment: datetime) -> int:
    """Microseconds since the Unix epoch of a timezone-aware moment, exactly."""
    return (moment - _EPOCH) // timedelta(microseconds=1)


def capture_id(source_uri: str, fetched_at: datetime, payload: bytes) -> bytes:
    """Id of one capture event: these bytes, from this URI, at this moment."""
    return typed_hash(
        "capture",
        source_uri.encode("utf-8"),
        epoch_micros(fetched_at).to_bytes(8, "big", signed=True),
        payload,
    )


def to_text(identifier: bytes) -> str:
    if len(identifier) != ID_BYTES:
        raise ValueError(f"an id has {ID_BYTES} bytes, got {len(identifier)}")
    return TEXT_PREFIX + identifier.hex()


def from_text(text: str) -> bytes:
    if not _TEXT_ID.fullmatch(text):
        raise ValueError(f"an id looks like sha256:<64 lower-case hex digits>, got {text!r}")
    return bytes.fromhex(text.removeprefix(TEXT_PREFIX))
