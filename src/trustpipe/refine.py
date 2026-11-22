"""Silver refinement: turn bronze captures into canonical documents and trust rows.

Each capture yields one silver.documents row (canonical text) and one silver.trust row
(provenance, license and the tier the registry assigns), written in the same order so
the two parts stay aligned row for row.
"""

from email.message import Message
from pathlib import Path
from urllib.parse import urlsplit

from trustpipe.layout import parse_bronze_captures_key, silver_documents_key, silver_trust_key
from trustpipe.registry import UNREGISTERED_LICENSE, UNREGISTERED_REASON, Registry
from trustpipe.schema import (
    BRONZE_CAPTURES,
    Document,
    RawCapture,
    TrustRecord,
    TrustTier,
    from_table,
    to_table,
)
from trustpipe.storage import read_table, write_table


def decode(capture: RawCapture) -> str:
    """Decode the payload with the charset its content type declares (UTF-8 if none).

    Decoding is strict: bytes that do not match the declared charset are an error, not
    something to paper over with replacement characters.
    """
    header = Message()
    header["content-type"] = capture.content_type or "text/plain"
    charset = header.get_content_charset() or "utf-8"
    return capture.raw.decode(charset)


def detect_lang(text: str) -> str | None:
    """Guess the dominant script: "ko" for Hangul, "en" for Latin, otherwise None.

    A deliberate simplification: production pipelines use a trained language identifier
    (fastText-style models); this keeps the single-record pipeline dependency-free.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    hangul = sum("가" <= ch <= "힣" or "ᄀ" <= ch <= "ᇿ" for ch in letters)
    latin = sum(ch.isascii() for ch in letters)
    if hangul / len(letters) >= 0.3:
        return "ko"
    if latin / len(letters) >= 0.5:
        return "en"
    return None


def refine(
    captures: list[RawCapture], registry: Registry
) -> tuple[list[Document], list[TrustRecord]]:
    documents: list[Document] = []
    trust: list[TrustRecord] = []
    for capture in captures:
        text = decode(capture)
        doc = Document.from_text(text, capture_id=capture.capture_id, lang=detect_lang(text))
        rule = registry.match(capture.source_uri)
        documents.append(doc)
        trust.append(
            TrustRecord(
                doc_id=doc.doc_id,
                source_uri=capture.source_uri,
                source_domain=(urlsplit(capture.source_uri).hostname or "").lower(),
                publisher=rule.publisher if rule else None,
                license=rule.license if rule else UNREGISTERED_LICENSE,
                retrieved_at=capture.fetched_at,
                tier=rule.tier if rule else TrustTier.UNVERIFIED,
                tier_reason=rule.reason if rule else UNREGISTERED_REASON,
                policy_version=registry.version,
            )
        )
    return documents, trust


def refine_part(
    root: Path, bronze_key: str, registry: Registry
) -> tuple[list[TrustRecord], str, str]:
    """Refine one bronze part into its mirrored silver parts; return trust rows and keys."""
    source, day, shard = parse_bronze_captures_key(bronze_key)
    captures = from_table(read_table(root, bronze_key, schema=BRONZE_CAPTURES), RawCapture)
    documents, trust = refine(captures, registry)
    documents_key = silver_documents_key(source, day, shard)
    trust_key = silver_trust_key(source, day, shard)
    write_table(to_table(documents, Document), root, documents_key)
    write_table(to_table(trust, TrustRecord), root, trust_key)
    return trust, documents_key, trust_key
