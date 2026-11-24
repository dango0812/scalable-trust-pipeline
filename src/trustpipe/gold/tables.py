"""Record schemas of the gold tables.

Gold is built as a snapshot: one deduplication over every silver part produces one
consistent set of training documents, rejects and retrieval chunks, named by a snapshot id
derived from its inputs. Rebuilding from the same silver parts and policies gives the same
snapshot id and the same files.

    gold.signals    per capture: the gold verdict before deduplication, and MinHash bands
    gold.dedup      per capture: keep, exact_duplicate or near_duplicate, and the copy kept
    gold.documents  per kept document: text, provenance, trust and corroboration
    gold.rejects    per capture that did not make it, with the stage and reason
    gold.chunks     per retrieval passage of a kept document, with its provenance

Every silver document ends up in exactly one of gold.documents and gold.rejects.
"""

import pyarrow as pa

from trustpipe.lake.schema import ID, table_schema

_UTC_MICROS = pa.timestamp("us", tz="UTC")

GOLD_SIGNALS = table_schema(
    "gold.signals",
    1,
    [
        ("capture_id", ID, False),
        ("doc_id", ID, False),
        ("host", pa.string(), False),
        ("fetched_at", _UTC_MICROS, False),
        ("tier_rank", pa.int8(), False),
        ("verdict", pa.string(), False),  # "candidate" or the reason it was rejected
        ("detail", pa.string(), True),
        ("bands", pa.list_(pa.uint64()), True),  # MinHash band hashes of candidates
    ],
)

GOLD_DEDUP = table_schema(
    "gold.dedup",
    1,
    [
        ("capture_id", ID, False),
        ("verdict", pa.string(), False),  # keep, exact_duplicate or near_duplicate
        ("kept_capture_id", ID, False),  # the copy that represents this text in gold
        ("corroboration", pa.int32(), False),  # distinct hosts that carry the same text
    ],
)

GOLD_DOCUMENTS = table_schema(
    "gold.documents",
    1,
    [
        ("doc_id", ID, False),
        ("capture_id", ID, False),
        ("source_uri", pa.string(), False),
        ("host", pa.string(), False),
        ("fetched_at", _UTC_MICROS, False),
        ("trust_policy_id", pa.string(), False),
        ("tier", pa.string(), False),
        ("tier_rank", pa.int8(), False),
        ("license", pa.string(), False),
        ("publisher", pa.string(), True),
        ("corroboration", pa.int32(), False),
        ("script", pa.string(), True),
        ("text", pa.large_string(), False),
    ],
)

GOLD_REJECTS = table_schema(
    "gold.rejects",
    1,
    [
        ("capture_id", ID, False),
        ("doc_id", ID, False),
        ("stage", pa.string(), False),
        ("reason", pa.string(), False),
        ("detail", pa.string(), True),
    ],
)

GOLD_CHUNKS = table_schema(
    "gold.chunks",
    1,
    [
        ("chunk_id", ID, False),
        ("doc_id", ID, False),
        ("ordinal", pa.int32(), False),
        ("char_start", pa.int32(), False),
        ("char_end", pa.int32(), False),
        ("source_uri", pa.string(), False),
        ("tier", pa.string(), False),
        ("tier_rank", pa.int8(), False),
        ("license", pa.string(), False),
        ("text", pa.large_string(), False),
    ],
)

GOLD_TOKEN_INDEX = table_schema(
    "gold.token_index",
    1,
    [
        ("doc_id", ID, False),
        ("offset", pa.int64(), False),  # position of the document's first token
        ("length", pa.int32(), False),  # tokens of the document, the end-of-text token included
    ],
)
