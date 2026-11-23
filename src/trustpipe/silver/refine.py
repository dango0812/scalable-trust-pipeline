"""Silver refinement: turn one bronze part into documents, rejects and trust rows.

Every capture of the part ends up in exactly one of two places:

- it decodes to non-empty text: one silver.documents row (canonical text and script) and
  one silver.trust row (what the policy decides about this capture), in the same order;
- it does not: one silver.rejects row with a reason code, and the part carries on.

So rows(bronze) == rows(documents) + rows(rejects) for every part, and one bad row never
fails a whole part.

Decoding (decoding.py) and the script guess (script.py) live in their own modules.

The silver files are written first and the receipt last (lake/receipts.py). The policy that
decided the trust rows is stored under _policies/, so every row can be explained later.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from trustpipe import __version__
from trustpipe.identity.ids import CANONICAL_FORM, canonical_text
from trustpipe.identity.uris import parse_uri
from trustpipe.lake.layout import (
    parse_bronze_captures_key,
    policy_key,
    silver_documents_key,
    silver_rejects_key,
    silver_trust_key,
)
from trustpipe.lake.receipts import Receipt, commit
from trustpipe.lake.schema import (
    BRONZE_CAPTURES,
    Document,
    RawCapture,
    Reject,
    TrustRecord,
    from_table,
    to_table,
)
from trustpipe.lake.storage import read_table, write_bytes, write_table
from trustpipe.silver.decoding import UnsupportedCharsetError, decode
from trustpipe.silver.script import detect_script
from trustpipe.trust.registry import Registry

STAGE = "silver.refine"


@dataclass(frozen=True, slots=True)
class Refined:
    documents: list[Document] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)
    trust: list[TrustRecord] = field(default_factory=list)


def _text_or_reason(capture: RawCapture) -> tuple[str | None, str, str]:
    try:
        text = canonical_text(decode(capture))
    except UnsupportedCharsetError as error:
        return None, "unsupported_charset", str(error)
    except UnicodeDecodeError as error:
        return None, "undecodable", f"{error.encoding}: {error.reason} at byte {error.start}"
    if not text.strip():
        return None, "empty_text", "the payload holds no visible text"
    return text, "", ""


def refine(captures: Iterable[RawCapture], registry: Registry) -> Refined:
    refined = Refined()
    for capture in captures:
        text, reason, detail = _text_or_reason(capture)
        if text is None:
            refined.rejects.append(Reject(capture.capture_id, STAGE, reason, detail))
            continue
        document = Document.from_text(
            text, capture_id=capture.capture_id, script=detect_script(text)
        )
        decision = registry.judge(parse_uri(capture.source_uri), capture.acquisition)
        refined.documents.append(document)
        refined.trust.append(
            TrustRecord(
                capture_id=capture.capture_id,
                doc_id=document.doc_id,
                host=capture.host,
                fetched_at=capture.fetched_at,
                policy_id=registry.policy_id,
                rule_id=decision.rule.uri_prefix if decision.rule else None,
                tier=decision.tier,
                tier_rank=decision.tier.rank,
                license=decision.license,
                publisher=decision.rule.publisher if decision.rule else None,
                reason=decision.reason,
            )
        )
    return refined


def refine_part(root: Path, bronze_key: str, registry: Registry) -> Receipt:
    """Refine one committed bronze part into its silver parts and commit them."""
    day, part = parse_bronze_captures_key(bronze_key)
    captures = from_table(read_table(root, bronze_key, schema=BRONZE_CAPTURES), RawCapture)
    refined = refine(captures, registry)
    write_bytes(root, policy_key(registry.policy_id), registry.canonical_bytes(), create_only=True)
    outputs = (
        write_table(
            to_table(refined.documents, Document),
            root,
            silver_documents_key(day, part),
            create_only=False,
        ),
        write_table(
            to_table(refined.rejects, Reject),
            root,
            silver_rejects_key(day, part),
            create_only=False,
        ),
        write_table(
            to_table(refined.trust, TrustRecord),
            root,
            silver_trust_key(registry.policy_id, day, part),
            create_only=False,
        ),
    )
    counters = Counter({"documents": len(refined.documents), "rejects": len(refined.rejects)})
    counters.update(f"tier.{record.tier.value}" for record in refined.trust)
    counters.update(f"reject.{reject.reason}" for reject in refined.rejects)
    receipt = Receipt(
        stage=STAGE,
        part_id=part,
        inputs=(bronze_key,),
        outputs=outputs,
        transform={
            "trustpipe": __version__,
            "canonical_form": CANONICAL_FORM,
            "policy_id": registry.policy_id,
        },
        counters=dict(sorted(counters.items())),
    )
    commit(root, receipt, create_only=False)
    return receipt
