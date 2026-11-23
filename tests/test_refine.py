import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustpipe.bronze.ingest import capture_file, land
from trustpipe.cli import main
from trustpipe.identity.ids import canonical_text
from trustpipe.lake.layout import (
    policy_key,
    silver_documents_key,
    silver_rejects_key,
    silver_trust_key,
)
from trustpipe.lake.receipts import is_committed
from trustpipe.lake.schema import (
    SILVER_DOCUMENTS,
    SILVER_TRUST,
    Acquisition,
    RawCapture,
    TrustTier,
)
from trustpipe.lake.storage import lake_path, read_table
from trustpipe.silver.decoding import decode
from trustpipe.silver.refine import STAGE, refine, refine_part
from trustpipe.silver.script import detect_script
from trustpipe.trust.registry import UNREGISTERED_REASON, load_registry

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "dango0812-github-nickname.txt"
REGISTRY = load_registry(ROOT / "config" / "sources.toml")
AT = datetime(2026, 11, 22, 13, 13, tzinfo=UTC)
PART = "4fd1996c3d4b2944"


def sample() -> RawCapture:
    return capture_file(SAMPLE, source_uri="https://github.com/dango0812", captured_at=AT)


def capture(
    raw: bytes,
    *,
    uri: str = "https://example.org/doc",
    content_type: str | None = "text/plain; charset=utf-8",
    acquisition: Acquisition = Acquisition.HTTP_FETCH,
) -> RawCapture:
    return RawCapture.from_bytes(
        raw, source_uri=uri, fetched_at=AT, acquisition=acquisition, content_type=content_type
    )


def test_the_sample_becomes_one_document_with_primary_trust() -> None:
    refined = refine([sample()], REGISTRY)
    assert refined.rejects == []
    [document] = refined.documents
    [trust] = refined.trust
    assert document.text == SAMPLE.read_text(encoding="utf-8")
    assert document.script == "Latn"
    assert trust.capture_id == sample().capture_id
    assert trust.doc_id == document.doc_id
    assert trust.host == "github.com"
    assert trust.policy_id == REGISTRY.policy_id
    assert trust.rule_id == "https://github.com/dango0812"
    assert trust.tier is TrustTier.PRIMARY
    assert trust.tier_rank == 3
    assert trust.license == "NOASSERTION"
    assert trust.publisher == "dango0812"
    assert trust.reason.startswith("first-party statement")


def test_a_rule_never_applies_to_an_acquisition_method_it_does_not_accept() -> None:
    # The same bytes claimed for the same URI, but fetched instead of handed in.
    fetched = capture(SAMPLE.read_bytes(), uri="https://github.com/dango0812")
    [trust] = refine([fetched], REGISTRY).trust
    assert trust.tier is TrustTier.UNVERIFIED
    assert trust.rule_id is None
    assert trust.reason == (
        "rule https://github.com/dango0812 accepts submission captures, not http_fetch"
    )


def test_sources_outside_the_registry_are_unverified() -> None:
    [trust] = refine([capture(b"Totally official facts!!")], REGISTRY).trust
    assert trust.tier is TrustTier.UNVERIFIED
    assert trust.reason == UNREGISTERED_REASON
    assert (trust.license, trust.publisher, trust.rule_id) == ("NOASSERTION", None, None)


@pytest.mark.parametrize(
    ("raw", "content_type", "reason"),
    [
        (b"\xc3\x28 broken", "text/plain; charset=utf-8", "undecodable"),
        (b"plain", "text/plain; charset=koi8-r", "unsupported_charset"),
        (b"  \n\t ", "text/plain", "empty_text"),
    ],
)
def test_captures_without_usable_text_become_rejects(
    raw: bytes, content_type: str, reason: str
) -> None:
    refined = refine([capture(raw, content_type=content_type)], REGISTRY)
    assert (refined.documents, refined.trust) == ([], [])
    [reject] = refined.rejects
    assert (reject.stage, reason) == (STAGE, reject.reason)
    assert reject.detail


def test_a_bad_capture_never_fails_the_others() -> None:
    captures = [capture(b"one"), capture(b"\xff"), capture(b"three", uri="https://example.org/3")]
    refined = refine(captures, REGISTRY)
    assert [d.text for d in refined.documents] == ["one", "three"]
    assert len(refined.documents) + len(refined.rejects) == len(captures)


@pytest.mark.parametrize(
    ("raw", "content_type", "text"),
    [
        (b"\xef\xbb\xbfhello", "text/plain; charset=iso-8859-1", "hello"),
        (b"\xff\xfe" + "hé".encode("utf-16-le"), None, "hé"),
        (b"\xfe\xff" + "hé".encode("utf-16-be"), None, "hé"),
        (b"\x80", "text/plain; charset=iso-8859-1", "€"),
        ("한국".encode("cp949"), "text/plain; charset=euc-kr", "한국"),
        ("hé".encode("utf-16-le"), "text/plain; charset=utf-16", "hé"),
        (b"plain", None, "plain"),
    ],
)
def test_decoding_follows_the_whatwg_rules(raw: bytes, content_type: str | None, text: str) -> None:
    assert decode(capture(raw, content_type=content_type)) == text


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("대한민국의 개발자", "Hang"),
        ("ㄱㄴㄷ 자모", "Hang"),
        ("A developer from South Korea.", "Latn"),
        ("café au lait", "Latn"),
        ("12345", None),
        ("", None),
        ("Привет, мир", None),
        ("★ ☆ ♪", None),
    ],
)
def test_detect_script_by_dominant_script(text: str, script: str | None) -> None:
    assert detect_script(text) == script


def test_the_script_guess_does_not_depend_on_unicode_composition() -> None:
    decomposed = unicodedata.normalize("NFD", "한국어 텍스트")
    assert detect_script(canonical_text(decomposed)) == detect_script("한국어 텍스트") == "Hang"


def test_refine_part_writes_aligned_silver_parts_and_commits_them(tmp_path: Path) -> None:
    bronze_key = land([sample()], tmp_path).outputs[0].key
    receipt = refine_part(tmp_path, bronze_key, REGISTRY)
    assert [(o.key, o.rows) for o in receipt.outputs] == [
        (silver_documents_key(AT.date(), PART), 1),
        (silver_rejects_key(AT.date(), PART), 0),
        (silver_trust_key(REGISTRY.policy_id, AT.date(), PART), 1),
    ]
    assert receipt.inputs == (bronze_key,)
    assert receipt.counters == {"documents": 1, "rejects": 0, "tier.primary": 1}
    assert receipt.transform["policy_id"] == REGISTRY.policy_id
    assert is_committed(tmp_path, STAGE, PART)

    documents = read_table(tmp_path, receipt.outputs[0].key, schema=SILVER_DOCUMENTS)
    trust = read_table(tmp_path, receipt.outputs[2].key, schema=SILVER_TRUST)
    assert documents.column("capture_id").equals(trust.column("capture_id"))

    # The stored policy is exactly the document the policy id is the hash of.
    policy = lake_path(tmp_path, policy_key(REGISTRY.policy_id)).read_bytes()
    assert "sha256:" + hashlib.sha256(policy).hexdigest() == REGISTRY.policy_id
    assert json.loads(policy)["rules"][0]["accepts"] == ["submission"]


def test_refining_again_reproduces_the_same_commit(tmp_path: Path) -> None:
    bronze_key = land([sample()], tmp_path).outputs[0].key
    assert refine_part(tmp_path, bronze_key, REGISTRY) == refine_part(
        tmp_path, bronze_key, REGISTRY
    )


def test_cli_refine_reports_counts_and_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bronze_key = land([sample()], tmp_path).outputs[0].key
    registry = str(ROOT / "config" / "sources.toml")
    assert main(["--lake", str(tmp_path), "refine", bronze_key, "--registry", registry]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "silver.refine"
    assert report["records"] == 1
    assert report["counters"] == {"documents": 1, "rejects": 0, "tier.primary": 1}
    assert report["policy_id"] == REGISTRY.policy_id
    assert report["receipt"] == f"_receipts/silver.refine/{PART}.json"
