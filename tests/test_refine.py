import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustpipe.cli import main
from trustpipe.ingest import capture_file, land
from trustpipe.refine import decode, detect_lang, refine, refine_part
from trustpipe.registry import UNPARSABLE_REASON, UNREGISTERED_REASON, Registry, load_registry
from trustpipe.schema import (
    SILVER_DOCUMENTS,
    SILVER_TRUST,
    Document,
    RawCapture,
    TrustRecord,
    TrustTier,
    from_table,
)
from trustpipe.storage import read_table

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "dango0812-github-nickname.txt"
REGISTRY = load_registry(ROOT / "config" / "sources.toml")
CAPTURED_AT = datetime(2026, 11, 22, 13, 13, tzinfo=UTC)
BRONZE_KEY = "bronze/captures/source=github.com/date=2026-11-22/part-00000.parquet"
DOCUMENTS_KEY = "silver/documents/source=github.com/date=2026-11-22/part-00000.parquet"
TRUST_KEY = "silver/trust/source=github.com/date=2026-11-22/part-00000.parquet"


def sample_capture() -> RawCapture:
    return capture_file(SAMPLE, source_uri="https://github.com/dango0812", captured_at=CAPTURED_AT)


def test_the_sample_becomes_an_english_document_with_primary_trust() -> None:
    documents, trust = refine([sample_capture()], REGISTRY)
    assert [d.text for d in documents] == [SAMPLE.read_text(encoding="utf-8")]
    assert documents[0].lang == "en"
    record = trust[0]
    assert record.doc_id == documents[0].doc_id
    assert record.tier is TrustTier.PRIMARY
    assert record.source_domain == "github.com"
    assert record.publisher == "dango0812"
    assert record.license == "NOASSERTION"
    assert record.retrieved_at == CAPTURED_AT
    assert record.policy_version == "2026-11-22"


def test_sources_outside_the_registry_are_unverified() -> None:
    capture = RawCapture.from_bytes(
        b"Totally official facts!!",
        source_uri="https://blog.example.net/post/1",
        fetched_at=CAPTURED_AT,
    )
    _, trust = refine([capture], Registry("empty", ()))
    assert trust[0].tier is TrustTier.UNVERIFIED
    assert trust[0].tier_reason == UNREGISTERED_REASON
    assert trust[0].license == "NOASSERTION"


def test_a_capture_whose_uri_has_no_canonical_form_fails_closed() -> None:
    capture = RawCapture.from_bytes(
        b"Looks official",
        source_uri="https://github.com@evil.example/dango0812",
        fetched_at=CAPTURED_AT,
    )
    _, trust = refine([capture], REGISTRY)
    assert trust[0].tier is TrustTier.UNVERIFIED
    assert trust[0].tier_reason == UNPARSABLE_REASON
    assert trust[0].source_domain == "evil.example"


def test_decoding_follows_the_declared_charset_and_is_strict() -> None:
    latin1 = RawCapture.from_bytes(
        "café".encode("latin-1"),
        source_uri="https://a.example/x",
        fetched_at=CAPTURED_AT,
        content_type="text/plain; charset=iso-8859-1",
    )
    assert decode(latin1) == "café"
    broken = RawCapture.from_bytes(
        b"\xff\xfe\xfa", source_uri="https://a.example/y", fetched_at=CAPTURED_AT
    )
    with pytest.raises(UnicodeDecodeError):
        decode(broken)


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("대한민국의 개발자", "ko"),
        ("A developer from South Korea.", "en"),
        ("12345", None),
        ("Привет, мир", None),
    ],
)
def test_detect_lang_by_dominant_script(text: str, lang: str | None) -> None:
    assert detect_lang(text) == lang


def test_refine_part_writes_aligned_silver_parts_and_is_idempotent(tmp_path: Path) -> None:
    land([sample_capture()], tmp_path, source="github.com")
    first = refine_part(tmp_path, BRONZE_KEY, REGISTRY)
    second = refine_part(tmp_path, BRONZE_KEY, REGISTRY)
    assert first == second
    assert first[1:] == (DOCUMENTS_KEY, TRUST_KEY)

    documents = from_table(read_table(tmp_path, DOCUMENTS_KEY, schema=SILVER_DOCUMENTS), Document)
    trust = from_table(read_table(tmp_path, TRUST_KEY, schema=SILVER_TRUST), TrustRecord)
    assert [d.doc_id for d in documents] == [t.doc_id for t in trust]
    assert sorted(p.name for p in (tmp_path / DOCUMENTS_KEY).parent.iterdir()) == [
        "part-00000.parquet"
    ]


def test_cli_refine_reports_tiers_and_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    land([sample_capture()], tmp_path, source="github.com")
    registry = str(ROOT / "config" / "sources.toml")
    assert main(["--lake", str(tmp_path), "refine", BRONZE_KEY, "--registry", registry]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "silver.refine"
    assert report["records"] == 1
    assert report["inputs"] == [BRONZE_KEY]
    assert report["outputs"] == [DOCUMENTS_KEY, TRUST_KEY]
    assert report["tiers"] == {"primary": 1}
    assert report["policy_version"] == "2026-11-22"
