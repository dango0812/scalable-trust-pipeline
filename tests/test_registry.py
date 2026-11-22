from pathlib import Path

import pytest

from trustpipe.registry import Registry, SourceRule, load_registry
from trustpipe.schema import TrustTier

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sources.toml"


def rule(prefix: str, tier: TrustTier = TrustTier.PRIMARY) -> SourceRule:
    return SourceRule(prefix, None, "CC0-1.0", tier, f"rule for {prefix}")


def test_repository_registry_trusts_the_first_party_source() -> None:
    registry = load_registry(CONFIG)
    assert registry.version == "2026-11-22"
    match = registry.match("https://github.com/dango0812")
    assert match is not None
    assert match.tier is TrustTier.PRIMARY
    assert match.license == "NOASSERTION"


def test_prefixes_match_on_path_boundaries_only() -> None:
    registry = Registry("v1", (rule("https://github.com/dango0812"),))
    assert registry.match("https://github.com/dango0812/repo") is not None
    assert registry.match("https://github.com/dango0812-impostor") is None
    assert registry.match("https://example.org/") is None


def test_the_most_specific_rule_wins() -> None:
    registry = Registry(
        "v1",
        (
            rule("https://example.org", TrustTier.UNVERIFIED),
            rule("https://example.org/standards", TrustTier.PRIMARY),
        ),
    )
    match = registry.match("https://example.org/standards/spec-1")
    assert match is not None
    assert match.tier is TrustTier.PRIMARY


def test_unknown_tiers_in_the_registry_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "sources.toml"
    config.write_text(
        'version = "v1"\n[[source]]\nuri_prefix = "https://a.example"\n'
        'license = "CC0-1.0"\ntier = "trust-me"\nreason = "x"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trust-me"):
        load_registry(config)
