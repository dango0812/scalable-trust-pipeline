from pathlib import Path

import pytest

from trustpipe.registry import Registry, RegistryError, SourceRule, load_registry
from trustpipe.schema import TrustTier

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sources.toml"


def rule(prefix: str, tier: TrustTier = TrustTier.PRIMARY) -> SourceRule:
    return SourceRule(prefix, None, "CC0-1.0", tier, f"rule for {prefix}")


def write_registry(tmp_path: Path, body: str) -> Path:
    config = tmp_path / "sources.toml"
    config.write_text(body, encoding="utf-8")
    return config


RULE = 'uri_prefix = "https://a.example/x"\nlicense = "CC0-1.0"\ntier = "primary"\nreason = "r"\n'


def test_repository_registry_trusts_the_first_party_source() -> None:
    registry = load_registry(CONFIG)
    assert registry.version == "2026-11-22"
    match = registry.match("https://github.com/dango0812")
    assert match is not None
    assert match.tier is TrustTier.PRIMARY
    assert match.license == "NOASSERTION"


@pytest.mark.parametrize(
    "uri",
    [
        "https://github.com/dango0812",
        "https://github.com/dango0812/",
        "https://github.com/dango0812/repo",
        "https://github.com/dango0812?tab=repositories",
        "https://GITHUB.com/dango0812",
        "https://github.com:443/dango0812",
        "https://github.com/dango0812#readme",
        "https://github.com/%64ango0812",
    ],
)
def test_every_spelling_of_a_trusted_location_is_covered(uri: str) -> None:
    assert load_registry(CONFIG).match(uri) is not None


@pytest.mark.parametrize(
    "uri",
    [
        "https://github.com/dango0812-impostor",
        "https://github.com/dango0812/../someone-else/page",
        "https://github.com/dango0812/%2e%2e/someone-else",
        "https://github.com@evil.example/dango0812",
        "https://github.com.evil.example/dango0812",
        "http://github.com/dango0812",
        "https://github.com:8443/dango0812",
        "https://github.com/dango0812%2Frepo",
        "not a uri at all",
    ],
)
def test_locations_that_only_look_trusted_are_not_covered(uri: str) -> None:
    assert load_registry(CONFIG).match(uri) is None


def test_the_most_specific_rule_wins_and_a_root_rule_covers_its_host() -> None:
    registry = Registry(
        "v1",
        (
            rule("https://example.org", TrustTier.UNVERIFIED),
            rule("https://example.org/standards/", TrustTier.PRIMARY),
        ),
    )
    standards = registry.match("https://example.org/standards/spec-1")
    blog = registry.match("https://example.org/blog/post")
    assert standards is not None
    assert standards.tier is TrustTier.PRIMARY
    assert blog is not None
    assert blog.tier is TrustTier.UNVERIFIED
    assert registry.match("https://other.example/standards/spec-1") is None


def test_rule_prefixes_are_stored_in_canonical_form() -> None:
    assert rule("https://example.org").uri_prefix == "https://example.org/"


@pytest.mark.parametrize(
    ("prefix", "problem"),
    [
        ("https://Example.org/a", "not canonical; write it as 'https://example.org/a'"),
        ("https://example.org/a/../b", "not canonical"),
        ("https://example.org/a?x=1", "query or a fragment"),
        ("https://example.org/a#top", "query or a fragment"),
        ("ftp://example.org/a", "not a valid source URI"),
    ],
)
def test_ambiguous_rule_prefixes_are_rejected(prefix: str, problem: str) -> None:
    with pytest.raises(RegistryError, match=problem):
        rule(prefix)


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ('version = "v1"\nsources = []\n', "unknown top-level keys: sources"),
        ("[[source]]\n" + RULE, "missing 'version'"),
        ('version = "  "\n', "version must not be empty"),
        ('version = "v1"\nsource = "x"\n', "array of tables"),
        ('version = "v1"\nsource = ["x"]\n', "entry 0 must be a table"),
        ('version = "v1"\n[[source]]\n' + RULE + 'teir = "primary"\n', "unknown keys: teir"),
        ('version = "v1"\n[[source]]\nuri_prefix = "https://a.example/"\n', "missing: license"),
        ('version = "v1"\n[[source]]\n' + RULE.replace('"primary"', '"trust-me"'), "trust-me"),
        ('version = "v1"\n[[source]]\n' + RULE.replace('"CC0-1.0"', '"CC0 1.0"'), "not an SPDX"),
        ('version = "v1"\n[[source]]\n' + RULE.replace('"r"', '" "'), "reason .* empty"),
        ('version = "v1"\n[[source]]\n' + RULE + "[[source]]\n" + RULE, "listed twice"),
    ],
)
def test_malformed_registries_are_rejected(tmp_path: Path, body: str, problem: str) -> None:
    with pytest.raises(RegistryError, match=problem):
        load_registry(write_registry(tmp_path, body))


def test_publisher_is_optional_and_kept_as_text(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path, 'version = "v1"\n[[source]]\n' + RULE))
    match = registry.match("https://a.example/x/y")
    assert match is not None
    assert match.publisher is None
    registry = load_registry(
        write_registry(tmp_path, 'version = "v1"\n[[source]]\n' + RULE + "publisher = 42\n")
    )
    match = registry.match("https://a.example/x")
    assert match is not None
    assert match.publisher == "42"
