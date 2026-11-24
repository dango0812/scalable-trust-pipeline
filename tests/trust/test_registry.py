import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from trustpipe.identity.uris import parse_uri
from trustpipe.lake.schema import Acquisition, TrustTier
from trustpipe.trust.registry import (
    UNPARSABLE_REASON,
    UNREGISTERED_REASON,
    Registry,
    RegistryError,
    SourceRule,
    load_registry,
)

CONFIG = Path(__file__).resolve().parents[2] / "config" / "sources.toml"


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


# --- acquisition rules, decisions and policy ids (contract v2) -----------------------------


def test_rules_accept_only_fetched_captures_unless_told_otherwise() -> None:
    assert rule("https://example.org").accepts == (Acquisition.HTTP_FETCH,)
    configured = load_registry(CONFIG).match("https://github.com/dango0812")
    assert configured is not None
    assert configured.accepts == (Acquisition.SUBMISSION,)


def test_judge_explains_every_outcome() -> None:
    registry = load_registry(CONFIG)
    trusted = parse_uri("https://github.com/dango0812")
    accepted = registry.judge(trusted, Acquisition.SUBMISSION)
    assert accepted.rule is not None
    assert (accepted.tier, accepted.license) == (TrustTier.PRIMARY, "NOASSERTION")
    refused = registry.judge(trusted, Acquisition.HTTP_FETCH)
    assert (refused.rule, refused.tier) == (None, TrustTier.UNVERIFIED)
    assert "accepts submission captures, not http_fetch" in refused.reason
    unknown = registry.judge(parse_uri("https://blog.example.net/"), Acquisition.HTTP_FETCH)
    assert (unknown.reason, unknown.license) == (UNREGISTERED_REASON, "NOASSERTION")
    assert registry.judge(None, Acquisition.HTTP_FETCH).reason == UNPARSABLE_REASON


@pytest.mark.parametrize(
    ("accepts", "problem"),
    [
        ("accepts = []\n", "at least one method"),
        ('accepts = ["telepathy"]\n', "unknown acquisition method"),
        ('accepts = "submission"\n', "must be a list"),
    ],
)
def test_malformed_acquisition_rules_are_rejected(
    tmp_path: Path, accepts: str, problem: str
) -> None:
    with pytest.raises(RegistryError, match=problem):
        load_registry(write_registry(tmp_path, 'version = "v1"\n[[source]]\n' + RULE + accepts))


def test_the_policy_id_is_the_sha256_of_the_canonical_policy() -> None:
    registry = load_registry(CONFIG)
    assert registry.policy_id == "sha256:" + hashlib.sha256(registry.canonical_bytes()).hexdigest()
    document = json.loads(registry.canonical_bytes())
    assert document["format"] == "trustpipe.policy/1"
    assert document["version"] == "2026-11-22"


def test_equal_policies_share_an_id_and_any_change_makes_a_new_one() -> None:
    a = rule("https://a.example")
    b = rule("https://b.example")
    base = Registry("v1", (a, b))
    assert Registry("v1", (b, a)).policy_id == base.policy_id  # rule order does not matter
    changed = [
        Registry("v2", (a, b)),
        Registry("v1", (a,)),
        Registry("v1", (a, rule("https://b.example", TrustTier.REVIEWED))),
        Registry("v1", (a, dataclasses.replace(b, accepts=(Acquisition.SUBMISSION,)))),
    ]
    assert len({base.policy_id, *(r.policy_id for r in changed)}) == 1 + len(changed)
