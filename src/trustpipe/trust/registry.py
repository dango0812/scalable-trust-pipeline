"""Source registry: decides how far a capture can be trusted from where it came from.

Trust is assigned by policy, not by content: a page that calls itself "official" gains
nothing. The registry is plain TOML (comments allowed, reviewed like code), and its
version is stamped on every trust row it produces.

Rules and source URIs are compared in canonical form (``trustpipe.identity.uris``) and by
structure, never as raw strings: scheme, host and port must be equal, and the rule's path
segments must be a prefix of the URI's path segments. A query string never matters, and
nothing that only looks like it is under a trusted path (``/trusted/../other``,
``/trusted-impostor``, ``trusted@evil.example``) is covered.

A rule also states which acquisition methods it accepts (default: only captures fetched by
the operator's own crawler), so a file handed in with a claimed URI, or a page taken from a
third-party archive, cannot inherit the tier of that URI unless the rule says so
explicitly.

Every registry has a policy id: the SHA-256 of its canonical JSON form. The id is written
on every trust row, and the policy document itself is kept in the lake, so any decision
can be traced to the exact rules that made it, and two equal registries share one id.

The loader is strict on purpose. A typo in this file decides what a model is trained on,
so an unknown key, a duplicate rule or an ambiguous prefix stops the load instead of
being ignored.
"""

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trustpipe.identity.uris import SourceUri, canonical_uri, parse_uri
from trustpipe.lake.schema import Acquisition, TrustTier

UNREGISTERED_LICENSE = "NOASSERTION"  # SPDX: no license information available
UNREGISTERED_REASON = "source is not in the registry"
UNPARSABLE_REASON = "source URI has no canonical form"
POLICY_FORMAT = "trustpipe.policy/1"

# SPDX short identifiers and expressions use letters, digits, ".", "-" and "+";
# NONE and NOASSERTION are SPDX keywords; LicenseRef-<name> names terms SPDX does not list.
_LICENSE = re.compile(r"NONE|NOASSERTION|LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.+-]*")
_TOP_LEVEL_KEYS = frozenset({"version", "source"})
_RULE_KEYS = frozenset({"uri_prefix", "publisher", "license", "tier", "reason", "accepts"})
_REQUIRED_RULE_KEYS = frozenset({"uri_prefix", "license", "tier", "reason"})


class RegistryError(ValueError):
    """The registry is malformed; nothing from it is used."""


def _canonical_prefix(prefix: str) -> str:
    """Return the prefix in canonical form, or raise when it is not already canonical."""
    if "?" in prefix or "#" in prefix:
        raise RegistryError(f"uri_prefix {prefix!r} must not contain a query or a fragment")
    try:
        canonical = canonical_uri(prefix)
    except ValueError as error:
        raise RegistryError(f"uri_prefix {prefix!r} is not a valid source URI: {error}") from None
    # The only spelling difference tolerated is an omitted root path: "https://host".
    if canonical not in (prefix, prefix + "/"):
        raise RegistryError(f"uri_prefix {prefix!r} is not canonical; write it as {canonical!r}")
    return canonical


@dataclass(frozen=True, slots=True)
class SourceRule:
    uri_prefix: str
    publisher: str | None
    license: str
    tier: TrustTier
    reason: str
    accepts: tuple[Acquisition, ...] = (Acquisition.HTTP_FETCH,)
    scope: SourceUri = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        canonical = _canonical_prefix(self.uri_prefix)
        object.__setattr__(self, "uri_prefix", canonical)
        object.__setattr__(self, "scope", parse_uri(canonical))
        object.__setattr__(self, "tier", TrustTier(self.tier))
        if not _LICENSE.fullmatch(self.license):
            raise RegistryError(
                f"license {self.license!r} of {canonical!r} is not an SPDX identifier, "
                "NONE, NOASSERTION or LicenseRef-<name>"
            )
        if not self.reason.strip():
            raise RegistryError(f"reason of {canonical!r} must not be empty")
        accepts = tuple(sorted({Acquisition(method) for method in self.accepts}))
        if not accepts:
            raise RegistryError(f"accepts of {canonical!r} must name at least one method")
        object.__setattr__(self, "accepts", accepts)

    def document(self) -> dict[str, Any]:
        return {
            "uri_prefix": self.uri_prefix,
            "publisher": self.publisher,
            "license": self.license,
            "tier": self.tier.value,
            "reason": self.reason,
            "accepts": [method.value for method in self.accepts],
        }

    @property
    def segments(self) -> tuple[str, ...]:
        """The rule's path segments; a trailing slash on a rule adds no empty segment."""
        segments = self.scope.segments
        return segments[:-1] if segments and segments[-1] == "" else segments

    def covers(self, uri: SourceUri) -> bool:
        segments = self.segments
        return (
            uri.scheme == self.scope.scheme
            and uri.host == self.scope.host
            and uri.port == self.scope.port
            and uri.segments[: len(segments)] == segments
        )


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of judging one capture: the rule that applied (if any) and why."""

    rule: SourceRule | None
    reason: str

    @property
    def tier(self) -> TrustTier:
        return self.rule.tier if self.rule else TrustTier.UNVERIFIED

    @property
    def license(self) -> str:
        return self.rule.license if self.rule else UNREGISTERED_LICENSE


@dataclass(frozen=True, slots=True)
class Registry:
    version: str
    rules: tuple[SourceRule, ...]
    _by_host: dict[str, tuple[SourceRule, ...]] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    policy_id: str = field(init=False, compare=False, default="")

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise RegistryError("version must not be empty")
        seen: set[str] = set()
        by_host: dict[str, list[SourceRule]] = {}
        for rule in self.rules:
            if rule.uri_prefix in seen:
                raise RegistryError(f"uri_prefix {rule.uri_prefix!r} is listed twice")
            seen.add(rule.uri_prefix)
            by_host.setdefault(rule.scope.host, []).append(rule)
        # Most specific first, so the first covering rule is the one that applies.
        for host, rules in by_host.items():
            rules.sort(key=lambda rule: len(rule.segments), reverse=True)
            self._by_host[host] = tuple(rules)
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        object.__setattr__(self, "policy_id", f"sha256:{digest}")

    def canonical_bytes(self) -> bytes:
        """Compact, key-sorted ASCII JSON of the policy; its SHA-256 is the policy id.

        The lake stores exactly these bytes under _policies/<hex>.json, so any standard
        sha256 tool can check a policy file against the id in the trust rows.
        """
        return json.dumps(self.document(), sort_keys=True, separators=(",", ":")).encode("ascii")

    def document(self) -> dict[str, Any]:
        """The canonical policy document whose SHA-256 is the policy id."""
        return {
            "format": POLICY_FORMAT,
            "version": self.version,
            "rules": [rule.document() for rule in sorted(self.rules, key=lambda r: r.uri_prefix)],
        }

    def match_uri(self, uri: SourceUri) -> SourceRule | None:
        """Return the most specific rule covering a parsed URI, if any."""
        return next((rule for rule in self._by_host.get(uri.host, ()) if rule.covers(uri)), None)

    def judge(self, uri: SourceUri | None, acquisition: Acquisition) -> Decision:
        """Decide the trust of one capture; fail closed on every doubt."""
        if uri is None:
            return Decision(None, UNPARSABLE_REASON)
        rule = self.match_uri(uri)
        if rule is None:
            return Decision(None, UNREGISTERED_REASON)
        if acquisition not in rule.accepts:
            accepted = ", ".join(method.value for method in rule.accepts)
            return Decision(
                None,
                f"rule {rule.uri_prefix} accepts {accepted} captures, not {acquisition.value}",
            )
        return Decision(rule, rule.reason)

    def match(self, source_uri: str) -> SourceRule | None:
        """Return the most specific rule covering the URI; None also for unparsable URIs."""
        try:
            uri = parse_uri(source_uri)
        except ValueError:
            return None
        return self.match_uri(uri)


def _rule(entry: Any, index: int) -> SourceRule:
    if not isinstance(entry, dict):
        raise RegistryError(f"[[source]] entry {index} must be a table")
    unknown = sorted(set(entry) - _RULE_KEYS)
    if unknown:
        raise RegistryError(f"[[source]] entry {index} has unknown keys: {', '.join(unknown)}")
    missing = sorted(_REQUIRED_RULE_KEYS - set(entry))
    if missing:
        raise RegistryError(f"[[source]] entry {index} is missing: {', '.join(missing)}")
    try:
        tier = TrustTier(entry["tier"])
    except ValueError:
        raise RegistryError(
            f"[[source]] entry {index} has an unknown tier {entry['tier']!r}"
        ) from None
    publisher = entry.get("publisher")
    accepts = entry.get("accepts", [Acquisition.HTTP_FETCH.value])
    if not isinstance(accepts, list):
        raise RegistryError(f"[[source]] entry {index}: accepts must be a list of methods")
    try:
        methods = tuple(Acquisition(str(method)) for method in accepts)
    except ValueError:
        raise RegistryError(
            f"[[source]] entry {index} has an unknown acquisition method in {accepts!r}"
        ) from None
    return SourceRule(
        uri_prefix=str(entry["uri_prefix"]),
        publisher=None if publisher is None else str(publisher),
        license=str(entry["license"]),
        tier=tier,
        reason=str(entry["reason"]),
        accepts=methods,
    )


def load_registry(path: Path) -> Registry:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown:
        raise RegistryError(f"{path.name} has unknown top-level keys: {', '.join(unknown)}")
    if "version" not in data:
        raise RegistryError(f"{path.name} is missing 'version'")
    entries = data.get("source", [])
    if not isinstance(entries, list):
        raise RegistryError(f"{path.name}: 'source' must be an array of tables ([[source]])")
    return Registry(
        version=str(data["version"]),
        rules=tuple(_rule(entry, index) for index, entry in enumerate(entries)),
    )
