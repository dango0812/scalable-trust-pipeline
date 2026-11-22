"""Source registry: decides how far a capture can be trusted from where it came from.

Trust is assigned by policy, not by content: a page that calls itself "official" gains
nothing. The registry is plain TOML (comments allowed, reviewed like code), and its
version is stamped on every trust row it produces.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trustpipe.schema import TrustTier

UNREGISTERED_LICENSE = "NOASSERTION"  # SPDX: no license information available
UNREGISTERED_REASON = "source is not in the registry"


@dataclass(frozen=True, slots=True)
class SourceRule:
    uri_prefix: str
    publisher: str | None
    license: str
    tier: TrustTier
    reason: str

    def covers(self, source_uri: str) -> bool:
        # Match on a path boundary: ".../dango0812" must not cover ".../dango0812-other".
        prefix = self.uri_prefix.rstrip("/")
        return source_uri == prefix or source_uri.startswith(prefix + "/")


@dataclass(frozen=True, slots=True)
class Registry:
    version: str
    rules: tuple[SourceRule, ...]

    def match(self, source_uri: str) -> SourceRule | None:
        """Return the most specific rule covering the URI, if any."""
        covering = [rule for rule in self.rules if rule.covers(source_uri)]
        return max(covering, key=lambda rule: len(rule.uri_prefix), default=None)


def _rule(entry: dict[str, Any]) -> SourceRule:
    return SourceRule(
        uri_prefix=str(entry["uri_prefix"]),
        publisher=entry.get("publisher"),
        license=str(entry["license"]),
        tier=TrustTier(entry["tier"]),
        reason=str(entry["reason"]),
    )


def load_registry(path: Path) -> Registry:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return Registry(
        version=str(data["version"]),
        rules=tuple(_rule(entry) for entry in data.get("source", [])),
    )
