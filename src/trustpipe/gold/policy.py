"""Gold policy: which silver documents become training data, and how they are prepared.

The policy is a reviewed TOML file (``config/gold.toml``) with four parts:

- ``min_tier``: the lowest trust tier that may be trained on. Below it a document is
  rejected, whatever its quality: trust comes first.
- ``[quality]``: heuristic quality rules. The thresholds are the defaults of datatrove's
  Gopher quality and repetition filters, which implement the rules of Rae et al. (2021),
  "Scaling Language Models: Methods, Analysis & Insights from Training Gopher", appendix A.
  The word-length and stop-word rules were calibrated on English, so they apply to texts
  whose script is Latin only; every other rule applies to every script.
- ``[dedup]``: MinHash near-duplicate detection, by default FineWeb's setting: word
  5-grams, 14 bands of 8 hashes (112 hashes), which catches pairs with a Jaccard similarity
  of about 0.75 and above.
- ``[chunks]``: how kept documents are cut into retrieval passages.

Like the source registry, the loader is strict (an unknown key stops the load), and the
policy id is the SHA-256 of its canonical JSON form, recorded on every gold part.
"""

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trustpipe.lake.schema import TrustTier

POLICY_FORMAT = "trustpipe.gold-policy/1"


class GoldPolicyError(ValueError):
    """The gold policy is malformed; nothing from it is used."""


@dataclass(frozen=True, slots=True)
class QualityRules:
    min_words: int = 50
    max_words: int = 100_000
    min_mean_word_length: float = 3.0  # Latin script only
    max_mean_word_length: float = 10.0  # Latin script only
    max_hash_ratio: float = 0.1  # "#" per word
    max_ellipsis_ratio: float = 0.1  # "..." and "…" per word
    max_bullet_lines: float = 0.9
    max_ellipsis_lines: float = 0.3
    min_alpha_words: float = 0.8
    stop_words: tuple[str, ...] = ("the", "be", "to", "of", "and", "that", "have", "with")
    min_stop_words: int = 2  # Latin script only
    max_dup_line_frac: float = 0.3
    max_dup_para_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    max_dup_para_char_frac: float = 0.2
    top_ngrams: tuple[tuple[int, float], ...] = ((2, 0.2), (3, 0.18), (4, 0.16))
    dup_ngrams: tuple[tuple[int, float], ...] = (
        (5, 0.15),
        (6, 0.14),
        (7, 0.13),
        (8, 0.12),
        (9, 0.11),
        (10, 0.10),
    )


@dataclass(frozen=True, slots=True)
class DedupRules:
    ngram: int = 5
    bands: int = 14
    rows: int = 8
    seed: int = 1

    def __post_init__(self) -> None:
        if min(self.ngram, self.bands, self.rows) < 1:
            raise GoldPolicyError("dedup ngram, bands and rows must be positive")


@dataclass(frozen=True, slots=True)
class ChunkRules:
    max_chars: int = 1200  # about 300 tokens of English, a common retrieval passage size
    min_chars: int = 200  # a shorter last piece is merged into the previous chunk

    def __post_init__(self) -> None:
        if not 0 < self.min_chars < self.max_chars:
            raise GoldPolicyError("chunks need 0 < min_chars < max_chars")


@dataclass(frozen=True, slots=True)
class GoldPolicy:
    version: str
    min_tier: TrustTier = TrustTier.REVIEWED
    quality: QualityRules = field(default_factory=QualityRules)
    dedup: DedupRules = field(default_factory=DedupRules)
    chunks: ChunkRules = field(default_factory=ChunkRules)
    policy_id: str = field(init=False, compare=False, default="")

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise GoldPolicyError("version must not be empty")
        object.__setattr__(self, "min_tier", TrustTier(self.min_tier))
        digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        object.__setattr__(self, "policy_id", f"sha256:{digest}")

    def document(self) -> dict[str, Any]:
        return {
            "format": POLICY_FORMAT,
            "version": self.version,
            "min_tier": self.min_tier.value,
            "quality": dataclasses.asdict(self.quality),
            "dedup": dataclasses.asdict(self.dedup),
            "chunks": dataclasses.asdict(self.chunks),
        }

    def canonical_bytes(self) -> bytes:
        """Compact, key-sorted JSON; its SHA-256 is the policy id."""
        return json.dumps(self.document(), sort_keys=True, separators=(",", ":")).encode()


def _tuples(value: Any) -> Any:
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def _section[T](cls: type[T], data: Any, name: str) -> T:
    if not isinstance(data, dict):
        raise GoldPolicyError(f"[{name}] must be a table")
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise GoldPolicyError(f"[{name}] has unknown keys: {', '.join(unknown)}")
    return cls(**{key: _tuples(value) for key, value in data.items()})


def load_gold_policy(path: Path) -> GoldPolicy:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(data) - {"version", "min_tier", "quality", "dedup", "chunks"})
    if unknown:
        raise GoldPolicyError(f"{path.name} has unknown top-level keys: {', '.join(unknown)}")
    if "version" not in data:
        raise GoldPolicyError(f"{path.name} is missing 'version'")
    try:
        min_tier = TrustTier(data.get("min_tier", TrustTier.REVIEWED.value))
    except ValueError:
        raise GoldPolicyError(f"unknown min_tier {data['min_tier']!r}") from None
    return GoldPolicy(
        version=str(data["version"]),
        min_tier=min_tier,
        quality=_section(QualityRules, data.get("quality", {}), "quality"),
        dedup=_section(DedupRules, data.get("dedup", {}), "dedup"),
        chunks=_section(ChunkRules, data.get("chunks", {}), "chunks"),
    )
