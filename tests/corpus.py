"""Deterministic test texts and a source registry shared by the tests."""

import hashlib
from datetime import UTC, datetime

AT = datetime(2026, 11, 24, 6, 0, tzinfo=UTC)


def words(text: str) -> list[str]:
    return text.split()


ENGLISH = words("""
source registry policy decides trust tier document model training data provenance record
capture archive crawler page text corpus evidence citation review publisher agency
university standard statistic official report audit receipt lake bronze silver gold
snapshot chunk passage retrieval question answer measure quality filter signal duplicate
similarity shingle signature band bucket worker executor part schema field
""")
STOP = words("the of and to with that have be in a")
KOREAN = words("""
신뢰 출처 정책 등급 문서 모델 학습 데이터 기록 수집 기관 대학 통계 보고서 검색 질문
답변 근거 품질 중복 유사도 단계 저장소 파일 결과 판정 사유 공개 기준 원문 기사
""")
PARTICLES = words("은 는 을 를 의 에서 로")


def pick(words: list[str], *key: object) -> str:
    """A word chosen by a hash of the key: varied like random choice, but reproducible."""
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return words[int.from_bytes(digest, "big") % len(words)]


def english(seed: int, words: int = 120) -> str:
    """A text that passes every quality rule: varied words, stop words, no repeats."""
    tokens = [pick(STOP if i % 3 == 0 else ENGLISH, seed, i) for i in range(words)]
    return "\n".join(" ".join(tokens[i : i + 15]) + "." for i in range(0, words, 15))


def korean(seed: int, words: int = 120) -> str:
    tokens = [pick(KOREAN, seed, i) + pick(PARTICLES, seed, i, "p") for i in range(words)]
    return "\n".join(" ".join(tokens[i : i + 12]) + "." for i in range(0, words, 12))


REGISTRY_TOML = """
version = "test"

[[source]]
uri_prefix = "https://primary.example.org"
license = "CC-BY-4.0"
tier = "primary"
reason = "the authoritative original"

[[source]]
uri_prefix = "https://agency.example.org"
license = "CC-BY-4.0"
tier = "institutional"
reason = "an accountable agency"

[[source]]
uri_prefix = "https://mirror.example.net"
license = "CC-BY-4.0"
tier = "reviewed"
reason = "an edited mirror"
"""
