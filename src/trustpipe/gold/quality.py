"""Heuristic quality rules for training documents (Gopher quality and repetition rules).

Each rule computes one statistic of the text and compares it with a threshold of the gold
policy (policy.py). ``assess`` returns the first rule a document breaks, as a reason code
and a detail that shows the measured value, or None when it passes them all.

Definitions follow datatrove's GopherQualityFilter and GopherRepetitionFilter:

- words are whitespace-separated tokens (datatrove uses a language-specific word
  tokenizer; whitespace tokens work for every script, Korean eojeol included);
- a "symbol word" consists of punctuation only and does not count as a word;
- lines are separated by one or more newlines, paragraphs by two or more;
- a duplicate line or paragraph is any repeat of one seen before; character fractions
  divide the characters of the repeats by the length of the whole text;
- the top n-gram fraction is the length of the most frequent word n-gram times its count;
  the duplicated n-gram fraction sums the characters of n-grams that repeat an earlier one,
  skipping past each repeat.
"""

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from trustpipe.gold.policy import QualityRules

# ASCII punctuation, plus the ellipsis, curly quotes, guillemets, CJK brackets and the
# ideographic and full-width marks Korean, Chinese and Japanese text uses.
_WIDE_PUNCTUATION = (
    "2026 201C 201D 2018 2019 00AB 00BB 00B7 3001 3002 3008 3009"
    " 300A 300B 300C 300D 300E 300F 3010 3011 FF01 FF0C FF0E FF1A FF1B FF1F"
)
_PUNCTUATION = frozenset(string.punctuation) | {
    chr(int(code, 16)) for code in _WIDE_PUNCTUATION.split()
}
_LINES = re.compile(r"\n+")
_PARAGRAPHS = re.compile(r"\n{2,}")


@dataclass(frozen=True, slots=True)
class Verdict:
    reason: str
    detail: str


def _repeats(elements: Sequence[str]) -> tuple[int, int]:
    """Number and total length of the elements that repeat an earlier one."""
    seen: set[str] = set()
    count = chars = 0
    for element in elements:
        if element in seen:
            count += 1
            chars += len(element)
        else:
            seen.add(element)
    return count, chars


def _top_ngram_chars(words: Sequence[str], n: int) -> int | None:
    grams = Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return None
    gram, count = grams.most_common(1)[0]
    return len(gram) * count


def _duplicated_ngram_chars(words: Sequence[str], n: int) -> int:
    seen: set[str] = set()
    chars = 0
    index = 0
    while index <= len(words) - n:
        gram = "".join(words[index : index + n])
        if gram in seen:
            chars += len(gram)
            index += n
        else:
            seen.add(gram)
            index += 1
    return chars


def _quality(
    text: str, words: list[str], script: str | None, rules: QualityRules
) -> Verdict | None:
    real = [w for w in words if any(c not in _PUNCTUATION for c in w)]
    if len(real) < rules.min_words:
        return Verdict("short_doc", f"{len(real)} words < {rules.min_words}")
    if len(real) > rules.max_words:
        return Verdict("long_doc", f"{len(real)} words > {rules.max_words}")
    if script == "Latn":
        mean = sum(map(len, real)) / len(real)
        if not rules.min_mean_word_length <= mean <= rules.max_mean_word_length:
            return Verdict("mean_word_length", f"{mean:.2f} characters per word")
    hashes = text.count("#") / len(words)
    if hashes > rules.max_hash_ratio:
        return Verdict("hash_ratio", f"{hashes:.3f} '#' per word")
    ellipses = (text.count("...") + text.count("…")) / len(words)
    if ellipses > rules.max_ellipsis_ratio:
        return Verdict("ellipsis_ratio", f"{ellipses:.3f} ellipses per word")
    lines = text.splitlines()
    bullets = sum(line.lstrip().startswith(("•", "-")) for line in lines) / len(lines)
    if bullets > rules.max_bullet_lines:
        return Verdict("bullet_lines", f"{bullets:.2f} of lines start with a bullet")
    trailing = sum(line.rstrip().endswith(("...", "…")) for line in lines) / len(lines)
    if trailing > rules.max_ellipsis_lines:
        return Verdict("ellipsis_lines", f"{trailing:.2f} of lines end with an ellipsis")
    alpha = sum(any(c.isalpha() for c in w) for w in words) / len(words)
    if alpha < rules.min_alpha_words:
        return Verdict("alpha_words", f"{alpha:.2f} of words contain a letter")
    if script == "Latn" and rules.min_stop_words > 0:
        found = len(set(rules.stop_words).intersection(words))
        if found < rules.min_stop_words:
            return Verdict("stop_words", f"{found} distinct stop words < {rules.min_stop_words}")
    return None


def _repetition(text: str, words: list[str], rules: QualityRules) -> Verdict | None:
    paragraphs = _PARAGRAPHS.split(text.strip())
    count, chars = _repeats(paragraphs)
    if count / len(paragraphs) > rules.max_dup_para_frac:
        return Verdict("dup_para_frac", f"{count} of {len(paragraphs)} paragraphs repeat")
    if chars / len(text) > rules.max_dup_para_char_frac:
        return Verdict("dup_para_char_frac", f"{chars / len(text):.2f} of characters")
    lines = _LINES.split(text)
    count, chars = _repeats(lines)
    if count / len(lines) > rules.max_dup_line_frac:
        return Verdict("dup_line_frac", f"{count} of {len(lines)} lines repeat")
    if chars / len(text) > rules.max_dup_line_char_frac:
        return Verdict("dup_line_char_frac", f"{chars / len(text):.2f} of characters")
    for n, limit in rules.top_ngrams:
        top = _top_ngram_chars(words, n)
        if top is not None and top / len(text) > limit:
            return Verdict(f"top_{n}gram", f"{top / len(text):.2f} of characters")
    for n, limit in rules.dup_ngrams:
        duplicated = _duplicated_ngram_chars(words, n) / len(text)
        if duplicated > limit:
            return Verdict(f"dup_{n}gram", f"{duplicated:.2f} of characters")
    return None


def assess(text: str, script: str | None, rules: QualityRules) -> Verdict | None:
    """The first quality rule the text breaks, or None. ``text`` must not be blank."""
    words = text.split()
    return _quality(text, words, script, rules) or _repetition(text, words, rules)
