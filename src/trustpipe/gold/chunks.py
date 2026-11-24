"""Cut documents into retrieval passages.

A passage is what a search returns and what a model reads as evidence, so it should hold
one coherent piece of text and say exactly where it came from. Lines are packed into a
passage until the next line would make it longer than ``max_chars``; a line that is longer
on its own is cut at the last space before the limit (or at the limit, for text written
without spaces). A last passage shorter than ``min_chars`` is merged into the one before.

Every passage keeps its character offsets into the document, so a citation can point at
the exact span, and ``text`` is always ``document[char_start:char_end]``.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from trustpipe.gold.policy import ChunkRules
from trustpipe.identity.ids import typed_hash

_LINE = re.compile(r"[^\n]+")


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    char_start: int
    char_end: int
    text: str


def chunk_id(doc_id: bytes, ordinal: int) -> bytes:
    return typed_hash("chunk", doc_id, ordinal.to_bytes(4, "big"))


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _pieces(text: str, max_chars: int) -> Iterator[tuple[int, int]]:
    """Non-blank lines as (start, end) spans, long lines cut to at most max_chars."""
    for line in _LINE.finditer(text):
        start, end = _strip(text, line.start(), line.end())
        while end - start > max_chars:
            cut = text.rfind(" ", start + 1, start + max_chars + 1)
            if cut == -1:
                cut = start + max_chars
            yield _strip(text, start, cut)
            start = _strip(text, cut, end)[0]
        if start < end:
            yield start, end


def chunk_text(text: str, rules: ChunkRules) -> list[Chunk]:
    spans: list[tuple[int, int]] = []
    for start, end in _pieces(text, rules.max_chars):
        if spans and end - spans[-1][0] <= rules.max_chars:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < rules.min_chars:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    return [Chunk(i, start, end, text[start:end]) for i, (start, end) in enumerate(spans)]
