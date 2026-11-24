"""Training examples for answering from retrieved sources (RAFT).

RAFT (Zhang et al., 2024, "RAFT: Adapting Language Model to Domain Specific RAG") trains a
model on questions shown together with retrieved documents: the "oracle" document that
holds the answer, mixed with distractor documents that do not. For a share ``oracle_rate``
of the questions the oracle is included; for the rest only distractors are shown. The
model learns to find the answer among distractors and to ignore what does not answer.

One deliberate change from the paper: when the oracle is withheld, RAFT still trains the
original answer, pushing the model to answer from memory. A model whose answers must be
traceable to trusted sources should not do that, so the target here is the no-answer
sentence instead, and answers with the oracle end with a citation of its number ([n]).

Questions come from a JSON Lines file of question, answer and the source URI where the
answer can be read; that source must be in the index. Distractors are the passages the
index itself retrieves for the question from other sources, so they are hard negatives.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trustpipe.index.search import Hit, Retriever
from trustpipe.train.prompt import Source, no_answer

_FIELDS = ("question", "answer", "source_uri")


@dataclass(frozen=True, slots=True)
class QAPair:
    question: str
    answer: str
    source_uri: str


@dataclass(frozen=True, slots=True)
class RagExample:
    question: str
    sources: tuple[Source, ...]
    target: str
    oracle: int | None  # 1-based position of the oracle source; None when withheld


def load_qa(path: Path) -> list[QAPair]:
    """Read question, answer and source_uri from every line of a JSON Lines file."""
    pairs: list[QAPair] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [f for f in _FIELDS if not isinstance(row.get(f), str) or not row[f].strip()]
        if missing:
            raise ValueError(f"{path.name} line {number}: missing {', '.join(missing)}")
        pairs.append(QAPair(row["question"], row["answer"], row["source_uri"]))
    return pairs


def _source(hit: Hit) -> Source:
    return Source(hit.text, hit.source_uri, hit.tier)


def rag_examples(
    pairs: list[QAPair],
    retriever: Retriever,
    *,
    sources: int = 4,
    oracle_rate: float = 0.8,
    seed: int = 0,
) -> list[RagExample]:
    """One training example per question, with the oracle at a random position or withheld."""
    if sources < 1 or not 0.0 <= oracle_rate <= 1.0:
        raise ValueError("sources must be positive and oracle_rate within [0, 1]")
    generator = np.random.default_rng(seed)
    examples: list[RagExample] = []
    for pair in pairs:
        hits = retriever.search(pair.question, sources + 8)
        oracles = [h for h in hits if h.source_uri == pair.source_uri]
        oracles = oracles or retriever.passages_of(pair.source_uri)
        if not oracles:
            raise ValueError(f"no passage of {pair.source_uri} is in the index")
        distractors = [_source(h) for h in hits if h.source_uri != pair.source_uri]
        if generator.random() < oracle_rate:
            chosen = distractors[: sources - 1]
            position = int(generator.integers(0, len(chosen) + 1))
            chosen.insert(position, _source(oracles[0]))
            target = f"{pair.answer} [{position + 1}]"
            examples.append(RagExample(pair.question, tuple(chosen), target, position + 1))
        else:
            withheld = tuple(distractors[:sources])
            examples.append(RagExample(pair.question, withheld, no_answer(pair.question), None))
    return examples
