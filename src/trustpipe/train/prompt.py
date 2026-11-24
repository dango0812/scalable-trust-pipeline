"""The retrieval-augmented prompt, shared by training and serving.

A model learns the exact format it is trained on, so the server must build prompts the
same way. This module is the single definition: numbered sources (passage text with its
URI and trust tier), the question, an instruction to answer only from the sources and cite
them as [n], and a fixed sentence for "the sources do not answer this".

The prompt goes through the tokenizer's chat template when the base model has one
(instruction-tuned models), so the model sees its native turn markers; base models without
a template get a plain "### System / ### User / ### Assistant" layout.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

SYSTEM = (
    "Answer the question using only the numbered sources. Cite every source you use as [n]. "
    "If the sources do not contain the answer, reply exactly with the no-answer sentence."
)
NO_ANSWER_EN = "The sources do not contain the answer."
NO_ANSWER_KO = "제공된 출처에서 답을 찾을 수 없습니다."
_HANGUL = re.compile("[가-힣]")
_CITATION = re.compile(r"\[(\d{1,3})\]")


@dataclass(frozen=True, slots=True)
class Source:
    text: str
    source_uri: str
    tier: str


class ChatTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...


def no_answer(question: str) -> str:
    """The no-answer sentence in the language of the question (Korean or English)."""
    return NO_ANSWER_KO if _HANGUL.search(question) else NO_ANSWER_EN


def user_message(question: str, sources: Sequence[Source]) -> str:
    blocks = [
        f"[{n}] {source.source_uri} (tier: {source.tier})\n{source.text}"
        for n, source in enumerate(sources, start=1)
    ]
    listing = "\n\n".join(blocks) if blocks else "(no sources)"
    return (
        f"Sources:\n\n{listing}\n\nQuestion: {question}\nNo-answer sentence: {no_answer(question)}"
    )


def render(tokenizer: ChatTokenizer, question: str, sources: Sequence[Source]) -> str:
    """The full prompt text, ending where the model's answer begins."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(question, sources)},
    ]
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return str(rendered)
    return f"### System\n{SYSTEM}\n\n### User\n{messages[1]['content']}\n\n### Assistant\n"


def citations(answer: str, sources: Sequence[Source]) -> list[Source]:
    """The sources an answer cites, in order of first citation, ignoring invalid numbers."""
    cited: list[Source] = []
    for match in _CITATION.finditer(answer):
        number = int(match[1])
        if 1 <= number <= len(sources) and sources[number - 1] not in cited:
            cited.append(sources[number - 1])
    return cited
