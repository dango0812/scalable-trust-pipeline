"""Guess the dominant writing system of a text."""

import re

_ASCII_LETTER = re.compile(r"[A-Za-z]")
_LETTER = re.compile(r"[^\W\d_]")
_HANGUL = re.compile("[가-힣ᄀ-ᇿ㄰-㆏]")


def detect_script(text: str) -> str | None:
    """ISO 15924 code of the dominant script: "Hang", "Latn", or None for anything else.

    A deliberate simplification, named for what it is: a script guess, not a language. A
    production pipeline runs a trained language identifier (fastText-style) and stores its
    label, score and model version. All counting happens in C (str.isascii, compiled
    regular expressions), so the guess costs microseconds, not milliseconds, per document.
    """
    if text.isascii():
        return "Latn" if _ASCII_LETTER.search(text) else None
    letters = len(_LETTER.findall(text))
    if letters == 0:
        return None
    if len(_HANGUL.findall(text)) / letters >= 0.3:
        return "Hang"
    if len(_ASCII_LETTER.findall(text)) / letters >= 0.5:
        return "Latn"
    return None
