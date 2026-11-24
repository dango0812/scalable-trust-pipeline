import pytest
from corpus import ENGLISH, KOREAN, english, korean, pick

from trustpipe.gold.policy import QualityRules
from trustpipe.gold.quality import assess

RULES = QualityRules()


def reason(text: str, script: str | None = "Latn", rules: QualityRules = RULES) -> str | None:
    verdict = assess(text, script, rules)
    return None if verdict is None else verdict.reason


def lines(text: str) -> list[str]:
    return text.split("\n")


def test_varied_english_and_korean_pass() -> None:
    assert reason(english(1)) is None
    assert reason(korean(1), "Hang") is None


def test_documents_need_enough_words_and_not_too_many() -> None:
    assert reason(english(1, words=30)) == "short_doc"
    assert reason(english(1), rules=QualityRules(max_words=100)) == "long_doc"
    # Punctuation-only tokens are not words.
    assert reason(english(1, words=45) + " - - - - - - - - - -") == "short_doc"


def test_word_length_is_checked_for_latin_script_only() -> None:
    long_words = " ".join(pick(ENGLISH, i) * 3 + "ization" for i in range(80))
    assert reason(long_words) == "mean_word_length"
    long_eojeol = " ".join(pick(KOREAN, i) * 6 for i in range(80))
    assert reason(long_eojeol, "Hang") != "mean_word_length"


def test_symbols_bullets_and_ellipses() -> None:
    assert reason(english(1) + "\n" + " ".join(["#"] * 20)) == "hash_ratio"
    assert reason(english(1) + "\n" + " ".join(["..."] * 20)) == "ellipsis_ratio"
    assert reason("\n".join("- " + line for line in lines(english(1)))) == "bullet_lines"
    trailing = [line[:-1] + "..." if n % 2 else line for n, line in enumerate(lines(english(1)))]
    assert reason("\n".join(trailing)) == "ellipsis_lines"


def test_words_must_mostly_contain_letters() -> None:
    assert reason(english(1) + "\n" + " ".join(str(n) for n in range(40))) == "alpha_words"


def test_stop_words_are_required_of_latin_script_only() -> None:
    no_stop = "\n".join(
        " ".join(pick(ENGLISH, line, n) for n in range(15)) + "." for line in range(8)
    )
    assert reason(no_stop) == "stop_words"
    assert reason(no_stop, None) is None
    assert reason(no_stop, rules=QualityRules(min_stop_words=0)) is None


def test_repeated_paragraphs() -> None:
    a, b, c = english(1, 40), english(2, 40), english(3, 40)
    assert reason("\n\n".join([a, b, a, a])) == "dup_para_frac"
    short_b, short_c = english(2, 15), english(3, 15)
    long_a = english(1, 75)
    assert reason("\n\n".join([long_a, short_b, short_c, long_a])) == "dup_para_char_frac"
    assert reason("\n\n".join([a, b, c])) is None


def test_repeated_lines() -> None:
    body = lines(english(1, 150))
    assert reason("\n".join(body[:6] + body[:4])) == "dup_line_frac"
    one_long = " ".join(lines(english(4, 90)))
    short = lines(english(5, 60))
    assert reason("\n".join([one_long, *short[:3], one_long])) == "dup_line_char_frac"


def test_repeated_n_grams() -> None:
    tokens = ["the", "of"]
    for i in range(60):
        tokens += ["data", "model", pick(ENGLISH, i)]
    assert reason(" ".join(tokens)) == "top_2gram"
    half = english(6, 60).replace("\n", " ")
    assert reason(half + " " + half) == "dup_5gram"


@pytest.mark.parametrize("script", ["Latn", "Hang", None])
def test_every_verdict_explains_itself(script: str | None) -> None:
    verdict = assess(english(1, words=10), script, RULES)
    assert verdict is not None
    assert verdict.detail == "10 words < 50"


def test_texts_shorter_than_an_n_gram_skip_that_rule() -> None:
    rules = QualityRules(min_words=1, min_stop_words=0)
    assert assess("word", "Latn", rules) is None
