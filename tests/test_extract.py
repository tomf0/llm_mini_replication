import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract import is_correct, parse_output  # noqa: E402
from prompts import LANGUAGES, build_messages  # noqa: E402

E = "<|im_end|>"


@pytest.mark.parametrize(
    "response, expected, method",
    [
        ("So she makes $18.\n\nAnswer: 18", 18, "answer_line"),
        ("**Answer:** 18", 18, "answer_line"),
        ("**Answer: 18**", 18, "answer_line"),
        ("Answer: $18", 18, "answer_line"),
        ("Answer: \\$18", 18, "answer_line"),
        ("Answer: 1,800", 1800, "answer_line"),
        ("Answer: 18.00", 18, "answer_line"),
        ("Answer: 2.5", 2.5, "answer_line"),
        ("Answer: -3", -3, "answer_line"),
        ("Answer： 18", 18, "answer_line"),                  # full-width colon
        ("answer: 18 dollars.", 18, "answer_line"),
        ("Answer: \\boxed{18}", 18, "answer_line"),
        ("The answer is \\boxed{18}.", 18, "answer_is"),
        ("Answer: 20\nWait, correction.\nAnswer: 18", 18, "answer_line"),  # last one wins
        ("Answer: 18." , 18, "answer_line"),               # sentence-final period
        ("Answer: 1 800", 1800, "answer_line"),             # French-style thousands
        ("The final answer is 18.", 18, "answer_is"),
        ("So the answer = 1,800 dollars", 1800, "answer_is"),
    ],
)
def test_answer_formats(response, expected, method):
    out = parse_output(f"<think>\nreasoning\n</think>\n\n{response}{E}")
    assert out["parse_status"] == "ok"
    assert out["extracted_answer"] == expected
    assert out["extraction_method"] == method


def test_truncated_mid_thought_has_no_answer():
    # Even though "Answer: 18" appears in the thinking, the trace never closed.
    out = parse_output("<think>\nI think Answer: 18 but let me check again and")
    assert out["parse_status"] == "no_think_close"
    assert out["extracted_answer"] is None
    assert not out["think_closed"]


def test_answer_inside_thinking_is_ignored():
    out = parse_output(f"<think>\nAnswer: 18\n</think>\n\nShe makes eighteen dollars.{E}")
    assert out["parse_status"] == "no_answer"
    assert out["extracted_answer"] is None


def test_fraction_is_not_parsed_as_integer():
    out = parse_output(f"<think>x</think>\nAnswer: 3/4{E}")
    assert out["parse_status"] == "no_answer"


@pytest.mark.parametrize("response, expected", [("18", 18), ("<18>", 18), ("3 × 60 = 180", 180), ("$70,000", 70000)])
def test_short_bare_response(response, expected):
    out = parse_output(f"<think>x</think>\n\n{response}{E}")
    assert out["parse_status"] == "ok"
    assert out["extracted_answer"] == expected
    assert out["extraction_method"] == "short_response"


def test_prose_without_answer_marker_is_not_mined_for_numbers():
    out = parse_output(f"<think>x</think>\nShe starts with 16 eggs and eats 3 of them for breakfast every single day of the week.{E}")
    assert out["parse_status"] == "no_answer"


def test_answer_with_no_number():
    out = parse_output(f"<think>x</think>\nAnswer: eighteen{E}")
    assert out["parse_status"] == "no_answer"


def test_thinking_not_expected():
    out = parse_output("Answer: 7", thinking_expected=False)
    assert out["extracted_answer"] == 7


def test_split_contents():
    out = parse_output(f"<think>\nfoo\n</think>\n\nAnswer: 1{E}")
    assert out["thinking"] == "foo"
    assert out["response"].startswith("Answer: 1")


def test_is_correct():
    assert is_correct(18, 18) is True
    assert is_correct(18.0, 18) is True
    assert is_correct(17, 18) is False
    assert is_correct(None, 18) is None


def test_prompts_differ_only_in_language_name():
    q = "What is 2+2?"
    texts = {k: build_messages(q, k)[0]["content"] for k in LANGUAGES}
    # Every condition equals the French prompt with only the reasoning-language name swapped.
    for k, name in LANGUAGES.items():
        assert texts[k] == texts["fr"].replace("French", name)
    assert all(t.startswith(q) for t in texts.values())
    assert all('"Answer:"' in t and "<number>" not in t for t in texts.values())


def test_variants():
    q = "What is 2+2?"
    native = build_messages(q, "zh", "instruct_native")
    assert native[0]["role"] == "system" and native[1]["content"].startswith(q)
    assert build_messages(q, "sw", "prefill") == build_messages(q, "sw", "instruct_en")
    with pytest.raises(ValueError):
        build_messages(q, "fr", "nope")
