"""Split a model output into thinking / response, and extract the numeric answer.

Unparseable outputs are never silently scored as wrong: parse_output returns a
status explaining what happened, which is logged in every results row.

Statuses:
  "ok"             an answer was parsed from the response (after </think>)
  "no_think_close" thinking never closed (almost always truncation): no answer exists
  "no_answer"      thinking closed, but no parseable answer in the response

Extraction methods, in priority order:
  "answer_line"    the requested format:  Answer: 18   (also **Answer:** $1,800 etc.)
  "answer_is"      prose fallback:        the final answer is 18 / answer = 18
  "boxed"          LaTeX fallback:        \\boxed{18}
  "short_response" the whole response is one short line: "18", "<18>", "3 x 60 = 180"
"""
import re

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Digits with optional "," or " " thousands groups ("1,800", "1 800") and optional decimals.
# The lookahead rejects fractions like 3/4 rather than silently reading them as 3.
NUMBER = r"-?\d+(?:[ ,]\d{3})*(?:\.\d+)?(?!\d*/)"

# Junk allowed between the marker and the number: spaces, *, $, \$, \boxed{.
_JUNK = r"\s*\**[\s$\\]*(?:boxed\{)?[\s$\\]*"

ANSWER_RE = re.compile(r"answer\s*\**\s*[:：]" + _JUNK + "(" + NUMBER + ")", re.IGNORECASE)
ANSWER_IS_RE = re.compile(r"answer\s+(?:is|=)" + _JUNK + "(" + NUMBER + ")", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{[\s$\\]*(" + NUMBER + ")")

METHODS = [("answer_line", ANSWER_RE), ("answer_is", ANSWER_IS_RE), ("boxed", BOXED_RE)]

SPECIAL_RE = re.compile(r"<\|[^|>]+\|>")  # <|im_end|>, <|endoftext|>, ...
NUMBER_RE = re.compile(NUMBER)


def _short_response_number(response: str):
    """Fallback 3: the response is essentially just a number ("18", "<18>", "3 x 60 = 180").

    Only applies to a single short line, so we never pull a stray number out of prose.
    """
    text = SPECIAL_RE.sub("", response).strip()
    if "\n" in text or len(text) > 60 or re.search(r"\d\s*/\s*\d", text):  # multi-line, long, or a fraction
        return None
    nums = NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def split_output(text: str, thinking_expected: bool = True) -> tuple[str, str, bool]:
    """Return (thinking, response, think_closed).

    If </think> is present, thinking is everything before it and response is after.
    If it is absent and thinking was expected, the whole output is (unfinished)
    thinking and the response is empty: we must NOT parse answers out of a trace
    that was cut off mid-thought.
    """
    if THINK_CLOSE in text:
        thinking, response = text.split(THINK_CLOSE, 1)
        return thinking.replace(THINK_OPEN, "").strip(), response.strip(), True
    if thinking_expected:
        return text.replace(THINK_OPEN, "").strip(), "", False
    return "", text.strip(), False


def to_number(s: str) -> int | float:
    """'1,800' -> 1800, '1 800' -> 1800, '18.00' -> 18, '2.5' -> 2.5."""
    x = float(s.replace(",", "").replace(" ", ""))
    return int(x) if x.is_integer() else x


def extract_answer(response: str) -> tuple[int | float | None, str | None]:
    """Return (value, method); the LAST match of the highest-priority method wins."""
    for method, regex in METHODS:
        matches = regex.findall(response)
        if matches:
            return to_number(matches[-1]), method
    bare = _short_response_number(response)
    if bare is not None:
        return to_number(bare), "short_response"
    return None, None


def parse_output(text: str, thinking_expected: bool = True) -> dict:
    """Everything the results row needs from one raw output."""
    thinking, response, closed = split_output(text, thinking_expected)
    value, method = extract_answer(response) if response else (None, None)
    if value is not None:
        status = "ok"
    elif thinking_expected and not closed:
        status = "no_think_close"
    else:
        status = "no_answer"
    return {
        "thinking": thinking,
        "response": response,
        "think_closed": closed,
        "extracted_answer": value,
        "extraction_method": method,
        "parse_status": status,
    }


def is_correct(value, gold) -> bool | None:
    """None when there is no parsed answer, so analysis can separate 'wrong' from 'unparseable'."""
    if value is None:
        return None
    return abs(float(value) - float(gold)) < 1e-6
