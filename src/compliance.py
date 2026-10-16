"""Which language is a reasoning trace actually written in?

We split the thinking text into sentence-like segments, discard segments that are
mostly numbers/symbols (arithmetic has no language), label each remaining segment
with lingua restricted to the four study languages, and aggregate by character
count. The row gets: dominant language, fraction of text in the target language,
and a boolean `complied`.

Restricting the detector to {en, fr, zh, sw} makes it far more accurate at telling
those apart, at the cost of mislabelling drift into a fifth language. Low-confidence
segments are labelled "unk" to soften that.
"""
import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

LANG_CODES = {Language.ENGLISH: "en", Language.FRENCH: "fr", Language.CHINESE: "zh", Language.SWAHILI: "sw"}

# Sentence-ish boundaries: Latin punctuation followed by whitespace, CJK punctuation
# (no whitespace follows it in Chinese), or newlines.
SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])|\n+")
LETTER_RE = re.compile(r"[^\W\d_]")  # unicode letters (includes CJK), no digits/underscore

MIN_LETTERS = 12      # shorter segments are too noisy to label
MIN_CONFIDENCE = 0.6  # below this the segment is "unk"
COMPLY_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_languages(*LANG_CODES).build()


def segments(text: str) -> list[str]:
    """Sentence-like chunks with enough letters to carry a language signal."""
    return [s.strip() for s in SPLIT_RE.split(text) if len(LETTER_RE.findall(s)) >= MIN_LETTERS]


def label_segment(seg: str) -> tuple[str, float]:
    """(language code or 'unk', confidence of the top language)."""
    values = _detector().compute_language_confidence_values(seg)
    if not values:
        return "unk", 0.0
    top = values[0]
    code = LANG_CODES.get(top.language, "unk")
    return (code if top.value >= MIN_CONFIDENCE else "unk"), float(top.value)


def analyse(thinking: str, target: str) -> dict:
    """Language profile of one trace against its target reasoning language."""
    segs = segments(thinking)
    weights: dict[str, int] = {}
    labelled = []
    for seg in segs:
        code, conf = label_segment(seg)
        n = len(LETTER_RE.findall(seg))
        weights[code] = weights.get(code, 0) + n
        labelled.append((code, conf, seg))
    total = sum(weights.values())
    if total == 0:
        return {"dominant_lang": None, "target_frac": None, "lang_fracs": {}, "complied": None,
                "n_segments": 0, "first_noncompliant": None}
    fracs = {k: round(v / total, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}
    dominant = next(iter(fracs))
    target_frac = fracs.get(target, 0.0)
    first_bad = next((seg for code, _, seg in labelled if code not in (target, "unk")), None)
    return {
        "dominant_lang": dominant,
        "target_frac": target_frac,
        "lang_fracs": fracs,
        "complied": dominant == target and target_frac >= COMPLY_THRESHOLD,
        "n_segments": len(segs),
        "first_noncompliant": first_bad[:120] if first_bad else None,
    }
