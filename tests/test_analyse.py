import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from analyse import accuracy_table, bootstrap_ci, paired_bootstrap_ci, paired_table  # noqa: E402


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    x = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    lo, hi = bootstrap_ci(x, B=2000, seed=1)
    assert lo <= x.mean() <= hi
    assert (lo, hi) == bootstrap_ci(x, B=2000, seed=1)
    assert 0.2 < lo < 0.6 < hi < 0.95


def test_bootstrap_ci_degenerate_and_empty():
    assert bootstrap_ci(np.ones(5)) == (1.0, 1.0)
    assert all(np.isnan(bootstrap_ci(np.array([]))))


def test_paired_ci_is_zero_for_identical_and_tight_for_constant_shift():
    a = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0)
    lo, hi = paired_bootstrap_ci(np.ones(8), np.zeros(8))
    assert (lo, hi) == (1.0, 1.0)


def _fake(rows):
    df = pd.DataFrame(rows)
    df["correct_bin"] = (df["correct"] == True).astype(int)  # noqa: E712
    df["unparsed"] = df["parse_status"] != "ok"
    df["complied"] = df["complied"] == True  # noqa: E712
    return df


def test_paired_table_uses_same_questions_and_counts_discordant():
    rows = []
    for q in range(4):
        rows.append(dict(question_id=q, variant="v", reasoning_lang="en", correct=q < 3, parse_status="ok",
                         complied=True, truncated=False, target_frac=1.0, trace_tokens=100))
        rows.append(dict(question_id=q, variant="v", reasoning_lang="fr", correct=q < 1, parse_status="ok",
                         complied=True, truncated=False, target_frac=1.0, trace_tokens=100))
    p = paired_table(_fake(rows), B=500)
    r = p.iloc[0]
    assert r.n_pairs == 4 and abs(r.diff_vs_en - (-0.5)) < 1e-9
    assert r.lang_right_en_wrong == 0 and r.lang_wrong_en_right == 2
    assert r.ci_lo <= -0.5 <= r.ci_hi


def test_unparsed_counts_as_wrong_in_accuracy():
    rows = [dict(question_id=0, variant="v", reasoning_lang="en", correct=None, parse_status="no_think_close",
                 complied=True, truncated=True, target_frac=1.0, trace_tokens=50),
            dict(question_id=1, variant="v", reasoning_lang="en", correct=True, parse_status="ok",
                 complied=True, truncated=False, target_frac=1.0, trace_tokens=50)]
    t = accuracy_table(_fake(rows), B=200).iloc[0]
    assert t.accuracy == 0.5 and t.unparsed_rate == 0.5 and t.truncated_rate == 0.5
