"""Analysis of a results JSONL: accuracy with bootstrap CIs, paired differences vs
English, confound tables, and two figures.

Scoring rule: a row is correct only if a parsed answer equals the gold answer.
Unparseable and truncated rows count as INCORRECT in the headline accuracy (the honest
number) and their rates are reported alongside so nothing is silently dropped.

Usage:
  python analysis/analyse.py results/dev/local_dev.jsonl --out analysis/out/local_dev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "en"
LANG_NAMES = {"en": "English", "fr": "French", "zh": "Chinese", "sw": "Swahili"}
ARM_NAMES = {"instruct_en": "Instruction only", "prefill": "Instruction + prefilled opener",
             "instruct_native": "Instruction in target language"}


# ----------------------------------------------------------------------------- data

def load(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["correct_bin"] = (df["correct"] == True).astype(int)  # noqa: E712  None/False -> 0
    df["unparsed"] = df["parse_status"] != "ok"
    df["complied"] = df["complied"] == True  # noqa: E712
    if "variant" not in df:
        df["variant"] = "instruct_en"
    return df


def _order(df: pd.DataFrame, lang_col: str = "lang") -> pd.DataFrame:
    """Fixed row order: arms as in ARM_NAMES, languages as in LANG_NAMES."""
    if df.empty:
        return df
    a = {k: i for i, k in enumerate(ARM_NAMES)}; l = {k: i for i, k in enumerate(LANG_NAMES)}
    return (df.assign(_a=df["variant"].map(a).fillna(99), _l=df[lang_col].map(l).fillna(99))
              .sort_values(["_a", "_l"], kind="stable").drop(columns=["_a", "_l"]).reset_index(drop=True))


# ----------------------------------------------------------------------------- stats

def bootstrap_ci(x: np.ndarray, B: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of x (resampling items with replacement)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, B: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """CI for mean(a - b) where a[i], b[i] are the same question under two conditions.
    Resamples QUESTIONS, keeping each question's pair together."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), B=B, seed=seed)


def accuracy_table(df: pd.DataFrame, B: int) -> pd.DataFrame:
    recs = []
    for (v, lang), g in df.groupby(["variant", "reasoning_lang"], sort=False):
        c = g["correct_bin"].to_numpy()
        lo, hi = bootstrap_ci(c, B=B)
        recs.append({
            "variant": v, "lang": lang, "n": len(g),
            "accuracy": c.mean(), "ci_lo": lo, "ci_hi": hi,
            "unparsed_rate": g["unparsed"].mean(), "truncated_rate": g["truncated"].mean(),
            "compliance_rate": g["complied"].mean(), "mean_target_frac": g["target_frac"].mean(),
            "median_trace_tokens": g["trace_tokens"].median(),
        })
    return _order(pd.DataFrame(recs))


def compliant_only_table(df: pd.DataFrame, B: int) -> pd.DataFrame:
    recs = []
    for (v, lang), g in df[df["complied"]].groupby(["variant", "reasoning_lang"], sort=False):
        c = g["correct_bin"].to_numpy()
        lo, hi = bootstrap_ci(c, B=B)
        recs.append({"variant": v, "lang": lang, "n_compliant": len(g), "accuracy": c.mean(), "ci_lo": lo, "ci_hi": hi})
    return _order(pd.DataFrame(recs))


def paired_table(df: pd.DataFrame, B: int) -> pd.DataFrame:
    """Each non-English language vs English on the SAME questions, within each variant."""
    recs = []
    for v, gv in df.groupby("variant", sort=False):
        wide = gv.pivot_table(index="question_id", columns="reasoning_lang", values="correct_bin", aggfunc="first")
        if BASELINE not in wide:
            continue
        for lang in [c for c in wide.columns if c != BASELINE]:
            both = wide[[lang, BASELINE]].dropna()
            a, b = both[lang].to_numpy(), both[BASELINE].to_numpy()
            lo, hi = paired_bootstrap_ci(a, b, B=B)
            recs.append({
                "variant": v, "lang": lang, "n_pairs": len(both),
                "diff_vs_en": (a - b).mean(), "ci_lo": lo, "ci_hi": hi,
                "lang_right_en_wrong": int(((a == 1) & (b == 0)).sum()),
                "lang_wrong_en_right": int(((a == 0) & (b == 1)).sum()),
            })
    return _order(pd.DataFrame(recs))


def length_bins_table(df: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
    """Accuracy by trace-length bin. Bins are pooled quantiles over non-truncated rows so
    every language is cut at the same token thresholds."""
    d = df[~df["truncated"] & df["trace_tokens"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    edges = np.unique(np.quantile(d["trace_tokens"], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame()
    d["length_bin"] = pd.cut(d["trace_tokens"], bins=edges, include_lowest=True)
    out = (d.groupby(["variant", "reasoning_lang", "length_bin"], observed=True)
             .agg(n=("correct_bin", "size"), accuracy=("correct_bin", "mean"),
                  mean_tokens=("trace_tokens", "mean")).reset_index())
    out["length_bin"] = out["length_bin"].map(lambda iv: f"{int(np.ceil(iv.left))}-{int(iv.right)}")
    return _order(out, "reasoning_lang")


# ----------------------------------------------------------------------------- plots
# Palette: reference dataviz palette, light surface. Slot 1 blue, slot 2 orange for the
# two arms (adjacent-pair validated in the palette reference); text in ink tokens.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"instruct_en": "#2a78d6", "prefill": "#eb6834", "instruct_native": "#1baf7a"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7"); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def plot_accuracy(acc: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = [l for l in LANG_NAMES if l in set(acc["lang"])]
    variants = [v for v in ARM_NAMES if v in set(acc["variant"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160, facecolor=SURFACE)
    _style(ax)
    width = 0.22 if len(variants) > 1 else 0.3   # thin marks; the slot's leftover is air
    x = np.arange(len(langs))
    for i, v in enumerate(variants):
        sub = acc[acc["variant"] == v].set_index("lang").reindex(langs)
        xs = x + (i - (len(variants) - 1) / 2) * (width + 0.06)
        ax.bar(xs, sub["accuracy"], width, color=SERIES.get(v, INK2), label=ARM_NAMES.get(v, v), zorder=2)
        ax.errorbar(xs, sub["accuracy"], yerr=[sub["accuracy"] - sub["ci_lo"], sub["ci_hi"] - sub["accuracy"]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        for xi, (a, n) in zip(xs, zip(sub["accuracy"], sub["n"])):
            ax.text(xi, min(a, 0.97) + 0.015, f"{a:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [LANG_NAMES[l] for l in langs], color=INK2, fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_yticks(np.linspace(0, 1, 6), [f"{t:.0%}" for t in np.linspace(0, 1, 6)])
    ax.set_ylabel("Accuracy (unparsed = wrong)", color=INK2, fontsize=9.5)
    n = int(acc["n"].iloc[0])
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.905, f"Error bars: 95% bootstrap CI over {n} questions. Unparsed or truncated = wrong.",
             fontsize=8.5, color=MUTED)
    if len(variants) > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


def plot_length(df: pd.DataFrame, out: Path, title: str) -> None:
    """Small multiples: one panel per language, trace length (x) vs correctness, with a
    binned accuracy line. Single hue per panel (identity is carried by the panel title)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[~df["truncated"] & df["trace_tokens"].notna()]
    langs = [l for l in LANG_NAMES if l in set(d["reasoning_lang"])]
    variants = [v for v in ARM_NAMES if v in set(d["variant"])]
    fig, axes = plt.subplots(len(variants), len(langs), figsize=(2.3 * len(langs) + 1, 2.4 * len(variants) + 0.8),
                             dpi=160, facecolor=SURFACE, sharex=True, sharey=True, squeeze=False)
    xmax = d["trace_tokens"].quantile(0.99) * 1.05
    for r, v in enumerate(variants):
        for c, lang in enumerate(langs):
            ax = axes[r, c]; _style(ax)
            g = d[(d["variant"] == v) & (d["reasoning_lang"] == lang)]
            rng = np.random.default_rng(0)
            jitter = rng.uniform(-0.04, 0.04, len(g))
            ax.scatter(g["trace_tokens"], g["correct_bin"] + jitter, s=18, color=SERIES.get(v, INK2), alpha=0.55,
                       edgecolors=SURFACE, linewidths=1, zorder=3)
            if len(g) >= 6:
                edges = np.unique(np.quantile(g["trace_tokens"], [0, 1 / 3, 2 / 3, 1]))
                if len(edges) >= 3:
                    bins = pd.cut(g["trace_tokens"], edges, include_lowest=True)
                    agg = g.groupby(bins, observed=True).agg(x=("trace_tokens", "mean"), y=("correct_bin", "mean"))
                    ax.plot(agg["x"], agg["y"], color=SERIES.get(v, INK2), linewidth=2, marker="o", markersize=5,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            if r == 0:
                ax.set_title(LANG_NAMES[lang], fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(ARM_NAMES.get(v, v).replace(" + ", "\n+ "), fontsize=8.5, color=INK2)
            ax.set_yticks([0, 1], ["wrong", "right"]); ax.set_ylim(-0.25, 1.25); ax.set_xlim(0, xmax)
    for ax in axes[-1]:
        ax.set_xlabel("thinking tokens", fontsize=8.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.text(0.01, 0.93, "Dots: individual questions (jittered). Line: accuracy in three length tertiles. "
             "Truncated traces excluded.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, facecolor=SURFACE); plt.close(fig)


# ----------------------------------------------------------------------------- report

def fmt_pct(x) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.0f}%"


def md_table(df: pd.DataFrame, cols: dict[str, str], pct: set[str]) -> str:
    if df.empty:
        return "_(no rows)_\n"
    lines = ["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            if k in pct:
                cells.append(fmt_pct(v))
            elif isinstance(v, float):
                cells.append(f"{v:.0f}" if abs(v) >= 10 else f"{v:+.2f}" if "diff" in k else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def report(df: pd.DataFrame, acc: pd.DataFrame, comp: pd.DataFrame, paired: pd.DataFrame, lb: pd.DataFrame,
           B: int, src: str) -> str:
    model = df["model"].iloc[0]
    acc = acc.copy(); acc["ci"] = acc.apply(lambda r: f"[{fmt_pct(r.ci_lo)}, {fmt_pct(r.ci_hi)}]", axis=1)
    comp = comp.copy()
    if not comp.empty:
        comp["ci"] = comp.apply(lambda r: f"[{fmt_pct(r.ci_lo)}, {fmt_pct(r.ci_hi)}]", axis=1)
    paired = paired.copy()
    if not paired.empty:
        paired["ci"] = paired.apply(lambda r: f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", axis=1)
    parts = [
        f"# Results: {model}\n",
        f"Source `{src}`: {len(df)} rows, {df['question_id'].nunique()} questions, seed {df['seed'].iloc[0]}, "
        f"max_new_tokens {df['generation'].iloc[0]['max_new_tokens']}. CIs are percentile bootstrap "
        f"({B:,} resamples over questions). Unparseable and truncated rows count as incorrect.\n",
        "## Accuracy by reasoning language (all items)\n",
        md_table(acc, {"variant": "arm", "lang": "lang", "n": "n", "accuracy": "accuracy", "ci": "95% CI",
                       "unparsed_rate": "unparsed", "truncated_rate": "truncated", "compliance_rate": "complied",
                       "mean_target_frac": "mean % in target lang", "median_trace_tokens": "median trace tokens"},
                 {"accuracy", "unparsed_rate", "truncated_rate", "compliance_rate", "mean_target_frac"}),
        "## Paired difference vs English (same questions, within arm)\n",
        md_table(paired, {"variant": "arm", "lang": "lang", "n_pairs": "pairs", "diff_vs_en": "acc diff",
                          "ci": "95% paired CI", "lang_right_en_wrong": "lang right / en wrong",
                          "lang_wrong_en_right": "lang wrong / en right"}, set()),
        "## Accuracy on compliant items only\n",
        md_table(comp, {"variant": "arm", "lang": "lang", "n_compliant": "n compliant", "accuracy": "accuracy",
                        "ci": "95% CI"}, {"accuracy"}),
        "## Accuracy by trace-length bin (pooled tertiles, truncated excluded)\n",
        md_table(lb, {"variant": "arm", "reasoning_lang": "lang", "length_bin": "tokens", "n": "n",
                      "accuracy": "accuracy"}, {"accuracy"}),
        "## Figures\n",
        "![accuracy by language](accuracy_by_language.png)\n\n![accuracy vs trace length](accuracy_vs_trace_length.png)\n",
    ]
    return "\n".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=None, help="output dir (default analysis/out/<file stem>)")
    ap.add_argument("--B", type=int, default=10_000, help="bootstrap resamples")
    a = ap.parse_args(argv)
    out = Path(a.out or Path("analysis/out") / Path(a.path).stem)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.path)
    acc, comp, paired, lb = accuracy_table(df, a.B), compliant_only_table(df, a.B), paired_table(df, a.B), length_bins_table(df)
    model = df["model"].iloc[0].split("/")[-1]
    plot_accuracy(acc, out / "accuracy_by_language.png", f"MGSM accuracy by reasoning language, {model}")
    plot_length(df, out / "accuracy_vs_trace_length.png", f"Correctness vs thinking length, {model}")
    text = report(df, acc, comp, paired, lb, a.B, a.path)
    (out / "summary.md").write_text(text)
    for name, t in (("accuracy", acc), ("compliant_only", comp), ("paired_vs_en", paired), ("length_bins", lb)):
        t.to_csv(out / f"{name}.csv", index=False)
    print(text)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
