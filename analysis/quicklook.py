"""Quick per-condition summary of a results JSONL. No stats, no plots: a sanity check.

Usage: python analysis/quicklook.py results/dev/local_dev.jsonl [--examples 2]
"""
import argparse
import json
from collections import defaultdict


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("variant", "-"), r["reasoning_lang"])].append(r)
    header = f"{'variant':16} {'lang':4} {'n':>3} {'acc(all)':>9} {'acc(cmpl)':>9} {'parsed':>6} {'trunc':>5} {'comply':>6} {'tgt%':>5} {'trace_tok':>9}"
    print(header); print("-" * len(header))
    for (v, lang), rs in sorted(groups.items()):
        n = len(rs)
        correct = sum(r["correct"] is True for r in rs)
        parsed = sum(r.get("parse_status") == "ok" for r in rs)
        trunc = sum(bool(r.get("truncated")) for r in rs)
        comp = [r for r in rs if r.get("complied") is True]
        acc_c = f"{sum(r['correct'] is True for r in comp)}/{len(comp)}" if comp else "-"
        tgt = [r["target_frac"] for r in rs if r.get("target_frac") is not None]
        tgt_s = f"{100*sum(tgt)/len(tgt):4.0f}%" if tgt else "    -"
        toks = [r["trace_tokens"] for r in rs if r.get("trace_tokens") is not None]
        tok_s = f"{sum(toks)/len(toks):9.0f}" if toks else f"{'-':>9}"
        print(f"{v:16} {lang:4} {n:3d} {correct:>4}/{n:<4} {acc_c:>9} {parsed:6d} {trunc:5d} {len(comp):6d} {tgt_s:>5} {tok_s}")


def examples(rows, k):
    print(f"\n=== {k} example(s) per condition: first 160 chars of thinking ===")
    seen = defaultdict(int)
    for r in rows:
        key = (r.get("variant", "-"), r["reasoning_lang"])
        if seen[key] >= k:
            continue
        seen[key] += 1
        text = r["full_output"].split("</think>")[0].replace("<think>", "").strip().replace("\n", " ")
        print(f"[{key[0]} {key[1]} q{r['question_id']}] lang={r.get('dominant_lang')} correct={r['correct']} "
              f"status={r.get('parse_status')} | {text[:160]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--examples", type=int, default=0)
    a = ap.parse_args()
    rows = load(a.path)
    print(f"{len(rows)} rows from {a.path}\n")
    summarise(rows)
    if a.examples:
        examples(rows, a.examples)
