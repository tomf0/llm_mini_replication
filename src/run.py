"""Run the experiment: one JSONL row per (question, reasoning language, variant).

Rows are written as they are produced and existing rows are skipped on restart,
so an interrupted Kaggle session can be resumed by re-running the same command.

Usage:
  python src/run.py --config config/local_dev.yaml
  python src/run.py --config config/kaggle_main.yaml --backend hf   # if vllm won't install
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402

from compliance import analyse  # noqa: E402
from config import RunConfig, parse_args  # noqa: E402
from extract import is_correct, parse_output  # noqa: E402
from generate import Generation, make_backend  # noqa: E402
from prompts import THINK_PREFILL, build_messages, render_prompt  # noqa: E402

GIT_SHA = None
try:
    import subprocess
    GIT_SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent).stdout.strip() or None
except Exception:
    pass


def build_row(cfg: RunConfig, qid: int, question: str, gold, lang: str, variant: str,
              prompt: str, gen: Generation, count_tokens) -> dict:
    """Pure function from one generation to one results row (testable without a model)."""
    parsed = parse_output(gen.text, thinking_expected=cfg.enable_thinking)
    comp = analyse(parsed["thinking"], lang)
    prefill = THINK_PREFILL[lang] if variant == "prefill" else None
    return {
        # identity
        "question_id": qid, "reasoning_lang": lang, "variant": variant,
        "question": question, "gold_answer": gold,
        # what the model saw and said
        "prompt": prompt, "prefill": prefill, "full_output": gen.text,
        # answer
        "extracted_answer": parsed["extracted_answer"],
        "extraction_method": parsed["extraction_method"],
        "parse_status": parsed["parse_status"],
        "correct": is_correct(parsed["extracted_answer"], gold),
        # length / truncation
        "trace_tokens": count_tokens(parsed["thinking"]),       # excludes any prefill
        "response_tokens": count_tokens(parsed["response"]),
        "new_tokens": gen.n_tokens,
        "think_closed": parsed["think_closed"],
        "finished": gen.finished, "truncated": gen.truncated,
        # compliance
        "dominant_lang": comp["dominant_lang"], "target_frac": comp["target_frac"],
        "lang_fracs": comp["lang_fracs"], "complied": comp["complied"],
        "first_noncompliant": comp["first_noncompliant"],
        # provenance
        "model": cfg.model, "seed": cfg.seed, "generation": cfg.generation_settings,
        "git_sha": GIT_SHA, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                keys.add((r["question_id"], r["reasoning_lang"], r["variant"]))
    return keys


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main(argv=None) -> None:
    cfg = parse_args(argv)
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"config: {json.dumps(cfg.to_dict())}", flush=True)

    ds = load_dataset(cfg.dataset, cfg.dataset_config, split=cfg.dataset_split)
    done = existing_keys(out)
    # Question-major order so that a partial file is balanced across conditions.
    jobs = [(q, lang, v) for q in cfg.qids for v in cfg.variants for lang in cfg.languages
            if (q, lang, v) not in done]
    print(f"{len(done)} rows already present, {len(jobs)} to generate -> {out}", flush=True)
    if not jobs:
        return

    backend = make_backend(cfg)
    tok = backend.tokenizer
    count_tokens = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0  # noqa: E731

    t0 = time.time()
    n_done = n_correct = n_parsed = n_trunc = n_comply = 0
    with out.open("a") as f:
        for batch in chunks(jobs, cfg.batch_size):
            prompts = []
            for q, lang, v in batch:
                msgs = build_messages(ds[q]["question"], lang, v)
                prompts.append(render_prompt(tok, msgs, lang, v, enable_thinking=cfg.enable_thinking))
            gens = backend.generate(prompts)
            for (q, lang, v), prompt, gen in zip(batch, prompts, gens):
                row = build_row(cfg, q, ds[q]["question"], ds[q]["answer_number"], lang, v, prompt, gen, count_tokens)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                n_correct += row["correct"] is True
                n_parsed += row["parse_status"] == "ok"
                n_trunc += row["truncated"]
                n_comply += row["complied"] is True
            f.flush()
            el = time.time() - t0
            eta = el / n_done * (len(jobs) - n_done)
            print(f"[{n_done}/{len(jobs)}] acc {n_correct}/{n_parsed} parsed | trunc {n_trunc} | "
                  f"complied {n_comply} | {el/60:.1f} min elapsed, ~{eta/60:.0f} min left", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
