# Project: Does reasoning language affect LLM accuracy? (mini replication)

## Context
I'm Tom, a final-year Maths & Statistics student. I'm building a small, weekend-sized
replication of "Phase 1" of a research project on how the language of a reasoning
model's chain of thought affects its accuracy. The goal is a clean, honest GitHub repo
I can point to in an application, AND for me to learn the LLM evaluation stack.

**Research question (mini version):** On MGSM maths problems asked in English, does a
small open reasoning model get worse when instructed to *reason* in another language,
while the question and final answer stay in English?

## How I want you to work
- Go step by step. Before running anything non-trivial, explain briefly what it does
  and why. I want to understand the stack, not just get output.
- Prefer small, verifiable steps: get one question working before running many.
- Ask before installing large packages or downloading models over ~2 GB.
- Keep code simple and readable; no premature abstraction beyond what's noted below.
- Never add paid cloud services. Budget is zero.
- When results look surprising, check for bugs (parsing, truncation, templates)
  before interpreting them.

## Environment (important constraints)
- **Local machine: Apple M1 (modest).** Use it ONLY to develop and debug the pipeline
  on ~5–10 questions with a capped token budget, and later for analysis/plots.
  - Inference locally via Hugging Face `transformers` on MPS (or CPU fallback).
  - vLLM does NOT use the M1 GPU; don't try to use it locally.
  - Local outputs are for checking the code runs, not for judging model quality.
- **Real run: Kaggle notebook, free GPU (T4 or P100).** Claude cannot see this
  environment, so the script must be robust, configurable, and tested locally first.
  - T4 has no bf16: load models in float16.
  - vLLM may be used on Kaggle if it installs cleanly; otherwise `transformers`.
  - Results are saved to files I download and analyse locally.

## Design
- Put generation behind ONE function, e.g. `generate(prompts, cfg) -> list[str]`,
  with two backends: `hf` (transformers, local + fallback) and `vllm` (Kaggle).
  Everything else is backend-agnostic.
- One config (YAML or dataclass): model id, backend, languages, n_questions,
  max_new_tokens, temperature, seed, output path.
- Every run writes a JSONL file with one row per (question, reasoning_lang):
  `question_id, reasoning_lang, prompt, full_output, extracted_answer, gold_answer,
  correct, trace_tokens, complied, truncated, model, seed, generation settings`.
- Save full traces: reading failures is part of the analysis.

## Experiment spec
- **Model:** a small open reasoning model that emits visible thinking, ~1.5B
  (e.g. a DeepSeek-R1-Distill-Qwen-1.5B-class model — check Hugging Face for what's
  current and suitable). Use the model's own chat template and thinking format.
- **Data:** MGSM, English split (250 questions). Verify the dataset id on Hugging Face.
- **Conditions:** reasoning in English (baseline), French, Chinese, Swahili.
  Question and final answer stay in English. Prompting only; no constrained decoding.
- **Answer format:** ask for a final line `Answer: <number>`; parse robustly, and log
  unparseable outputs rather than silently scoring them wrong.
- **Token budget:** generous enough not to truncate typical traces; record truncation.
- **Compliance:** detect the language of the reasoning trace (e.g. fastText language
  ID or `lingua`); record the dominant language and whether it matches the target.

## Analysis (statistics matter most here)
- Accuracy per language with bootstrap 95% CIs.
- **Paired** differences vs English on the same questions, paired bootstrap CIs.
- Confounds to check and report: trace length (tokenisation differs by language),
  truncation rate, non-compliance (model drifting back to English), unparseable answers.
- Report accuracy both on all items and on compliant-only items.
- Plots: accuracy by language with error bars; accuracy vs trace length by language.

## Milestones
1. Environment set up locally; load a small model on MPS; generate for 1 question.
2. Prompt builder for each reasoning language + answer extraction, tested on ~5 items.
3. Compliance + length + truncation logging; full JSONL row written correctly.
4. Config + backend switch; script runs end-to-end locally on ~10 items.
5. Kaggle-ready script/notebook cell + instructions for me to run it there.
6. Analysis script on downloaded results: tables, CIs, plots.
7. README: setup, findings (2–3 sentences, honest), caveats (one small model, small
   sample, prompting not constrained decoding, possible benchmark contamination).

Stretch goals (only if time): second model, second benchmark (multilingual MMLU-style),
seed-to-seed variance, logit-lens check for English in intermediate layers.

## Repo layout (suggested)
```
config/        run configs
src/           generate.py, prompts.py, extract.py, compliance.py, run.py
analysis/      analyse.py, plots
results/       raw JSONL from runs (gitignored if large)
README.md
```