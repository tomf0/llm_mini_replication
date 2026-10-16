# Does the language a reasoning model *thinks* in affect its accuracy?

A weekend-sized replication of one question from a larger research project: on English
maths word problems, does a small open reasoning model get worse when it is made to
reason in another language, while the question and the final answer stay in English?

**Status:** pipeline complete and tested locally; main run on Kaggle pending.
Findings below are placeholders until the run is in.

## Setup

- **Model:** [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), a 1.7B-parameter
  model with a visible `<think>...</think>` reasoning block, run in float16. Qwen3-0.6B is
  used locally to debug the pipeline only.
- **Data:** [MGSM](https://huggingface.co/datasets/juletxara/mgsm), English split, 250
  grade-school maths problems with integer answers.
- **Conditions:** reasoning language in {English, French, Chinese, Swahili}, crossed with
  two prompting arms:
  - `instruct_en`: an English instruction asking the model to reason only in the target
    language and finish with `Answer: <number>` in English.
  - `prefill`: the same instruction, plus the thinking block is opened for the model with a
    short phrase in the target language (e.g. `<think>\nD'accord, voyons voir.`).
  Prompting only; no constrained decoding. Sampling at the model's recommended settings
  (temperature 0.6, top-p 0.95, top-k 20), one seed.
- **Measures per trace:** parsed answer and correctness; parse status (so unparseable
  outputs are counted, not hidden); thinking length in tokens; truncation; and the language
  the thinking is *actually* in, detected per sentence with
  [lingua](https://github.com/pemistahl/lingua-py), summarised as the dominant language and
  the fraction of text in the target language.

## Findings

_Pending the Kaggle run. This section will hold 2-3 sentences and two figures:_

- _accuracy by reasoning language with 95% bootstrap CIs, both arms;_
- _paired differences vs English on the same questions;_
- _accuracy on compliant traces only, and the compliance rate per language._

What the local debugging runs on the 0.6B model already show, for context only (10
questions, not evidence about the 1.7B model): the plain instruction is ignored entirely
(0% of traces in the requested language), and the prefilled opener works for Chinese and
mostly for French but not for Swahili, where the model reverts to English within a sentence.

## Caveats

- One small model, one seed, 250 questions. CIs are wide; treat differences as suggestive.
- Prompting, not constrained decoding: the model may not comply, so compliance is reported
  as a result and accuracy is given both on all items and on compliant items only.
- MGSM/GSM8K is old and widely published, and the model's training data cannot be audited.
  Absolute accuracy is probably inflated by contamination, and the paired differences could
  be biased in either direction (memorised English solutions could survive non-English
  reasoning, or be disrupted by it). MGSM also contains French and Chinese translations of
  these exact problems but no Swahili ones.
- Trace length in tokens is not comparable across languages: the tokenizer spends about
  2.5 tokens per Swahili word against 1.1 per English word. Length is a confound, not a
  clean measure of reasoning effort.
- The language detector is restricted to the four study languages, so drift into a fifth
  language would be mislabelled. Low-confidence sentences are labelled unknown.

## Repository layout

```
config/       run configs (local_dev.yaml, kaggle_compliance_check.yaml, kaggle_main.yaml)
src/          prompts.py, extract.py, compliance.py, generate.py (hf + vllm), config.py, run.py
analysis/     analyse.py (bootstrap CIs, paired diffs, figures), quicklook.py
kaggle/       run_experiment.ipynb + README.md: how to run on a free Kaggle T4
tests/        pytest suite (extractor, prompts, compliance, row schema, statistics)
results/      raw JSONL, one row per (question, language, arm); dev/ is from the 0.6B model
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                            # 48 tests, no model needed
python src/run.py --config config/local_dev.yaml     # ~40 min on an M1, Qwen3-0.6B
python analysis/analyse.py results/dev/local_dev.jsonl
```

For the real run see [kaggle/README.md](kaggle/README.md).
