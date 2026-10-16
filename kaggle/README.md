# Running on Kaggle

Claude cannot see the Kaggle environment, so everything here was tested locally on an M1
with Qwen3-0.6B and the `hf` backend. The `vllm` backend is untested until it runs on Kaggle.

## One-time setup
1. Kaggle account with phone verification (required for GPU + internet).
2. The repo pushed to a public GitHub repository.

## Each run
1. kaggle.com -> Create -> New Notebook -> File -> Import Notebook -> upload `kaggle/run_experiment.ipynb`.
2. Right-hand panel: Accelerator **GPU T4 x1** (P100 also fine), Internet **On**.
3. Cell 1 already points at https://github.com/tomf0/llm_mini_replication.
4. Run cells 1-3 interactively. Read the compliance-check table and examples.
5. For the main run, use **Save Version -> Save & Run All (Commit)** so it runs headless (max 12 h).
6. When it finishes, download `results.zip` from the notebook's **Output** tab and unzip into
   `results/kaggle/` locally. Then run the analysis.

## If something breaks
- `pip install vllm` fails or vLLM crashes loading the model: set `BACKEND = "hf"` in cell 2 and rerun
  from cell 3. The hf path is slower (batch of 16 runs until its longest trace finishes) but was tested locally.
- Out of GPU memory with hf: lower `--batch-size` (8, then 4).
- Session died mid-run: rerun cell 4. It resumes from the rows already written.
- Many `truncated: true` rows: raise `max_new_tokens` in `config/kaggle_main.yaml` (and `max_model_len`
  to at least prompt + max_new_tokens) and rerun; only the missing rows are generated.
- The `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` at the end of a
  run is harmless noise from the `datasets` library on Python 3.12.

## Budget
Free tier: ~30 GPU-hours/week, 12 h per session. The main run is 2000 generations; at ~1000
tokens each that is ~2M tokens. vLLM on a T4 should finish in 1-2 h; `hf` could take 4-6 h.
