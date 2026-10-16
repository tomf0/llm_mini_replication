"""Milestone 1 smoke test: load a small reasoning model on MPS (or CPU) and
generate an answer for ONE maths question, with a capped token budget.

This is deliberately minimal. It exists to check that:
  * the model downloads and loads in float16 on this machine,
  * the chat template produces a prompt with the model's thinking format,
  * generation runs and we can see the raw output (thinking + answer).

Usage:
  python src/smoke_test.py --model Qwen/Qwen3-0.6B --max-new-tokens 512
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_QUESTION = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
    "and bakes muffins for her friends every day with four. She sells the remainder "
    "at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
    "does she make every day at the farmers' market?"
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # float16 everywhere: matches the Kaggle T4, which has no bf16.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"loaded {args.model} ({n_params:.2f}B params) in {time.time() - t0:.1f}s")

    # Build the prompt with the model's OWN chat template. enable_thinking is
    # honoured by Qwen3 templates and silently ignored by others (Jinja).
    messages = [
        {
            "role": "user",
            "content": args.question
            + "\n\nFinish with a final line of the form 'Answer: <number>'.",
        }
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    print("\n===== PROMPT (exact text fed to the model) =====")
    print(prompt)

    inputs = tok(prompt, return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]

    torch.manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    dt = time.time() - t0

    new_ids = out[0, n_prompt:]
    text = tok.decode(new_ids, skip_special_tokens=False)
    n_new = new_ids.shape[0]
    truncated = n_new >= args.max_new_tokens

    print("\n===== MODEL OUTPUT =====")
    print(text)
    print("\n===== STATS =====")
    print(f"prompt tokens: {n_prompt} | new tokens: {n_new} | "
          f"{n_new / dt:.1f} tok/s | truncated: {truncated}")


if __name__ == "__main__":
    main()
