"""generate(prompts, cfg) -> list[Generation], behind two interchangeable backends.

  hf    Hugging Face transformers. Works on MPS (M1), CUDA, CPU. Batched with LEFT
        padding so every prompt ends in the same column and generation continues
        from the last real token.
  vllm  vLLM, for the Kaggle T4. Untested locally (vLLM does not run on M1), so it
        is kept minimal.

Both return the same Generation record, so nothing downstream knows which ran.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import RunConfig


@dataclass
class Generation:
    text: str            # decoded new tokens only (prompt excluded), special tokens kept
    n_tokens: int        # new tokens generated, including the end token if any
    finished: bool       # model emitted an end-of-sequence token
    truncated: bool      # hit max_new_tokens without finishing


class HFBackend:
    def __init__(self, cfg: RunConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:        # some models ship without one
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        print(f"[hf] {cfg.model} on {self.device} in {cfg.dtype}; eos ids {sorted(self.eos_ids)}, "
              f"pad id {self.tokenizer.pad_token_id}", flush=True)

    def generate(self, prompts: list[str]) -> list[Generation]:
        torch, tok, cfg = self.torch, self.tokenizer, self.cfg
        # add_special_tokens=False: the chat template already placed every control token.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        torch.manual_seed(cfg.seed)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.temperature > 0,
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                top_p=cfg.top_p if cfg.temperature > 0 else None,
                top_k=cfg.top_k if cfg.temperature > 0 else None,
                pad_token_id=tok.pad_token_id,
            )
        results = []
        for row in out[:, n_prompt:].tolist():
            # Cut at the first end token; everything after it is padding.
            end = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
            finished = end is not None
            ids = row[: end + 1] if finished else row
            results.append(Generation(
                text=tok.decode(ids, skip_special_tokens=False),
                n_tokens=len(ids),
                finished=finished,
                truncated=not finished and len(ids) >= cfg.max_new_tokens,
            ))
        if self.device == "mps":
            torch.mps.empty_cache()
        return results


class VLLMBackend:
    def __init__(self, cfg: RunConfig):
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.llm = LLM(model=cfg.model, dtype=cfg.dtype, seed=cfg.seed,
                       max_model_len=cfg.max_model_len, gpu_memory_utilization=0.90)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
            max_tokens=cfg.max_new_tokens, seed=cfg.seed, skip_special_tokens=False,
        )
        print(f"[vllm] {cfg.model} in {cfg.dtype}, max_model_len {cfg.max_model_len}", flush=True)

    def generate(self, prompts: list[str]) -> list[Generation]:
        outs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        results = []
        for o in outs:
            c = o.outputs[0]
            finished = c.finish_reason == "stop"
            results.append(Generation(text=c.text, n_tokens=len(c.token_ids),
                                      finished=finished, truncated=c.finish_reason == "length"))
        return results


def make_backend(cfg: RunConfig):
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")
