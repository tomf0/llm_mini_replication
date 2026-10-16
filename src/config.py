"""One run configuration, loaded from YAML, with a few CLI overrides.

Everything that could differ between the local M1 debug run and the Kaggle run
lives here, so the code never has to be edited between the two.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    model: str = "Qwen/Qwen3-0.6B"
    backend: str = "hf"                     # "hf" (transformers) or "vllm"
    dtype: str = "float16"                  # T4 has no bf16; keep fp16 everywhere
    languages: list[str] = field(default_factory=lambda: ["en", "fr", "zh", "sw"])
    variants: list[str] = field(default_factory=lambda: ["instruct_en", "prefill"])
    n_questions: int = 10                   # first N MGSM items; ignored if question_ids is set
    question_ids: list[int] | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 0
    batch_size: int = 4                     # prompts per generate() call
    enable_thinking: bool = True
    output: str = "results/dev/run.jsonl"
    dataset: str = "juletxara/mgsm"
    dataset_config: str = "en"
    dataset_split: str = "test"
    max_model_len: int = 4096               # vllm only: prompt + generation budget

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def qids(self) -> list[int]:
        return list(self.question_ids) if self.question_ids else list(range(self.n_questions))

    @property
    def generation_settings(self) -> dict:
        """What gets stamped into every results row."""
        return {k: getattr(self, k) for k in
                ("max_new_tokens", "temperature", "top_p", "top_k", "dtype", "backend", "enable_thinking")}


def parse_args(argv: list[str] | None = None) -> RunConfig:
    """`--config path.yaml` plus optional overrides for the things you tweak at the last minute."""
    ap = argparse.ArgumentParser(description="Run the reasoning-language experiment.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["hf", "vllm"])
    ap.add_argument("--n-questions", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--output")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    for key in ("model", "backend", "n_questions", "max_new_tokens", "batch_size", "output", "seed"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)
    return cfg
