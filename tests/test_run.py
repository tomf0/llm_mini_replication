import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import RunConfig  # noqa: E402
from generate import Generation  # noqa: E402
from run import build_row, existing_keys  # noqa: E402

REQUIRED = {"question_id", "reasoning_lang", "prompt", "full_output", "extracted_answer", "gold_answer",
            "correct", "trace_tokens", "complied", "truncated", "model", "seed", "generation"}

fake_count = lambda s: len(s.split())  # noqa: E731


def test_row_has_every_field_from_the_spec():
    cfg = RunConfig()
    gen = Generation(text="<think>\nOkay, nine eggs at two dollars is eighteen.\n</think>\n\nAnswer: 18<|im_end|>",
                     n_tokens=40, finished=True, truncated=False)
    row = build_row(cfg, 0, "Janet...", 18, "en", "instruct_en", "PROMPT", gen, fake_count)
    assert REQUIRED <= set(row)
    assert row["correct"] is True and row["parse_status"] == "ok"
    assert row["complied"] is True and row["dominant_lang"] == "en"
    assert row["trace_tokens"] == 8 and row["response_tokens"] == 2
    assert row["prefill"] is None
    json.dumps(row)  # serialisable


def test_truncated_row_is_unparsed_not_wrong():
    cfg = RunConfig()
    gen = Generation(text="<think>\nOkay, let me think about this again and", n_tokens=cfg.max_new_tokens,
                     finished=False, truncated=True)
    row = build_row(cfg, 3, "q", 540, "fr", "instruct_en", "P", gen, fake_count)
    assert row["correct"] is None and row["parse_status"] == "no_think_close"
    assert row["truncated"] is True and row["think_closed"] is False
    assert row["complied"] is False and row["dominant_lang"] == "en"


def test_prefill_row_records_prefill_and_handles_missing_open_tag():
    cfg = RunConfig()
    gen = Generation(text="Les canards de Janet pondent seize oeufs par jour et elle en vend neuf.\n</think>\n\nAnswer: 18<|im_end|>",
                     n_tokens=30, finished=True, truncated=False)
    row = build_row(cfg, 0, "q", 18, "fr", "prefill", "P<think>\nD'accord, voyons voir. ", gen, fake_count)
    assert row["prefill"] == "D'accord, voyons voir."
    assert row["complied"] is True and row["correct"] is True


def test_config_yaml_roundtrip_and_unknown_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: X\nn_questions: 3\nvariants: [prefill]\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.model == "X" and cfg.qids == [0, 1, 2] and cfg.variants == ["prefill"]
    p.write_text("modle: X\n")
    try:
        RunConfig.from_yaml(p)
        assert False, "should reject typo"
    except ValueError as e:
        assert "modle" in str(e)


def test_shipped_configs_load():
    for name in ("local_dev", "kaggle_compliance_check", "kaggle_main"):
        cfg = RunConfig.from_yaml(Path(__file__).resolve().parents[1] / "config" / f"{name}.yaml")
        assert cfg.dtype == "float16"


def test_existing_keys_resume(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"question_id": 0, "reasoning_lang": "fr", "variant": "prefill"}) + "\n\n")
    assert existing_keys(p) == {(0, "fr", "prefill")}
    assert existing_keys(tmp_path / "missing.jsonl") == set()
