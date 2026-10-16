import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compliance import analyse, segments  # noqa: E402

EN = "Okay, let's see. Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes with four. So 16 - 3 - 4 = 9 eggs. Nine eggs at two dollars each is eighteen dollars."
FR = "D'accord, voyons voir. Les canards de Janet pondent 16 oeufs par jour. Elle en mange trois au petit-déjeuner et en utilise quatre pour les muffins. Il reste donc 9 oeufs, vendus deux dollars chacun."
ZH = "好的，让我们来看看。Janet的鸭子每天产蛋16枚。她早餐吃掉3枚，做松饼用4枚，剩下9枚鸡蛋。每枚卖2美元，所以一共是18美元。"
SW = "Sawa, hebu tuone. Bata wa Janet hutaga mayai 16 kwa siku. Anakula matatu kwa kifungua kinywa na anatumia manne kuoka. Kwa hivyo yanabaki mayai tisa, kila moja kwa dola mbili."


def test_pure_language_traces():
    for text, code in [(EN, "en"), (FR, "fr"), (ZH, "zh"), (SW, "sw")]:
        a = analyse(text, code)
        assert a["dominant_lang"] == code, (code, a)
        assert a["complied"] is True
        assert a["target_frac"] >= 0.9


def test_english_trace_fails_french_target():
    a = analyse(EN, "fr")
    assert a["dominant_lang"] == "en"
    assert a["complied"] is False
    assert a["target_frac"] == 0.0
    assert a["first_noncompliant"].startswith("Janet")  # "Okay, let's see." is too short to label


def test_drift_is_measured_not_binary():
    a = analyse(FR + " " + EN + " " + EN, "fr")
    assert a["dominant_lang"] == "en"
    assert 0.0 < a["target_frac"] < 0.5
    assert a["complied"] is False


def test_arithmetic_only_segments_are_dropped():
    assert segments("16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18.") == []
    a = analyse("16 - 3 = 13. 13 - 4 = 9.", "fr")
    assert a["dominant_lang"] is None and a["complied"] is None and a["n_segments"] == 0


def test_cjk_counts_as_letters():
    assert len(segments(ZH)) >= 2
