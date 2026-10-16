"""Prompt construction for each reasoning-language condition.

Design rule: within a variant, every condition uses the same template; only the
language changes. The question and the required answer line stay in English.

Variants (all are prompting only; none use constrained decoding):
  "instruct_en"      English instruction naming the reasoning language (baseline design).
  "instruct_native"  The same instruction, written IN the target language, plus a short
                     system prompt in that language.
  "prefill"          English instruction + the thinking block is opened for the model with
                     a short phrase in the target language ("<think>\\nD'accord, voyons voir.").
"""

LANGUAGES = {"en": "English", "fr": "French", "zh": "Chinese", "sw": "Swahili"}
VARIANTS = ("instruct_en", "instruct_native", "prefill")

# NB: no "<number>" placeholder: Qwen3-0.6B copied it literally and answered "<18>".
INSTRUCTION_EN = (
    "Think step by step. Write all of your thinking and reasoning in {language} only, "
    "and do not switch to any other language while reasoning.\n"
    "When you have finished, write the final answer in English on its own line as the "
    "word \"Answer:\" followed by the number only."
)

# The same instruction translated. (Translations are mine; have a speaker check them
# before the real run if this variant is adopted.)
INSTRUCTION_NATIVE = {
    "en": INSTRUCTION_EN.format(language="English"),
    "fr": (
        "Réfléchis étape par étape. Écris tout ton raisonnement uniquement en français, "
        "sans jamais passer à une autre langue pendant que tu réfléchis.\n"
        "Quand tu as terminé, écris la réponse finale en anglais sur une ligne à part, "
        "sous la forme du mot « Answer: » suivi uniquement du nombre."
    ),
    "zh": (
        "请一步一步地思考。所有的思考和推理过程只能用中文书写，推理过程中不要切换到任何其他语言。\n"
        "完成后，请用英文在单独一行给出最终答案，格式为单词“Answer:”后面只跟数字。"
    ),
    "sw": (
        "Fikiria hatua kwa hatua. Andika mawazo na hoja zako zote kwa Kiswahili pekee, "
        "na usibadilishe kwenda lugha nyingine yoyote unapofikiri.\n"
        "Ukimaliza, andika jibu la mwisho kwa Kiingereza kwenye mstari wake peke yake, "
        "kama neno \"Answer:\" likifuatiwa na namba tu."
    ),
}

SYSTEM_NATIVE = {
    "en": "You are a careful assistant who reasons in English.",
    "fr": "Tu es un assistant rigoureux qui raisonne en français.",
    "zh": "你是一个严谨的助手，用中文进行推理。",
    "sw": "Wewe ni msaidizi makini anayefikiri kwa Kiswahili.",
}

# Opening words placed inside <think> for the "prefill" variant.
THINK_PREFILL = {
    "en": "Okay, let's see.",
    "fr": "D'accord, voyons voir.",
    "zh": "好的，让我们来看看。",
    "sw": "Sawa, hebu tuone.",
}


def build_messages(question: str, lang: str, variant: str = "instruct_en") -> list[dict]:
    """Chat messages for one (question, reasoning language, variant)."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown reasoning language {lang!r}; expected one of {list(LANGUAGES)}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    q = question.strip()
    if variant == "instruct_native":
        return [
            {"role": "system", "content": SYSTEM_NATIVE[lang]},
            {"role": "user", "content": f"{q}\n\n{INSTRUCTION_NATIVE[lang]}"},
        ]
    content = f"{q}\n\n{INSTRUCTION_EN.format(language=LANGUAGES[lang])}"
    return [{"role": "user", "content": content}]


def render_prompt(tokenizer, messages: list[dict], lang: str = "en",
                  variant: str = "instruct_en", enable_thinking: bool = True) -> str:
    """Apply the model's OWN chat template and return the exact prompt string.

    enable_thinking is read by Qwen3 templates and silently ignored by others,
    so always eyeball the rendered prompt when switching model families.
    For the prefill variant the assistant turn is started for the model.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    if variant == "prefill":
        prompt += "<think>\n" + THINK_PREFILL[lang] + " "
    return prompt
