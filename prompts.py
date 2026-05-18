TARGET_LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "kk": "Kazakh",
    "uz": "Uzbek",
    "tr": "Turkish",
}

SOURCE_LANG_NAMES = {
    "chg": "Chagatai",
}


def get_target_name(target_lang: str) -> str:
    return TARGET_LANG_NAMES.get(target_lang, target_lang)


def get_source_name(source_lang: str) -> str:
    return SOURCE_LANG_NAMES.get(source_lang, source_lang)


def build_prompt(
    examples,
    input_text: str,
    source_lang: str = "chg",
    target_lang: str = "en",
) -> str:
    """
    Standard ICL prompt.

    This follows the article's idea:
    instruction + formatted demonstrations + input sentence.
    """

    source_name = get_source_name(source_lang)
    target_name = get_target_name(target_lang)

    prompt = (
        f"Task: Translate the following {source_name} texts to {target_name}.\n"
        f"Output only the translation. Do not explain anything.\n\n"
        f"Examples:\n"
    ) 


    for _, ex in examples.iterrows():
        source = str(ex["source_text"]).strip()
        target = str(ex["target_text"]).strip()

        prompt += f"{source_name}: {source}\n"
        prompt += f"{target_name}: {target}\n\n"

    prompt += "Now translate:\n"
    prompt += f"{source_name}: {input_text}\n"
    prompt += f"{target_name}:"

    print("[INFO] Generated prompt:\n", prompt)
    return prompt