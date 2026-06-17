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


def format_lexicon_entries(entries: list[dict] | None) -> str:
    if not entries:
        return "Vocabulary hints: no matching dictionary entries found.\n"

    lines = ["Vocabulary hints:\n"]

    for entry in entries:
        source = str(entry.get("source", "")).strip()
        query = str(entry.get("query", "")).strip()
        translations = [
            str(value).strip()
            for value in entry.get("translations", [])
            if str(value).strip()
        ]

        if not source or not translations:
            continue

        quoted_translations = " or ".join(
            f'"{translation}"' for translation in translations
        )
        match = str(entry.get("match", "exact")).strip()

        if match == "fuzzy" and query and query != source:
            lines.append(
                f'- "{query}" may contain dictionary word "{source}", '
                f"which means {quoted_translations}.\n"
            )
        else:
            lines.append(f'- "{source}" means {quoted_translations}.\n')

    if len(lines) == 1:
        return "Vocabulary hints: no matching dictionary entries found.\n"

    return "".join(lines)


def build_prompt(
    examples,
    input_text: str,
    source_lang: str = "chg",
    target_lang: str = "en",
    prompt_mode: str = "standard",
    example_lexicon_entries: list[list[dict]] | None = None,
    input_lexicon_entries: list[dict] | None = None,
) -> str:
    """
    Standard ICL prompt.

    This follows the article's idea:
    instruction + formatted demonstrations + input sentence.
    """

    source_name = get_source_name(source_lang)
    target_name = get_target_name(target_lang)
    prompt_mode = str(prompt_mode).strip().lower()

    if prompt_mode == "dipmt_plus":
        return build_dipmt_plus_prompt(
            examples=examples,
            input_text=input_text,
            source_name=source_name,
            target_name=target_name,
            example_lexicon_entries=example_lexicon_entries,
            input_lexicon_entries=input_lexicon_entries,
        )

    if prompt_mode != "standard":
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")

    parts = [
        f"Task: Translate the following {source_name} texts to {target_name}.\n"
        f"Output only the translation. Do not explain anything.\n\n"
        f"Examples:\n"
    ]

    for _, ex in examples.iterrows():
        source = str(ex["source_text"]).strip()
        target = str(ex["target_text"]).strip()

        parts.append(f"{source_name}: {source}\n")
        parts.append(f"{target_name}: {target}\n\n")

    parts.append("Now translate:\n")
    parts.append(f"{source_name}: {input_text}\n")
    parts.append(f"{target_name}:")

    return "".join(parts)


def build_dipmt_plus_prompt(
    examples,
    input_text: str,
    source_name: str,
    target_name: str,
    example_lexicon_entries: list[list[dict]] | None = None,
    input_lexicon_entries: list[dict] | None = None,
) -> str:
    example_lexicon_entries = example_lexicon_entries or []

    parts = [
        f"Task: Translate the following {source_name} texts to {target_name}.\n"
        "Use the vocabulary hints and the examples as in-context guidance.\n"
        "Output only the translation. Do not explain anything.\n\n"
    ]

    for example_idx, (_, ex) in enumerate(examples.iterrows(), start=1):
        source = str(ex["source_text"]).strip()
        target = str(ex["target_text"]).strip()
        lexicon_entries = (
            example_lexicon_entries[example_idx - 1]
            if example_idx - 1 < len(example_lexicon_entries)
            else None
        )

        parts.append(f"Example {example_idx}:\n")
        parts.append(f"{source_name}: {source}\n")
        parts.append(format_lexicon_entries(lexicon_entries))
        parts.append(f"{target_name}: {target}\n\n")

    parts.append("Now translate:\n")
    parts.append(f"{source_name}: {input_text}\n")
    parts.append(format_lexicon_entries(input_lexicon_entries))
    parts.append(f"{target_name}:")

    return "".join(parts)
