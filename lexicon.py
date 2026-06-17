import unicodedata
from collections import defaultdict

import pandas as pd

TOKEN_EXTRA_CHARS = {"ʿ", "ʾ", "'", "-", "’"}


def is_token_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "N", "M"} or char in TOKEN_EXTRA_CHARS


def normalize_lexicon_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(char for char in text if is_token_char(char))


def tokenize_for_lexicon(text: str) -> list[str]:
    tokens = []
    current = []

    for char in unicodedata.normalize("NFKC", str(text)):
        if is_token_char(char):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return tokens


def dedupe_keep_order(values) -> list[str]:
    seen = set()
    result = []

    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


class Lexicon:
    def __init__(
        self,
        mapping: dict[str, list[str]],
        surfaces: dict[str, str],
        top_n: int = 2,
        max_entries: int = 80,
        use_fuzzy: bool = True,
        fuzzy_strategy: str = "max_matching",
        fuzzy_min_chars: int = 3,
    ):
        self.mapping = {
            key: dedupe_keep_order(values)
            for key, values in mapping.items()
            if key and dedupe_keep_order(values)
        }
        self.surfaces = dict(surfaces)
        self.top_n = max(1, int(top_n))
        self.max_entries = max(1, int(max_entries))
        self.use_fuzzy = bool(use_fuzzy)
        self.fuzzy_strategy = str(fuzzy_strategy).strip().lower()
        self.fuzzy_min_chars = max(1, int(fuzzy_min_chars))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        source_lang: str,
        target_lang: str,
        top_n: int = 2,
        max_entries: int = 80,
        use_fuzzy: bool = True,
        fuzzy_strategy: str = "max_matching",
        fuzzy_min_chars: int = 3,
    ):
        required = {"source_text", "target_text", "source_lang", "target_lang", "type"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing lexicon columns: {', '.join(missing)}")

        lexicon_df = df.copy()
        lexicon_df = lexicon_df[lexicon_df["source_lang"] == source_lang]
        lexicon_df = lexicon_df[lexicon_df["target_lang"] == target_lang]
        lexicon_df = lexicon_df[lexicon_df["type"] == "word"]

        mapping = defaultdict(list)
        surfaces = {}

        for row in lexicon_df.itertuples(index=False):
            source = str(getattr(row, "source_text", "")).strip()
            target = str(getattr(row, "target_text", "")).strip()
            key = normalize_lexicon_key(source)

            if not key or not source or not target:
                continue

            mapping[key].append(target)
            surfaces.setdefault(key, source)

        return cls(
            mapping=mapping,
            surfaces=surfaces,
            top_n=top_n,
            max_entries=max_entries,
            use_fuzzy=use_fuzzy,
            fuzzy_strategy=fuzzy_strategy,
            fuzzy_min_chars=fuzzy_min_chars,
        )

    def add_entries(
        self,
        rows,
        source_field: str = "source_text",
        target_field: str = "target_text",
    ):
        for row in rows:
            source = str(row.get(source_field, "")).strip()
            target = str(row.get(target_field, "")).strip()
            key = normalize_lexicon_key(source)

            if not key or not source or not target:
                continue

            values = self.mapping.setdefault(key, [])
            if target not in values:
                values.append(target)
            self.surfaces.setdefault(key, source)

    def stats(self) -> dict:
        return {
            "source_terms": len(self.mapping),
            "translation_entries": sum(len(values) for values in self.mapping.values()),
            "top_n": self.top_n,
            "max_entries": self.max_entries,
            "use_fuzzy": self.use_fuzzy,
            "fuzzy_strategy": self.fuzzy_strategy,
            "fuzzy_min_chars": self.fuzzy_min_chars,
        }

    def lookup(self, text: str) -> list[dict]:
        entries = []
        seen_keys = set()

        for token in tokenize_for_lexicon(text):
            key = normalize_lexicon_key(token)
            if not key:
                continue

            if key in self.mapping:
                self._append_entry(
                    entries=entries,
                    seen_keys=seen_keys,
                    key=key,
                    query=token,
                    match_type="exact",
                )
            elif self.use_fuzzy:
                for fuzzy_key in self._fuzzy_keys(key):
                    self._append_entry(
                        entries=entries,
                        seen_keys=seen_keys,
                        key=fuzzy_key,
                        query=token,
                        match_type="fuzzy",
                    )
                    if len(entries) >= self.max_entries:
                        return entries

            if len(entries) >= self.max_entries:
                return entries

        return entries

    def _append_entry(
        self,
        entries: list[dict],
        seen_keys: set[str],
        key: str,
        query: str,
        match_type: str,
    ):
        if key in seen_keys or key not in self.mapping:
            return

        seen_keys.add(key)
        entries.append(
            {
                "query": query,
                "source": self.surfaces.get(key, query),
                "translations": self.mapping[key][: self.top_n],
                "match": match_type,
            }
        )

    def _fuzzy_keys(self, key: str) -> list[str]:
        if self.fuzzy_strategy == "substring":
            return self._substring_fuzzy_keys(key)

        return self._maximum_matching_keys(key)

    def _substring_fuzzy_keys(self, key: str) -> list[str]:
        if len(key) < self.fuzzy_min_chars:
            return []

        candidates = set()
        length = len(key)

        for start in range(length):
            for end in range(length, start + self.fuzzy_min_chars - 1, -1):
                piece = key[start:end]
                if piece in self.mapping:
                    candidates.add(piece)

        return sorted(candidates, key=lambda item: (-len(item), item))

    def _maximum_matching_keys(self, key: str) -> list[str]:
        candidates = []

        for segment in self._forward_maximum_match(key):
            if segment in self.mapping:
                candidates.append(segment)

        for segment in self._backward_maximum_match(key):
            if segment in self.mapping:
                candidates.append(segment)

        return dedupe_keep_order(candidates)

    def _forward_maximum_match(self, key: str) -> list[str]:
        segments = []
        index = 0

        while index < len(key):
            match = self._longest_match_at(key, index)
            if match:
                segments.append(match)
                index += len(match)
            else:
                index += 1

        return segments

    def _backward_maximum_match(self, key: str) -> list[str]:
        segments = []
        index = len(key)

        while index > 0:
            match = self._longest_match_ending_at(key, index)
            if match:
                segments.append(match)
                index -= len(match)
            else:
                index -= 1

        return list(reversed(segments))

    def _longest_match_at(self, key: str, start: int) -> str | None:
        for end in range(len(key), start + self.fuzzy_min_chars - 1, -1):
            piece = key[start:end]
            if piece in self.mapping:
                return piece

        return None

    def _longest_match_ending_at(self, key: str, end: int) -> str | None:
        for start in range(0, end - self.fuzzy_min_chars + 1):
            piece = key[start:end]
            if piece in self.mapping:
                return piece

        return None
