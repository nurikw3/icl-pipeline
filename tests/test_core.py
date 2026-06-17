import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from bli import parse_bli_lexicon
from lexicon import Lexicon
from prompts import build_prompt
from retrieval import GraphICLRetriever
from run_baseline import (
    OpenAIClientPool,
    call_openai,
    load_dictionary_df,
    load_static_examples,
    make_data_fingerprint,
    make_output_paths,
    normalize_schema,
    retrieve_random_examples,
    sample_df_fraction,
)


class FakeUsage:
    def model_dump(self):
        return {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


class FakeMessage:
    content = " translated text "


class EmptyMessage:
    content = "   "


class FakeChoice:
    message = FakeMessage()
    finish_reason = "stop"


class EmptyChoice:
    message = EmptyMessage()
    finish_reason = "length"


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()


class EmptyResponse:
    choices = [EmptyChoice()]
    usage = FakeUsage()


class FlakyCompletions:
    def __init__(self, failures_before_success):
        self.failures_before_success = failures_before_success
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise RuntimeError("temporary failure")
        return FakeResponse()


class EmptyThenSuccessCompletions:
    def __init__(self):
        self.calls = 0
        self.max_tokens_seen = []

    def create(self, **kwargs):
        self.calls += 1
        self.max_tokens_seen.append(kwargs.get("max_tokens"))
        if self.calls == 1:
            return EmptyResponse()
        return FakeResponse()


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


class Args:
    run_name = None
    strategies = ["random", "similarity"]
    k_list = [1, 3]
    target_lang = "en"
    retrieval_backend = "dense"
    embedding_model = "intfloat/multilingual-e5-base"


class CoreTests(unittest.TestCase):
    def test_build_prompt_does_not_print(self):
        examples = pd.DataFrame(
            [
                {
                    "source_text": "dur",
                    "target_text": "is",
                }
            ]
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            prompt = build_prompt(examples, "nan")

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Chagatai: dur", prompt)
        self.assertTrue(prompt.endswith("English:"))

    def test_dipmt_plus_prompt_includes_vocabulary_hints(self):
        examples = pd.DataFrame(
            [
                {
                    "source_text": "dur",
                    "target_text": "is",
                }
            ]
        )
        prompt = build_prompt(
            examples,
            "qanddur",
            prompt_mode="dipmt_plus",
            example_lexicon_entries=[
                [
                    {
                        "query": "dur",
                        "source": "dur",
                        "translations": ["is"],
                        "match": "exact",
                    }
                ]
            ],
            input_lexicon_entries=[
                {
                    "query": "qanddur",
                    "source": "qand",
                    "translations": ["sugar"],
                    "match": "fuzzy",
                }
            ],
        )

        self.assertIn("Vocabulary hints:", prompt)
        self.assertIn('"dur" means "is"', prompt)
        self.assertIn('"qanddur" may contain dictionary word "qand"', prompt)
        self.assertTrue(prompt.endswith("English:"))

    def test_zero_shot_prompt_has_no_examples(self):
        examples = pd.DataFrame(columns=["source_text", "target_text"])

        prompt = build_prompt(examples, "tuz aççıqdur")

        self.assertIn("Examples:\nNow translate:", prompt)
        self.assertNotIn("English: salt is bitter", prompt)
        self.assertTrue(prompt.endswith("English:"))

    def test_load_static_examples_filters_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "static.csv"
            pd.DataFrame(
                [
                    {
                        "source_text": "a",
                        "target_text": "A",
                        "source_lang": "chg",
                        "target_lang": "en",
                        "type": "phrase",
                    },
                    {
                        "source_text": "b",
                        "target_text": "B",
                        "source_lang": "chg",
                        "target_lang": "ru",
                        "type": "phrase",
                    },
                    {
                        "source_text": "c",
                        "target_text": "C",
                        "source_lang": "chg",
                        "target_lang": "en",
                        "type": "word",
                    },
                ]
            ).to_csv(path, index=False)

            static_examples = load_static_examples(
                path=path,
                target_lang="en",
                source_lang="chg",
                only_sentences=True,
            )

        self.assertEqual(static_examples["source_text"].tolist(), ["a"])

    def test_lexicon_uses_word_rows_and_fuzzy_matching(self):
        df = pd.DataFrame(
            [
                {
                    "source_text": "qand",
                    "target_text": "sugar",
                    "source_lang": "chg",
                    "target_lang": "en",
                    "type": "word",
                },
                {
                    "source_text": "qand şı̇̄rı̇̄ndür",
                    "target_text": "sugar is sweet",
                    "source_lang": "chg",
                    "target_lang": "en",
                    "type": "phrase",
                },
            ]
        )
        lexicon = Lexicon.from_dataframe(df, source_lang="chg", target_lang="en")

        entries = lexicon.lookup("qanddur")

        self.assertEqual(entries[0]["source"], "qand")
        self.assertEqual(entries[0]["translations"], ["sugar"])
        self.assertEqual(entries[0]["match"], "fuzzy")

    def test_lexicon_maximum_matching_splits_compound_tokens(self):
        df = pd.DataFrame(
            [
                {
                    "source_text": "tuz",
                    "target_text": "salt",
                    "source_lang": "chg",
                    "target_lang": "en",
                    "type": "word",
                },
                {
                    "source_text": "dur",
                    "target_text": "is",
                    "source_lang": "chg",
                    "target_lang": "en",
                    "type": "word",
                },
            ]
        )
        lexicon = Lexicon.from_dataframe(
            df,
            source_lang="chg",
            target_lang="en",
            fuzzy_strategy="max_matching",
        )

        entries = lexicon.lookup("tuzdur")

        self.assertEqual([entry["source"] for entry in entries], ["tuz", "dur"])

    def test_parse_bli_lexicon_filters_by_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bli.tsv"
            path.write_text("süt\tmilk\t0.8\nara\tbetween\t0.2\n", encoding="utf-8")

            rows = parse_bli_lexicon(path, threshold=0.6)

        self.assertEqual(rows, [
            {
                "source_text": "süt",
                "target_text": "milk",
                "score": 0.8,
            }
        ])

    def test_load_dictionary_df_supports_wide_word_file(self):
        df = pd.DataFrame(
            [
                {
                    "source_text": "dur",
                    "English (eng_Latn)": "is",
                    "type": "word",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "words.csv"
            df.to_csv(path, index=False)
            loaded = load_dictionary_df(path, source_lang="chg", target_lang="en")

        self.assertEqual(loaded.iloc[0]["source_text"], "dur")
        self.assertEqual(loaded.iloc[0]["target_text"], "is")
        self.assertEqual(loaded.iloc[0]["target_lang"], "en")

    def test_data_fingerprint_is_stable_and_data_sensitive(self):
        df = pd.DataFrame(
            [
                {
                    "source_text": "a",
                    "target_text": "b",
                    "target_lang": "en",
                    "type": "word",
                }
            ]
        )
        changed_df = df.copy()
        changed_df.loc[0, "target_text"] = "c"

        self.assertEqual(
            make_data_fingerprint(df),
            make_data_fingerprint(df.copy()),
        )
        self.assertNotEqual(
            make_data_fingerprint(df),
            make_data_fingerprint(changed_df),
        )

    def test_output_paths_include_manifest_and_status(self):
        paths = make_output_paths(
            output_dir=Path("outputs"),
            args=Args(),
            model="openai/gpt",
        )

        self.assertIn("manifest", paths)
        self.assertIn("status", paths)
        self.assertEqual(
            paths["base_name"],
            "en_random-similarity_k1-3_intfloat_multilingual-e5-base_openai_gpt",
        )

    def test_random_examples_are_deterministic_per_test_id(self):
        train_df = pd.DataFrame({"source_text": list("abcdef")})

        first = retrieve_random_examples(train_df, k=3, seed=42, test_id=7)
        second = retrieve_random_examples(train_df, k=3, seed=42, test_id=7)

        self.assertEqual(first.index.tolist(), second.index.tolist())
        self.assertEqual(len(first), 3)

    def test_sample_fraction_is_deterministic_and_stratified_by_type(self):
        df = pd.DataFrame(
            {
                "source_text": [f"x{i}" for i in range(20)],
                "type": ["word"] * 10 + ["phrase"] * 10,
            }
        )

        first = sample_df_fraction(df, fraction=0.2, seed=42)
        second = sample_df_fraction(df, fraction=0.2, seed=42)

        self.assertEqual(first["source_text"].tolist(), second["source_text"].tolist())
        self.assertEqual(first["type"].value_counts().to_dict(), {
            "word": 2,
            "phrase": 2,
        })

    def test_graph_common_retrieves_structurally_related_examples(self):
        train_df = pd.DataFrame(
            [
                {
                    "source_text": "tuz aççıqdur",
                    "target_text": "salt is bitter",
                    "type": "phrase",
                    "length": 2,
                },
                {
                    "source_text": "qand aqdur",
                    "target_text": "sugar is white",
                    "type": "phrase",
                    "length": 2,
                },
                {
                    "source_text": "kitab",
                    "target_text": "book",
                    "type": "word",
                    "length": 1,
                },
            ]
        )
        retriever = GraphICLRetriever(train_df)

        result = retriever.retrieve(
            query="muz qattıqdur",
            strategy="graph_common",
            k=2,
            query_metadata={
                "type": "phrase",
                "length": 2,
            },
        )

        self.assertEqual(result["type"].tolist(), ["phrase", "phrase"])

    def test_graph_ppr_bm25_dense_strategy_combines_scores(self):
        train_df = pd.DataFrame(
            [
                {
                    "source_text": "tuz aççıqdur",
                    "target_text": "salt is bitter",
                    "type": "phrase",
                    "length": 2,
                },
                {
                    "source_text": "kitab",
                    "target_text": "book",
                    "type": "word",
                    "length": 1,
                },
            ]
        )
        retriever = GraphICLRetriever(train_df)
        retriever._dense_scores = lambda query: np.array([0.9, 0.1])

        result = retriever.retrieve(
            query="tuz aççıqdur",
            strategy="graph_ppr_bm25_dense",
            k=1,
            query_metadata={
                "type": "phrase",
                "length": 2,
            },
        )

        self.assertEqual(result.iloc[0]["source_text"], "tuz aççıqdur")

    def test_call_openai_retries_and_strips_prediction(self):
        completions = FlakyCompletions(failures_before_success=2)
        client = FakeClient(completions)
        stdout = io.StringIO()

        with (
            patch("run_baseline.time.sleep") as sleep,
            contextlib.redirect_stdout(stdout),
        ):
            prediction, usage, info = call_openai(
                client=client,
                request_payload={"model": "x", "messages": []},
                max_retries=3,
                retry_delay=0.1,
            )

        self.assertEqual(prediction, "translated text")
        self.assertEqual(usage["total_tokens"], 5)
        self.assertEqual(info["attempts"], 3)
        self.assertEqual(completions.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_call_openai_retries_empty_prediction_with_more_tokens(self):
        completions = EmptyThenSuccessCompletions()
        client = FakeClient(completions)

        with patch("run_baseline.time.sleep") as sleep:
            prediction, usage, info = call_openai(
                client=client,
                request_payload={"model": "x", "messages": [], "max_tokens": 10},
                max_retries=0,
                retry_delay=0.1,
                empty_retries=1,
                empty_retry_max_tokens=40,
            )

        self.assertEqual(prediction, "translated text")
        self.assertEqual(usage["total_tokens"], 5)
        self.assertEqual(info["empty_retries"], 1)
        self.assertEqual(completions.max_tokens_seen, [10, 40])
        self.assertEqual(sleep.call_count, 1)

    def test_call_openai_switches_api_key_pool(self):
        failing = FlakyCompletions(failures_before_success=99)
        succeeding = FlakyCompletions(failures_before_success=0)
        client = OpenAIClientPool(
            [
                FakeClient(failing),
                FakeClient(succeeding),
            ]
        )

        prediction, usage, info = call_openai(
            client=client,
            request_payload={"model": "x", "messages": []},
            max_retries=0,
        )

        self.assertEqual(prediction, "translated text")
        self.assertEqual(usage["total_tokens"], 5)
        self.assertEqual(info["api_key_index"], 2)
        self.assertEqual(info["api_key_count"], 2)
        self.assertEqual(failing.calls, 1)
        self.assertEqual(succeeding.calls, 1)

    def test_normalize_schema_converts_new_format_to_canonical(self):
        raw = pd.DataFrame(
            [
                {
                    "src_text": "اولار",
                    "tgt_text": "they",
                    "src_lang": "chg_Arab",
                    "tgt_lang": "eng_Latn",
                },
                {
                    "src_text": "اولار بيز آط",
                    "tgt_text": "we have a horse",
                    "src_lang": "chg_Arab",
                    "tgt_lang": "kaz_Cyrl",
                },
            ]
        )

        result = normalize_schema(raw)

        # Columns renamed to canonical names
        self.assertIn("source_text", result.columns)
        self.assertIn("target_text", result.columns)
        self.assertIn("source_lang", result.columns)
        self.assertIn("target_lang", result.columns)
        self.assertNotIn("src_text", result.columns)
        self.assertNotIn("tgt_text", result.columns)

        # Lang codes normalised
        self.assertEqual(result["source_lang"].tolist(), ["chg", "chg"])
        self.assertEqual(result["target_lang"].tolist(), ["en", "kk"])

        # type derived from token count: 1 token -> word, multi-token -> phrase
        self.assertIn("type", result.columns)
        self.assertEqual(result["type"].tolist(), ["word", "phrase"])

    def test_normalize_schema_is_idempotent_on_canonical_frame(self):
        canonical = pd.DataFrame(
            [
                {
                    "source_text": "dur",
                    "target_text": "is",
                    "source_lang": "chg",
                    "target_lang": "en",
                    "type": "word",
                }
            ]
        )

        result = normalize_schema(canonical)

        self.assertEqual(result.columns.tolist(), canonical.columns.tolist())
        self.assertEqual(result["source_lang"].tolist(), ["chg"])
        self.assertEqual(result["target_lang"].tolist(), ["en"])
        self.assertEqual(result["type"].tolist(), ["word"])


if __name__ == "__main__":
    unittest.main()
