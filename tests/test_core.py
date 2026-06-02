import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from prompts import build_prompt
from retrieval import GraphICLRetriever
from run_baseline import (
    call_openai,
    make_data_fingerprint,
    make_output_paths,
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
                    "sheet": "Ch 1",
                },
                {
                    "source_text": "qand aqdur",
                    "target_text": "sugar is white",
                    "type": "phrase",
                    "length": 2,
                    "sheet": "Ch 1",
                },
                {
                    "source_text": "kitab",
                    "target_text": "book",
                    "type": "word",
                    "length": 1,
                    "sheet": "Ch 2",
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
                "sheet": "Ch 1",
            },
        )

        self.assertEqual(result["type"].tolist(), ["phrase", "phrase"])

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


if __name__ == "__main__":
    unittest.main()
