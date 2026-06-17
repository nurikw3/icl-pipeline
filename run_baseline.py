import argparse
import hashlib
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from bli import parse_bli_lexicon, run_giza_py_lexicon
from experiment_config import (
    DEFAULT_EMBEDDING_MODEL,
    GRAPH_RETRIEVAL_STRATEGIES,
    PROMPT_MODES,
    RETRIEVAL_BACKENDS,
    RETRIEVAL_STRATEGIES,
)
from lexicon import Lexicon
from prompts import build_prompt
from retrieval import (
    BM25ICLRetriever,
    GraphICLRetriever,
    ICLRetriever,
    tokenize_for_graph,
)

# ---------------------------------------------------------------------------
# Dataset schema normalisation
# ---------------------------------------------------------------------------

# Explicit map from new-style lang codes to canonical internal codes.
_LANG_CODE_MAP: dict[str, str] = {
    "chg_Arab": "chg",
    "eng_Latn": "en",
    "kaz_Cyrl": "kk",
    "rus_Cyrl": "ru",
    "uzb_Latn": "uz",
    "uzb_Cyrl": "uz",
    "tur_Latn": "tr",
}

# Fallback: ISO-639-3 prefix (before "_") → canonical code.
_ISO3_TO_CANONICAL: dict[str, str] = {
    "eng": "en",
    "kaz": "kk",
    "rus": "ru",
    "uzb": "uz",
    "tur": "tr",
    "chg": "chg",
}


def _normalize_lang_code(code: str) -> str:
    """Map a raw lang code to the canonical internal code."""
    if code in _LANG_CODE_MAP:
        return _LANG_CODE_MAP[code]
    if "_" in code:
        prefix = code.split("_", 1)[0]
        return _ISO3_TO_CANONICAL.get(prefix, code)
    return code


def _derive_type(source_text: str) -> str:
    """Derive a row ``type`` from the source text token count.

    A single-token source is a dictionary ``word``; anything longer is a
    ``phrase``. Used only when the dataset has no explicit ``type`` column.
    """
    return "word" if len(tokenize_for_graph(source_text)) == 1 else "phrase"


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw CSV DataFrame to the canonical internal schema.

    - Renames src_text/tgt_text/src_lang/tgt_lang → source_text/target_text/
      source_lang/target_lang (idempotent when canonical names already present).
    - Normalises lang codes in source_lang / target_lang columns.
    - Derives a ``type`` column when absent: single-token sources become
      ``"word"``, multi-token sources become ``"phrase"``.
    """
    df = df.copy()

    column_renames = {
        "src_text": "source_text",
        "tgt_text": "target_text",
        "src_lang": "source_lang",
        "tgt_lang": "target_lang",
    }
    rename_map = {k: v for k, v in column_renames.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ("source_lang", "target_lang"):
        if col in df.columns:
            df[col] = df[col].astype(str).map(_normalize_lang_code)

    if "type" not in df.columns:
        if "source_text" in df.columns:
            df["type"] = df["source_text"].astype(str).map(_derive_type)
        else:
            df["type"] = "phrase"

    return df


def resolve_split_path(data_dir: Path, split: str) -> Path:
    """
    Return the path for *split* (``"train"`` or ``"test"``), preferring
    ``<split>_icl.csv`` and falling back to ``<split>.csv``.

    Raises :class:`FileNotFoundError` (listing both candidates) if neither
    file exists.
    """
    primary = data_dir / f"{split}_icl.csv"
    fallback = data_dir / f"{split}.csv"

    if primary.exists():
        return primary
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find data file for split '{split}'. "
        f"Tried: {primary}, {fallback}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")

    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--source_lang", type=str, default="chg")

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "similarity", "diversity"],
        choices=RETRIEVAL_STRATEGIES,
    )

    parser.add_argument(
        "--k_list",
        nargs="+",
        type=int,
        default=[8],
        help="Example: --k_list 1 3 5 8 10 15",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="If omitted, OPENAI_MODEL from .env is used.",
    )

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=700)
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Pass enable_thinking=false in extra_body (for Qwen3 and similar models).",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Dense retriever model. Try BAAI/bge-m3 for BGE.",
    )

    parser.add_argument(
        "--retrieval_backend",
        type=str,
        default="dense",
        choices=RETRIEVAL_BACKENDS,
        help="dense uses sentence embeddings; bm25 uses lexical retrieval.",
    )
    parser.add_argument(
        "--static_examples_path",
        type=str,
        default=None,
        help=(
            "Optional CSV with fixed in-context examples. Use with "
            "--strategies static to reuse the same examples for every test row."
        ),
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="standard",
        choices=PROMPT_MODES,
        help="standard uses plain ICL; dipmt_plus adds dictionary hints.",
    )
    parser.add_argument(
        "--lexicon_top_n",
        type=int,
        default=2,
        help="Number of dictionary translations to show per source word.",
    )
    parser.add_argument(
        "--lexicon_max_entries",
        type=int,
        default=80,
        help="Maximum vocabulary hint rows inserted per prompt section.",
    )
    parser.add_argument(
        "--dictionary_path",
        type=str,
        default=None,
        help=(
            "Optional CSV dictionary. Supports this project's long schema or "
            "the wide chagatai_words_only.csv format."
        ),
    )
    parser.add_argument(
        "--disable_lexicon_fuzzy",
        action="store_false",
        dest="lexicon_use_fuzzy",
        help="Disable maximum-substring matching for out-of-dictionary tokens.",
    )
    parser.set_defaults(lexicon_use_fuzzy=True)
    parser.add_argument(
        "--fuzzy_strategy",
        type=str,
        default="max_matching",
        choices=["max_matching", "substring"],
        help="Fuzzy dictionary lookup strategy. The paper uses max matching.",
    )
    parser.add_argument(
        "--enable_bli",
        action="store_true",
        help="Expand the dictionary with bilingual lexicon induction.",
    )
    parser.add_argument(
        "--bli_threshold",
        type=float,
        default=0.6,
        help="Minimum BLI lexicon score. The paper uses 0.6.",
    )
    parser.add_argument(
        "--bli_lexicon_path",
        type=str,
        default=None,
        help="Optional precomputed BLI lexicon TSV from GIZA++/giza-py.",
    )
    parser.add_argument(
        "--giza_py_path",
        type=str,
        default=None,
        help="Path to sillsdev/giza-py giza.py for generating BLI lexicon.",
    )

    parser.add_argument("--candidate_n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=None,
        help=(
            "Deterministically sample this fraction of filtered train/test. "
            "Use 0.1 for a quick 10 percent smoke run."
        ),
    )

    parser.add_argument(
        "--only_sentences",
        action="store_true",
        help="Use only phrase/sentence examples.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name for output files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run_name by skipping completed prediction rows.",
    )

    parser.add_argument(
        "--print_prompts",
        action="store_true",
        help="Print first N prompts to terminal/log.",
    )

    parser.add_argument(
        "--print_prompt_limit",
        type=int,
        default=3,
        help="How many prompts to print when --print_prompts is enabled.",
    )

    parser.add_argument(
        "--save_prompts",
        action="store_true",
        default=True,
        help="Save prompts and request payloads to JSONL files.",
    )
    parser.add_argument(
        "--disable_langfuse",
        action="store_false",
        dest="langfuse_enabled",
        help="Disable Langfuse tracing even when Langfuse credentials are configured.",
    )
    parser.set_defaults(langfuse_enabled=True)

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between API calls.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent LLM workers. Set >1 to parallelise API calls. "
            "Match LM Studio's n_parallel setting for best results."
        ),
    )

    parser.add_argument(
        "--api_retries",
        type=int,
        default=3,
        help="How many times to retry a failed API call.",
    )

    parser.add_argument(
        "--api_retry_delay",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds. Retries use exponential backoff.",
    )

    parser.add_argument(
        "--empty_retries",
        type=int,
        default=2,
        help="Retry LLM calls that return an empty prediction.",
    )

    parser.add_argument(
        "--max_tokens_cap",
        type=int,
        default=1500,
        help="Upper limit for adaptive max_tokens and empty-output retries.",
    )

    parser.add_argument(
        "--chars_per_output_token",
        type=float,
        default=3.0,
        help="Used to estimate max_tokens for long source texts.",
    )

    parser.add_argument(
        "--disable_adaptive_max_tokens",
        action="store_false",
        dest="adaptive_max_tokens",
        help="Disable automatic max_tokens increase for long inputs.",
    )
    parser.set_defaults(adaptive_max_tokens=True)

    return parser.parse_args()


def validate_args(args):
    if not args.strategies:
        raise ValueError("At least one strategy is required.")

    bad_strategies = sorted(set(args.strategies) - set(RETRIEVAL_STRATEGIES))
    if bad_strategies:
        raise ValueError(f"Unknown strategies: {', '.join(bad_strategies)}")

    if not args.k_list:
        raise ValueError("At least one k value is required.")

    if any(k < 0 for k in args.k_list):
        raise ValueError("--k_list values must be non-negative integers.")

    if 0 in args.k_list and set(args.strategies) != {"zero"}:
        raise ValueError("--k_list 0 is only supported with --strategies zero.")

    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max_examples must be positive when provided.")

    if args.retrieval_backend not in RETRIEVAL_BACKENDS:
        raise ValueError(f"Unknown retrieval backend: {args.retrieval_backend}")

    if "static" in args.strategies and not args.static_examples_path:
        raise ValueError("--strategies static requires --static_examples_path.")

    if args.static_examples_path and set(args.strategies) != {"static"}:
        raise ValueError("--static_examples_path is only supported with static.")

    if args.prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {args.prompt_mode}")

    if args.lexicon_top_n < 1:
        raise ValueError("--lexicon_top_n must be positive.")

    if args.lexicon_max_entries < 1:
        raise ValueError("--lexicon_max_entries must be positive.")

    if args.bli_threshold < 0:
        raise ValueError("--bli_threshold cannot be negative.")

    if args.enable_bli and args.prompt_mode != "dipmt_plus":
        raise ValueError("--enable_bli requires --prompt_mode dipmt_plus.")

    selected_graph_strategies = set(args.strategies) & set(GRAPH_RETRIEVAL_STRATEGIES)
    if selected_graph_strategies and args.retrieval_backend != "graph":
        raise ValueError(
            "Graph strategies require --retrieval_backend graph: "
            f"{', '.join(sorted(selected_graph_strategies))}"
        )

    if args.retrieval_backend == "graph":
        unsupported = (
            set(args.strategies)
            - set(GRAPH_RETRIEVAL_STRATEGIES)
            - {"random", "zero", "static"}
        )
        if unsupported:
            raise ValueError(
                "Graph backend supports random plus graph strategies only: "
                f"{', '.join(sorted(unsupported))}"
            )

    if args.max_tokens < 1:
        raise ValueError("--max_tokens must be positive.")

    if args.max_tokens_cap < args.max_tokens:
        raise ValueError(
            "--max_tokens_cap must be greater than or equal to --max_tokens."
        )

    if args.empty_retries < 0:
        raise ValueError("--empty_retries cannot be negative.")

    if args.api_retries < 0:
        raise ValueError("--api_retries cannot be negative.")

    if args.candidate_n < 1:
        raise ValueError("--candidate_n must be positive.")

    if args.sample_fraction is not None and not (0 < args.sample_fraction <= 1):
        raise ValueError("--sample_fraction must be in the interval (0, 1].")


def load_api_settings(args):
    load_dotenv()
    ensure_langfuse_env_aliases()

    api_keys = load_openai_api_keys()
    base_url = os.getenv("OPENAI_BASE_URL") or None
    env_model = os.getenv("OPENAI_MODEL")

    if not api_keys:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_API_KEYS is missing. Add it to your .env file."
        )

    if args.model is None:
        if not env_model:
            raise ValueError("OPENAI_MODEL is missing in .env or pass --model.")
        args.model = env_model

    return api_keys, base_url, args.model


def load_openai_api_keys() -> list[str]:
    raw_keys = []
    multi_key_value = os.getenv("OPENAI_API_KEYS", "")
    if multi_key_value:
        raw_keys.extend(multi_key_value.replace("\n", ",").split(","))
    else:
        single_key = os.getenv("OPENAI_API_KEY")
        if single_key:
            raw_keys.append(single_key)

    keys = []
    seen = set()
    for raw_key in raw_keys:
        key = raw_key.strip().strip('"').strip("'")
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def ensure_langfuse_env_aliases():
    base_url = os.getenv("LANGFUSE_BASE_URL")
    if base_url and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url


def has_langfuse_credentials() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


class OpenAIClientPool:
    def __init__(self, clients: list[Any]):
        if not clients:
            raise ValueError("At least one OpenAI client is required.")
        self.clients = clients
        self.active_index = 0

    @property
    def active_key_index(self) -> int:
        return self.active_index

    @property
    def key_count(self) -> int:
        return len(self.clients)

    def create_chat_completion(self, **request_payload):
        last_error = None
        start_index = self.active_index

        for offset in range(len(self.clients)):
            index = (start_index + offset) % len(self.clients)
            client = self.clients[index]

            try:
                response = client.chat.completions.create(**request_payload)
                if index != self.active_index:
                    print(f"LLM API switched to key {index + 1}/{len(self.clients)}.")
                self.active_index = index
                return response
            except Exception as exc:
                last_error = exc
                if len(self.clients) > 1 and offset < len(self.clients) - 1:
                    next_index = (index + 1) % len(self.clients)
                    print(
                        f"LLM API key {index + 1}/{len(self.clients)} failed: {exc}. "
                        f"Trying key {next_index + 1}/{len(self.clients)}."
                    )

        raise last_error


def build_openai_client(
    api_keys: list[str],
    base_url: str | None,
    langfuse_enabled: bool,
) -> tuple[Any, Any | None]:
    if langfuse_enabled and has_langfuse_credentials():
        from langfuse import get_client
        from langfuse.openai import OpenAI

        clients = [
            OpenAI(
                **{
                    "api_key": api_key,
                    **({"base_url": base_url} if base_url else {}),
                }
            )
            for api_key in api_keys
        ]
        return OpenAIClientPool(clients), get_client()

    from openai import OpenAI

    if langfuse_enabled:
        print("Langfuse disabled: LANGFUSE_PUBLIC_KEY/SECRET_KEY are not configured.")
    clients = [
        OpenAI(
            **{
                "api_key": api_key,
                **({"base_url": base_url} if base_url else {}),
            }
        )
        for api_key in api_keys
    ]
    return OpenAIClientPool(clients), None


def make_safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def make_base_name(args, model: str) -> str:
    if args.run_name:
        return args.run_name

    strategy_name = "-".join(args.strategies)
    k_name = "-".join(str(k) for k in args.k_list)
    model_name = make_safe_name(model)
    prompt_mode = getattr(args, "prompt_mode", "standard")
    prompt_prefix = "" if prompt_mode == "standard" else f"{prompt_mode}_"
    dictionary_path = getattr(args, "dictionary_path", None)
    dictionary_name = ""
    if prompt_mode != "standard" and dictionary_path:
        dictionary_name = f"dict_{make_safe_name(Path(dictionary_path).stem)}_"
    bli_name = "bli_" if getattr(args, "enable_bli", False) else ""
    retrieval_backend = getattr(args, "retrieval_backend", "dense")
    retriever_name = retrieval_backend
    static_examples_path = getattr(args, "static_examples_path", None)

    if retrieval_backend == "dense":
        retriever_name = make_safe_name(
            getattr(args, "embedding_model", "intfloat/multilingual-e5-base")
        )

    if static_examples_path:
        retriever_name = f"static_{make_safe_name(Path(static_examples_path).stem)}"

    return (
        f"{args.target_lang}_"
        f"{prompt_prefix}"
        f"{dictionary_name}"
        f"{bli_name}"
        f"{strategy_name}_"
        f"k{k_name}_"
        f"{retriever_name}_"
        f"{model_name}"
    )


def make_output_paths(output_dir: Path, args, model: str):
    base_name = make_base_name(args, model)

    return {
        "base_name": base_name,
        "predictions": output_dir / f"predictions_{base_name}.csv",
        "prompts": output_dir / f"prompts_{base_name}.jsonl",
        "requests": output_dir / f"requests_{base_name}.jsonl",
        "retrieved": output_dir / f"retrieved_examples_{base_name}.jsonl",
        "manifest": output_dir / f"manifest_{base_name}.json",
        "status": output_dir / f"status_{base_name}.json",
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def make_data_fingerprint(df: pd.DataFrame) -> str:
    columns = ["source_text", "target_text", "target_lang", "type"]
    digest = hashlib.sha256()
    digest.update(str(len(df)).encode("utf-8"))

    for row in df[columns].itertuples(index=False, name=None):
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")

    return digest.hexdigest()[:16]


def retrieve_random_examples(
    train_df: pd.DataFrame,
    k: int,
    seed: int,
    test_id: int,
) -> pd.DataFrame:
    rng = random.Random(seed + int(test_id))
    indices = rng.sample(range(len(train_df)), k=min(k, len(train_df)))
    return train_df.iloc[indices].copy()


def compute_request_max_tokens(
    source_text: str,
    base_max_tokens: int,
    max_tokens_cap: int,
    chars_per_output_token: float,
    adaptive_max_tokens: bool,
) -> int:
    base_max_tokens = max(1, int(base_max_tokens))
    max_tokens_cap = max(base_max_tokens, int(max_tokens_cap))

    if not adaptive_max_tokens:
        return min(base_max_tokens, max_tokens_cap)

    chars_per_output_token = max(0.1, float(chars_per_output_token))
    estimated_tokens = int(len(str(source_text)) / chars_per_output_token) + 1

    return min(max(base_max_tokens, estimated_tokens), max_tokens_cap)


WIDE_DICTIONARY_TARGET_COLUMNS = {
    "en": "English (eng_Latn)",
    "kk": "Kazakh (kaz_Cyrl)",
    "ru": "Russian (rus_Cyrl)",
    "uz": "Uzbek (uzn_Latn)",
    "tr": "Turkish (tur_Latn)",
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "source_text",
        "target_text",
        "source_lang",
        "target_lang",
        "type",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()

    for col in required_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[df["source_text"] != ""]
    df = df[df["target_text"] != ""]
    df = df[df["target_lang"] != ""]

    return df.reset_index(drop=True)


def resolve_optional_data_path(path_text: str | None, data_dir: Path) -> Path | None:
    if not path_text:
        return None

    path = Path(path_text)
    if path.exists():
        return path

    data_path = data_dir / path_text
    if data_path.exists():
        return data_path

    raise FileNotFoundError(f"Dictionary file not found: {path_text}")


def resolve_optional_path(path_text: str | None, data_dir: Path) -> Path | None:
    if not path_text:
        return None

    path = Path(path_text)
    if path.exists():
        return path

    data_path = data_dir / path_text
    if data_path.exists():
        return data_path

    raise FileNotFoundError(f"File not found: {path_text}")


def resolve_giza_py_path(path_text: str | None) -> Path | None:
    if path_text:
        path = Path(path_text)
        if path.exists():
            return path
        raise FileNotFoundError(f"giza.py file not found: {path_text}")

    found = shutil.which("giza.py")
    return Path(found) if found else None


def load_dictionary_df(
    path: Path,
    source_lang: str,
    target_lang: str,
) -> pd.DataFrame:
    df = normalize_schema(pd.read_csv(path))
    long_schema = {"source_text", "target_text", "source_lang", "target_lang", "type"}

    if long_schema.issubset(df.columns):
        return clean_df(df)

    target_column = WIDE_DICTIONARY_TARGET_COLUMNS.get(target_lang)
    if not target_column or target_column not in df.columns:
        raise ValueError(
            "Dictionary file is not in long schema and has no supported "
            f"wide column for target_lang={target_lang}."
        )

    source_column = "source_text"
    if source_column not in df.columns:
        source_column = "Chagatai Latin (chg_Latn)"

    if source_column not in df.columns:
        raise ValueError("Wide dictionary file has no source text column.")

    dictionary_df = pd.DataFrame(
        {
            "source_text": df[source_column],
            "target_text": df[target_column],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "type": "word",
        }
    )

    return clean_df(dictionary_df)


def filter_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_lang: str,
    only_sentences: bool,
):
    train_df = train_df[train_df["target_lang"] == target_lang].reset_index(drop=True)
    test_df = test_df[test_df["target_lang"] == target_lang].reset_index(drop=True)

    if only_sentences:
        allowed_types = ["phrase", "sentence"]

        train_df = train_df[train_df["type"].isin(allowed_types)].reset_index(drop=True)
        test_df = test_df[test_df["type"].isin(allowed_types)].reset_index(drop=True)

    return train_df, test_df


def load_static_examples(
    path: Path,
    target_lang: str,
    source_lang: str,
    only_sentences: bool,
) -> pd.DataFrame:
    static_df = clean_df(pd.read_csv(path))
    static_df = static_df[
        (static_df["target_lang"] == target_lang)
        & (static_df["source_lang"] == source_lang)
    ].reset_index(drop=True)

    if only_sentences:
        static_df = static_df[
            static_df["type"].isin(["phrase", "sentence"])
        ].reset_index(drop=True)

    if len(static_df) == 0:
        raise ValueError(
            "Static examples file has no rows after filtering. "
            "Check source_lang, target_lang, and --only_sentences."
        )

    return static_df


def sample_df_fraction(
    df: pd.DataFrame,
    fraction: float | None,
    seed: int,
) -> pd.DataFrame:
    if fraction is None or fraction >= 1:
        return df.reset_index(drop=True)

    sampled_parts = []

    for _, group_df in df.groupby("type", sort=False):
        n = max(1, int(round(len(group_df) * fraction)))
        n = min(n, len(group_df))
        sampled_parts.append(group_df.sample(n=n, random_state=seed))

    if not sampled_parts:
        return df.head(0).copy().reset_index(drop=True)

    sampled_df = pd.concat(sampled_parts).sort_index()
    return sampled_df.reset_index(drop=True)


def sample_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fraction: float | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        sample_df_fraction(train_df, fraction=fraction, seed=seed),
        sample_df_fraction(test_df, fraction=fraction, seed=seed + 1),
    )


def build_request_payload(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool = False,
):
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    if disable_thinking:
        payload["extra_body"] = {"enable_thinking": False}
    return payload


def extract_response_info(response, attempts: int, empty_retries: int) -> dict:
    finish_reason = None

    try:
        finish_reason = response.choices[0].finish_reason
    except Exception:
        finish_reason = None

    return {
        "finish_reason": finish_reason,
        "attempts": attempts,
        "empty_retries": empty_retries,
    }


def call_openai(
    client: Any,
    request_payload: dict,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    empty_retries: int = 2,
    empty_retry_max_tokens: int | None = None,
):
    max_retries = max(0, int(max_retries))
    retry_delay = max(0.0, float(retry_delay))
    empty_retries = max(0, int(empty_retries))
    request_payload = dict(request_payload)
    last_error = None
    attempts = 0

    for empty_attempt in range(empty_retries + 1):
        response = None

        for attempt in range(max_retries + 1):
            attempts += 1

            try:
                if hasattr(client, "create_chat_completion"):
                    response = client.create_chat_completion(**request_payload)
                else:
                    response = client.chat.completions.create(**request_payload)
                break
            except Exception as e:
                last_error = e
                if attempt >= max_retries:
                    raise

                delay = retry_delay * (2 ** attempt)
                print(
                    f"LLM call failed on attempt {attempt + 1}/"
                    f"{max_retries + 1}: {e}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)

        if response is None:
            raise last_error

        prediction = response.choices[0].message.content

        if prediction is None:
            prediction = ""

        prediction = prediction.strip()

        usage = None
        try:
            usage = response.usage.model_dump()
        except Exception:
            usage = None

        info = extract_response_info(
            response=response,
            attempts=attempts,
            empty_retries=empty_attempt,
        )
        info["max_tokens"] = request_payload.get("max_tokens")
        if hasattr(client, "active_key_index"):
            info["api_key_index"] = client.active_key_index + 1
            info["api_key_count"] = client.key_count

        if prediction != "" or empty_attempt >= empty_retries:
            return prediction, usage, info

        if empty_retry_max_tokens is not None:
            request_payload["max_tokens"] = max(
                int(request_payload.get("max_tokens", 0) or 0),
                int(empty_retry_max_tokens),
            )

        print(
            "LLM returned an empty prediction "
            f"(finish_reason={info['finish_reason']}). "
            f"Retrying with max_tokens={request_payload.get('max_tokens')}."
        )

        if retry_delay > 0:
            time.sleep(retry_delay * (2 ** empty_attempt))

    raise last_error


def save_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def result_key(record: dict) -> tuple[str, int, int]:
    return (
        str(record["strategy"]),
        int(record["k"]),
        int(record["test_index"]),
    )


def load_existing_predictions(
    path: Path,
) -> tuple[list[dict], set[tuple[str, int, int]]]:
    if not path.exists():
        return [], set()

    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    keys = set()

    for record in records:
        try:
            keys.add(result_key(record))
        except Exception:
            continue

    return records, keys


def append_csv_record(path: Path, record: dict):
    header = not path.exists()
    pd.DataFrame([record]).to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def remove_old_output_files(paths):
    for path in paths:
        if path.exists():
            path.unlink()


def print_run_header(
    train_df,
    test_df,
    args,
    model,
    base_url,
    paths,
    data_fingerprint,
):
    print("=" * 100)
    print("ICL BASELINE RUN")
    print("=" * 100)
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Source language:", args.source_lang)
    print("Target language:", args.target_lang)
    print("Strategies:", args.strategies)
    print("k_list:", args.k_list)
    print("Only sentences:", args.only_sentences)
    print("Max examples:", args.max_examples)
    print("Sample fraction:", args.sample_fraction)
    print("Model:", model)
    print("Base URL:", base_url or "(default)")
    print("Temperature:", args.temperature)
    print("Max tokens:", args.max_tokens)
    print("Adaptive max tokens:", args.adaptive_max_tokens)
    print("Max tokens cap:", args.max_tokens_cap)
    print("Empty retries:", args.empty_retries)
    print("Retrieval backend:", args.retrieval_backend)
    print("Static examples path:", args.static_examples_path or "(none)")
    print("Prompt mode:", args.prompt_mode)
    print("Lexicon top N:", args.lexicon_top_n)
    print("Lexicon max entries:", args.lexicon_max_entries)
    print("Lexicon fuzzy matching:", args.lexicon_use_fuzzy)
    print("Fuzzy strategy:", args.fuzzy_strategy)
    print("BLI enabled:", args.enable_bli)
    print("BLI threshold:", args.bli_threshold)
    print("Embedding model:", args.embedding_model)
    print("Candidate N:", args.candidate_n)
    print("Data fingerprint:", data_fingerprint)
    print("Predictions:", paths["predictions"])
    print("Prompts:", paths["prompts"])
    print("Requests:", paths["requests"])
    print("Retrieved examples:", paths["retrieved"])
    print("Manifest:", paths["manifest"])
    print("Status:", paths["status"])
    print("=" * 100)


def write_manifest(
    path: Path,
    args,
    model: str,
    base_url: str | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_fingerprint: str,
    paths: dict,
    lexicon_stats: dict | None = None,
):
    manifest = {
        "created_at": utc_now(),
        "base_name": paths["base_name"],
        "model": model,
        "base_url": base_url,
        "args": vars(args),
        "data": {
            "fingerprint": data_fingerprint,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_target_lang_counts": train_df[
                "target_lang"
            ].value_counts().to_dict(),
            "test_target_lang_counts": test_df["target_lang"].value_counts().to_dict(),
            "train_type_counts": train_df["type"].value_counts().to_dict(),
            "test_type_counts": test_df["type"].value_counts().to_dict(),
        },
        "lexicon": lexicon_stats,
        "paths": {
            key: str(value)
            for key, value in paths.items()
            if key != "base_name"
        },
    }
    write_json(path, manifest)


def write_status(path: Path, **updates):
    payload = {
        "updated_at": utc_now(),
        **updates,
    }
    write_json(path, payload)


def make_langfuse_trace_id(
    langfuse_client,
    session_id: str,
    strategy: str,
    k: int,
    test_idx: int,
) -> str:
    if langfuse_client is None:
        return ""

    seed = f"{session_id}:{strategy}:k{k}:test_idx{test_idx}"
    return langfuse_client.create_trace_id(seed=seed)


def make_langfuse_score_id(trace_id: str, name: str) -> str:
    return hashlib.sha256(f"{trace_id}:{name}".encode()).hexdigest()[:32]


def add_langfuse_context_to_payload(
    request_payload: dict,
    *,
    trace_id: str,
    session_id: str,
    args,
    paths: dict,
    strategy: str,
    k: int,
    test_idx: int,
    row_id: Any,
    source_text: str,
    reference: str,
    retrieved_records: list[dict],
):
    if not trace_id:
        return request_payload

    payload = dict(request_payload)
    payload["name"] = "icl-translation"
    payload["trace_id"] = trace_id
    payload["metadata"] = {
        "langfuse_session_id": session_id,
        "langfuse_tags": [
            "icl",
            f"target:{args.target_lang}",
            f"retrieval:{args.retrieval_backend}",
            f"prompt:{args.prompt_mode}",
        ],
        "run_name": paths["base_name"],
        "strategy": strategy,
        "k": int(k),
        "test_index": int(test_idx),
        "row_id": str(row_id),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "retrieval_backend": args.retrieval_backend,
        "prompt_mode": args.prompt_mode,
        "embedding_model": args.embedding_model,
        "reference": reference,
        "source_preview": source_text[:300],
        "retrieved_count": len(retrieved_records),
        "retrieved_examples": retrieved_records,
    }
    return payload


def score_langfuse_translation(
    langfuse_client,
    *,
    trace_id: str,
    prediction: str,
    reference: str,
    error: str,
):
    if langfuse_client is None or not trace_id:
        return

    normalized_prediction = prediction.strip()
    normalized_reference = reference.strip()
    scores = [
        (
            "translation_exact_match",
            1 if normalized_prediction == normalized_reference else 0,
            "BOOLEAN",
            "Strict normalized prediction/reference match.",
        ),
        (
            "empty_prediction",
            1 if normalized_prediction == "" else 0,
            "BOOLEAN",
            "Prediction is empty after stripping whitespace.",
        ),
        (
            "llm_error",
            1 if error else 0,
            "BOOLEAN",
            error[:500] if error else "LLM call completed without exception.",
        ),
    ]

    for name, value, data_type, comment in scores:
        langfuse_client.create_score(
            name=name,
            value=value,
            trace_id=trace_id,
            data_type=data_type,
            comment=comment,
            score_id=make_langfuse_score_id(trace_id, name),
        )


def main():
    args = parse_args()
    validate_args(args)

    api_keys, base_url, model = load_api_settings(args)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_split_path(data_dir, "train")
    test_path = resolve_split_path(data_dir, "test")

    raw_train_df = clean_df(normalize_schema(pd.read_csv(train_path)))
    raw_test_df = clean_df(normalize_schema(pd.read_csv(test_path)))
    dictionary_path = resolve_optional_data_path(args.dictionary_path, data_dir)
    static_examples_path = resolve_optional_data_path(
        args.static_examples_path,
        data_dir,
    )
    dictionary_df = (
        load_dictionary_df(
            dictionary_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
        if dictionary_path
        else raw_train_df
    )

    train_df, test_df = filter_data(
        train_df=raw_train_df,
        test_df=raw_test_df,
        target_lang=args.target_lang,
        only_sentences=args.only_sentences,
    )

    train_df, test_df = sample_dataframes(
        train_df=train_df,
        test_df=test_df,
        fraction=args.sample_fraction,
        seed=args.seed,
    )

    if args.max_examples is not None:
        test_df = test_df.head(args.max_examples).reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(
            "Train set is empty after filtering. "
            "Check --target_lang and --only_sentences."
        )

    if len(test_df) == 0:
        raise ValueError(
            "Test set is empty after filtering. "
            "Check --target_lang and --only_sentences."
        )

    static_examples_df = None
    if static_examples_path:
        static_examples_df = load_static_examples(
            path=static_examples_path,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            only_sentences=args.only_sentences,
        )

        max_static_k = max(args.k_list)
        if max_static_k > len(static_examples_df):
            raise ValueError(
                f"Static examples file has {len(static_examples_df)} rows, "
                f"but max requested k is {max_static_k}."
            )

    lexicon = None
    lexicon_stats = None
    if args.prompt_mode == "dipmt_plus":
        lexicon = Lexicon.from_dataframe(
            dictionary_df,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            top_n=args.lexicon_top_n,
            max_entries=args.lexicon_max_entries,
            use_fuzzy=args.lexicon_use_fuzzy,
            fuzzy_strategy=args.fuzzy_strategy,
        )
        lexicon_stats = lexicon.stats()
        lexicon_stats["dictionary_path"] = (
            str(dictionary_path) if dictionary_path else str(train_path)
        )
        lexicon_stats["bli_enabled"] = False

    paths = make_output_paths(
        output_dir=output_dir,
        args=args,
        model=model,
    )
    data_fingerprint = make_data_fingerprint(train_df)

    if lexicon is not None and args.enable_bli:
        bli_lexicon_path = resolve_optional_path(args.bli_lexicon_path, data_dir)
        bli_rows = []
        bli_source = None

        if bli_lexicon_path:
            bli_rows = parse_bli_lexicon(
                bli_lexicon_path,
                threshold=args.bli_threshold,
            )
            bli_source = str(bli_lexicon_path)
        else:
            giza_py_path = resolve_giza_py_path(args.giza_py_path)
            if not giza_py_path:
                raise FileNotFoundError(
                    "BLI is enabled, but no GIZA++ lexicon or giza.py runner was "
                    "found. Pass --bli_lexicon_path or --giza_py_path."
                )

            generated_bli_path = (
                output_dir / f"bli_lexicon_{paths['base_name']}_{data_fingerprint}.tsv"
            )
            bli_rows = run_giza_py_lexicon(
                parallel_df=train_df,
                output_path=generated_bli_path,
                giza_py_path=giza_py_path,
                threshold=args.bli_threshold,
            )
            bli_source = str(generated_bli_path)

        lexicon.add_entries(bli_rows)
        lexicon_stats = lexicon.stats()
        lexicon_stats["dictionary_path"] = (
            str(dictionary_path) if dictionary_path else str(train_path)
        )
        lexicon_stats["bli_enabled"] = True
        lexicon_stats["bli_source"] = bli_source
        lexicon_stats["bli_threshold"] = args.bli_threshold
        lexicon_stats["bli_entries_added_or_seen"] = len(bli_rows)

    if not args.resume:
        remove_old_output_files(
            [
                paths["predictions"],
                paths["prompts"],
                paths["requests"],
                paths["retrieved"],
                paths["manifest"],
                paths["status"],
            ]
        )

    write_manifest(
        path=paths["manifest"],
        args=args,
        model=model,
        base_url=base_url,
        train_df=train_df,
        test_df=test_df,
        data_fingerprint=data_fingerprint,
        paths=paths,
        lexicon_stats=lexicon_stats,
    )
    write_status(
        paths["status"],
        status="running",
        total_rows=len(test_df) * len(args.k_list) * len(args.strategies),
        completed_rows=0,
        total_errors=0,
        total_empty_predictions=0,
    )

    print_run_header(
        train_df=train_df,
        test_df=test_df,
        args=args,
        model=model,
        base_url=base_url,
        paths=paths,
        data_fingerprint=data_fingerprint,
    )
    if lexicon_stats:
        print("Lexicon source terms:", lexicon_stats["source_terms"])
        print("Lexicon translation entries:", lexicon_stats["translation_entries"])

    client, langfuse_client = build_openai_client(
        api_keys=api_keys,
        base_url=base_url,
        langfuse_enabled=args.langfuse_enabled,
    )
    langfuse_session_id = paths["base_name"]
    print("OpenAI-compatible API keys configured:", len(api_keys))
    if langfuse_client is not None:
        print("Langfuse tracing: enabled")
        print("Langfuse session:", langfuse_session_id)

    all_results, completed_keys = load_existing_predictions(paths["predictions"])

    total_errors = sum(1 for row in all_results if str(row.get("error", "")).strip())
    total_empty_predictions = sum(
        1 for row in all_results if str(row.get("prediction", "")).strip() == ""
    )
    completed_rows = len(completed_keys)

    if args.resume and completed_rows:
        print(f"Resume enabled: skipping {completed_rows} existing prediction rows.")

    retriever = None

    def get_retriever():
        nonlocal retriever
        if retriever is not None:
            return retriever

        if args.retrieval_backend == "bm25":
            retriever = BM25ICLRetriever(train_df=train_df)
            return retriever

        safe_embedding_name = make_safe_name(args.embedding_model)

        if args.retrieval_backend == "graph":
            cache_name = (
                f"graph_dense_"
                f"{args.target_lang}_"
                f"{safe_embedding_name}_"
                f"{data_fingerprint}_"
                f"{'sentences' if args.only_sentences else 'all'}"
                f".npy"
            )

            retriever = GraphICLRetriever(
                train_df=train_df,
                embedding_model_name=args.embedding_model,
                cache_dir="embeddings",
                cache_name=cache_name,
            )
            return retriever

        cache_name = (
            f"train_"
            f"{args.target_lang}_"
            f"{safe_embedding_name}_"
            f"{data_fingerprint}_"
            f"{'sentences' if args.only_sentences else 'all'}"
            f".npy"
        )

        retriever = ICLRetriever(
            train_df=train_df,
            embedding_model_name=args.embedding_model,
            cache_dir="embeddings",
            cache_name=cache_name,
        )
        return retriever

    for k in args.k_list:
        for strategy in args.strategies:
            print("\n" + "-" * 100)
            print(f"Running strategy={strategy}, k={k}")
            print("-" * 100)

            write_status(
                paths["status"],
                status="running",
                current_strategy=strategy,
                current_k=int(k),
                total_rows=len(test_df) * len(args.k_list) * len(args.strategies),
                completed_rows=completed_rows,
                total_errors=total_errors,
                total_empty_predictions=total_empty_predictions,
            )

            strategy_errors = 0
            strategy_empty_predictions = 0

            # ── Phase 1: retrieval + prompt building (sequential) ──────────
            work_items = []
            prompt_prints = 0
            for test_idx, row in test_df.iterrows():
                source_text = str(row["source_text"]).strip()
                reference = str(row["target_text"]).strip()
                current_key = (str(strategy), int(k), int(test_idx))

                if current_key in completed_keys:
                    continue

                langfuse_trace_id = make_langfuse_trace_id(
                    langfuse_client=langfuse_client,
                    session_id=langfuse_session_id,
                    strategy=str(strategy),
                    k=int(k),
                    test_idx=int(test_idx),
                )

                if strategy == "zero":
                    examples = train_df.head(0).copy()
                elif strategy == "static":
                    examples = static_examples_df.head(k).copy()
                elif strategy == "random":
                    examples = retrieve_random_examples(
                        train_df=train_df,
                        k=k,
                        seed=args.seed,
                        test_id=test_idx,
                    )
                else:
                    retrieve_kwargs = {
                        "query": source_text,
                        "strategy": strategy,
                        "k": k,
                        "seed": args.seed,
                        "test_id": test_idx,
                        "candidate_n": args.candidate_n,
                    }
                    if args.retrieval_backend == "graph":
                        retrieve_kwargs["query_metadata"] = row.to_dict()
                    examples = get_retriever().retrieve(**retrieve_kwargs)

                example_lexicon_entries = None
                input_lexicon_entries = None
                if lexicon is not None:
                    example_lexicon_entries = [
                        lexicon.lookup(str(ex["source_text"]).strip())
                        for _, ex in examples.iterrows()
                    ]
                    input_lexicon_entries = lexicon.lookup(source_text)

                prompt = build_prompt(
                    examples=examples,
                    input_text=source_text,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    prompt_mode=args.prompt_mode,
                    example_lexicon_entries=example_lexicon_entries,
                    input_lexicon_entries=input_lexicon_entries,
                )
                request_max_tokens = compute_request_max_tokens(
                    source_text=source_text,
                    base_max_tokens=args.max_tokens,
                    max_tokens_cap=args.max_tokens_cap,
                    chars_per_output_token=args.chars_per_output_token,
                    adaptive_max_tokens=args.adaptive_max_tokens,
                )
                empty_retry_max_tokens = min(
                    max(args.max_tokens, args.max_tokens_cap),
                    max(request_max_tokens * 2, args.max_tokens),
                )
                request_payload = build_request_payload(
                    prompt=prompt,
                    model=model,
                    temperature=args.temperature,
                    max_tokens=request_max_tokens,
                    disable_thinking=args.disable_thinking,
                )

                if args.print_prompts and prompt_prints < args.print_prompt_limit:
                    print("\n" + "=" * 100)
                    print(
                        f"PROMPT DEBUG | "
                        f"strategy={strategy} | "
                        f"k={k} | "
                        f"test_idx={test_idx}"
                    )
                    print("=" * 100)
                    print(prompt)
                    print("=" * 100 + "\n")
                    prompt_prints += 1

                retrieved_records = examples[
                    ["source_text", "target_text", "target_lang", "type"]
                ].to_dict(orient="records")
                request_payload = add_langfuse_context_to_payload(
                    request_payload,
                    trace_id=langfuse_trace_id,
                    session_id=langfuse_session_id,
                    args=args,
                    paths=paths,
                    strategy=str(strategy),
                    k=int(k),
                    test_idx=int(test_idx),
                    row_id=row.get("id", ""),
                    source_text=source_text,
                    reference=reference,
                    retrieved_records=retrieved_records,
                )

                work_items.append({
                    "test_idx": test_idx,
                    "row": row,
                    "source_text": source_text,
                    "reference": reference,
                    "current_key": current_key,
                    "langfuse_trace_id": langfuse_trace_id,
                    "request_payload": request_payload,
                    "request_max_tokens": request_max_tokens,
                    "empty_retry_max_tokens": empty_retry_max_tokens,
                    "prompt": prompt,
                    "retrieved_records": retrieved_records,
                    "example_lexicon_entries": example_lexicon_entries,
                    "input_lexicon_entries": input_lexicon_entries,
                })

            # ── Phase 2: LLM calls (parallel when workers > 1) ────────────
            def _call_llm(item):
                prediction, error, usage, response_info = "", "", None, {}
                try:
                    prediction, usage, response_info = call_openai(
                        client=client,
                        request_payload=item["request_payload"],
                        max_retries=args.api_retries,
                        retry_delay=args.api_retry_delay,
                        empty_retries=args.empty_retries,
                        empty_retry_max_tokens=item["empty_retry_max_tokens"],
                    )
                    if args.sleep > 0 and args.workers == 1:
                        time.sleep(args.sleep)
                except Exception as e:
                    error = str(e)
                return {**item, "prediction": prediction, "usage": usage,
                        "response_info": response_info, "error": error}

            if args.workers > 1:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    completed_items = list(tqdm(
                        executor.map(_call_llm, work_items),
                        total=len(work_items),
                        desc=f"{strategy}, k={k}",
                    ))
            else:
                completed_items = [
                    _call_llm(item)
                    for item in tqdm(work_items, desc=f"{strategy}, k={k}")
                ]

            # ── Phase 3: write results (sequential) ───────────────────────
            for result in completed_items:
                test_idx = result["test_idx"]
                row = result["row"]
                source_text = result["source_text"]
                reference = result["reference"]
                current_key = result["current_key"]
                langfuse_trace_id = result["langfuse_trace_id"]
                request_payload = result["request_payload"]
                request_max_tokens = result["request_max_tokens"]
                prompt = result["prompt"]
                retrieved_records = result["retrieved_records"]
                example_lexicon_entries = result["example_lexicon_entries"]
                input_lexicon_entries = result["input_lexicon_entries"]
                prediction = result["prediction"]
                usage = result["usage"]
                response_info = result["response_info"]
                error = result["error"]

                if error:
                    strategy_errors += 1
                    total_errors += 1
                    print(f"\nLLM error at test_idx={test_idx}: {error}")

                if prediction.strip() == "":
                    strategy_empty_predictions += 1
                    total_empty_predictions += 1

                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    completion_tokens = usage.get("completion_tokens", 0) or 0
                    total_tokens = usage.get("total_tokens", 0) or 0

                result_record = {
                    "test_index": int(test_idx),
                    "id": row.get("id", ""),
                    "source_text": source_text,
                    "reference": reference,
                    "prediction": prediction,
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "type": row.get("type", ""),
                    "strategy": strategy,
                    "k": int(k),
                    "retrieval_backend": args.retrieval_backend,
                    "prompt_mode": args.prompt_mode,
                    "embedding_model": args.embedding_model,
                    "model": model,
                    "temperature": args.temperature,
                    "max_tokens": request_max_tokens,
                    "response_max_tokens": response_info.get(
                        "max_tokens",
                        request_max_tokens,
                    ),
                    "finish_reason": response_info.get("finish_reason", ""),
                    "llm_attempts": response_info.get("attempts", 0),
                    "empty_retries": response_info.get("empty_retries", 0),
                    "api_key_index": response_info.get("api_key_index", ""),
                    "api_key_count": response_info.get("api_key_count", len(api_keys)),
                    "error": error,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "langfuse_trace_id": langfuse_trace_id,
                    "langfuse_session_id": langfuse_session_id,
                }
                score_langfuse_translation(
                    langfuse_client=langfuse_client,
                    trace_id=langfuse_trace_id,
                    prediction=prediction,
                    reference=reference,
                    error=error,
                )

                all_results.append(result_record)
                completed_keys.add(current_key)

                if args.save_prompts:
                    prompt_record = {
                        "test_index": int(test_idx),
                        "id": str(row.get("id", "")),
                        "source_text": source_text,
                        "reference": reference,
                        "source_lang": args.source_lang,
                        "target_lang": args.target_lang,
                        "type": str(row.get("type", "")),
                        "strategy": strategy,
                        "k": int(k),
                        "retrieval_backend": args.retrieval_backend,
                        "prompt_mode": args.prompt_mode,
                        "embedding_model": args.embedding_model,
                        "model": model,
                        "max_tokens": request_max_tokens,
                        "prompt": prompt,
                        "retrieved_examples": retrieved_records,
                        "input_lexicon_entries": input_lexicon_entries,
                        "example_lexicon_entries": example_lexicon_entries,
                    }

                    request_record = {
                        "test_index": int(test_idx),
                        "id": str(row.get("id", "")),
                        "source_text": source_text,
                        "reference": reference,
                        "source_lang": args.source_lang,
                        "target_lang": args.target_lang,
                        "type": str(row.get("type", "")),
                        "strategy": strategy,
                        "k": int(k),
                        "retrieval_backend": args.retrieval_backend,
                        "prompt_mode": args.prompt_mode,
                        "embedding_model": args.embedding_model,
                        "request_payload": request_payload,
                        "prediction": prediction,
                        "error": error,
                        "usage": usage,
                        "response_info": response_info,
                        "langfuse_trace_id": langfuse_trace_id,
                        "langfuse_session_id": langfuse_session_id,
                    }

                    retrieved_record = {
                        "test_index": int(test_idx),
                        "id": str(row.get("id", "")),
                        "source_text": source_text,
                        "reference": reference,
                        "source_lang": args.source_lang,
                        "target_lang": args.target_lang,
                        "type": str(row.get("type", "")),
                        "strategy": strategy,
                        "k": int(k),
                        "retrieval_backend": args.retrieval_backend,
                        "prompt_mode": args.prompt_mode,
                        "embedding_model": args.embedding_model,
                        "retrieved_examples": retrieved_records,
                        "input_lexicon_entries": input_lexicon_entries,
                    }

                    save_jsonl(paths["prompts"], prompt_record)
                    save_jsonl(paths["requests"], request_record)
                    save_jsonl(paths["retrieved"], retrieved_record)

                append_csv_record(paths["predictions"], result_record)
                completed_rows += 1

                write_status(
                    paths["status"],
                    status="running",
                    current_strategy=strategy,
                    current_k=int(k),
                    current_test_index=int(test_idx),
                    total_rows=len(test_df) * len(args.k_list) * len(args.strategies),
                    completed_rows=completed_rows,
                    total_errors=total_errors,
                    total_empty_predictions=total_empty_predictions,
                )

            print(f"\nFinished strategy={strategy}, k={k}")
            print("Strategy errors:", strategy_errors)
            print("Strategy empty predictions:", strategy_empty_predictions)

    if paths["predictions"].exists():
        results_df = pd.read_csv(paths["predictions"])
    else:
        results_df = pd.DataFrame(all_results)
    write_status(
        paths["status"],
        status="finished",
        total_rows=len(results_df),
        completed_rows=len(results_df),
        total_errors=total_errors,
        total_empty_predictions=total_empty_predictions,
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved predictions:", paths["predictions"])
    print("Saved prompts:", paths["prompts"])
    print("Saved requests:", paths["requests"])
    print("Saved retrieved examples:", paths["retrieved"])
    print("Saved manifest:", paths["manifest"])
    print("Saved status:", paths["status"])
    print("Rows:", len(results_df))
    print("Total errors:", total_errors)
    print("Total empty predictions:", total_empty_predictions)
    if langfuse_client is not None:
        langfuse_client.flush()
        print("Langfuse flushed:", langfuse_session_id)
    print("=" * 100)


if __name__ == "__main__":
    main()
