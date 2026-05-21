import argparse
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from experiment_config import (
    DEFAULT_EMBEDDING_MODEL,
    RETRIEVAL_BACKENDS,
    RETRIEVAL_STRATEGIES,
)
from prompts import build_prompt
from retrieval import BM25ICLRetriever, ICLRetriever


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

    parser.add_argument("--candidate_n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)

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
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep between API calls.",
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

    if any(k < 1 for k in args.k_list):
        raise ValueError("--k_list values must be positive integers.")

    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max_examples must be positive when provided.")

    if args.retrieval_backend not in RETRIEVAL_BACKENDS:
        raise ValueError(f"Unknown retrieval backend: {args.retrieval_backend}")

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


def load_api_settings(args):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    env_model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    if args.model is None:
        if not env_model:
            raise ValueError("OPENAI_MODEL is missing in .env or pass --model.")
        args.model = env_model

    return api_key, base_url, args.model


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
    retrieval_backend = getattr(args, "retrieval_backend", "dense")
    retriever_name = retrieval_backend

    if retrieval_backend == "dense":
        retriever_name = make_safe_name(
            getattr(args, "embedding_model", "intfloat/multilingual-e5-base")
        )

    return (
        f"{args.target_lang}_"
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


def build_request_payload(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
):
    return {
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
    client: OpenAI,
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
    print("Model:", model)
    print("Base URL:", base_url or "(default)")
    print("Temperature:", args.temperature)
    print("Max tokens:", args.max_tokens)
    print("Adaptive max tokens:", args.adaptive_max_tokens)
    print("Max tokens cap:", args.max_tokens_cap)
    print("Empty retries:", args.empty_retries)
    print("Retrieval backend:", args.retrieval_backend)
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


def main():
    args = parse_args()
    validate_args(args)

    api_key, base_url, model = load_api_settings(args)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = clean_df(train_df)
    test_df = clean_df(test_df)

    train_df, test_df = filter_data(
        train_df=train_df,
        test_df=test_df,
        target_lang=args.target_lang,
        only_sentences=args.only_sentences,
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

    paths = make_output_paths(
        output_dir=output_dir,
        args=args,
        model=model,
    )
    data_fingerprint = make_data_fingerprint(train_df)

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

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

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

            for test_idx, row in tqdm(
                test_df.iterrows(),
                total=len(test_df),
                desc=f"{strategy}, k={k}",
            ):
                source_text = str(row["source_text"]).strip()
                reference = str(row["target_text"]).strip()
                current_key = (str(strategy), int(k), int(test_idx))

                if current_key in completed_keys:
                    continue

                if strategy == "random":
                    examples = retrieve_random_examples(
                        train_df=train_df,
                        k=k,
                        seed=args.seed,
                        test_id=test_idx,
                    )
                else:
                    examples = get_retriever().retrieve(
                        query=source_text,
                        strategy=strategy,
                        k=k,
                        seed=args.seed,
                        test_id=test_idx,
                        candidate_n=args.candidate_n,
                    )

                prompt = build_prompt(
                    examples=examples,
                    input_text=source_text,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
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
                )

                if args.print_prompts and test_idx < args.print_prompt_limit:
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

                retrieved_records = examples[
                    ["source_text", "target_text", "target_lang", "type"]
                ].to_dict(orient="records")

                prediction = ""
                error = ""
                usage = None
                response_info = {}

                try:
                    prediction, usage, response_info = call_openai(
                        client=client,
                        request_payload=request_payload,
                        max_retries=args.api_retries,
                        retry_delay=args.api_retry_delay,
                        empty_retries=args.empty_retries,
                        empty_retry_max_tokens=empty_retry_max_tokens,
                    )
                except Exception as e:
                    error = str(e)
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
                    "error": error,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

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
                        "embedding_model": args.embedding_model,
                        "model": model,
                        "max_tokens": request_max_tokens,
                        "prompt": prompt,
                        "retrieved_examples": retrieved_records,
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
                        "embedding_model": args.embedding_model,
                        "request_payload": request_payload,
                        "prediction": prediction,
                        "error": error,
                        "usage": usage,
                        "response_info": response_info,
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
                        "embedding_model": args.embedding_model,
                        "retrieved_examples": retrieved_records,
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

                if args.sleep > 0:
                    time.sleep(args.sleep)

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
    print("=" * 100)


if __name__ == "__main__":
    main()
