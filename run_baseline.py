import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from retrieval import ICLRetriever
from prompts import build_prompt


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
        choices=["random", "similarity", "diversity"],
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
    parser.add_argument("--max_tokens", type=int, default=300)

    parser.add_argument(
        "--embedding_model",
        type=str,
        default="intfloat/multilingual-e5-base",
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

    return parser.parse_args()


def load_api_settings(args):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    env_model = os.getenv("OPENAI_MODEL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    if not base_url:
        raise ValueError("OPENAI_BASE_URL is missing. Add it to your .env file.")

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


def make_output_paths(output_dir: Path, args, model: str):
    if args.run_name:
        base_name = args.run_name
    else:
        strategy_name = "-".join(args.strategies)
        k_name = "-".join(str(k) for k in args.k_list)
        model_name = make_safe_name(model)

        base_name = (
            f"{args.target_lang}_"
            f"{strategy_name}_"
            f"k{k_name}_"
            f"{model_name}"
        )

    predictions_path = output_dir / f"predictions_{base_name}.csv"
    prompts_path = output_dir / f"prompts_{base_name}.jsonl"
    requests_path = output_dir / f"requests_{base_name}.jsonl"
    retrieved_path = output_dir / f"retrieved_examples_{base_name}.jsonl"

    return predictions_path, prompts_path, requests_path, retrieved_path


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


def call_openai(client: OpenAI, request_payload: dict):
    response = client.chat.completions.create(**request_payload)

    prediction = response.choices[0].message.content

    if prediction is None:
        prediction = ""

    prediction = prediction.strip()

    usage = None
    try:
        usage = response.usage.model_dump()
    except Exception:
        usage = None

    return prediction, usage


def save_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    predictions_path,
    prompts_path,
    requests_path,
    retrieved_path,
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
    print("Base URL:", base_url)
    print("Temperature:", args.temperature)
    print("Max tokens:", args.max_tokens)
    print("Embedding model:", args.embedding_model)
    print("Candidate N:", args.candidate_n)
    print("Predictions:", predictions_path)
    print("Prompts:", prompts_path)
    print("Requests:", requests_path)
    print("Retrieved examples:", retrieved_path)
    print("=" * 100)


def main():
    args = parse_args()

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

    predictions_path, prompts_path, requests_path, retrieved_path = make_output_paths(
        output_dir=output_dir,
        args=args,
        model=model,
    )

    remove_old_output_files(
        [
            predictions_path,
            prompts_path,
            requests_path,
            retrieved_path,
        ]
    )

    print_run_header(
        train_df=train_df,
        test_df=test_df,
        args=args,
        model=model,
        base_url=base_url,
        predictions_path=predictions_path,
        prompts_path=prompts_path,
        requests_path=requests_path,
        retrieved_path=retrieved_path,
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    all_results = []

    total_errors = 0
    total_empty_predictions = 0

    for k in args.k_list:
        for strategy in args.strategies:
            print("\n" + "-" * 100)
            print(f"Running strategy={strategy}, k={k}")
            print("-" * 100)

            cache_name = (
                f"train_"
                f"{args.target_lang}_"
                f"{args.embedding_model.replace('/', '_')}_"
                f"{'sentences' if args.only_sentences else 'all'}"
                f".npy"
            )

            retriever = ICLRetriever(
                train_df=train_df,
                embedding_model_name=args.embedding_model,
                cache_dir="embeddings",
                cache_name=cache_name,
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

                examples = retriever.retrieve(
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

                request_payload = build_request_payload(
                    prompt=prompt,
                    model=model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
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

                try:
                    prediction, usage = call_openai(
                        client=client,
                        request_payload=request_payload,
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
                    "model": model,
                    "temperature": args.temperature,
                    "error": error,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

                all_results.append(result_record)

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
                        "model": model,
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
                        "request_payload": request_payload,
                        "prediction": prediction,
                        "error": error,
                        "usage": usage,
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
                        "retrieved_examples": retrieved_records,
                    }

                    save_jsonl(prompts_path, prompt_record)
                    save_jsonl(requests_path, request_record)
                    save_jsonl(retrieved_path, retrieved_record)

                # save progress after every example
                pd.DataFrame(all_results).to_csv(
                    predictions_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                if args.sleep > 0:
                    time.sleep(args.sleep)

            print(f"\nFinished strategy={strategy}, k={k}")
            print("Strategy errors:", strategy_errors)
            print("Strategy empty predictions:", strategy_empty_predictions)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Saved predictions:", predictions_path)
    print("Saved prompts:", prompts_path)
    print("Saved requests:", requests_path)
    print("Saved retrieved examples:", retrieved_path)
    print("Rows:", len(results_df))
    print("Total errors:", total_errors)
    print("Total empty predictions:", total_empty_predictions)
    print("=" * 100)


if __name__ == "__main__":
    main()