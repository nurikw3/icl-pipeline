import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from metrics import compute_all_metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/summary_metrics.csv",
    )
    parser.add_argument(
        "--disable_langfuse_scores",
        action="store_false",
        dest="langfuse_scores_enabled",
        help="Do not send aggregate evaluation scores to Langfuse.",
    )
    parser.set_defaults(langfuse_scores_enabled=True)

    return parser.parse_args()


def ensure_langfuse_env_aliases():
    base_url = os.getenv("LANGFUSE_BASE_URL")
    if base_url and not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = base_url


def has_langfuse_credentials() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def make_score_id(session_id: str, metric_name: str, metadata: dict) -> str:
    raw = (
        f"{session_id}:"
        f"{metadata['model']}:"
        f"{metadata['target_lang']}:"
        f"{metadata['strategy']}:"
        f"k{metadata['k']}:"
        f"{metric_name}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def resolve_langfuse_session_id(group_df: pd.DataFrame, fallback: str) -> str:
    if "langfuse_session_id" not in group_df.columns:
        return fallback

    session_ids = [
        str(value).strip()
        for value in group_df["langfuse_session_id"].dropna().unique().tolist()
        if str(value).strip()
    ]
    if len(session_ids) == 1:
        return session_ids[0]
    return fallback


def send_langfuse_scores(
    summary_df: pd.DataFrame,
    source_df: pd.DataFrame,
    predictions_path: Path,
    enabled: bool,
):
    if not enabled:
        return

    load_dotenv()
    ensure_langfuse_env_aliases()

    if not has_langfuse_credentials():
        print(
            "Langfuse scores skipped: "
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY are not configured."
        )
        return

    from langfuse import get_client

    langfuse = get_client()
    fallback_session_id = predictions_path.stem
    metric_names = [
        "BLEU",
        "chrF",
        "BERTScore",
        "format_error_rate",
        "empty_prediction_count",
        "empty_reference_count",
        "n",
    ]

    group_cols = ["model", "target_lang", "strategy", "k"]
    grouped = {
        group_key: group_df
        for group_key, group_df in source_df.groupby(group_cols)
    }

    sent = 0
    for _, row in summary_df.iterrows():
        group_key = (
            row["model"],
            row["target_lang"],
            row["strategy"],
            row["k"],
        )
        group_df = grouped.get(group_key)
        session_id = (
            resolve_langfuse_session_id(group_df, fallback_session_id)
            if group_df is not None
            else fallback_session_id
        )
        metadata = {
            "model": str(row["model"]),
            "target_lang": str(row["target_lang"]),
            "strategy": str(row["strategy"]),
            "k": int(row["k"]),
            "predictions_path": str(predictions_path),
        }

        for metric_name in metric_names:
            value = row.get(metric_name)
            if pd.isna(value):
                continue
            langfuse.create_score(
                name=metric_name,
                value=float(value),
                session_id=session_id,
                data_type="NUMERIC",
                metadata=metadata,
                score_id=make_score_id(session_id, metric_name, metadata),
            )
            sent += 1

    langfuse.flush()
    print(f"Sent {sent} aggregate scores to Langfuse.")


def main():
    args = parse_args()

    predictions_path = Path(args.predictions_path)
    df = pd.read_csv(predictions_path)

    required = [
        "prediction",
        "reference",
        "target_lang",
        "strategy",
        "k",
        "model",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["prediction"] = df["prediction"].fillna("").astype(str).str.strip()
    df["reference"] = df["reference"].fillna("").astype(str).str.strip()

    rows = []

    group_cols = [
        "model",
        "target_lang",
        "strategy",
        "k",
    ]

    for group_key, group_df in df.groupby(group_cols):
        model, target_lang, strategy, k = group_key

        predictions = group_df["prediction"].tolist()
        references = group_df["reference"].tolist()

        metrics = compute_all_metrics(
            predictions=predictions,
            references=references,
        )

        rows.append(
            {
                "model": model,
                "target_lang": target_lang,
                "strategy": strategy,
                "k": k,
                **metrics,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["target_lang", "k", "strategy"],
        ascending=[True, True, True],
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    send_langfuse_scores(
        summary_df=summary_df,
        source_df=df,
        predictions_path=predictions_path,
        enabled=args.langfuse_scores_enabled,
    )

    print(summary_df)
    print("\nSaved summary:", output_path)


if __name__ == "__main__":
    main()
