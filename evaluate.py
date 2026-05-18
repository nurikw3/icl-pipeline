import argparse
from pathlib import Path

import pandas as pd

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

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.predictions_path)

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

    print(summary_df)
    print("\nSaved summary:", output_path)


if __name__ == "__main__":
    main()