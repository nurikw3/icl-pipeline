import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from lexicon import tokenize_for_lexicon


def tokenized_line(text: str) -> str:
    return " ".join(tokenize_for_lexicon(text))


def write_parallel_corpus(
    parallel_df: pd.DataFrame,
    source_path: Path,
    target_path: Path,
):
    source_lines = []
    target_lines = []

    for row in parallel_df.itertuples(index=False):
        source = tokenized_line(getattr(row, "source_text", ""))
        target = tokenized_line(getattr(row, "target_text", ""))

        if not source or not target:
            continue

        source_lines.append(source)
        target_lines.append(target)

    source_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    target_path.write_text("\n".join(target_lines) + "\n", encoding="utf-8")


def parse_bli_lexicon(path: Path, threshold: float = 0.6) -> list[dict]:
    rows = []
    threshold = float(threshold)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()

            if len(parts) < 2:
                continue

            source = parts[0].strip()
            target = parts[1].strip()
            score = 1.0

            if len(parts) >= 3:
                try:
                    score = float(parts[2])
                except ValueError:
                    score = 1.0

            if source and target and score >= threshold:
                rows.append(
                    {
                        "source_text": source,
                        "target_text": target,
                        "score": score,
                    }
                )

    return rows


def run_giza_py_lexicon(
    parallel_df: pd.DataFrame,
    output_path: Path,
    giza_py_path: Path,
    threshold: float = 0.6,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    giza_py_path = giza_py_path.resolve()
    giza_bin_dir = giza_py_path.parent / ".bin"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source_path = tmpdir / "source.txt"
        target_path = tmpdir / "target.txt"
        write_parallel_corpus(parallel_df, source_path, target_path)

        cmd = [
            sys.executable,
            str(giza_py_path),
        ]
        if giza_bin_dir.is_dir():
            cmd.extend(["--bin", str(giza_bin_dir)])
        cmd.extend(
            [
                "--source",
                str(source_path),
                "--target",
                str(target_path),
                "--lexicon",
                str(output_path),
                "--lexicon-threshold",
                str(threshold),
                "--quiet",
            ]
        )
        subprocess.run(cmd, check=True)

    return parse_bli_lexicon(output_path, threshold=threshold)
