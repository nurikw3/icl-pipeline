import csv
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from experiment_config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    GRAPH_RETRIEVAL_STRATEGIES,
    PROMPT_MODES,
    RETRIEVAL_BACKENDS,
    RETRIEVAL_STRATEGIES,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "web_logs"
RUNS_PATH = OUTPUT_DIR / "web_runs.json"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def load_runs():
    if not RUNS_PATH.exists():
        return {}

    try:
        runs = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    for run in runs.values():
        if run.get("status") in {"queued", "running", "evaluating"}:
            run["status"] = "interrupted"

    return runs


def save_runs():
    RUNS_PATH.write_text(
        json.dumps(RUNS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


RUNS = load_runs()


def parse_k_list(k_text: str):
    values = [int(x.strip()) for x in k_text.split(",") if x.strip()]
    if not values:
        raise ValueError("k_list must contain at least one positive integer.")
    if any(value < 1 for value in values):
        raise ValueError("k_list values must be positive integers.")
    return values


def parse_positive_int(text: str, default: int) -> int:
    if not str(text).strip():
        return default

    value = int(str(text).strip())
    if value < 1:
        raise ValueError("Value must be positive.")
    return value


def parse_non_negative_int(text: str, default: int) -> int:
    if not str(text).strip():
        return default

    value = int(str(text).strip())
    if value < 0:
        raise ValueError("Value cannot be negative.")
    return value


def parse_sample_fraction(text: str):
    if not str(text).strip():
        return None

    value = float(str(text).strip())
    if value <= 0 or value > 1:
        raise ValueError("Sample fraction must be in the interval (0, 1].")
    return value


def make_safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def read_csv_preview(path: Path, limit: int = 200):
    if not path.exists():
        return [], []

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)

    return reader.fieldnames or [], rows


def read_jsonl_preview(path: Path, limit: int = 100):
    if not path.exists():
        return []

    rows = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"raw": line})

    return rows


def shorten(text, max_len=300):
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def run_experiment_process(run_id: str, config: dict):
    log_path = LOG_DIR / f"{run_id}.log"
    run_name = config["run_name"]

    cmd = [
        "uv", "run", "run_baseline.py",
        "--target_lang", config["target_lang"],
        "--strategies", *config["strategies"],
        "--k_list", *[str(k) for k in config["k_list"]],
        "--run_name", run_name,
        "--retrieval_backend", config["retrieval_backend"],
        "--prompt_mode", config.get("prompt_mode", "standard"),
        "--embedding_model", config["embedding_model"],
        "--max_tokens", str(config["max_tokens"]),
        "--empty_retries", str(config["empty_retries"]),
        "--max_tokens_cap", str(config["max_tokens_cap"]),
        "--lexicon_top_n", str(config.get("lexicon_top_n", 2)),
        "--lexicon_max_entries", str(config.get("lexicon_max_entries", 80)),
    ]

    if config.get("dictionary_path"):
        cmd.extend(["--dictionary_path", config["dictionary_path"]])

    if config.get("enable_bli"):
        cmd.extend(
            [
                "--enable_bli",
                "--bli_threshold",
                str(config.get("bli_threshold", 0.6)),
            ]
        )
        if config.get("bli_lexicon_path"):
            cmd.extend(["--bli_lexicon_path", config["bli_lexicon_path"]])
        if config.get("giza_py_path"):
            cmd.extend(["--giza_py_path", config["giza_py_path"]])

    if not config.get("lexicon_use_fuzzy", True):
        cmd.append("--disable_lexicon_fuzzy")

    if config.get("fuzzy_strategy"):
        cmd.extend(["--fuzzy_strategy", config["fuzzy_strategy"]])

    if config["sample_fraction"] is not None:
        cmd.extend(["--sample_fraction", str(config["sample_fraction"])])

    if config["only_sentences"]:
        cmd.append("--only_sentences")

    if config["max_examples"] is not None:
        cmd.extend(["--max_examples", str(config["max_examples"])])

    if config["print_prompts"]:
        cmd.append("--print_prompts")

    if config.get("resume"):
        cmd.append("--resume")

    RUNS[run_id]["status"] = "running"
    RUNS[run_id]["command"] = " ".join(cmd)
    save_runs()

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"RUN ID: {run_id}\n")
        log_file.write(f"RUN NAME: {run_name}\n")
        log_file.write(f"COMMAND: {' '.join(cmd)}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        RUNS[run_id]["pid"] = process.pid
        save_runs()
        code = process.wait()

        RUNS[run_id]["return_code"] = code
        RUNS[run_id]["status"] = "finished" if code == 0 else "failed"
        save_runs()

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"PROCESS FINISHED WITH CODE: {code}\n")

    predictions_path = OUTPUT_DIR / f"predictions_{run_name}.csv"
    metrics_path = OUTPUT_DIR / f"metrics_{run_name}.csv"

    if code == 0 and predictions_path.exists():
        eval_cmd = [
            "uv", "run", "evaluate.py",
            "--predictions_path", str(predictions_path),
            "--output_path", str(metrics_path),
        ]
        RUNS[run_id]["status"] = "evaluating"
        save_runs()

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n\nRUNNING EVALUATION\n")
            log_file.write(f"COMMAND: {' '.join(eval_cmd)}\n")
            log_file.write("=" * 80 + "\n")

            eval_process = subprocess.Popen(
                eval_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            eval_code = eval_process.wait()
            RUNS[run_id]["eval_return_code"] = eval_code
            RUNS[run_id]["status"] = "finished" if eval_code == 0 else "eval_failed"
            save_runs()

            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"EVALUATION FINISHED WITH CODE: {eval_code}\n")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "runs": RUNS,
            "embedding_models": EMBEDDING_MODELS,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "prompt_modes": PROMPT_MODES,
        },
    )


@app.post("/run")
def run_experiment(
    background_tasks: BackgroundTasks,
    target_lang: str = Form(...),
    strategies: list[str] = Form(default=[]),
    k_list: str = Form(...),
    max_examples: str = Form(""),
    retrieval_backend: str = Form("dense"),
    prompt_mode: str = Form("standard"),
    embedding_model: str = Form(DEFAULT_EMBEDDING_MODEL),
    max_tokens: str = Form("700"),
    empty_retries: str = Form("2"),
    max_tokens_cap: str = Form("1500"),
    lexicon_top_n: str = Form("2"),
    lexicon_max_entries: str = Form("80"),
    dictionary_path: str = Form(""),
    fuzzy_strategy: str = Form("max_matching"),
    lexicon_use_fuzzy: bool = Form(False),
    enable_bli: bool = Form(False),
    bli_threshold: str = Form("0.6"),
    bli_lexicon_path: str = Form(""),
    giza_py_path: str = Form(""),
    sample_fraction: str = Form(""),
    only_sentences: bool = Form(False),
    print_prompts: bool = Form(False),
):
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        clean_strategies = [strategy.strip().lower() for strategy in strategies]
        clean_strategies = [strategy for strategy in clean_strategies if strategy]

        if not clean_strategies:
            raise ValueError("Choose at least one retrieval strategy.")

        bad_strategies = sorted(set(clean_strategies) - set(RETRIEVAL_STRATEGIES))
        if bad_strategies:
            raise ValueError(f"Unknown strategies: {', '.join(bad_strategies)}")

        clean_k_list = parse_k_list(k_list)
        clean_retrieval_backend = retrieval_backend.strip().lower()

        if clean_retrieval_backend not in RETRIEVAL_BACKENDS:
            raise ValueError("Unknown retrieval backend.")

        clean_prompt_mode = prompt_mode.strip().lower()
        if clean_prompt_mode not in PROMPT_MODES:
            raise ValueError("Unknown prompt mode.")

        selected_graph_strategies = set(clean_strategies) & set(
            GRAPH_RETRIEVAL_STRATEGIES
        )
        if selected_graph_strategies and clean_retrieval_backend != "graph":
            raise ValueError("Graph strategies require graph backend.")

        if clean_retrieval_backend == "graph":
            unsupported = (
                set(clean_strategies)
                - set(GRAPH_RETRIEVAL_STRATEGIES)
                - {"random"}
            )
            if unsupported:
                raise ValueError(
                    "Graph backend supports random plus graph strategies only."
                )

        clean_embedding_model = embedding_model.strip() or DEFAULT_EMBEDDING_MODEL
        clean_max_tokens = parse_positive_int(max_tokens, default=700)
        clean_empty_retries = parse_non_negative_int(empty_retries, default=2)
        clean_max_tokens_cap = parse_positive_int(max_tokens_cap, default=1500)
        clean_lexicon_top_n = parse_positive_int(lexicon_top_n, default=2)
        clean_lexicon_max_entries = parse_positive_int(
            lexicon_max_entries,
            default=80,
        )
        clean_dictionary_path = dictionary_path.strip()
        clean_fuzzy_strategy = fuzzy_strategy.strip().lower()
        if clean_fuzzy_strategy not in {"max_matching", "substring"}:
            raise ValueError("Unknown fuzzy strategy.")
        clean_bli_threshold = float(bli_threshold)
        if clean_bli_threshold < 0:
            raise ValueError("BLI threshold cannot be negative.")
        if clean_max_tokens_cap < clean_max_tokens:
            raise ValueError("Max tokens cap must be >= max tokens.")

        clean_sample_fraction = parse_sample_fraction(sample_fraction)
        clean_max_examples = (
            parse_positive_int(max_examples, default=0)
            if max_examples.strip()
            else None
        )
    except ValueError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    retriever_name = (
        "bm25"
        if clean_retrieval_backend == "bm25"
        else make_safe_name(clean_embedding_model)
    )

    sample_name = ""
    if clean_sample_fraction:
        sample_value = str(clean_sample_fraction).replace(".", "_")
        sample_name = f"sample{sample_value}_"

    dictionary_name = ""
    if clean_dictionary_path:
        dictionary_name = f"dict_{make_safe_name(Path(clean_dictionary_path).stem)}_"
    bli_name = "bli_" if enable_bli else ""

    run_name = (
        f"{target_lang}_"
        f"{clean_prompt_mode}_"
        f"{dictionary_name}"
        f"{bli_name}"
        f"{retriever_name}_"
        f"{'-'.join(clean_strategies)}_"
        f"k{'-'.join(str(k) for k in clean_k_list)}_"
        f"{sample_name}"
        f"{timestamp}"
    )

    config = {
        "target_lang": target_lang,
        "strategies": clean_strategies,
        "k_list": clean_k_list,
        "retrieval_backend": clean_retrieval_backend,
        "prompt_mode": clean_prompt_mode,
        "embedding_model": clean_embedding_model,
        "max_tokens": clean_max_tokens,
        "empty_retries": clean_empty_retries,
        "max_tokens_cap": clean_max_tokens_cap,
        "lexicon_top_n": clean_lexicon_top_n,
        "lexicon_max_entries": clean_lexicon_max_entries,
        "dictionary_path": clean_dictionary_path,
        "fuzzy_strategy": clean_fuzzy_strategy,
        "lexicon_use_fuzzy": lexicon_use_fuzzy,
        "enable_bli": enable_bli,
        "bli_threshold": clean_bli_threshold,
        "bli_lexicon_path": bli_lexicon_path.strip(),
        "giza_py_path": giza_py_path.strip(),
        "sample_fraction": clean_sample_fraction,
        "max_examples": clean_max_examples,
        "only_sentences": only_sentences,
        "print_prompts": print_prompts,
        "resume": False,
        "run_name": run_name,
    }

    RUNS[run_id] = {
        "run_id": run_id,
        "run_name": run_name,
        "status": "queued",
        "created_at": timestamp,
        "config": config,
        "log_path": str(LOG_DIR / f"{run_id}.log"),
        "predictions_path": str(OUTPUT_DIR / f"predictions_{run_name}.csv"),
        "prompts_path": str(OUTPUT_DIR / f"prompts_{run_name}.jsonl"),
        "requests_path": str(OUTPUT_DIR / f"requests_{run_name}.jsonl"),
        "retrieved_path": str(OUTPUT_DIR / f"retrieved_examples_{run_name}.jsonl"),
        "metrics_path": str(OUTPUT_DIR / f"metrics_{run_name}.csv"),
        "manifest_path": str(OUTPUT_DIR / f"manifest_{run_name}.json"),
        "status_path": str(OUTPUT_DIR / f"status_{run_name}.json"),
    }
    save_runs()

    background_tasks.add_task(run_experiment_process, run_id, config)

    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.post("/resume/{run_id}")
def resume_run(background_tasks: BackgroundTasks, run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    if run.get("status") in {"queued", "running", "evaluating"}:
        return PlainTextResponse("Run is already active", status_code=400)

    config = dict(run["config"])
    config["resume"] = True
    run["config"] = config
    run["status"] = "queued"
    run["return_code"] = None
    run["eval_return_code"] = None
    save_runs()

    background_tasks.add_task(run_experiment_process, run_id, config)

    return RedirectResponse(url=f"/run/{run_id}", status_code=303)


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run": run,
        },
    )


@app.get("/logs/{run_id}", response_class=HTMLResponse)
def logs_page(request: Request, run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    path = Path(run["log_path"])
    text = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "Log file not created yet."
    )

    return templates.TemplateResponse(
        request,
        "logs.html",
        {
            "run": run,
            "log_text": text,
        },
    )


@app.get("/table/{run_id}/{file_type}", response_class=HTMLResponse)
def table_page(request: Request, run_id: str, file_type: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    key_map = {
        "predictions": "predictions_path",
        "metrics": "metrics_path",
    }

    if file_type not in key_map:
        return PlainTextResponse("Unknown table type", status_code=400)

    path_value = run.get(key_map[file_type])
    if not path_value:
        return PlainTextResponse("File not available for this run", status_code=404)

    path = Path(path_value)
    columns, rows = read_csv_preview(path, limit=500)

    return templates.TemplateResponse(
        request,
        "table.html",
        {
            "run": run,
            "file_type": file_type,
            "path": path,
            "columns": columns,
            "rows": rows,
        },
    )


@app.get("/prompts/{run_id}", response_class=HTMLResponse)
def prompts_page(request: Request, run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    path = Path(run["prompts_path"])
    records = read_jsonl_preview(path, limit=100)

    compact = []
    for rec in records:
        compact.append({
            "test_index": rec.get("test_index", ""),
            "strategy": rec.get("strategy", ""),
            "k": rec.get("k", ""),
            "source_text": rec.get("source_text", ""),
            "reference": rec.get("reference", ""),
            "prompt": rec.get("prompt", ""),
            "retrieved_examples": rec.get("retrieved_examples", []),
            "input_lexicon_entries": rec.get("input_lexicon_entries", []),
            "example_lexicon_entries": rec.get("example_lexicon_entries", []),
        })

    return templates.TemplateResponse(
        request,
        "prompts.html",
        {
            "run": run,
            "records": compact,
            "path": path,
        },
    )


@app.get("/requests/{run_id}", response_class=HTMLResponse)
def requests_page(request: Request, run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    path = Path(run["requests_path"])
    records = read_jsonl_preview(path, limit=100)

    compact = []
    for rec in records:
        payload = rec.get("request_payload", {})
        messages = payload.get("messages", [])
        content = ""
        if messages:
            content = messages[0].get("content", "")

        compact.append({
            "test_index": rec.get("test_index", ""),
            "strategy": rec.get("strategy", ""),
            "k": rec.get("k", ""),
            "model": payload.get("model", ""),
            "temperature": payload.get("temperature", ""),
            "max_tokens": payload.get("max_tokens", ""),
            "source_text": rec.get("source_text", ""),
            "reference": rec.get("reference", ""),
            "request_content": content,
            "prediction": rec.get("prediction", ""),
            "error": rec.get("error", ""),
            "usage": rec.get("usage", ""),
        })

    return templates.TemplateResponse(
        request,
        "requests.html",
        {
            "run": run,
            "records": compact,
            "path": path,
        },
    )


@app.get("/artifact/{run_id}/{file_type}", response_class=PlainTextResponse)
def artifact_page(run_id: str, file_type: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    key_map = {
        "manifest": "manifest_path",
        "status": "status_path",
        "retrieved": "retrieved_path",
    }

    if file_type not in key_map:
        return PlainTextResponse("Unknown artifact type", status_code=400)

    path_value = run.get(key_map[file_type])
    if not path_value:
        return PlainTextResponse("Artifact not available for this run", status_code=404)

    path = Path(path_value)
    text = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "Artifact not created yet."
    )

    return PlainTextResponse(text)


@app.post("/clear_outputs")
def clear_outputs():
    for path in OUTPUT_DIR.glob("*"):
        if path.is_file():
            path.unlink()

    RUNS.clear()
    save_runs()

    return RedirectResponse(url="/", status_code=303)


@app.post("/clear_run/{run_id}")
def clear_run(run_id: str):
    run = RUNS.get(run_id)

    if not run:
        return PlainTextResponse("Run not found", status_code=404)

    file_keys = [
        "log_path",
        "predictions_path",
        "prompts_path",
        "requests_path",
        "retrieved_path",
        "metrics_path",
        "manifest_path",
        "status_path",
    ]

    for key in file_keys:
        path_value = run.get(key)
        if not path_value:
            continue

        path = Path(path_value)
        if path.exists() and path.is_file():
            path.unlink()

    RUNS.pop(run_id, None)
    save_runs()

    return RedirectResponse(url="/", status_code=303)
