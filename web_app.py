import csv
import json
import uuid
import subprocess
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")

OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "web_logs"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

RUNS = {}


def parse_k_list(k_text: str):
    return [int(x.strip()) for x in k_text.split(",") if x.strip()]


def read_csv_preview(path: Path, limit: int = 200):
    if not path.exists():
        return [], []

    with open(path, "r", encoding="utf-8-sig") as f:
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

    with open(path, "r", encoding="utf-8") as f:
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
    ]

    if config["only_sentences"]:
        cmd.append("--only_sentences")

    if config["max_examples"] is not None:
        cmd.extend(["--max_examples", str(config["max_examples"])])

    if config["print_prompts"]:
        cmd.append("--print_prompts")

    RUNS[run_id]["status"] = "running"
    RUNS[run_id]["command"] = " ".join(cmd)

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
        code = process.wait()

        RUNS[run_id]["return_code"] = code
        RUNS[run_id]["status"] = "finished" if code == 0 else "failed"

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"PROCESS FINISHED WITH CODE: {code}\n")

    predictions_path = OUTPUT_DIR / f"predictions_{run_name}.csv"
    metrics_path = OUTPUT_DIR / f"metrics_{run_name}.csv"

    if predictions_path.exists():
        eval_cmd = [
            "uv", "run", "evaluate.py",
            "--predictions_path", str(predictions_path),
            "--output_path", str(metrics_path),
        ]

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

            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"EVALUATION FINISHED WITH CODE: {eval_code}\n")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "runs": RUNS,
        },
    )


@app.post("/run")
def run_experiment(
    background_tasks: BackgroundTasks,
    target_lang: str = Form(...),
    strategies: list[str] = Form(...),
    k_list: str = Form(...),
    max_examples: str = Form(""),
    only_sentences: bool = Form(False),
    print_prompts: bool = Form(False),
):
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    clean_strategies = strategies
    clean_k_list = parse_k_list(k_list)

    run_name = (
        f"{target_lang}_"
        f"{'-'.join(clean_strategies)}_"
        f"k{'-'.join(str(k) for k in clean_k_list)}_"
        f"{timestamp}"
    )

    config = {
        "target_lang": target_lang,
        "strategies": clean_strategies,
        "k_list": clean_k_list,
        "max_examples": int(max_examples) if max_examples.strip() else None,
        "only_sentences": only_sentences,
        "print_prompts": print_prompts,
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
    }

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
    text = path.read_text(encoding="utf-8") if path.exists() else "Log file not created yet."

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

    path = Path(run[key_map[file_type]])
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


@app.post("/clear_outputs")
def clear_outputs():
    for folder in [OUTPUT_DIR, Path("embeddings")]:
        folder.mkdir(exist_ok=True)

        for path in folder.glob("*"):
            if path.is_file():
                path.unlink()

    RUNS.clear()

    return RedirectResponse(url="/", status_code=303)