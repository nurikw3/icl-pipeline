<h1>
  <img src="https://api.iconify.design/lucide:workflow.svg?color=%23f97316" width="32" alt="Pipeline icon" align="left">
  ICL Pipeline
</h1>

<p align="center">
  <img src="assets/image.png" alt="ICL Pipeline preview" width="720">
</p>

## Run

```bash
uv sync
uv run uvicorn web_app:app
```

## CLI examples

Dense E5 retrieval:

```bash
uv run run_baseline.py --target_lang en --strategies similarity diversity --k_list 8
```

BGE-M3 retrieval:

```bash
uv run run_baseline.py --target_lang en --embedding_model BAAI/bge-m3 --strategies similarity --k_list 8
```

BM25 retrieval:

```bash
uv run run_baseline.py --target_lang en --retrieval_backend bm25 --strategies similarity diversity --k_list 8
```

Graph-aware retrieval comparison on a 10% sample:

```bash
uv run run_baseline.py \
  --target_lang en \
  --retrieval_backend graph \
  --strategies graph_common graph_ppr hybrid_graph \
  --k_list 8 \
  --sample_fraction 0.1 \
  --only_sentences
```

Resume an interrupted run:

```bash
uv run run_baseline.py --run_name my_run --resume
```

## Environment

Create `.env` with:

```bash
OPENAI_API_KEY=...
OPENAI_MODEL=...
OPENAI_BASE_URL=...
```

`OPENAI_BASE_URL` is optional. Outputs are written to `outputs/`; dense
embedding caches are written to `embeddings/` and are keyed by model and data
fingerprint.

The graph backend uses an in-memory feature graph over the filtered train split.
`hybrid_graph` lazily uses the same local cached embedding store as dense
retrieval, so no external vector database or graph database is required for the
first comparison.

## Data schema

`data/train.csv` and `data/test.csv` must include:

- `source_text`
- `target_text`
- `source_lang`
- `target_lang`
- `type`
