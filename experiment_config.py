DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

EMBEDDING_MODELS = {
    "intfloat/multilingual-e5-base": {
        "label": "E5 multilingual base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "BAAI/bge-m3": {
        "label": "BGE-M3",
        "query_prefix": "",
        "passage_prefix": "",
    },
}

RETRIEVAL_BACKENDS = ("dense", "bm25")
RETRIEVAL_STRATEGIES = ("random", "similarity", "diversity")


def get_embedding_model_config(model_name: str) -> dict:
    model_name = str(model_name).strip()
    return EMBEDDING_MODELS.get(
        model_name,
        {
            "label": model_name,
            "query_prefix": "query: " if "e5" in model_name.lower() else "",
            "passage_prefix": "passage: " if "e5" in model_name.lower() else "",
        },
    )
