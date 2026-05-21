from sacrebleu.metrics import BLEU, CHRF

bleu_metric = BLEU()
chrf_metric = CHRF(word_order=2)


def compute_bleu(predictions, references) -> float:
    """
    Corpus BLEU.
    predictions: list[str]
    references: list[str]
    """
    return bleu_metric.corpus_score(
        predictions,
        [references],
    ).score


def compute_chrf(predictions, references) -> float:
    """
    Corpus chrF.
    """
    return chrf_metric.corpus_score(
        predictions,
        [references],
    ).score


def compute_bertscore(
    predictions,
    references,
    model_type: str = "bert-base-multilingual-cased",
) -> float:
    """
    BERTScore F1 using multilingual BERT.
    Skips empty prediction/reference pairs because bert-score can crash on
    empty strings.
    """
    from bert_score import score

    clean_pairs = []

    for pred, ref in zip(predictions, references, strict=False):
        pred = str(pred).strip()
        ref = str(ref).strip()

        if pred == "" or ref == "":
            continue

        clean_pairs.append((pred, ref))

    if len(clean_pairs) == 0:
        return None

    clean_predictions = [p for p, r in clean_pairs]
    clean_references = [r for p, r in clean_pairs]

    _, _, f1 = score(
        clean_predictions,
        clean_references,
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )

    return float(f1.mean().item() * 100)


def has_bad_format(text: str) -> bool:
    """
    Not from article, but useful for debugging.
    We do not include this in BLEU/chrF/BERTScore, only report it.
    """
    text = str(text).lower()

    bad_markers = [
        "translation:",
        "explanation:",
        "here is",
        "the translation is",
        "```",
        "source:",
        "target:",
    ]

    return any(marker in text for marker in bad_markers)


def compute_format_error_rate(predictions) -> float:
    if len(predictions) == 0:
        return 0.0

    errors = sum(1 for p in predictions if has_bad_format(p))
    return errors / len(predictions) * 100


def compute_all_metrics(predictions, references) -> dict:
    predictions = ["" if x is None else str(x).strip() for x in predictions]
    references = ["" if x is None else str(x).strip() for x in references]

    empty_prediction_count = sum(1 for x in predictions if x == "")
    empty_reference_count = sum(1 for x in references if x == "")

    try:
        bert = compute_bertscore(predictions, references)
    except Exception as e:
        print(f"BERTScore failed: {e}")
        bert = None

    return {
        "BLEU": compute_bleu(predictions, references),
        "chrF": compute_chrf(predictions, references),
        "BERTScore": bert,
        "format_error_rate": compute_format_error_rate(predictions),
        "empty_prediction_count": empty_prediction_count,
        "empty_reference_count": empty_reference_count,
        "n": len(predictions),
    }
