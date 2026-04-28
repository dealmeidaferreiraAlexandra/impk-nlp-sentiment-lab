import json
from pathlib import Path

from src.config import METRICS_PATH, EVAL_SAMPLE_SIZE
from src.data import load_imdb_dataset
from src.metrics import binary_metrics
from src.traditional import load_traditional_pipeline, predict_traditional_batch
from src.bert_utils import load_bert_components, predict_bert_batch


def main():
    print(f"📦 Loading IMDb test subset ({EVAL_SAMPLE_SIZE} samples)...")
    data = load_imdb_dataset(sample_test_size=EVAL_SAMPLE_SIZE)

    traditional = load_traditional_pipeline()
    if traditional is None:
        raise FileNotFoundError(
            "Traditional model not found. Run train_traditional.py first."
        )

    print("🧠 Evaluating TF-IDF + Logistic Regression...")
    trad_preds, _ = predict_traditional_batch(data["test_texts"], traditional)
    trad_metrics = binary_metrics(data["test_labels"], trad_preds)

    print("🤖 Loading BERT sentiment model...")
    tokenizer, bert_model, device = load_bert_components()

    print("🧠 Evaluating BERT...")
    bert_preds, _ = predict_bert_batch(
        data["test_texts"],
        tokenizer,
        bert_model,
        batch_size=16,
        device=device,
    )
    bert_metrics = binary_metrics(data["test_labels"], bert_preds)

    metrics = {
        "dataset": "IMDb",
        "evaluation_subset_size": EVAL_SAMPLE_SIZE,
        "traditional": trad_metrics,
        "bert": bert_metrics,
        "note": "BERT uses the pretrained textattack/bert-base-uncased-imdb checkpoint.",
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("✅ Saved metrics to artifacts/metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()