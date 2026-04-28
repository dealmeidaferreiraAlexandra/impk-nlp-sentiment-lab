from src.data import load_imdb_dataset
from src.metrics import binary_metrics
from src.traditional import (
    train_traditional_pipeline,
    save_traditional_pipeline,
    predict_traditional_batch,
)

from pathlib import Path


def main():
    print("📦 Loading IMDb dataset...")
    data = load_imdb_dataset()

    print("🧠 Training TF-IDF + Logistic Regression...")
    pipeline = train_traditional_pipeline(data["train_texts"], data["train_labels"])

    save_traditional_pipeline(pipeline)
    print("✅ Saved traditional pipeline to artifacts/traditional_pipeline.joblib")

    print("📊 Quick evaluation on IMDb test set...")
    y_pred, _ = predict_traditional_batch(data["test_texts"], pipeline)
    metrics = binary_metrics(data["test_labels"], y_pred)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()