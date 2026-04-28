from pathlib import Path
from typing import List, Tuple, Dict, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import TRADITIONAL_MODEL_PATH


def build_traditional_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=40000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                ),
            ),
        ]
    )


def train_traditional_pipeline(texts: List[str], labels: List[int]) -> Pipeline:
    pipeline = build_traditional_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


def save_traditional_pipeline(pipeline: Pipeline, path=TRADITIONAL_MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_traditional_pipeline(path=TRADITIONAL_MODEL_PATH) -> Optional[Pipeline]:
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def predict_traditional_text(text: str, pipeline: Pipeline) -> Dict[str, float | str | int]:
    probs = pipeline.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))

    label = "Positive" if pred_idx == 1 else "Negative"

    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
        "positive_prob": float(probs[1]),
        "negative_prob": float(probs[0]),
        "pred_index": pred_idx,
    }


def predict_traditional_batch(texts: List[str], pipeline: Pipeline) -> Tuple[List[int], List[float]]:
    probs = pipeline.predict_proba(texts)
    preds = probs.argmax(axis=1).tolist()
    positive_probs = probs[:, 1].tolist()
    return preds, positive_probs