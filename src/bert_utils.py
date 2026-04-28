from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BERT_MODEL_NAME, MAX_LENGTH


def _find_label_index(model, target_label: str, default_idx: int) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    target = target_label.lower()

    for label_name, idx in label2id.items():
        if str(label_name).lower() == target:
            return int(idx)

    return default_idx


def load_bert_components(model_name: str = BERT_MODEL_NAME, device: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_bert_text(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
) -> Dict[str, float | str | int]:
    pos_idx = _find_label_index(model, "positive", 1)
    neg_idx = _find_label_index(model, "negative", 0)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    label = "Positive" if pred_idx == pos_idx else "Negative"

    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
        "positive_prob": float(probs[pos_idx]),
        "negative_prob": float(probs[neg_idx]),
        "pred_index": pred_idx,
    }


def predict_bert_batch(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tuple[List[int], List[float]]:
    pos_idx = _find_label_index(model, "positive", 1)

    preds: List[int] = []
    positive_probs: List[float] = []

    model.eval()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        batch_preds = probs.argmax(axis=1).tolist()
        batch_pos_probs = probs[:, pos_idx].tolist()

        preds.extend([int(x) for x in batch_preds])
        positive_probs.extend([float(x) for x in batch_pos_probs])

    return preds, positive_probs