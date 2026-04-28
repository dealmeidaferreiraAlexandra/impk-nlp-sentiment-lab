from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

from .config import RANDOM_SEED


def _sample_split(split, sample_size: Optional[int], seed: int) -> Tuple[List[str], List[int]]:
    texts = split["text"]
    labels = split["label"]

    if sample_size is None or sample_size >= len(texts):
        return list(texts), [int(x) for x in labels]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=sample_size, replace=False)

    sampled_texts = [texts[i] for i in idx]
    sampled_labels = [int(labels[i]) for i in idx]
    return sampled_texts, sampled_labels


def load_imdb_dataset(
    sample_train_size: Optional[int] = None,
    sample_test_size: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Dict[str, List]:
    dataset = load_dataset("imdb")

    train_texts, train_labels = _sample_split(dataset["train"], sample_train_size, seed)
    test_texts, test_labels = _sample_split(dataset["test"], sample_test_size, seed)

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }