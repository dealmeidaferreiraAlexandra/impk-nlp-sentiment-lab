from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
TRADITIONAL_MODEL_PATH = ARTIFACTS_DIR / "traditional_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

BERT_MODEL_NAME = "textattack/bert-base-uncased-imdb"
MAX_LENGTH = 256
EVAL_SAMPLE_SIZE = 1000
RANDOM_SEED = 42