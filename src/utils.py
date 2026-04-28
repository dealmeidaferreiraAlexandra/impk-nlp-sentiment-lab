import io
import re
from typing import Optional

import pandas as pd


def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".txt") or name.endswith(".md"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty:
            return ""

        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        if text_cols:
            first_col = text_cols[0]
            val = df[first_col].dropna()
            if not val.empty:
                return str(val.iloc[0])

        return str(df.iloc[0, 0])

    return raw.decode("utf-8", errors="ignore")