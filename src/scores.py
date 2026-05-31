from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "scored"

COLUMNS = ["section", "text", "label", "positive", "negative", "neutral"]


def _cache_path(ticker: str, call_date: pd.Timestamp) -> Path:
    iso = pd.Timestamp(call_date).date().isoformat()
    return CACHE_DIR / f"{ticker.upper()}_{iso}.parquet"


def is_cached(ticker: str, call_date: pd.Timestamp) -> bool:
    return _cache_path(ticker, call_date).exists()


def score_call(ticker: str, call_date: pd.Timestamp, transcript: str) -> pd.DataFrame:
    path = _cache_path(ticker, call_date)
    if path.exists():
        return pd.read_parquet(path)

    from .sentiment import score_sentences
    from .transcripts import split_sections, tokenize_sentences

    sections = split_sections(transcript)
    rows: list[dict] = []
    for key, section_label in (("prepared", "Prepared"), ("qa", "Q&A")):
        sents = tokenize_sentences(sections[key])
        for s in score_sentences(sents):
            rows.append({**s.to_dict(), "section": section_label})
    df = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def summarize_call(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {
            "n": 0,
            "avg_signed": float("nan"),
            "pct_positive": float("nan"),
            "pct_negative": float("nan"),
        }
    signed = (scored["positive"] - scored["negative"]).mean()
    return {
        "n": int(len(scored)),
        "avg_signed": float(signed),
        "pct_positive": float((scored["label"] == "positive").mean()),
        "pct_negative": float((scored["label"] == "negative").mean()),
    }
