from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

PKL_PATH = Path(__file__).resolve().parent.parent / "motley-fool-data.pkl"

_DATE_PREFIX = re.compile(r"^([A-Z][a-z]{2} \d{1,2}, \d{4})")


@lru_cache(maxsize=1)
def load_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else PKL_PATH
    df = pd.read_pickle(p).copy()
    df["call_date"] = df["date"].map(_parse_date)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["call_date", "ticker", "transcript"])
    return df.sort_values(["ticker", "call_date"]).reset_index(drop=True)


def _parse_date(raw) -> pd.Timestamp | None:
    if not isinstance(raw, str):
        return None
    m = _DATE_PREFIX.match(raw)
    if not m:
        return None
    return pd.to_datetime(m.group(1), errors="coerce")


def list_tickers(df: pd.DataFrame) -> list[str]:
    return sorted(df["ticker"].unique().tolist())


def get_calls_for(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    mask = df["ticker"] == ticker.upper()
    cols = ["call_date", "q", "exchange", "transcript"]
    return df.loc[mask, cols].reset_index(drop=True)


def split_sections(transcript: str) -> dict[str, str]:
    """Return {'prepared', 'qa', 'participants'} — missing sections are ''."""
    t = transcript.replace("Questions & Answers:", "Questions and Answers:")
    t = t.replace("Call Participants:", "Call participants:")
    markers = [
        ("prepared", "Prepared Remarks:"),
        ("qa", "Questions and Answers:"),
        ("participants", "Call participants:"),
    ]
    found = []
    for key, header in markers:
        idx = t.find(header)
        if idx >= 0:
            found.append((idx, key, header))
    found.sort()
    out = {"prepared": "", "qa": "", "participants": ""}
    for i, (start, key, header) in enumerate(found):
        body_start = start + len(header)
        body_end = found[i + 1][0] if i + 1 < len(found) else len(t)
        out[key] = t[body_start:body_end].strip()
    return out


def tokenize_sentences(text: str) -> list[str]:
    if not text:
        return []
    sent_tokenize = _sent_tokenizer()
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


@lru_cache(maxsize=1)
def _sent_tokenizer():
    import nltk
    from nltk.tokenize import sent_tokenize

    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return sent_tokenize
        except LookupError:
            nltk.download(pkg, quiet=True)
    return sent_tokenize
