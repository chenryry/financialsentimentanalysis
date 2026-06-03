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
    # The raw corpus has many duplicate rows per call (e.g. AAPL: 62 rows for 14
    # actual calls). Collapse to one row per (ticker, call_date), keeping the
    # longest transcript, so downstream counts, the call picker, and the
    # cross-sectional panel each treat a call exactly once.
    df = df.assign(_len=df["transcript"].str.len())
    df = (
        df.sort_values(["ticker", "call_date", "_len"])
        .drop_duplicates(subset=["ticker", "call_date"], keep="last")
        .drop(columns="_len")
    )
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

    # Don't guess resource names: NLTK 3.9 needs 'punkt_tab' while older
    # releases use 'punkt', and a partial install can make nltk.data.find raise
    # OSError pointing at the wrong package. Probe by actually tokenizing, and
    # only download (trying both packages) if that fails.
    def _works() -> bool:
        try:
            sent_tokenize("Ready. Set. Go.")
            return True
        except (LookupError, OSError):
            return False

    if _works():
        return sent_tokenize
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
        if _works():
            return sent_tokenize
    return sent_tokenize
