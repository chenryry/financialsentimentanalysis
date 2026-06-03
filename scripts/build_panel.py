"""Assemble the call-level panel for the cross-sectional study.

For every call that has been scored (data/scored/*.parquet), join its FinBERT
sentiment summary to a beta-adjusted event return and write one tidy row to
data/panel.parquet. This is the dataset the corpus-wide regression runs on.

Run scripts/score_all.py first to populate the sentiment cache.

Examples:
    python scripts/build_panel.py
    python scripts/build_panel.py --tickers AAPL,MSFT,NVDA
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prices import market_model_car
from src.scores import is_cached, score_call
from src.transcripts import load_dataframe

warnings.filterwarnings("ignore")

PANEL_PATH = Path(__file__).resolve().parent.parent / "data" / "panel.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tickers", help="Comma-separated tickers to restrict to.")
    p.add_argument(
        "--out", default=str(PANEL_PATH), help="Output parquet path."
    )
    return p.parse_args()


def _section_signed(scored: pd.DataFrame, section: str) -> float:
    sub = scored[scored["section"] == section]
    if sub.empty:
        return float("nan")
    return float((sub["positive"] - sub["negative"]).mean())


def main() -> int:
    args = parse_args()
    df = load_dataframe()
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        df = df[df["ticker"].isin(wanted)].copy()

    cached = df[df.apply(lambda r: is_cached(r["ticker"], r["call_date"]), axis=1)]
    print(f"scored calls available: {len(cached):,}")
    if cached.empty:
        print("Nothing to build. Run scripts/score_all.py first.")
        return 1

    rows: list[dict] = []
    no_prices = 0
    from tqdm import tqdm

    for r in tqdm(cached.itertuples(index=False), total=len(cached), unit="call"):
        scored = score_call(r.ticker, r.call_date, r.transcript)
        if scored.empty:
            continue
        signed = scored["positive"] - scored["negative"]
        mm = market_model_car(r.ticker, r.call_date)
        if mm is None:
            no_prices += 1
        rows.append(
            {
                "ticker": r.ticker,
                "call_date": r.call_date,
                "q": r.q,
                "n_sentences": int(len(scored)),
                "avg_signed": float(signed.mean()),
                "prepared_signed": _section_signed(scored, "Prepared"),
                "qa_signed": _section_signed(scored, "Q&A"),
                "pct_positive": float((scored["label"] == "positive").mean()),
                "pct_negative": float((scored["label"] == "negative").mean()),
                "beta": mm.beta if mm else float("nan"),
                "car": mm.car if mm else float("nan"),
                "raw_abnormal": mm.raw_abnormal if mm else float("nan"),
                "stock_return": mm.stock_return if mm else float("nan"),
                "market_return": mm.market_return if mm else float("nan"),
            }
        )

    panel = pd.DataFrame(rows).sort_values(["ticker", "call_date"]).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    usable = int(panel["car"].notna().sum())
    print(
        f"wrote {len(panel):,} rows to {out}  "
        f"({usable:,} with returns, {no_prices:,} missing price data)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
