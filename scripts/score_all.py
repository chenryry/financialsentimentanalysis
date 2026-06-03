"""Batch-score earnings call transcripts with FinBERT.

Walks the corpus and writes per-call parquet files under data/scored/.
Already-cached calls are skipped, so this script is safe to re-run and
resume after interruption.

Examples:
    python scripts/score_all.py --dry-run
    python scripts/score_all.py --top-tickers 50
    python scripts/score_all.py --tickers AAPL,MSFT,NVDA
    python scripts/score_all.py --limit 500 --shuffle
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scores import is_cached, score_call
from src.transcripts import load_dataframe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", help="Comma-separated tickers (e.g. AAPL,MSFT).")
    p.add_argument("--top-tickers", type=int, metavar="N", help="Score only the N tickers with the most calls.")
    p.add_argument("--limit", type=int, metavar="N", help="Stop after scoring N call(s) this run.")
    p.add_argument("--shuffle", action="store_true", help="Shuffle call order (useful for partial runs).")
    p.add_argument("--dry-run", action="store_true", help="Report counts without scoring.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataframe()
    print(f"corpus: {len(df):,} rows, {df['ticker'].nunique():,} tickers")

    if args.top_tickers:
        top = df["ticker"].value_counts().head(args.top_tickers).index
        df = df[df["ticker"].isin(top)].copy()
        print(f"limiting to top {args.top_tickers} tickers: {len(df):,} rows")

    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        df = df[df["ticker"].isin(wanted)].copy()
        print(f"limiting to {len(wanted)} ticker(s): {len(df):,} rows")

    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    if args.shuffle:
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    cached_mask = df.apply(lambda r: is_cached(r["ticker"], r["call_date"]), axis=1)
    todo = df.loc[~cached_mask].reset_index(drop=True)
    print(f"already cached: {int(cached_mask.sum()):,}  ·  to score: {len(todo):,}")

    if args.limit and len(todo) > args.limit:
        todo = todo.head(args.limit)
        print(f"limited to {len(todo):,} call(s) this run")

    if args.dry_run or todo.empty:
        return 0

    from tqdm import tqdm

    errors = 0
    start = time.time()
    bar = tqdm(todo.itertuples(index=False), total=len(todo), unit="call")
    for r in bar:
        bar.set_postfix_str(f"{r.ticker} {r.call_date.date()}")
        try:
            score_call(r.ticker, r.call_date, r.transcript)
        except Exception:
            errors += 1
            tqdm.write(f"[error] {r.ticker} {r.call_date.date()}: {traceback.format_exc().splitlines()[-1]}")

    elapsed = time.time() - start
    ok = len(todo) - errors
    print(f"done: scored {ok:,} call(s) in {elapsed / 60:.1f} min  ({errors} error(s))")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
