from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

BENCHMARK = "^GSPC"


@dataclass
class EventReturn:
    window: tuple[int, int]
    stock_return: float
    market_return: float
    abnormal_return: float
    prices: pd.DataFrame  # indexed by date, cols: stock, market


def fetch_event_return(
    ticker: str,
    event_date: pd.Timestamp,
    pre_days: int = 1,
    post_days: int = 1,
) -> EventReturn | None:
    pad = 10
    start = (event_date - pd.Timedelta(days=pad)).date()
    end = (event_date + pd.Timedelta(days=pad)).date()
    data = yf.download(
        [ticker, BENCHMARK],
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
    )
    if data is None or data.empty:
        return None
    try:
        stock = data[ticker]["Close"].dropna()
        market = data[BENCHMARK]["Close"].dropna()
    except KeyError:
        return None
    joined = pd.concat([stock.rename("stock"), market.rename("market")], axis=1).dropna()
    if joined.empty:
        return None
    on_or_after = joined.index >= pd.Timestamp(event_date)
    if not on_or_after.any():
        return None
    anchor = joined.index[on_or_after][0]
    pos = joined.index.get_loc(anchor)
    lo = max(0, pos - pre_days)
    hi = min(len(joined) - 1, pos + post_days)
    window = joined.iloc[lo : hi + 1]
    if len(window) < 2:
        return None
    stock_ret = window["stock"].iloc[-1] / window["stock"].iloc[0] - 1
    mkt_ret = window["market"].iloc[-1] / window["market"].iloc[0] - 1
    return EventReturn(
        window=(-pre_days, +post_days),
        stock_return=float(stock_ret),
        market_return=float(mkt_ret),
        abnormal_return=float(stock_ret - mkt_ret),
        prices=window,
    )
