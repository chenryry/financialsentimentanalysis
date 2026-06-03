from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK = "^GSPC"
HIST_DIR = Path(__file__).resolve().parent.parent / "data" / "prices"

# Corpus spans 2019-05 .. 2023-02; pad generously so estimation windows for the
# earliest calls and event windows for the latest calls are both covered.
HIST_START = "2018-06-01"
HIST_END = "2023-06-30"


@dataclass
class EventReturn:
    window: tuple[int, int]
    stock_return: float
    market_return: float
    abnormal_return: float  # raw: stock - market
    prices: pd.DataFrame  # indexed by date, cols: stock, market


@dataclass
class MarketModelCAR:
    """Beta-adjusted cumulative abnormal return around an earnings call.

    The market model R_i = alpha + beta * R_m is fit on an estimation window
    ending `gap` trading days before the event; the CAR is the sum of daily
    (actual - predicted) returns over the [-pre, +post] event window.
    """

    event_date: pd.Timestamp
    alpha: float
    beta: float
    n_estimation: int
    window: tuple[int, int]
    car: float  # cumulative abnormal return over the event window
    raw_abnormal: float  # stock - market over the same window (model-free)
    stock_return: float
    market_return: float


def _hist_path(ticker: str) -> Path:
    safe = ticker.upper().replace("^", "_")
    return HIST_DIR / f"{safe}.parquet"


def load_history(
    ticker: str,
    start: str = HIST_START,
    end: str = HIST_END,
    force: bool = False,
) -> pd.Series | None:
    """Return a cached daily adjusted-close series for `ticker` (or None).

    The first call for a ticker downloads the full history once and caches it
    to data/prices/<TICKER>.parquet; subsequent calls read from disk.
    """
    path = _hist_path(ticker)
    if path.exists() and not force:
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        return s

    data = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if data is None or data.empty:
        return None
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # multi-ticker shape; take the column
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    close.rename("close").to_frame().to_parquet(path)
    return close


def _event_anchor(index: pd.DatetimeIndex, event_date: pd.Timestamp) -> int | None:
    """Position of the first trading day on or after the event date."""
    on_or_after = index >= pd.Timestamp(event_date)
    if not on_or_after.any():
        return None
    return int(index.get_loc(index[on_or_after][0]))


def fetch_event_return(
    ticker: str,
    event_date: pd.Timestamp,
    pre_days: int = 1,
    post_days: int = 1,
) -> EventReturn | None:
    """Raw (model-free) event return: stock vs. benchmark over a small window."""
    stock = load_history(ticker)
    market = load_history(BENCHMARK)
    if stock is None or market is None:
        return None
    joined = pd.concat(
        [stock.rename("stock"), market.rename("market")], axis=1
    ).dropna()
    if joined.empty:
        return None
    pos = _event_anchor(joined.index, event_date)
    if pos is None:
        return None
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


def market_model_car(
    ticker: str,
    event_date: pd.Timestamp,
    pre_days: int = 1,
    post_days: int = 1,
    est_window: int = 120,
    est_gap: int = 10,
) -> MarketModelCAR | None:
    """Beta-adjusted cumulative abnormal return around `event_date`.

    Fits R_i = alpha + beta * R_m by OLS over `est_window` trading days ending
    `est_gap` days before the event, then sums (actual - predicted) daily
    returns across the [-pre_days, +post_days] event window.
    """
    stock = load_history(ticker)
    market = load_history(BENCHMARK)
    if stock is None or market is None:
        return None
    rets = pd.concat(
        [stock.pct_change().rename("stock"), market.pct_change().rename("market")],
        axis=1,
    ).dropna()
    if rets.empty:
        return None
    pos = _event_anchor(rets.index, event_date)
    if pos is None:
        return None

    est_hi = pos - est_gap
    est_lo = est_hi - est_window
    if est_lo < 0 or est_hi - est_lo < 30:  # need a usable estimation window
        return None
    est = rets.iloc[est_lo:est_hi]
    x = est["market"].to_numpy()
    y = est["stock"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)  # slope, intercept

    lo = max(0, pos - pre_days)
    hi = min(len(rets) - 1, pos + post_days)
    ev = rets.iloc[lo : hi + 1]
    if ev.empty:
        return None
    predicted = alpha + beta * ev["market"].to_numpy()
    abnormal = ev["stock"].to_numpy() - predicted
    car = float(np.sum(abnormal))
    raw_abnormal = float((ev["stock"] - ev["market"]).sum())
    return MarketModelCAR(
        event_date=pd.Timestamp(event_date),
        alpha=float(alpha),
        beta=float(beta),
        n_estimation=int(len(est)),
        window=(-pre_days, +post_days),
        car=car,
        raw_abnormal=raw_abnormal,
        stock_return=float(ev["stock"].sum()),
        market_return=float(ev["market"].sum()),
    )
