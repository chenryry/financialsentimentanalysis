"""Unit tests for event-window logic, with yfinance stubbed out.

These exercise the date-anchoring and market-model maths against a synthetic
price history so they run offline and deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import prices


@pytest.fixture
def synthetic_history(monkeypatch):
    """Patch load_history to return deterministic series for STOCK and ^GSPC."""
    days = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    market_ret = rng.normal(0.0005, 0.01, len(days))
    market = pd.Series(100 * np.cumprod(1 + market_ret), index=days)
    # Stock = beta 1.5 on market + idiosyncratic noise, with a +5% jump on the
    # event day (the 150th trading day).
    stock_ret = 1.5 * market_ret + rng.normal(0, 0.005, len(days))
    stock_ret[150] += 0.05
    stock = pd.Series(50 * np.cumprod(1 + stock_ret), index=days)

    def fake_load(ticker, *a, **k):
        return market.copy() if ticker == prices.BENCHMARK else stock.copy()

    monkeypatch.setattr(prices, "load_history", fake_load)
    return days


def test_event_anchor_picks_first_trading_day_on_or_after():
    idx = pd.bdate_range("2020-01-01", periods=5)  # Wed..Tue
    # Saturday 2020-01-04 -> next trading day is Monday 2020-01-06 (position 3).
    assert prices._event_anchor(idx, pd.Timestamp("2020-01-04")) == 3
    assert prices._event_anchor(idx, pd.Timestamp("2020-01-01")) == 0


def test_event_anchor_returns_none_past_end():
    idx = pd.bdate_range("2020-01-01", periods=5)
    assert prices._event_anchor(idx, pd.Timestamp("2021-01-01")) is None


def test_fetch_event_return_captures_jump(synthetic_history):
    days = synthetic_history
    ev = prices.fetch_event_return("STOCK", days[150])
    assert ev is not None
    assert ev.window == (-1, 1)
    # The stock jumped ~5% on the event day, so abnormal return is clearly positive.
    assert ev.abnormal_return > 0.02


def test_market_model_recovers_beta_and_positive_car(synthetic_history):
    days = synthetic_history
    mm = prices.market_model_car("STOCK", days[150])
    assert mm is not None
    assert mm.n_estimation == 120
    assert mm.beta == pytest.approx(1.5, abs=0.2)  # true beta is 1.5
    assert mm.car > 0.03  # the +5% shock survives beta adjustment


def test_market_model_none_without_estimation_window(synthetic_history):
    days = synthetic_history
    # Too early: no room for a 120-day estimation window before the event.
    assert prices.market_model_car("STOCK", days[5]) is None
