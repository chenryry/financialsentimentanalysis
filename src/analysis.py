"""Cross-sectional study: does earnings-call sentiment predict abnormal return?

Loads the call-level panel built by scripts/build_panel.py and runs the pooled
regression CAR ~ sentiment, plus a sentiment-quantile portfolio sort. The
regression uses heteroskedasticity-robust (HC1) standard errors, and can
cluster by ticker so repeated calls from the same firm aren't treated as
independent observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_PATH = Path(__file__).resolve().parent.parent / "data" / "panel.parquet"


@dataclass
class RegressionResult:
    x: str
    y: str
    n: int
    slope: float
    intercept: float
    se: float  # standard error of the slope
    t: float
    p: float  # two-sided p-value for slope == 0
    r2: float
    ci_low: float  # 95% CI on the slope
    ci_high: float
    cluster: str | None

    def summary(self) -> str:
        stars = "***" if self.p < 0.01 else "**" if self.p < 0.05 else "*" if self.p < 0.1 else ""
        return (
            f"{self.y} ~ {self.x}   (n={self.n}"
            + (f", clustered by {self.cluster}" if self.cluster else "")
            + ")\n"
            f"  slope     = {self.slope:+.4f} {stars}\n"
            f"  std err   = {self.se:.4f}\n"
            f"  t / p     = {self.t:+.2f} / {self.p:.4f}\n"
            f"  95% CI    = [{self.ci_low:+.4f}, {self.ci_high:+.4f}]\n"
            f"  R-squared = {self.r2:.4f}"
        )


def load_panel(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else PANEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Build it with: python scripts/build_panel.py"
        )
    return pd.read_parquet(p)


def regress(
    panel: pd.DataFrame,
    x: str = "avg_signed",
    y: str = "car",
    cluster: str | None = "ticker",
) -> RegressionResult:
    """OLS of y on x with robust (or ticker-clustered) standard errors."""
    import statsmodels.formula.api as smf

    cols = [x, y] + ([cluster] if cluster else [])
    data = panel[cols].dropna()
    if len(data) < 3:
        raise ValueError(f"need >=3 complete rows, got {len(data)}")

    model = smf.ols(f"{y} ~ {x}", data=data)
    if cluster:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster]})
    else:
        fit = model.fit(cov_type="HC1")

    ci = fit.conf_int().loc[x]
    return RegressionResult(
        x=x,
        y=y,
        n=int(fit.nobs),
        slope=float(fit.params[x]),
        intercept=float(fit.params["Intercept"]),
        se=float(fit.bse[x]),
        t=float(fit.tvalues[x]),
        p=float(fit.pvalues[x]),
        r2=float(fit.rsquared),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        cluster=cluster,
    )


def quantile_portfolios(
    panel: pd.DataFrame,
    x: str = "avg_signed",
    y: str = "car",
    q: int = 5,
) -> pd.DataFrame:
    """Sort calls into `q` sentiment bins; report mean abnormal return per bin.

    The long-short spread (top bin minus bottom bin) is the headline number: how
    much extra abnormal return the most-positive calls earn over the most
    negative ones.
    """
    data = panel[[x, y]].dropna()
    if len(data) < q:
        raise ValueError(f"need >={q} rows, got {len(data)}")
    bins = pd.qcut(data[x], q, labels=False, duplicates="drop")
    grouped = data.groupby(bins)[y]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            f"mean_{x}": data.groupby(bins)[x].mean(),
            f"mean_{y}": grouped.mean(),
            f"std_{y}": grouped.std(),
        }
    )
    out.index.name = "sentiment_bin"
    return out.reset_index()


def describe(panel: pd.DataFrame) -> dict:
    usable = panel.dropna(subset=["avg_signed", "car"])
    return {
        "n_calls": int(len(panel)),
        "n_usable": int(len(usable)),
        "n_tickers": int(panel["ticker"].nunique()),
        "date_min": panel["call_date"].min(),
        "date_max": panel["call_date"].max(),
        "mean_car": float(usable["car"].mean()) if len(usable) else float("nan"),
    }
