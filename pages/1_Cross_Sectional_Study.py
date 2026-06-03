"""Corpus-wide view: does earnings-call sentiment predict abnormal return?

Reads the panel built by scripts/build_panel.py and shows the pooled
regression, a sentiment-quantile portfolio sort, and the underlying scatter.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analysis import describe, load_panel, quantile_portfolios, regress

st.set_page_config(page_title="Cross-Sectional Study", layout="wide")
st.title("Does call sentiment predict abnormal return?")
st.caption(
    "Pooled across every scored call. Abnormal return = market-model CAR over a "
    "±1-day window; sentiment = mean FinBERT (positive − negative) per call."
)

try:
    panel = load_panel()
except FileNotFoundError:
    st.warning(
        "No panel yet. Score calls with `python scripts/score_all.py` then build "
        "the panel with `python scripts/build_panel.py`."
    )
    st.stop()

meta = describe(panel)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Calls scored", f"{meta['n_calls']:,}")
c2.metric("Usable (with prices)", f"{meta['n_usable']:,}")
c3.metric("Tickers", f"{meta['n_tickers']:,}")
c4.metric("Mean CAR", f"{meta['mean_car'] * 100:+.2f}%")

if meta["n_usable"] < 10:
    st.info("Score more calls for a meaningful cross-section (need ≥10 with prices).")
    st.stop()

x_choice = st.selectbox(
    "Sentiment measure",
    ["avg_signed", "qa_signed", "prepared_signed", "pct_positive"],
    format_func={
        "avg_signed": "Whole call (pos − neg)",
        "qa_signed": "Q&A section only",
        "prepared_signed": "Prepared remarks only",
        "pct_positive": "% positive sentences",
    }.get,
)
cluster = st.checkbox("Cluster standard errors by ticker", value=True)

res = regress(panel, x=x_choice, y="car", cluster="ticker" if cluster else None)

st.subheader("Pooled regression")
m1, m2, m3 = st.columns(3)
m1.metric("Slope", f"{res.slope:+.4f}", help="Change in CAR per unit of sentiment")
m2.metric("p-value", f"{res.p:.4f}", help="Two-sided, H0: slope = 0")
m3.metric("R²", f"{res.r2:.4f}")
verdict = (
    "statistically significant" if res.p < 0.05 else "not statistically significant"
)
sign = "higher" if res.slope > 0 else "lower"
st.markdown(
    f"More positive sentiment is associated with **{sign}** abnormal returns, and "
    f"the relationship is **{verdict}** (n={res.n}, 95% CI "
    f"[{res.ci_low:+.4f}, {res.ci_high:+.4f}])."
)
st.code(res.summary(), language="text")

st.subheader("Sentiment-quantile portfolios")
st.caption(
    "Calls sorted into bins by sentiment; bars show mean abnormal return per bin. "
    "A monotonic rise from left to right is the signal we're after."
)
q = st.slider("Number of bins", 3, 10, 5)
ports = quantile_portfolios(panel, x=x_choice, y="car", q=q)
fig_q = px.bar(ports, x="sentiment_bin", y="mean_car")
fig_q.update_yaxes(tickformat=".2%", title_text="Mean abnormal return (CAR)")
fig_q.update_xaxes(title_text="Sentiment bin (low → high)")
fig_q.update_layout(height=360)
st.plotly_chart(fig_q, use_container_width=True)
spread = ports["mean_car"].iloc[-1] - ports["mean_car"].iloc[0]
st.caption(f"Top-minus-bottom spread: **{spread * 100:+.2f}%** abnormal return.")

st.subheader("All calls")
valid = panel.dropna(subset=[x_choice, "car"])
fig_s = px.scatter(
    valid, x=x_choice, y="car", hover_data=["ticker", "q", "call_date"],
    trendline="ols", trendline_color_override="#6b7280",
)
fig_s.update_yaxes(tickformat=".1%", title_text="Abnormal return (CAR, ±1d)")
fig_s.update_layout(height=420)
st.plotly_chart(fig_s, use_container_width=True)
