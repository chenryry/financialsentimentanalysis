from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.prices import fetch_event_return
from src.scores import is_cached, score_call, summarize_call
from src.sentiment import device_label
from src.transcripts import get_calls_for, list_tickers, load_dataframe

st.set_page_config(page_title="Earnings Call Sentiment", layout="wide")
st.title("Earnings Call Sentiment Dashboard")
st.caption(f"FinBERT device: `{device_label()}`  ·  model: ProsusAI/finbert")


@st.cache_resource(show_spinner="Loading transcript corpus…")
def _corpus():
    return load_dataframe()


@st.cache_data(show_spinner="Scoring transcript with FinBERT…")
def _score(ticker: str, call_date: pd.Timestamp, _transcript: str) -> pd.DataFrame:
    return score_call(ticker, call_date, _transcript)


@st.cache_data(show_spinner="Fetching price data…")
def _returns(ticker: str, event_date: pd.Timestamp):
    r = fetch_event_return(ticker, event_date)
    if r is None:
        return None
    return {
        "window": r.window,
        "stock_return": r.stock_return,
        "market_return": r.market_return,
        "abnormal_return": r.abnormal_return,
        "prices": r.prices.reset_index().to_dict("records"),
    }


df = _corpus()

col_a, col_b = st.columns([1, 2])
with col_a:
    tickers = list_tickers(df)
    default = tickers.index("AAPL") if "AAPL" in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=default)

calls = get_calls_for(df, ticker)
with col_b:
    labels = calls.apply(lambda r: f"{r['call_date'].date()}  ({r['q']})", axis=1)
    choice = st.selectbox(
        "Earnings call", options=range(len(calls)), format_func=lambda i: labels.iloc[i]
    )

row = calls.iloc[choice]
all_rows = _score(ticker, row["call_date"], row["transcript"])
if all_rows.empty:
    st.warning("No sentences could be extracted from this transcript.")
    st.stop()

all_rows = all_rows.copy()
all_rows["signed"] = all_rows["positive"] - all_rows["negative"]
all_rows["idx"] = range(len(all_rows))

summary = summarize_call(all_rows)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Sentences", f"{summary['n']:,}")
m2.metric("Avg signed", f"{summary['avg_signed']:+.3f}")
m3.metric("% positive", f"{summary['pct_positive'] * 100:.1f}%")
m4.metric("% negative", f"{summary['pct_negative'] * 100:.1f}%")

st.subheader("Sentiment timeline")
rolling = all_rows["signed"].rolling(15, min_periods=1).mean()
fig_tl = go.Figure()
fig_tl.add_scatter(
    x=all_rows["idx"], y=all_rows["signed"], mode="markers", name="sentence",
    marker=dict(size=5, color=all_rows["signed"], colorscale="RdYlGn", cmin=-1, cmax=1),
)
fig_tl.add_scatter(
    x=all_rows["idx"], y=rolling, mode="lines", name="rolling(15)", line=dict(color="black"),
)
qa_mask = all_rows["section"] == "Q&A"
if qa_mask.any():
    fig_tl.add_vline(x=int(qa_mask.idxmax()), line_dash="dash", annotation_text="Q&A begins")
fig_tl.update_layout(height=360, yaxis_title="pos − neg", xaxis_title="sentence index")
st.plotly_chart(fig_tl, use_container_width=True)

st.subheader("Section breakdown")
breakdown = all_rows.groupby("section")[["positive", "negative", "neutral"]].mean().reset_index()
fig_sec = px.bar(
    breakdown.melt(id_vars="section", var_name="label", value_name="avg_prob"),
    x="section", y="avg_prob", color="label", barmode="group",
    color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"},
)
st.plotly_chart(fig_sec, use_container_width=True)

st.subheader("Price reaction")
ev = _returns(ticker, row["call_date"])
if ev is None:
    st.info("Price data unavailable (ticker may be delisted or outside yfinance coverage).")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Stock return", f"{ev['stock_return'] * 100:.2f}%")
    c2.metric("S&P 500 return", f"{ev['market_return'] * 100:.2f}%")
    c3.metric("Abnormal return", f"{ev['abnormal_return'] * 100:.2f}%")
    prices = pd.DataFrame(ev["prices"])
    prices["Date"] = pd.to_datetime(prices["Date"])
    fig_px = go.Figure()
    fig_px.add_scatter(
        x=prices["Date"], y=prices["stock"] / prices["stock"].iloc[0] - 1, name=ticker,
    )
    fig_px.add_scatter(
        x=prices["Date"], y=prices["market"] / prices["market"].iloc[0] - 1, name="S&P 500",
    )
    fig_px.update_layout(yaxis_tickformat=".1%", height=320)
    st.plotly_chart(fig_px, use_container_width=True)

st.subheader("Key sentences")
c1, c2 = st.columns(2)
c1.markdown("**Most positive**")
c1.dataframe(
    all_rows.nlargest(5, "positive")[["section", "positive", "text"]],
    use_container_width=True, hide_index=True,
)
c2.markdown("**Most negative**")
c2.dataframe(
    all_rows.nlargest(5, "negative")[["section", "negative", "text"]],
    use_container_width=True, hide_index=True,
)

st.divider()
st.subheader(f"Cross-quarter trend — {ticker}")

cached_mask = calls.apply(lambda r: is_cached(ticker, r["call_date"]), axis=1)
n_total = len(calls)
n_cached = int(cached_mask.sum())

if n_cached < n_total:
    missing = n_total - n_cached
    st.caption(
        f"{n_cached}/{n_total} calls scored. Score the rest to see the full trend."
    )
    if st.button(f"Score remaining {missing} call(s)"):
        progress = st.progress(0.0, text="Scoring…")
        to_score = calls.loc[~cached_mask].reset_index(drop=True)
        for i, r in to_score.iterrows():
            _score(ticker, r["call_date"], r["transcript"])
            progress.progress((i + 1) / len(to_score), text=f"Scored {i + 1}/{len(to_score)}")
        progress.empty()
        st.rerun()
else:
    st.caption(f"All {n_total} calls scored.")

cached_calls = calls.loc[cached_mask].reset_index(drop=True)
trend_rows: list[dict] = []
for _, r in cached_calls.iterrows():
    scored = _score(ticker, r["call_date"], r["transcript"])
    s = summarize_call(scored)
    ev_i = _returns(ticker, r["call_date"])
    trend_rows.append(
        {
            "call_date": r["call_date"],
            "q": r["q"],
            "avg_signed": s["avg_signed"],
            "n": s["n"],
            "abnormal_return": ev_i["abnormal_return"] if ev_i else None,
        }
    )
trend = pd.DataFrame(trend_rows).sort_values("call_date").reset_index(drop=True)

if len(trend) >= 1:
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_scatter(
        x=trend["call_date"], y=trend["avg_signed"], mode="lines+markers",
        name="Avg signed sentiment", line=dict(color="#3b82f6"),
    )
    if trend["abnormal_return"].notna().any():
        fig_trend.add_scatter(
            x=trend["call_date"], y=trend["abnormal_return"], mode="lines+markers",
            name="Abnormal return (±1d)", line=dict(color="#f59e0b", dash="dot"),
            secondary_y=True,
        )
        fig_trend.update_yaxes(title_text="Abnormal return", tickformat=".1%", secondary_y=True)
    fig_trend.update_yaxes(title_text="Avg signed sentiment", secondary_y=False)
    fig_trend.update_layout(height=360, xaxis_title="Call date", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    valid = trend.dropna(subset=["avg_signed", "abnormal_return"])
    if len(valid) >= 2:
        st.subheader("Sentiment vs. abnormal return")
        fig_scatter = px.scatter(
            valid, x="avg_signed", y="abnormal_return",
            hover_data={"q": True, "n": True, "call_date": "|%Y-%m-%d"},
        )
        if len(valid) >= 3:
            import numpy as np
            x = valid["avg_signed"].to_numpy()
            y = valid["abnormal_return"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            fig_scatter.add_scatter(
                x=xs, y=slope * xs + intercept, mode="lines",
                name="linear fit", line=dict(color="#6b7280", dash="dash"),
            )
        corr = valid["avg_signed"].corr(valid["abnormal_return"])
        st.caption(f"Pearson correlation across {len(valid)} call(s): **{corr:+.3f}**")
        fig_scatter.update_yaxes(tickformat=".1%", title_text="Abnormal return (±1d)")
        fig_scatter.update_xaxes(title_text="Avg signed sentiment")
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)
