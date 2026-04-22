from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.priceconda import fetch_event_return
from src.sentiment import device_label, score_sentences
from src.transcripts import (
    get_calls_for,
    list_tickers,
    load_dataframe,
    split_sections,
    tokenize_sentences,
)

st.set_page_config(page_title="Earnings Call Sentiment", layout="wide")
st.title("Earnings Call Sentiment Dashboard")
st.caption(f"FinBERT device: `{device_label()}`  ·  model: ProsusAI/finbert")


@st.cache_resource(show_spinner="Loading transcript corpus…")
def _corpus():
    return load_dataframe()


@st.cache_data(show_spinner="Scoring transcript with FinBERT…")
def _score(transcript: str) -> dict[str, list[dict]]:
    sections = split_sections(transcript)
    out: dict[str, list[dict]] = {}
    for key in ("prepared", "qa"):
        sents = tokenize_sentences(sections[key])
        out[key] = [s.to_dict() for s in score_sentences(sents)]
    return out


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
scored = _score(row["transcript"])
prepared = pd.DataFrame(scored["prepared"])
qa = pd.DataFrame(scored["qa"])
if prepared.empty and qa.empty:
    st.warning("No sentences could be extracted from this transcript.")
    st.stop()

frames = []
if not prepared.empty:
    frames.append(prepared.assign(section="Prepared"))
if not qa.empty:
    frames.append(qa.assign(section="Q&A"))
all_rows = pd.concat(frames, ignore_index=True)
all_rows["signed"] = all_rows["positive"] - all_rows["negative"]
all_rows["idx"] = range(len(all_rows))

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
