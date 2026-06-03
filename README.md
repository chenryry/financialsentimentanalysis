# Earnings Call Sentiment Dashboard

**Research question:** *Does the tone of an earnings call predict the stock's
abnormal return around that call — and is any of that signal not already priced
in?*

This project scores S&P earnings-call transcripts with [FinBERT], measures each
stock's market reaction with an event study, and pools the results into a
corpus-wide test of whether call sentiment explains abnormal returns.

[FinBERT]: https://huggingface.co/ProsusAI/finbert

## What it does

1. **Parse** Motley Fool transcripts into Prepared Remarks / Q&A sections and
   sentences (`src/transcripts.py`).
2. **Score** every sentence with FinBERT → positive / negative / neutral
   probabilities, cached per call as parquet (`src/sentiment.py`,
   `src/scores.py`).
3. **Measure the reaction** with a market-model event study: estimate
   `R_i = α + β·R_m` over a 120-day window, then sum abnormal returns over the
   ±1-day window around the call (`src/prices.py`). Price history is cached per
   ticker.
4. **Pool & test** across all scored calls — a ticker-clustered regression of
   abnormal return on sentiment, plus sentiment-quantile portfolios
   (`src/analysis.py`).

## Layout

```
app.py                              Streamlit home — per-call exploration
pages/1_Cross_Sectional_Study.py    Corpus-wide regression & portfolio sort
src/transcripts.py                  Load corpus, split sections, tokenize
src/sentiment.py                    FinBERT scoring
src/scores.py                       Per-call scoring + parquet cache
src/prices.py                       Cached history, raw & market-model returns
src/analysis.py                     Cross-sectional regression / portfolios
scripts/score_all.py               Batch-score the corpus into data/scored/
scripts/build_panel.py             Join sentiment + returns into data/panel.parquet
```

## Quickstart

```bash
conda env create -f environment.yml
conda activate earningsdashboard

# 1. Score transcripts (CPU is slow ~1 call/min; start with a subset)
python scripts/score_all.py --top-tickers 25

# 2. Build the call-level panel (sentiment joined to abnormal returns)
python scripts/build_panel.py

# 3. Explore
streamlit run app.py
```

The corpus (`motley-fool-data.pkl`) and all generated caches
(`data/scored/`, `data/prices/`, `data/panel.parquet`) are git-ignored;
regenerate them with the scripts above.

## Notes & caveats

- **Abnormal returns are beta-adjusted** (market model), not raw stock − index.
- The ±1-day window captures the immediate reaction; it does not test
  post-earnings drift.
- FinBERT is sentence-level and finance-tuned but still a proxy for "tone."
- yfinance coverage gaps (delistings, etc.) drop some calls from the panel.
