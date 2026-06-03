from __future__ import annotations

import pandas as pd

from src.transcripts import _parse_date, split_sections, tokenize_sentences

SAMPLE = """Apple Inc. (AAPL) Q1 2020 Earnings Call

Prepared Remarks:
Thank you. Revenue grew strongly this quarter. We are pleased with results.

Questions and Answers:
What about margins? Margins expanded year over year.

Call participants:
Tim Cook -- CEO
"""


def test_split_sections_separates_prepared_and_qa():
    out = split_sections(SAMPLE)
    assert "Revenue grew strongly" in out["prepared"]
    assert "What about margins" in out["qa"]
    assert "Tim Cook" in out["participants"]
    # Prepared body must not leak into Q&A.
    assert "Revenue grew strongly" not in out["qa"]


def test_split_sections_normalizes_header_variants():
    text = SAMPLE.replace("Questions and Answers:", "Questions & Answers:")
    out = split_sections(text)
    assert "What about margins" in out["qa"]


def test_split_sections_missing_sections_are_empty():
    out = split_sections("No recognizable headers here at all.")
    assert out == {"prepared": "", "qa": "", "participants": ""}


def test_parse_date_extracts_leading_date():
    assert _parse_date("Jan 28, 2020, 5:00 p.m. ET") == pd.Timestamp("2020-01-28")


def test_parse_date_rejects_non_date():
    assert _parse_date("no date here") is None
    assert _parse_date(None) is None


def test_tokenize_sentences_splits_and_strips():
    sents = tokenize_sentences("Revenue grew.  Margins fell. ")
    assert sents == ["Revenue grew.", "Margins fell."]


def test_tokenize_sentences_empty():
    assert tokenize_sentences("") == []
