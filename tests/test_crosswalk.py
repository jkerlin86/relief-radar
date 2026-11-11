import pandas as pd

from relief_radar.crosswalk import match_companies, prepare_tickers


def _tickers():
    raw = pd.DataFrame(
        {
            "cik_str": [111, 222, 333, 444],
            "name": [
                "ACME BANK CORP",
                "BETA FINANCIAL GROUP INC",
                "GAMMA HOLDINGS LLC",
                "DELTA NATIONAL BANK",
            ],
            "ticker": ["ACME", "BETA", "GAMA", "DLTA"],
        }
    )
    return prepare_tickers(raw)


def test_prepare_tickers_sec_title_schema():
    # The SEC publishes company_tickers.json with a "title" field.
    raw = pd.DataFrame(
        {
            "cik_str": [320193],
            "title": ["Apple Inc."],
            "ticker": ["AAPL"],
        }
    )
    out = prepare_tickers(raw)
    assert out.iloc[0]["cik"] == 320193
    assert out.iloc[0]["sec_name"] == "Apple Inc."
    assert out.iloc[0]["ticker"] == "AAPL"


def test_exact_tier():
    out = match_companies(["ACME BANK"], _tickers())
    row = out.iloc[0]
    assert row["tier"] == "exact"
    assert row["cik"] == 111


def test_containment_tier():
    # CFPB name carries an extra substantive token the SEC name lacks.
    out = match_companies(["DELTA NATIONAL BANK OF TEXAS"], _tickers())
    row = out.iloc[0]
    assert row["tier"] == "containment"
    assert row["cik"] == 444


def test_fuzzy_tier():
    out = match_companies(["BETA FINANCIL GROUP"], _tickers())
    row = out.iloc[0]
    assert row["tier"] == "fuzzy"
    assert row["cik"] == 222


def test_unmatched():
    out = match_companies(["ZZZ UNKNOWN STANDALONE"], _tickers())
    assert out.iloc[0]["tier"] == "unmatched"
    assert out.iloc[0]["cik"] is None or pd.isna(out.iloc[0]["cik"])
