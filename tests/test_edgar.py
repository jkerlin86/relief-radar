from relief_radar.edgar import annual_facts


def _facts():
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"form": "10-K", "fp": "FY", "fy": 2023,
                             "val": 100, "filed": "2024-02-01"},
                            # Restated / later filing for the same year wins.
                            {"form": "10-K", "fp": "FY", "fy": 2023,
                             "val": 105, "filed": "2025-02-01"},
                            {"form": "10-Q", "fp": "Q1", "fy": 2024,
                             "val": 30, "filed": "2024-05-01"},
                            {"form": "10-K", "fp": "FY", "fy": 2024,
                             "val": 120, "filed": "2025-02-15"},
                        ]
                    }
                }
            }
        }
    }


def test_annual_facts_picks_latest_filed_10k():
    rows = annual_facts(_facts(), "Revenues")
    assert [r["year"] for r in rows] == [2023, 2024]
    assert rows[0]["value"] == 105.0
    assert rows[1]["value"] == 120.0


def test_annual_facts_missing_concept():
    assert annual_facts(_facts(), "NetIncomeLoss") == []
    assert annual_facts({}, "Revenues") == []
