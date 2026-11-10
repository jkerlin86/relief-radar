import numpy as np
import pandas as pd

from relief_radar.selection import (
    add_narrative_flag,
    company_size_tiers,
    rate_table,
    rates_from_aggregates,
    tier_labels,
)


def test_add_narrative_flag(complaints_frame):
    ids = pd.Series([1000, 1005, 1006])
    out = add_narrative_flag(complaints_frame, ids)
    flagged = out.loc[out["has_narrative"] == 1, "complaint_id"].tolist()
    assert flagged == [1000, 1005, 1006]


def test_rate_table_math(complaints_frame):
    out = add_narrative_flag(complaints_frame, pd.Series([1000, 1001, 1004]))
    table = rate_table(out, "product")
    cc = table[table["product"] == "Credit card"].iloc[0]
    assert cc["n"] == 8
    assert cc["narrative_n"] == 3
    # Narrative-bearing credit-card rows: ids 1000 (y=1), 1001 (0), 1004 (1).
    assert abs(cc["relief_rate_narrative"] - 2 / 3) < 1e-9
    assert abs(cc["narrative_share"] - 3 / 8) < 1e-9


def test_rates_from_aggregates_matches_rate_table(complaints_frame):
    out = add_narrative_flag(complaints_frame, pd.Series([1000, 1001, 1004]))
    out["relief_with"] = out["target"] * out["has_narrative"]
    out["relief_without"] = out["target"] * (1 - out["has_narrative"])
    agg = (
        out.groupby("product", observed=True)
        .agg(
            n=("complaint_id", "size"),
            narrative_n=("has_narrative", "sum"),
            relief_with=("relief_with", "sum"),
            relief_without=("relief_without", "sum"),
        )
        .reset_index()
    )
    via_agg = rates_from_aggregates(agg).set_index("product")
    via_rows = rate_table(out, "product").set_index("product")
    cols = ["narrative_share", "relief_rate_narrative", "relief_rate_no_narrative"]
    for col in cols:
        assert np.allclose(
            via_agg[col].to_numpy(dtype=float),
            via_rows[col].to_numpy(dtype=float),
            atol=1e-9,
            equal_nan=True,
        )


def test_tier_labels(complaints_frame):
    totals = complaints_frame.groupby("company_norm")["complaint_id"].size()
    labels = tier_labels(totals, n_tiers=2)
    assert labels["size_tier"].isin([1, 2]).all()
    assert len(labels) == totals.shape[0]


def test_company_size_tiers(complaints_frame):
    out = add_narrative_flag(complaints_frame, pd.Series([], dtype="int64"))
    tiered = company_size_tiers(out, n_tiers=2)
    assert tiered["size_tier"].isin([1, 2]).all()
    assert tiered.groupby("company_norm")["size_tier"].nunique().max() == 1
