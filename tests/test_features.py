import numpy as np
import pandas as pd

from relief_radar.features import (
    add_features,
    apply_category_caps,
    build_history_tables,
    fit_category_caps,
    month_index,
    monthly_company_stats,
    monthly_product_stats,
)


def _history(frame):
    return build_history_tables(
        monthly_company_stats(frame),
        monthly_product_stats(frame),
        float(frame["target"].mean()),
    )


def test_month_index_monotone_across_years():
    idx = month_index(pd.Series([202312, 202401, 202402]))
    assert list(idx) == sorted(idx)
    assert idx.iloc[1] - idx.iloc[0] == 1


def test_history_excludes_own_month(complaints_frame):
    history = _history(complaints_frame)
    featured = add_features(complaints_frame, history, prior_strength=20.0)
    jan = featured[featured["month"] == 202401]
    # January complaints see no history at all.
    assert (jan["prior_n"] == 0).all()
    assert (jan["company_seen_before"] == 0).all()
    feb_acme = featured[
        (featured["month"] == 202402) & (featured["company_norm"] == "ACME BANK")
    ]
    # ACME had 3 January complaints, 1 with relief.
    assert (feb_acme["prior_n"] == 3).all()
    assert (feb_acme["prior_relief"] == 1).all()
    # Previous-month volume for February is January's count.
    assert (feb_acme["prev_month_n"] == 3).all()


def test_prior_rate_shrinks_toward_product_rate(complaints_frame):
    history = _history(complaints_frame)
    featured = add_features(complaints_frame, history, prior_strength=20.0)
    feb = featured[featured["month"] == 202402]
    # Rates stay in [0, 1] and are shrunk: ACME credit-card complaints in
    # February see prior_n=3, prior_relief=1, product prior rate 1/3, so
    # the shrunk rate is (1 + 20/3) / (3 + 20) = 1/3, not the raw 1/3 vs
    # an unshrunk alternative — and crucially not a degenerate value.
    assert feb["company_prior_relief_rate"].between(0, 1).all()
    acme_cc = feb[
        (feb["company_norm"] == "ACME BANK") & (feb["product"] == "Credit card")
    ]
    expected = (1 + 20 * (1 / 3)) / (3 + 20)
    assert (abs(acme_cc["company_prior_relief_rate"] - expected) < 1e-9).all()


def test_category_caps(complaints_frame):
    caps = fit_category_caps(complaints_frame, ["product"], top_k=1)
    capped = apply_category_caps(complaints_frame, caps)
    kept = set(capped["product"].unique())
    assert "__OTHER__" in kept
    assert len(kept) == 2
    assert np.isin(["Credit card"], list(kept)).all()
