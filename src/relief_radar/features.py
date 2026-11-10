"""Point-in-time feature construction.

Every feature in this module is computable at the moment a complaint is
received. Company and product history features are *expanding* aggregates
over strictly earlier calendar months: a complaint received in month m
only sees complaints received in months < m. The outcome fields of the
complaint itself (response, timeliness, public response, date sent) are
never used; see docs/leakage_audit.md for the field-by-field audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "submitted_via",
    "state",
]

NUMERIC_FEATURES = [
    "year",
    "month_of_year",
    "day_of_week",
    "zip_missing",
    "tag_older_american",
    "tag_servicemember",
    "company_seen_before",
    "company_log_prior_volume",
    "company_prior_relief_rate",
    "company_log_prev_month_volume",
    "product_prior_relief_rate",
    "product_log_prev_month_volume",
]


def month_index(month_yyyymm: pd.Series) -> pd.Series:
    """Convert YYYYMM ints to a monotone month index."""
    return (month_yyyymm // 100) * 12 + (month_yyyymm % 100)


def monthly_company_stats(primary: pd.DataFrame) -> pd.DataFrame:
    """Per company-month complaint and relief counts (primary population)."""
    stats = (
        primary.groupby(["company_norm", "month"], observed=True)
        .agg(n=("target", "size"), relief=("target", "sum"))
        .reset_index()
    )
    stats["month_idx"] = month_index(stats["month"])
    return stats.sort_values(["company_norm", "month_idx"]).reset_index(drop=True)


def _expanding_prior(
    stats: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    """Add prior_n / prior_relief: expanding sums over strictly earlier months."""
    out = stats.sort_values(group_cols + ["month_idx"]).copy()
    grouped = out.groupby(group_cols, observed=True)
    out["prior_n"] = grouped["n"].cumsum() - out["n"]
    out["prior_relief"] = grouped["relief"].cumsum() - out["relief"]
    prev = out.copy()
    prev["month_idx"] = prev["month_idx"] + 1
    prev = prev.rename(columns={"n": "prev_month_n"})[
        group_cols + ["month_idx", "prev_month_n"]
    ]
    out = out.merge(prev, on=group_cols + ["month_idx"], how="left")
    out["prev_month_n"] = out["prev_month_n"].fillna(0)
    return out


def monthly_product_stats(primary: pd.DataFrame) -> pd.DataFrame:
    """Per product-month complaint and relief counts (primary population)."""
    stats = (
        primary.groupby(["product", "month"], observed=True)
        .agg(n=("target", "size"), relief=("target", "sum"))
        .reset_index()
    )
    stats["month_idx"] = month_index(stats["month"])
    return stats.sort_values(["product", "month_idx"]).reset_index(drop=True)


def build_history_tables(
    monthly_company: pd.DataFrame,
    monthly_product: pd.DataFrame,
    global_rate: float,
) -> dict[str, pd.DataFrame]:
    """Company and product expanding-history lookup tables.

    Takes monthly aggregate frames with columns [key, month, n, relief]
    (from monthly_company_stats / monthly_product_stats on a frame, or
    relief_radar.data.monthly_stats on the Arrow population table).
    """
    for frame in (monthly_company, monthly_product):
        if "month_idx" not in frame.columns:
            frame["month_idx"] = month_index(frame["month"])
    company = _expanding_prior(monthly_company, ["company_norm"])
    product = _expanding_prior(monthly_product, ["product"])
    product["prior_rate"] = product["prior_relief"] / product["prior_n"]
    product["prior_rate"] = product["prior_rate"].fillna(global_rate)
    return {"company": company, "product": product}


def add_features(
    df: pd.DataFrame, history: dict[str, pd.DataFrame], prior_strength: float = 20.0
) -> pd.DataFrame:
    """Attach receipt-time features to a complaints frame."""
    out = df.copy()
    out["month_idx"] = month_index(out["month"])
    if "month_of_year" not in out.columns or "day_of_week" not in out.columns:
        dates = pd.to_datetime(out["date_received"])
        out["month_of_year"] = dates.dt.month.astype("int16")
        out["day_of_week"] = dates.dt.dayofweek.astype("int16")

    company = history["company"][
        ["company_norm", "month_idx", "prior_n", "prior_relief", "prev_month_n"]
    ]
    out = out.merge(company, on=["company_norm", "month_idx"], how="left")
    product = history["product"][
        ["product", "month_idx", "prior_rate", "prev_month_n"]
    ].rename(
        columns={
            "prior_rate": "product_prior_relief_rate",
            "prev_month_n": "product_prev_month_n",
        }
    )
    out = out.merge(product, on=["product", "month_idx"], how="left")

    out["prior_n"] = out["prior_n"].fillna(0)
    out["prior_relief"] = out["prior_relief"].fillna(0)
    out["prev_month_n"] = out["prev_month_n"].fillna(0)
    out["company_seen_before"] = (out["prior_n"] > 0).astype("int8")
    out["company_log_prior_volume"] = np.log1p(out["prior_n"])
    out["company_log_prev_month_volume"] = np.log1p(out["prev_month_n"])
    out["company_prior_relief_rate"] = (
        out["prior_relief"] + prior_strength * out["product_prior_relief_rate"]
    ) / (out["prior_n"] + prior_strength)
    out["product_log_prev_month_volume"] = np.log1p(
        out["product_prev_month_n"].fillna(0)
    )
    return out


def fit_category_caps(
    train: pd.DataFrame, columns: list[str], top_k: int
) -> dict[str, list[str]]:
    """Top-K category values per column, fitted on training data only."""
    caps: dict[str, list[str]] = {}
    for col in columns:
        caps[col] = train[col].value_counts().head(top_k).index.tolist()
    return caps


def apply_category_caps(df: pd.DataFrame, caps: dict[str, list[str]]) -> pd.DataFrame:
    """Replace out-of-cap categories with __OTHER__."""
    out = df.copy()
    for col, keep in caps.items():
        out[col] = out[col].where(out[col].isin(keep), "__OTHER__")
    return out
