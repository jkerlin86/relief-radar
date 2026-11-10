"""Narrative-selection audit computations.

Question: how does the narrative-bearing minority of complaints differ
from all complaints? The audit window is restricted to complaints
received while narrative publication existed (2015-01-01 to 2026-08-14);
outside that window a complaint could not have carried a published
narrative, so including it would manufacture a selection effect.
"""

from __future__ import annotations

import pandas as pd

AUDIT_START = "2015-01-01"
AUDIT_END = "2026-08-14"  # last day of narrative publication


def add_narrative_flag(df: pd.DataFrame, narrative_ids: pd.Series) -> pd.DataFrame:
    """Attach has_narrative by Complaint ID join."""
    out = df.copy()
    out["has_narrative"] = out["complaint_id"].isin(set(narrative_ids)).astype("int8")
    return out


def rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Narrative share and relief rates by group."""
    table = (
        df.groupby(group_col, observed=True)
        .agg(
            n=("complaint_id", "size"),
            narrative_n=("has_narrative", "sum"),
            relief_rate_all=("target", "mean"),
        )
        .reset_index()
    )
    narr = df[df["has_narrative"] == 1].groupby(group_col, observed=True)["target"].mean()
    plain = df[df["has_narrative"] == 0].groupby(group_col, observed=True)["target"].mean()
    table["narrative_share"] = table["narrative_n"] / table["n"]
    table["relief_rate_narrative"] = table[group_col].map(narr)
    table["relief_rate_no_narrative"] = table[group_col].map(plain)
    table["relief_rate_diff"] = (
        table["relief_rate_narrative"] - table["relief_rate_no_narrative"]
    )
    return table.sort_values("n", ascending=False)


def rates_from_aggregates(agg: pd.DataFrame) -> pd.DataFrame:
    """Audit table from pre-aggregated counts (e.g. Arrow group-bys).

    Input columns beyond the group keys: n, narrative_n, relief_with,
    relief_without. Output mirrors rate_table's columns so the two
    paths cannot drift apart semantically.
    """
    out = agg.copy()
    out["narrative_share"] = out["narrative_n"] / out["n"]
    with_n = out["narrative_n"]
    without_n = out["n"] - out["narrative_n"]
    out["relief_rate_all"] = (out["relief_with"] + out["relief_without"]) / out["n"]
    out["relief_rate_narrative"] = out["relief_with"] / with_n.where(with_n > 0)
    out["relief_rate_no_narrative"] = out["relief_without"] / without_n.where(
        without_n > 0
    )
    out["relief_rate_diff"] = (
        out["relief_rate_narrative"] - out["relief_rate_no_narrative"]
    )
    return out.sort_values("n", ascending=False)


def tier_labels(totals: pd.Series, n_tiers: int = 5) -> pd.DataFrame:
    """Company size-tier labels from per-company totals (rank quintiles)."""
    ranked = totals.rank(method="first")
    tiers = pd.qcut(ranked, q=n_tiers, labels=False) + 1
    return pd.DataFrame({"company_norm": totals.index, "size_tier": tiers.to_numpy()})


def company_size_tiers(df: pd.DataFrame, n_tiers: int = 5) -> pd.DataFrame:
    """Assign each company a complaint-volume tier (1 = smallest)."""
    totals = df.groupby("company_norm", observed=True)["complaint_id"].size()
    ranked = totals.rank(method="first")
    tiers = pd.qcut(ranked, q=n_tiers, labels=False) + 1
    mapping = pd.DataFrame(
        {"company_norm": totals.index, "size_tier": tiers.to_numpy()}
    )
    return df.merge(mapping, on="company_norm", how="left")
