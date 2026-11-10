"""Narrative-selection audit: who writes, and does it matter?

Joins the FOIA narrative IDs onto the primary population, restricts to
the publication window (2015-01-01..2026-08-14), and measures narrative
share and relief-rate differences by era, product, channel, and company
size. All aggregation runs in Arrow over the full population (pandas
conversion of 18M rows exceeds this machine's memory); only small
aggregate tables cross into pandas, through the same tested rate math
(relief_radar.selection.rates_from_aggregates).

Outputs reports/parts/selection_audit.json and CSV tables.
"""

from __future__ import annotations

import json

import pandas as pd

from relief_radar.config import project_path
from relief_radar.selection import AUDIT_END, AUDIT_START, rates_from_aggregates, tier_labels


def _agg(table, keys: list[str]) -> pd.DataFrame:
    grouped = table.group_by(keys).aggregate(
        [
            ("complaint_id", "count"),
            ("narr_flag", "sum"),
            ("relief_with", "sum"),
            ("relief_without", "sum"),
        ]
    ).to_pandas()
    return grouped.rename(
        columns={
            "complaint_id_count": "n",
            "narr_flag_sum": "narrative_n",
            "relief_with_sum": "relief_with",
            "relief_without_sum": "relief_without",
        }
    )


def main() -> None:
    import pyarrow as pa
    import pyarrow.compute as pc

    from relief_radar.data import load_primary_arrow

    table = load_primary_arrow(extra_columns=("submitted_via",))
    dates = pc.cast(table["date_received"], pa.string())
    window = table.filter(
        pc.and_(pc.greater_equal(dates, AUDIT_START), pc.less_equal(dates, AUDIT_END))
    )
    del table
    narrative_ids = pd.read_parquet(
        project_path("data", "processed", "narratives.parquet"),
        columns=["complaint_id"],
    )["complaint_id"]
    flag = pc.is_in(
        window["complaint_id"], value_set=pa.array(narrative_ids.to_numpy())
    )
    window = window.append_column("narr_flag", pc.cast(flag, pa.int32()))
    target = pc.cast(window["target"], pa.int32())
    relief_with = pc.multiply(target, window["narr_flag"])
    window = window.append_column("relief_with", relief_with)
    window = window.append_column("relief_without", pc.subtract(target, relief_with))

    n_window = window.num_rows
    n_narr = int(pc.sum(window["narr_flag"]).as_py())
    relief_narr = int(pc.sum(window["relief_with"]).as_py())
    relief_plain = int(pc.sum(window["relief_without"]).as_py())
    out: dict = {
        "window": {"start": AUDIT_START, "end": AUDIT_END},
        "population": {
            "primary_rows_in_window": n_window,
            "narrative_rows_in_window": n_narr,
            "narrative_share": n_narr / n_window,
            "relief_rate_narrative": relief_narr / n_narr,
            "relief_rate_no_narrative": relief_plain / (n_window - n_narr),
        },
    }

    parts = project_path("reports", "parts")
    tables = {}
    for group in ["year", "product", "submitted_via"]:
        table_out = rates_from_aggregates(_agg(window, [group]))
        table_out.to_csv(parts / f"selection_by_{group}.csv", index=False)
        tables[group] = table_out.to_dict(orient="records")

    # Company size tiers from per-company window totals (rank quintiles).
    totals = (
        window.group_by(["company_norm"])
        .aggregate([("complaint_id", "count")])
        .to_pandas()
    )
    labels = tier_labels(
        totals.set_index("company_norm")["complaint_id_count"], n_tiers=5
    )
    company_agg = _agg(window, ["company_norm"]).merge(labels, on="company_norm")
    tier_agg = (
        company_agg.groupby("size_tier", observed=True)[
            ["n", "narrative_n", "relief_with", "relief_without"]
        ]
        .sum()
        .reset_index()
    )
    tier_table = rates_from_aggregates(tier_agg)
    tier_table.to_csv(parts / "selection_by_size_tier.csv", index=False)
    tables["size_tier"] = tier_table.to_dict(orient="records")
    out["tables"] = tables

    # Within-cell check: relief-rate difference inside year x product
    # cells (volume-weighted), so era/product mix cannot manufacture it.
    cells = rates_from_aggregates(_agg(window, ["year", "product"]))
    cells = cells[(cells["narrative_n"] > 0) & (cells["narrative_n"] < cells["n"])]
    if len(cells):
        diff = cells["relief_rate_diff"]
        weights = cells["n"]
        out["within_cell_relief_diff"] = {
            "cells": len(cells),
            "volume_weighted_mean_diff": float(
                (diff * weights).sum() / weights.sum()
            ),
            "median_diff": float(diff.median()),
            "share_cells_narrative_higher": float((diff > 0).mean()),
        }
    with open(parts / "selection_audit.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out["population"], indent=1))
    print(json.dumps(out.get("within_cell_relief_diff", {}), indent=1))


if __name__ == "__main__":
    main()
