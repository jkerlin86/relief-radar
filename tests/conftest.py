"""Synthetic fixtures: tiny hand-checkable frames, no real data."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def complaints_frame() -> pd.DataFrame:
    """12 complaints across 3 months, 2 companies, 2 products."""
    rows = []
    specs = [
        # (month, company, product, target)
        (202401, "ACME BANK", "Credit card", 1),
        (202401, "ACME BANK", "Credit card", 0),
        (202401, "ACME BANK", "Mortgage", 0),
        (202401, "BETA FINANCIAL", "Credit card", 0),
        (202402, "ACME BANK", "Credit card", 1),
        (202402, "ACME BANK", "Credit card", 0),
        (202402, "BETA FINANCIAL", "Mortgage", 1),
        (202402, "BETA FINANCIAL", "Mortgage", 0),
        (202403, "ACME BANK", "Credit card", 0),
        (202403, "ACME BANK", "Mortgage", 1),
        (202403, "BETA FINANCIAL", "Credit card", 0),
        (202403, "BETA FINANCIAL", "Credit card", 0),
    ]
    for i, (month, company, product, target) in enumerate(specs):
        year, mm = divmod(month, 100)
        rows.append(
            {
                "complaint_id": 1000 + i,
                "date_received": f"{year}-{mm:02d}-15",
                "year": year,
                "month": month,
                "month_of_year": mm,
                "day_of_week": 2,
                "product": product,
                "sub_product": "",
                "issue": "Billing",
                "sub_issue": "",
                "company": company,
                "company_norm": company,
                "state": "FL",
                "zip3": "333",
                "zip_missing": 0,
                "tag_older_american": 0,
                "tag_servicemember": 0,
                "submitted_via": "Web",
                "response": "Closed with monetary relief"
                if target
                else "Closed with explanation",
                "target": target,
                "in_primary_population": True,
            }
        )
    return pd.DataFrame(rows)
