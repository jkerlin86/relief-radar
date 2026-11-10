"""Loading processed Relief Radar datasets.

The processed layer is written by scripts/build_dataset.py:
- complaints.parquet: one row per complaint in the pinned CFPB snapshot,
  with the modelling population flags and target derived from the raw
  "Company response to consumer" value.
- narratives.parquet: one row per narrative-bearing complaint from the
  CFPB FOIA Reading Room archive, keyed by Complaint ID.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from relief_radar.config import project_path

COMPLAINTS_PATH = ("data", "processed", "complaints.parquet")


def load_frame_light(columns: list[str]) -> pd.DataFrame:
    """Memory-bounded load of selected columns over the full snapshot.

    Arrow-backed strings, dictionary-encoded to pandas categories.
    Object-dtype loading of the full 19M-row frame OOM-kills this
    machine; this loader peaks well under the limit.
    """
    df = pd.read_parquet(
        project_path(*COMPLAINTS_PATH), columns=columns, dtype_backend="pyarrow"
    )
    for col in df.columns:
        if str(df[col].dtype).startswith("string[pyarrow]") and col != "date_received":
            df[col] = df[col].astype("category")
    return df


def load_primary_arrow(extra_columns: tuple[str, ...] = ()):
    """Pass-1 frame: primary-population rows as an Arrow table.

    Only light columns, string columns dictionary-encoded. Pandas
    conversion of the full population OOM-kills this machine even in
    Arrow-backed form (measured 3.2 GB peak for 8 columns), so all
    population-level aggregation happens in Arrow and only small
    results cross into pandas.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    columns = [
        "complaint_id",
        "date_received",
        "year",
        "month",
        "product",
        "company_norm",
        "target",
        "in_primary_population",
        *extra_columns,
    ]
    dataset = ds.dataset(project_path(*COMPLAINTS_PATH), format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("in_primary_population") == True,
    )
    table = table.drop_columns(["in_primary_population"])
    for name in table.column_names:
        if pa.types.is_string(table[name].type):
            table = table.set_column(
                table.column_names.index(name),
                name,
                pc.cast(
                    table[name],
                    pa.dictionary(index_type=pa.int32(), value_type=pa.string()),
                ),
            )
    return table


def monthly_stats(table, key: str) -> pd.DataFrame:
    """Monthly complaint counts and positives by a key column (Arrow)."""
    grouped = table.group_by([key, "month"]).aggregate(
        [("target", "sum"), ("target", "count")]
    )
    out = grouped.to_pandas()
    return out.rename(columns={"target_sum": "relief", "target_count": "n"})


def fetch_rows_by_ids(ids: set[int], columns: list[str]) -> pd.DataFrame:
    """Fetch full rows for a set of Complaint IDs, batch by batch."""
    pf = pq.ParquetFile(project_path(*COMPLAINTS_PATH))
    frames = []
    for batch in pf.iter_batches(columns=columns, batch_size=100_000):
        part = batch.to_pandas()
        mask = part["complaint_id"].isin(ids)
        if mask.any():
            frames.append(part[mask])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)

COMPLAINT_COLUMNS = [
    "complaint_id",
    "date_received",
    "year",
    "month",
    "month_of_year",
    "day_of_week",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "company_norm",
    "state",
    "zip3",
    "zip_missing",
    "tag_older_american",
    "tag_servicemember",
    "submitted_via",
    "response",
    "target",
    "in_primary_population",
]


def load_complaints(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the processed complaints parquet."""
    return pd.read_parquet(
        project_path("data", "processed", "complaints.parquet"),
        columns=columns or COMPLAINT_COLUMNS,
    )


def load_narratives() -> pd.DataFrame:
    """Load the processed narratives parquet (complaint_id, narrative)."""
    return pd.read_parquet(project_path("data", "processed", "narratives.parquet"))


def primary_population(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the primary modelling population (target observed)."""
    return df[df["in_primary_population"]].copy()
