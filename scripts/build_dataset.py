"""Build the processed datasets from the pinned raw downloads.

Outputs:
- data/processed/complaints.parquet — one row per complaint in the CFPB
  bulk snapshot, with target and population flags derived from the raw
  response value (mapping in configs/default.json).
- data/processed/narratives.parquet — one row per narrative-bearing
  complaint from the FOIA Reading Room archive exports (non-empty
  narrative text only), keyed by Complaint ID.
- reports/parts/dataset_build.json — measured build facts.

The bulk CSV is ~5.8 GB uncompressed; it is streamed in chunks and never
held in memory in full.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from relief_radar.config import load_config, project_path
from relief_radar.names import normalize_company

RAW_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Company response to consumer",
    "Complaint ID",
]


def _transform_chunk(chunk: pd.DataFrame, cfg: dict, norm_cache: dict) -> pd.DataFrame:
    pop = cfg["population"]
    # The snapshot mixes ISO (YYYY-MM-DD) and legacy US (MM/DD/YYYY)
    # date strings; parse per element. Verified from the file itself:
    # a strict "%m/%d/%Y" parse fails on ISO rows and vice versa.
    dates = pd.to_datetime(chunk["Date received"], format="mixed")
    response = chunk["Company response to consumer"].fillna("(missing)")
    target = pd.Series(-1, index=chunk.index, dtype="int8")
    target[response == pop["positive_response"]] = 1
    target[response.isin(pop["negative_responses"])] = 0
    companies = chunk["Company"].fillna("")
    normed = companies.map(lambda c: norm_cache.setdefault(c, normalize_company(c)))
    tags = chunk["Tags"].fillna("")
    zipc = chunk["ZIP code"].fillna("")
    out = pd.DataFrame(
        {
            "complaint_id": chunk["Complaint ID"].astype("int64"),
            "date_received": dates.dt.date.astype(str),
            "year": dates.dt.year.astype("int16"),
            "month": (dates.dt.year * 100 + dates.dt.month).astype("int32"),
            "month_of_year": dates.dt.month.astype("int8"),
            "day_of_week": dates.dt.dayofweek.astype("int8"),
            "product": chunk["Product"].fillna("(missing)").astype(str),
            "sub_product": chunk["Sub-product"].fillna("").astype(str),
            "issue": chunk["Issue"].fillna("").astype(str),
            "sub_issue": chunk["Sub-issue"].fillna("").astype(str),
            "company": companies.astype(str),
            "company_norm": normed.astype(str),
            "state": chunk["State"].fillna("").astype(str),
            "zip3": zipc.str.slice(0, 3),
            "zip_missing": (zipc.str.len() < 5).astype("int8"),
            "tag_older_american": tags.str.contains("Older American").astype("int8"),
            "tag_servicemember": tags.str.contains("Servicemember").astype("int8"),
            "submitted_via": chunk["Submitted via"].fillna("(missing)").astype(str),
            "response": response.astype(str),
            "target": target,
            "in_primary_population": (target >= 0).astype(bool),
        }
    )
    return out


def build_complaints(cfg: dict) -> dict:
    zip_path = project_path(cfg["snapshot"]["bulk_zip"])
    out_path = project_path("data", "processed", "complaints.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    norm_cache: dict[str, str] = {}
    stats = Counter()
    years = Counter()
    date_min: str | None = None
    date_max: str | None = None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as fh:
                reader = pd.read_csv(
                    fh, usecols=RAW_COLUMNS, dtype=str, chunksize=300_000
                )
                for chunk in reader:
                    out = _transform_chunk(chunk, cfg, norm_cache)
                    stats["rows"] += len(out)
                    stats["primary_rows"] += int(out["in_primary_population"].sum())
                    stats["positives"] += int((out["target"] == 1).sum())
                    stats["negatives"] += int((out["target"] == 0).sum())
                    lo, hi = out["date_received"].min(), out["date_received"].max()
                    date_min = lo if date_min is None or lo < date_min else date_min
                    date_max = hi if date_max is None or hi > date_max else date_max
                    years.update(out["year"].astype(int).tolist())
                    table = pa.Table.from_pandas(out, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {
        "date_min_parsed": date_min,
        "date_max_parsed": date_max,
        "rows": stats["rows"],
        "primary_population_rows": stats["primary_rows"],
        "positives_monetary_relief": stats["positives"],
        "negatives_other_closures": stats["negatives"],
        "unique_companies_raw": len(norm_cache),
        "rows_by_year": dict(sorted(years.items())),
    }


def _narrative_chunks(zpath: Path):
    """Yield (rows_seen, kept_frame) per chunk of one FOIA export zip."""
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with zf.open(member) as fh:
                reader = pd.read_csv(
                    fh,
                    usecols=["Complaint ID", "Consumer complaint narrative"],
                    dtype=str,
                    chunksize=300_000,
                )
                for chunk in reader:
                    text = chunk["Consumer complaint narrative"]
                    keep = text.notna() & (text.str.strip() != "")
                    part = pd.DataFrame(
                        {
                            "complaint_id": chunk.loc[keep, "Complaint ID"].astype(
                                "int64"
                            ),
                            "narrative": text[keep].astype(str),
                        }
                    )
                    yield len(chunk), part


def build_narratives(cfg: dict) -> dict:
    """Stream the FOIA exports into narratives.parquet.

    Two passes keep memory bounded: pass 1 collects kept Complaint IDs
    per file (ints only); pass 2 writes text for IDs not already written
    by an earlier file (exports are chronological windows; duplicates
    can only occur at window boundaries).
    """
    import numpy as np

    foia_dir = project_path(cfg["snapshot"]["foia_dir"])
    out_path = project_path("data", "processed", "narratives.parquet")
    files = sorted(Path(foia_dir).glob("*.zip"))
    per_file: dict[str, dict] = {}
    ids_by_file: dict[str, np.ndarray] = {}
    total_before = 0
    for zpath in files:
        rows = 0
        ids: list[np.ndarray] = []
        for n, part in _narrative_chunks(zpath):
            rows += n
            ids.append(part["complaint_id"].to_numpy())
        merged = np.concatenate(ids) if ids else np.array([], dtype="int64")
        ids_by_file[zpath.name] = merged
        total_before += len(merged)
        per_file[zpath.name] = {"rows_in_file": rows, "narratives_kept": len(merged)}

    writer: pq.ParquetWriter | None = None
    seen = np.array([], dtype="int64")
    written = 0
    try:
        for zpath in files:
            file_ids = np.sort(ids_by_file[zpath.name])
            fresh_mask = np.isin(file_ids, seen, assume_unique=False, invert=True)
            fresh = set(file_ids[fresh_mask].tolist())
            per_file[zpath.name]["duplicates_dropped"] = int((~fresh_mask).sum())
            for _n, part in _narrative_chunks(zpath):
                part = part[part["complaint_id"].isin(fresh)]
                part = part.drop_duplicates(subset="complaint_id", keep="first")
                if len(part):
                    table = pa.Table.from_pandas(part, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    written += len(part)
            seen = np.union1d(seen, file_ids)
    finally:
        if writer is not None:
            writer.close()
    return {
        "narrative_rows_before_dedup": total_before,
        "narrative_rows": written,
        "duplicate_ids_dropped": total_before - written,
        "per_file": per_file,
    }


def main() -> None:
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    out = project_path("reports", "parts", "dataset_build.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    facts: dict = {}
    if out.exists():
        with open(out, encoding="utf-8") as fh:
            facts = json.load(fh)
    if stage in ("all", "complaints"):
        facts["complaints"] = build_complaints(cfg)
    if stage in ("all", "narratives"):
        facts["narratives"] = build_narratives(cfg)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1)
    print(json.dumps(facts, indent=1)[:2000])


if __name__ == "__main__":
    main()
