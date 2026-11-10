"""Regenerate the exact modelling extracts for the clean-data archive.

The owner archive rule (2026-11-10) archives the capped samples the
published results were actually computed from, not the multi-GB
processed population. This script replays the identical sampling code
paths as run_modelling.py and run_narrative_arm.py (same seed, same
call order) and verifies the regenerated tables against artefacts the
original runs left behind:

- structured test sample: Complaint ID set and targets must match
  data/processed/test_predictions.parquet row for row, and the sample
  base rates must equal reports/parts/model_results.json exactly;
- narrative sample: IDs come from the encode sidecar (the exact lists
  the embeddings were computed for) and sample base rates must equal
  the sidecar values.

Outputs (to --out, default the portfolio clean-data-archive staging):
structured_train_sample.parquet, structured_test_sample.parquet,
narrative_arm_sample.parquet, narrative_embeddings.npz (copied).
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd

from relief_radar.config import load_config, project_path
from relief_radar.data import fetch_rows_by_ids, load_primary_arrow, monthly_stats
from relief_radar.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    apply_category_caps,
    build_history_tables,
    fit_category_caps,
)

LOAD_COLUMNS = [
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
    "company_norm",
    "state",
    "zip_missing",
    "tag_older_american",
    "tag_servicemember",
    "submitted_via",
    "target",
    "in_primary_population",
]
KEEP = (
    ["complaint_id", "date_received", "month"]
    + CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + ["target"]
)
CAP_COLS = ["sub_product", "issue", "sub_issue"]


def structured_extracts(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyarrow.compute as pc

    split = cfg["split"]
    rng = np.random.default_rng(cfg["seed"])
    table = load_primary_arrow()
    positives = int(pc.sum(table["target"]).as_py())
    history = build_history_tables(
        monthly_stats(table, "company_norm"),
        monthly_stats(table, "product"),
        positives / table.num_rows,
    )
    years = table["year"].to_numpy()
    ids_all = table["complaint_id"].to_numpy()
    del table
    train_mask = years <= 2023
    test_mask = years >= 2025

    def sample_ids(mask: np.ndarray, n: int) -> set[int]:
        pool = ids_all[mask]
        if len(pool) <= n:
            return set(pool.tolist())
        return set(rng.choice(pool, size=n, replace=False).tolist())

    train_ids = sample_ids(train_mask, split["train_sample_n"])
    test_ids = sample_ids(test_mask, split["test_sample_n"])
    del years, train_mask, test_mask

    sampled = fetch_rows_by_ids(train_ids | test_ids, LOAD_COLUMNS)
    strength = cfg["models"]["company_prior_strength"]
    train = add_features(
        sampled[sampled["complaint_id"].isin(train_ids)].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=strength,
    )
    test = add_features(
        sampled[sampled["complaint_id"].isin(test_ids)].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=strength,
    )
    del sampled
    caps = fit_category_caps(train, CAP_COLS, cfg["models"]["top_k_categories"])
    train = apply_category_caps(train, caps)
    test = apply_category_caps(test, caps)
    return train[KEEP], test[KEEP]


def verify_structured(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    reference = json.loads(
        project_path("reports", "parts", "model_results.json").read_text()
    )
    samples = reference["samples"]
    assert len(train) == samples["train_n"], (len(train), samples["train_n"])
    assert len(test) == samples["test_n"], (len(test), samples["test_n"])
    assert abs(train["target"].mean() - samples["train_base_rate"]) < 1e-12
    assert abs(test["target"].mean() - samples["test_base_rate"]) < 1e-12
    preds = pd.read_parquet(
        project_path("data", "processed", "test_predictions.parquet"),
        columns=["complaint_id", "target"],
    )
    merged = test[["complaint_id", "target"]].merge(
        preds, on="complaint_id", suffixes=("", "_pred"), validate="one_to_one"
    )
    assert len(merged) == len(test)
    assert (merged["target"] == merged["target_pred"]).all()
    return {
        "train_n": len(train),
        "test_n": len(test),
        "test_positives": int(test["target"].sum()),
        "verified_against": "model_results.json samples + test_predictions.parquet",
    }


def narrative_extract(cfg: dict) -> pd.DataFrame:
    sidecar = json.loads(
        project_path("data", "processed", "narrative_encode_sidecar.json").read_text()
    )
    train_ids = sidecar["train_ids"]
    test_ids = sidecar["test_ids"]
    pass1 = project_path("data", "processed", "narrative_pass1")
    history = {
        "company": pd.read_parquet(pass1 / "history_company.parquet"),
        "product": pd.read_parquet(pass1 / "history_product.parquet"),
    }
    full = fetch_rows_by_ids(set(train_ids) | set(test_ids), LOAD_COLUMNS)
    strength = cfg["models"]["company_prior_strength"]
    train = add_features(
        full[full["complaint_id"].isin(set(train_ids))].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=strength,
    )
    test = add_features(
        full[full["complaint_id"].isin(set(test_ids))].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=strength,
    )
    del full
    caps = fit_category_caps(train, CAP_COLS, cfg["models"]["top_k_categories"])
    train = apply_category_caps(train, caps)
    test = apply_category_caps(test, caps)
    train = train[KEEP].assign(split="train")
    test = test[KEEP].assign(split="test")
    out = pd.concat([train, test], ignore_index=True)
    stats = sidecar["samples"]
    assert len(train) == stats["train_n_with_text"]
    assert len(test) == stats["test_n_with_text"]
    assert abs(train["target"].mean() - stats["train_base_rate"]) < 1e-9
    assert abs(test["target"].mean() - stats["test_base_rate"]) < 1e-9
    emb = np.load(project_path("data", "processed", "narrative_embeddings.npz"))
    assert emb["train_ids"].tolist() == train_ids
    assert emb["test_ids"].tolist() == test_ids
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(
            project_path("..", "..", "clean-data-archive", "D1_relief_radar")
        ),
    )
    args = parser.parse_args()
    from pathlib import Path

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    train, test = structured_extracts(cfg)
    check = verify_structured(train, test)
    train.to_parquet(out / "structured_train_sample.parquet", index=False)
    test.to_parquet(out / "structured_test_sample.parquet", index=False)
    print("structured extracts verified:", json.dumps(check), flush=True)

    narrative = narrative_extract(cfg)
    narrative.to_parquet(out / "narrative_arm_sample.parquet", index=False)
    shutil.copyfile(
        project_path("data", "processed", "narrative_embeddings.npz"),
        out / "narrative_embeddings.npz",
    )
    print(
        "narrative extract verified:",
        len(narrative),
        "rows (train",
        int((narrative["split"] == "train").sum()),
        "/ test",
        int((narrative["split"] == "test").sum()),
        ")",
        flush=True,
    )


if __name__ == "__main__":
    main()
