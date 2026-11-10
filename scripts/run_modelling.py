"""Structured-outcome modelling: temporal split, baselines first.

Protocol (docs/leakage_audit.md):
- Primary population only (target observed, unambiguous closure).
- Train: received before 2024-01-01. Buffer: calendar 2024 (base-rate
  drift check only). Test: received on or after 2025-01-01.
- Uniform random samples within era (fixed seed) bound compute; era
  population counts and base rates are computed on the full data and
  reported alongside, so sample metrics can be read against them.
- History features are expanding aggregates over strictly earlier
  months, computed from the full primary population.

Outputs: reports/parts/model_results.json, calibration CSVs, test
predictions under data/processed/, fitted pipelines under models/.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from relief_radar.config import load_config, project_path
from relief_radar.evaluate import binary_metrics, calibration_table, subgroup_metrics
from relief_radar.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    apply_category_caps,
    build_history_tables,
    fit_category_caps,
)
from relief_radar.models import make_histgb_pipeline, make_logistic_pipeline

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


def main() -> None:
    cfg = load_config()
    split = cfg["split"]
    rng = np.random.default_rng(cfg["seed"])

    # Two-pass memory discipline: pass 1 aggregates the full primary
    # population in Arrow (pandas conversion of 18M rows OOM-kills
    # this machine; measured and recorded in the build log); pass 2
    # fetches full feature rows for the sampled IDs only.
    import pyarrow.compute as pc

    from relief_radar.data import (
        fetch_rows_by_ids,
        load_primary_arrow,
        monthly_stats,
    )

    table = load_primary_arrow()
    n_rows = table.num_rows
    positives = int(pc.sum(table["target"]).as_py())
    base_rate = positives / n_rows
    era = (
        table.group_by(["year"])
        .aggregate([("target", "sum"), ("target", "count")])
        .to_pandas()
    )
    era_rates = {
        int(r.year): {
            "n": int(r.target_count),
            "relief_rate": float(r.target_sum / r.target_count),
        }
        for r in era.itertuples()
    }
    history = build_history_tables(
        monthly_stats(table, "company_norm"),
        monthly_stats(table, "product"),
        base_rate,
    )
    years = table["year"].to_numpy()
    targets = table["target"].to_numpy()
    ids_all = table["complaint_id"].to_numpy()
    del table
    # Era boundaries fall on year lines: train_end 2024-01-01 means
    # year <= 2023; test_start 2025-01-01 means year >= 2025.
    train_mask = years <= 2023
    test_mask = years >= 2025

    results: dict = {
        "protocol": {
            "train": f"date_received < {split['train_end']}",
            "buffer_year": split["validation_year"],
            "test": f"date_received >= {split['test_start']}",
            "train_sample_n": split["train_sample_n"],
            "test_sample_n": split["test_sample_n"],
            "seed": cfg["seed"],
        },
        "population": {
            "primary_rows": n_rows,
            "primary_positives": positives,
            "primary_base_rate": base_rate,
            "train_era_rows": int(train_mask.sum()),
            "test_era_rows": int(test_mask.sum()),
            "train_era_base_rate": float(targets[train_mask].mean()),
            "test_era_base_rate": float(targets[test_mask].mean()),
        },
        "era_base_rates_full_population": era_rates,
    }

    def sample_ids(mask: np.ndarray, n: int) -> set[int]:
        pool = ids_all[mask]
        if len(pool) <= n:
            return set(pool.tolist())
        return set(rng.choice(pool, size=n, replace=False).tolist())

    train_ids = sample_ids(train_mask, split["train_sample_n"])
    test_ids = sample_ids(test_mask, split["test_sample_n"])
    del years, targets, train_mask, test_mask

    sampled = fetch_rows_by_ids(train_ids | test_ids, LOAD_COLUMNS)
    train = add_features(
        sampled[sampled["complaint_id"].isin(train_ids)].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=cfg["models"]["company_prior_strength"],
    )
    test = add_features(
        sampled[sampled["complaint_id"].isin(test_ids)].drop(
            columns=["in_primary_population"]
        ),
        history,
        prior_strength=cfg["models"]["company_prior_strength"],
    )
    del sampled

    caps = fit_category_caps(
        train, ["sub_product", "issue", "sub_issue"], cfg["models"]["top_k_categories"]
    )
    train = apply_category_caps(train, caps)
    test = apply_category_caps(test, caps)
    results["category_caps"] = {k: len(v) for k, v in caps.items()}

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    x_train, y_train = train[feature_cols], train["target"].astype(int)
    x_test, y_test = test[feature_cols], test["target"].astype(int)
    results["samples"] = {
        "train_n": len(train),
        "train_base_rate": float(y_train.mean()),
        "test_n": len(test),
        "test_base_rate": float(y_test.mean()),
    }

    models = {
        "logistic_regression": make_logistic_pipeline(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, cfg["models"]["logistic"]
        ),
        "hist_gradient_boosting": make_histgb_pipeline(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, cfg["models"]["hist_gb"]
        ),
        "logistic_regression_balanced_sensitivity": make_logistic_pipeline(
            CATEGORICAL_FEATURES,
            NUMERIC_FEATURES,
            cfg["models"]["logistic"],
            class_weight="balanced",
        ),
    }

    parts = project_path("reports", "parts")
    results["models"] = {}
    predictions = pd.DataFrame(
        {"complaint_id": test["complaint_id"], "target": y_test.to_numpy()}
    )
    for name, pipe in models.items():
        pipe.fit(x_train, y_train)
        prob = pipe.predict_proba(x_test)[:, 1]
        predictions[f"prob_{name}"] = prob
        entry = {"test": binary_metrics(y_test, prob)}
        calib = calibration_table(y_test, prob)
        calib.to_csv(parts / f"calibration_{name}.csv", index=False)
        entry["calibration_max_abs_gap"] = float(
            (calib["mean_predicted"] - calib["observed_rate"]).abs().max()
        )
        results["models"][name] = entry
        joblib.dump(pipe, project_path("models", f"{name}.joblib"))

    main_model = "hist_gradient_boosting"
    test_scored = test[["product", "year", "submitted_via"]].copy()
    test_scored["target"] = y_test.to_numpy()
    test_scored["prob"] = predictions[f"prob_{main_model}"].to_numpy()
    for group in ["product", "year", "submitted_via"]:
        table = subgroup_metrics(test_scored, group)
        table.to_csv(parts / f"subgroups_{main_model}_{group}.csv", index=False)
        results["models"][main_model][f"subgroups_{group}"] = table.to_dict(
            orient="records"
        )

    predictions.to_parquet(
        project_path("data", "processed", "test_predictions.parquet"), index=False
    )
    with open(parts / "model_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(json.dumps(results["models"], indent=1))
    print(json.dumps(results["samples"], indent=1))


if __name__ == "__main__":
    main()
