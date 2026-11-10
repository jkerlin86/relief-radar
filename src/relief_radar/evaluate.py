"""Evaluation metrics for imbalanced binary outcomes.

Headline metrics are ranking (ROC AUC, PR AUC) and probability quality
(Brier score, log loss, calibration tables). Accuracy is never reported:
at a ~1% base rate it is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob) -> dict:
    """Core metric bundle for one model on one evaluation set."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    out: dict = {
        "n": len(y),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(
            log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])
        ),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def calibration_table(y_true, y_prob, n_bins: int = 20) -> pd.DataFrame:
    """Quantile-binned calibration: mean prediction vs observed rate."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob, dtype=float)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    table = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_rate=("y", "mean"),
            min_predicted=("p", "min"),
            max_predicted=("p", "max"),
        )
        .reset_index(drop=True)
    )
    return table


def subgroup_metrics(
    frame: pd.DataFrame, group_col: str, y_col: str = "target", p_col: str = "prob"
) -> pd.DataFrame:
    """binary_metrics per subgroup; AUC fields are null when undefined."""
    rows = []
    for value, part in frame.groupby(group_col, observed=True):
        m = binary_metrics(part[y_col], part[p_col])
        m[group_col] = value
        rows.append(m)
    cols = [group_col] + [c for c in rows[0] if c != group_col] if rows else [group_col]
    return pd.DataFrame(rows)[cols].sort_values("n", ascending=False)
