import numpy as np
import pandas as pd

from relief_radar.evaluate import binary_metrics, calibration_table, subgroup_metrics


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = binary_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["base_rate"] == 0.5
    assert m["positives"] == 2


def test_constant_prediction_brier():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    m = binary_metrics(y, p)
    assert abs(m["brier"] - 0.25) < 1e-9
    assert m["roc_auc"] == 0.5


def test_single_class_auc_is_null():
    m = binary_metrics(np.zeros(5), np.full(5, 0.1))
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None


def test_calibration_table_sums():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    p = rng.random(500)
    table = calibration_table(y, p, n_bins=10)
    assert table["n"].sum() == 500
    assert table["mean_predicted"].between(0, 1).all()
    assert table["observed_rate"].between(0, 1).all()


def test_subgroup_metrics_shape():
    frame = pd.DataFrame(
        {
            "product": ["A", "A", "B", "B", "B"],
            "target": [0, 1, 0, 0, 1],
            "prob": [0.1, 0.9, 0.2, 0.3, 0.8],
        }
    )
    out = subgroup_metrics(frame, "product")
    assert set(out["product"]) == {"A", "B"}
    assert out["n"].sum() == 5
