import numpy as np
import pandas as pd

from relief_radar.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from relief_radar.models import make_histgb_pipeline, make_logistic_pipeline

LOG_CFG = {"C": 0.1, "max_iter": 200}
GB_CFG = {
    "max_iter": 20,
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 5,
    "early_stopping": False,
}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "product": rng.choice(["Credit card", "Mortgage"], size=n),
            "sub_product": rng.choice(["", "General"], size=n),
            "issue": rng.choice(["Billing", "Fraud"], size=n),
            "sub_issue": "",
            "submitted_via": rng.choice(["Web", "Phone"], size=n),
            "state": rng.choice(["FL", "TX"], size=n),
        }
    )
    for col in NUMERIC_FEATURES:
        df[col] = rng.random(n)
    df["year"] = 2023
    y = (df["company_prior_relief_rate"] > 0.5).astype(int)
    return df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], y


def test_logistic_pipeline_fits_and_predicts():
    x, y = _frame()
    pipe = make_logistic_pipeline(CATEGORICAL_FEATURES, NUMERIC_FEATURES, LOG_CFG)
    prob = pipe.fit(x, y).predict_proba(x)[:, 1]
    assert prob.shape == (len(x),)
    assert np.isfinite(prob).all()


def test_histgb_pipeline_fits_and_predicts():
    x, y = _frame()
    pipe = make_histgb_pipeline(CATEGORICAL_FEATURES, NUMERIC_FEATURES, GB_CFG)
    prob = pipe.fit(x, y).predict_proba(x)[:, 1]
    assert prob.shape == (len(x),)
    assert np.isfinite(prob).all()


def test_unseen_category_does_not_crash():
    x, y = _frame()
    pipe = make_logistic_pipeline(CATEGORICAL_FEATURES, NUMERIC_FEATURES, LOG_CFG)
    pipe.fit(x, y)
    x2 = x.copy()
    x2.loc[x2.index[0], "product"] = "Never seen before"
    prob = pipe.predict_proba(x2)[:, 1]
    assert np.isfinite(prob).all()
