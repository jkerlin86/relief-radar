"""Model factories.

Two model classes, both sklearn pipelines with all preprocessing fitted
inside the training data only:
- regularized logistic regression over one-hot categoricals (baseline);
- histogram gradient boosting over ordinal-encoded categoricals.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_logistic_pipeline(
    categorical: list[str],
    numeric: list[str],
    cfg: dict,
    class_weight: str | None = None,
) -> Pipeline:
    """Regularized logistic regression baseline."""
    pre = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    clf = LogisticRegression(
        C=cfg["C"], max_iter=cfg["max_iter"], class_weight=class_weight
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def make_histgb_pipeline(
    categorical: list[str], numeric: list[str], cfg: dict
) -> Pipeline:
    """Histogram gradient boosting challenger."""
    cat_idx = list(range(len(categorical)))
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=cfg["max_iter"],
        learning_rate=cfg["learning_rate"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        min_samples_leaf=cfg["min_samples_leaf"],
        early_stopping=cfg["early_stopping"],
        categorical_features=cat_idx,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
