"""Render figures from the measured report parts. No hand-typed data."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from relief_radar.config import project_path

PARTS = project_path("reports", "parts")
FIGS = project_path("figures")


def _load(name: str) -> dict | None:
    path = PARTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def calibration_figure() -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, label in [
        ("logistic_regression", "Logistic regression"),
        ("hist_gradient_boosting", "Gradient boosting"),
    ]:
        path = PARTS / f"calibration_{name}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            markersize=3,
            label=label,
        )
    ax.plot([0, 0.06], [0, 0.06], linestyle="--", color="grey", label="Perfect")
    ax.set_xlabel("Mean predicted probability (test sample, 2025+)")
    ax.set_ylabel("Observed monetary-relief rate")
    ax.set_title("Calibration on the held-out era")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=150)
    plt.close(fig)


def era_drift_figure(model_results: dict) -> None:
    rates = model_results["era_base_rates_full_population"]
    years = sorted(rates, key=int)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [int(y) for y in years],
        [rates[y]["relief_rate"] * 100 for y in years],
        color="#33638c",
    )
    ax.set_xlabel("Year received")
    ax.set_ylabel("Monetary-relief rate (%)")
    ax.set_title("Relief rate by era, full primary population (note 2011 = December only)")
    fig.tight_layout()
    fig.savefig(FIGS / "era_relief_rates.png", dpi=150)
    plt.close(fig)


def subgroup_figure() -> None:
    path = PARTS / "subgroups_hist_gradient_boosting_product.csv"
    if not path.exists():
        return
    table = pd.read_csv(path).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [str(p)[:45] for p in table["product"]]
    ax.barh(labels[::-1], table["roc_auc"][::-1], color="#33638c")
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set_xlabel("ROC AUC on the test sample (gradient boosting)")
    ax.set_title("Discrimination by product")
    fig.tight_layout()
    fig.savefig(FIGS / "subgroup_auc_product.png", dpi=150)
    plt.close(fig)


def narrative_share_figure() -> None:
    path = PARTS / "selection_by_year.csv"
    if not path.exists():
        return
    table = pd.read_csv(path).sort_values("year")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["year"], table["narrative_share"] * 100, color="#7a5195")
    ax.set_xlabel("Year received")
    ax.set_ylabel("Complaints with a published narrative (%)")
    ax.set_title("The narrative layer is a minority, and it closes in 2026")
    fig.tight_layout()
    fig.savefig(FIGS / "narrative_share_by_year.png", dpi=150)
    plt.close(fig)


def narrative_arm_figure(narrative: dict) -> None:
    models = narrative["models"]
    names = list(models)
    aucs = [models[n]["roc_auc"] for n in names]
    labels = [n.replace("_", " ") for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels[::-1], aucs[::-1], color="#2e7d5b")
    ax.axvline(0.5, color="grey", linestyle="--")
    ax.set_xlabel("ROC AUC on narrative-bearing test sample (2025+)")
    ax.set_title("Does the consumer's text help? Same rows, four models")
    fig.tight_layout()
    fig.savefig(FIGS / "narrative_arm_auc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    model_results = _load("model_results.json")
    calibration_figure()
    if model_results:
        era_drift_figure(model_results)
    subgroup_figure()
    narrative_share_figure()
    narrative = _load("narrative_results.json")
    if narrative:
        narrative_arm_figure(narrative)
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
