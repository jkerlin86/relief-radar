# Model card — Relief Radar outcome models

## Intended use

Triage and early-warning ranking of incoming consumer complaints by their likelihood of closing with monetary relief, for consumer-operations and conduct-risk teams. Not for use in decisions about individual consumers, and not a measure of whether any complaint is meritorious.

## Training and evaluation protocol

- Train: received before 2024-01-01 (1,000,000 sampled rows, base rate 3.28%).
- Test: received on or after 2025-01-01 (500,000 sampled rows, base rate 0.51%). Calendar 2024 is a buffer year used only for drift checks.
- Features: receipt-time fields only; company/product history as expanding aggregates over strictly earlier months. Full leakage audit in docs/leakage_audit.md.
- Imbalance: no resampling or class weighting in the headline models (a class-weighted sensitivity run is included in results.json); ranking and calibration metrics are reported instead of accuracy.

## Performance on the held-out era

| Model | ROC AUC | PR AUC | Brier | Max calibration gap |
|---|---|---|---|---|
| logistic_regression | 0.985 | 0.297 | 0.0040 | 0.012 |
| hist_gradient_boosting | 0.990 | 0.294 | 0.0041 | 0.015 |

Subgroup tables (product, year, channel) are in reports/parts/subgroups_*.csv and summarised in results.json. Performance varies by product; the model must not be read as uniformly reliable across the catalogue.

## Limitations

- Trained on a self-selected complaint stream; scores rank complaints, they do not estimate harm or fault.
- Relief rates drift by era, so calibrated probabilities are era-specific even when ranking transfers.
- Legacy response labels that do not distinguish monetary from non-monetary relief are excluded from training (counted in the dataset build report).
