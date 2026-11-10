# Leakage audit — field by field

The model predicts, at the moment a complaint is received, whether it
will close with monetary relief. A feature is admissible only if its
value is fixed at receipt. Every column of the CFPB snapshot was audited
against that rule before any modelling.

| Field | Known at receipt? | Use |
|---|---|---|
| Date received | Yes — it is the receipt timestamp | Used: year, month of year, day of week |
| Product | Yes — consumer-selected at submission | Used (categorical) |
| Sub-product | Yes | Used (categorical, top-K capped) |
| Issue | Yes | Used (categorical, top-K capped) |
| Sub-issue | Yes | Used (categorical, top-K capped) |
| Company | Yes | Used only through point-in-time history aggregates (below); raw company identity is not one-hot encoded — with thousands of companies and a temporal split it would mostly encode era-specific noise, and the aggregates carry the decision-relevant signal |
| State | Yes | Used (categorical) |
| ZIP code | Yes, as published (CFPB degrades it for privacy) | Only its missingness is used; the project does not model on a deliberately degraded geography when state is available |
| Tags | Yes — set at submission (Older American, Servicemember) | Used (two indicators) |
| Submitted via | Yes — the intake channel | Used (categorical) |
| Consumer complaint narrative | Yes — written by the consumer at submission | Used in the narrative arm only (separate model comparison) |
| Date sent to company | **No** — the Bureau forwards the complaint after receipt; the gap encodes Bureau handling | **Excluded** |
| Company public response | **No** — the company may publish it up to 180 days after the complaint is sent | **Excluded** |
| Timely response? | **No** — determined by how the company later responds | **Excluded** |
| Company response to consumer | **No** — this is the outcome | Target source only |
| Complaint ID | Identifier | Join key only, never a feature |

## Derived history features

Company and product history aggregates (prior volume, prior relief
rate, previous-month volume) are computed on **expanding windows over
strictly earlier calendar months**: a complaint received in month *m*
sees only complaints received in months < *m*. Using the complaint's own
month would leak — complaints close at different speeds, so a same-month
aggregate is partly determined by outcomes not yet observed at receipt.
The company prior relief rate is shrunk toward the product's prior rate
with a prior strength of 20 complaints (configs/default.json), so a
company with two historical complaints does not get a 0% or 100% rate.

## Split discipline

Training and evaluation are separated in time, never shuffled: train on
complaints received before 2024-01-01, test on complaints received on or
after 2025-01-01. Calendar 2024 is a buffer year, used only to check
base-rate drift, so no tuning decision touches the test era. Category
caps and all preprocessing are fitted on the training sample only.
