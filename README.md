# Relief Radar: What Can Be Known When a Complaint Arrives?

**Point-in-time outcome modelling on 19 million CFPB complaints, an audit of the opt-in narrative layer, and an honest account of what the text adds.**

When a consumer complaint arrives at the Consumer Financial Protection
Bureau, how much can be known at that moment about whether it will end
in monetary relief, the closest observable proxy for a substantiated
problem? Relief Radar answers that with a strict receipt-time feature
audit and a temporal evaluation, then turns the same discipline on the
complaint narratives themselves: they are an opt-in, self-selected
minority, and this project measures how far they can speak for all
complaints before it uses them.

Data: CFPB Consumer Complaint Database bulk snapshot of 2026-11-10
(19,045,222 complaints, received 2011-12-01 to 2026-11-10), plus the
narrative corpus from the CFPB FOIA Reading Room archive (3,851,415
narratives; narrative publication ceased 2026-08-14 and the live bulk
file no longer carries the narrative column, verified from the pinned
file itself before any design decisions were taken; see
`reports/snapshot_gate.json`).

## Headline results

All figures are produced by `scripts/run_all.sh` and recorded in
`reports/results.json`; the documents in `reports/` are rendered from
that file, never hand-typed.

**Receipt-time features rank relief outcomes, but the era does not sit
still.** Primary population: 18,367,254 unambiguous closures, monetary
relief rate 1.23%. Train era (received before 2024) relief rate 3.28%;
test era (2025 onward) 0.49%. On a 500,000-row held-out test sample
(base rate 0.51%):

| Model | ROC AUC | PR AUC | Brier |
|---|---|---|---|
| Logistic regression (baseline) | 0.985 | 0.297 | 0.0040 |
| Gradient boosting | 0.990 | 0.294 | 0.0041 |

Read those AUCs with the product mix: the test era is dominated by
credit-reporting complaints (448,755 of 500,000 sampled rows, relief
rate 0.02%), which are near-separable and flatter the pooled ranking.
Within-product gradient-boosting AUCs span 0.71 to 0.98; where relief
is actually common, discrimination is real but modest (credit card
0.80, checking or savings 0.71, prepaid card 0.84). The subgroup
tables in `reports/parts/` are the operationally honest read.

**The narrative layer is selected, measurably.** In the publication
window (2015-01-01 to 2026-08-14), 22.86% of complaints carry a
narrative (3,836,347 of 16,782,339). Their raw relief rate is 2.58%
versus 0.73% for complaints without narratives, but most of that gap
is composition: inside 132 year-by-product cells the volume-weighted
difference is +0.22 percentage points (median +0.34; narratives higher
in 76% of cells). Narratives speak for narrative-bearing complaints,
not for the complaint stream.

**The text itself helps a little, on its own turf.** On identical
narrative-bearing rows (stratified sample, 40,000 train / 20,000 test;
consumer text embedded with all-MiniLM-L6-v2), structured-only
logistic regression scores AUC 0.965, a text-only model 0.954, and the
combined model 0.969 (PR AUC 0.339 versus 0.299 structured-only). The
combined-minus-structured AUC difference is +0.004 (bootstrap 95% CI
+0.000 to +0.007): small, positive, and reported with its interval.

**The EDGAR alignment is measured, and says what it found.** CFPB
company names were normalized into a 7,639-company table and matched
against the SEC tickers file with the name-to-CIK matcher (exact /
containment / fuzzy tiers with ambiguity flags): 348 companies
matched (4.56% by count), covering 64.89% of primary-population
complaint volume, with 102 matches flagged ambiguous. The alignment
panel aligns annual complaint volume and relief rates with filed
financial facts (revenues, net income, assets) for the 15 largest
confidently matched filers across 407 company-year-concept rows; it
is descriptive context only, and `reports/alignment_report.md` gives
the full audit. At publication this step was blocked (SEC EDGAR
returned HTTP 403 to the original build environment and no
third-party mirror was substituted); it was completed on 2026-11-11
once the SEC fair-access route was established, and the report
records that history.

## What is in the repository

- `src/relief_radar/` — point-in-time feature construction (expanding
  company/product history over strictly earlier months), name
  normalization, evaluation, the narrative-selection audit math, the
  CIK matcher, and the EDGAR client.
- `scripts/` — the full pipeline: `download_data.py`,
  `inspect_snapshot.py` (the data gate), `build_dataset.py`,
  `run_modelling.py`, `run_selection_audit.py`, `run_crosswalk.py`,
  `run_narrative_arm.py`, `make_figures.py`, `make_reports.py`,
  `run_all.sh`.
- `docs/leakage_audit.md` — the field-by-field audit: every feature is
  something known at receipt; outcome-adjacent fields (company
  response, timeliness, date sent) are excluded and named.
- `docs/architecture.md` — pipeline design, including the measured
  compute discipline (Arrow population aggregation, subprocess phase
  isolation, checkpointed length-sorted embedding) that a 6 GB build
  machine required.
- `notebooks/` — executed walkthroughs that read only build
  artefacts, so they cannot drift from the pipeline outputs.
- `tests/` — 28 tests on synthetic fixtures with hand-computable
  expectations; the suite never touches the downloads.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,nlp]"
bash scripts/run_all.sh   # download -> gate -> build -> model -> audit -> reports -> tests
```

Raw downloads land in `data/raw/` (gitignored) with a SHA-256 manifest;
the SEC tickers step records a blocked status where EDGAR fair access
is unavailable rather than failing the pipeline.

## Data and licence

Code: MIT (see `LICENSE`). Data: CFPB complaint data and the FOIA
narrative archive are US Government works distributed by the CFPB;
SEC EDGAR data is public. `LICENSE` documents the data terms
separately from the code licence. Processed datasets are reproducible
from the pinned snapshot and are not committed.

## Limits

- Complaints are self-selected; the CFPB states the database is not a
  statistical sample. Complaint volume confounds harm with company
  size and consumer awareness. Nothing here estimates consumer harm,
  fault, or merit.
- Monetary relief is what the company reported closing with, not an
  adjudication.
- The narrative corpus is closed (publication ceased 2026-08-14) and
  opt-in; text findings describe that archive only.
- Probabilities are era-specific: relief base rates moved from 3.28%
  (train era) to 0.49% (test era), so calibration must be re-anchored
  per era even where ranking transfers.
