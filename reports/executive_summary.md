# Executive summary — Relief Radar

## The question

When a consumer complaint arrives at the CFPB, how much can be known at that moment about whether it will end in monetary relief — the closest observable proxy for a substantiated problem? And can the consumer's own written narrative, an opt-in and now-closed archive, be trusted to speak for complaints in general?

## What was built

- A point-in-time outcome model on the full CFPB snapshot: 19,045,222 complaints, of which 18,367,254 are in the primary population (unambiguous closures) with a monetary-relief rate of 1.23%. Features are only what is known at receipt (see docs/leakage_audit.md); evaluation is on a later era than training, never a shuffled split.
- A narrative-selection audit comparing the narrative-bearing minority with all complaints in the publication window.
- A narrative arm testing whether consumer text adds signal on top of the structured fields.
- A CFPB-to-EDGAR name crosswalk and complaint-vs-disclosure alignment (status in reports/alignment_report.md).

## Headline results (held-out era, 2025+ test sample)

- Logistic regression baseline: ROC AUC 0.985, PR AUC 0.297 against a base rate of 0.51%, Brier 0.0040.
- Gradient boosting: ROC AUC 0.990, PR AUC 0.294, Brier 0.0041.
- Era drift is real: the full-population relief rate in the training era was 3.28% and in the test era 0.49%. Absolute probabilities must be re-anchored per era; ranking is what carries across.
- Read the headline AUC with the product mix: the largest test subgroup, Credit reporting or other personal consumer reports, is 448,755 of the sampled test rows at a relief rate of 0.02%, and near-separable volume like that flatters the pooled ranking. Within-product AUCs for gradient boosting span 0.71 to 0.98; on the products where relief is common (Credit card 0.80, Checking or savings account 0.71, Prepaid card 0.84), discrimination is real but far more modest. The subgroup table is the operationally honest read.

## The narrative layer, audited

- In the publication window, 22.86% of complaints carry a narrative (3,836,347 of 16,782,339).
- Monetary-relief rate: 2.58% for narrative-bearing complaints vs 0.73% without. reports/selection_audit.md decomposes how much of that gap is composition (product and era mix) and how much survives inside year x product cells.

## Does the text help?

- On identical narrative-bearing rows, structured-only logistic regression scores ROC AUC 0.965; text-only 0.954; combined 0.969. The combined-minus-structured AUC difference is +0.004 (bootstrap 95% CI +0.000 to +0.007).

## Limits that bind every conclusion

- Complaints are self-selected; the CFPB states the database is not a statistical sample. Volume confounds harm with size and awareness. Nothing here estimates consumer harm.
- The narrative corpus is closed (publication ceased 2026-08-14) and opt-in; text findings describe that archive, not future complaints.
- Monetary relief is a proxy outcome: it records what the company reported closing with, not an adjudication.
