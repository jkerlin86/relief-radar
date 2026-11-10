# Problem framing

## The decision context

Consumer-operations and conduct-risk teams at consumer-facing firms
receive complaints continuously, but the consequential question — *is
this complaint one of the rare ones that ends in monetary relief, the
closest observable proxy for a substantiated problem?* — is only
answered weeks later, when the company closes the complaint. A
point-in-time model that scores complaints at receipt lets a team triage
review, watch emerging product issues months before they surface in
aggregate reports, and benchmark its own handling against the market.

Relief Radar builds that model on the full CFPB Consumer Complaint
Database (19,045,222 complaints in the pinned 2026-11-10 snapshot),
and then does the two things public work on this dataset usually skips:

1. **It audits its own text layer.** Complaint narratives were opt-in,
   scrubbed, and — since August 14, 2026 — no longer published at all.
   The narrative corpus is a closed, self-selected archive. The project
   measures how the narrative-bearing minority differs from all
   complaints before drawing any conclusion from text.
2. **It connects complaints to disclosure.** Complaints name companies
   in free text; SEC EDGAR knows them by CIK. The project builds an
   audited name → CIK crosswalk and aligns complaint experience with
   filed financials for the confidently matched large filers —
   descriptively, with no causal claims.

## What the model is and is not

- The target is "Closed with monetary relief" versus other unambiguous
  closures (explanation, non-monetary relief, without relief). In the
  primary population the monetary-relief rate is about 1.2%: this is a
  ranking and triage instrument, not a classifier to threshold naively,
  and accuracy is never reported.
- Complaints are self-selected. The CFPB itself states the database "is
  not a statistical sample of consumers' experiences." Volumes confound
  harm with company size, market share, and awareness of the Bureau.
  Nothing in this project estimates how many consumers were harmed.
- "Closed" and "Closed with relief" are legacy response labels that do
  not distinguish monetary from non-monetary relief; they are excluded
  from the primary population and counted, not silently dropped (see
  the gate note and dataset build report).

## Audiences

Consumer-operations, conduct-risk, and compliance teams; secondarily,
anyone evaluating public claims made from this dataset.
