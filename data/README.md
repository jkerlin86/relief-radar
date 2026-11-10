# Data

Two public sources, both obtained from the publisher, never from a
mirror. Raw files are not committed (see .gitignore); reproduce them
with `scripts/download_data.py`, then build the processed layer with
`scripts/build_dataset.py`.

## 1. CFPB Consumer Complaint Database (structured records + outcomes)

- Publisher: Consumer Financial Protection Bureau.
- File: `raw/complaints.csv.zip` from
  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
- Pinned snapshot: downloaded 2026-11-10 (server Last-Modified
  2026-11-10 10:17 UTC). 364,972,185 bytes.
  SHA-256: db66181722a6a49b6010b73fe32425bca393d718c8f45af40d84cadb2f4c1d9f
- Contents (verified from the file, reports/snapshot_gate.json):
  19,045,222 complaints received 2011-12-01 to 2026-11-10, 15 columns.
- **The current bulk file has no narrative column.** The CFPB ceased
  publishing narratives on August 14, 2026; see source 2.
- Licence/status: work of the U.S. federal government, published for
  public download. The CFPB states the database "is not a statistical
  sample of consumers' experiences in the marketplace"; complaints are
  self-selected and allegations are not verified by the Bureau.

## 2. Complaint narratives (FOIA Reading Room archive)

- Publisher: CFPB FOIA Reading Room, "CFPB Consumer Complaint Database
  Narratives Archive" (proactive disclosure of previously published
  records).
- Files: `raw/foia_narratives/CCDB_Export_*.zip` — 21 full-record
  exports covering complaints received December 2011 through
  August 2026, downloaded from files.consumerfinance.gov.
- A complaint counts as narrative-bearing only if its archive record
  carries non-empty narrative text (the exports are full-record files;
  in the March 2026 export, 21,640 of 614,945 records carry text).
  Joined to source 1 on Complaint ID.
- Status: narratives were published under the CFPB's 2015 Narrative
  Policy (consumer opt-in, Bureau scrubbing of personal information).
  The Bureau treats previously published narratives as public domain
  for FOIA purposes. The corpus is closed: no narratives exist before
  2015 or after August 14, 2026. Do not use narrative text to attempt
  to re-identify any consumer; redaction markers are part of the text
  as published.

## 3. SEC EDGAR (crosswalk and filed financials)

- Publisher: U.S. Securities and Exchange Commission.
- Files: `raw/company_tickers.json` (CIK mapping) and per-company XBRL
  company facts from data.sec.gov, fetched under the SEC's fair-access
  rules with the declared User-Agent in configs/default.json and a
  maximum of 5 requests per second.
- Note: during this build, EDGAR returned access/rate pages (HTTP 403)
  to this environment; see reports/parts/crosswalk_status.json for the
  recorded status. No third-party mirror was substituted.

## Processed layer

- `processed/complaints.parquet` — one row per complaint with parsed
  dates, the target (1 = Closed with monetary relief, 0 = other
  unambiguous closure, -1 = excluded), and the primary-population flag.
  Legacy labels "Closed" and "Closed with relief" are excluded from
  the primary population because they do not distinguish monetary from
  non-monetary relief; "In progress" and "Untimely response" are
  excluded because the outcome is not (yet) an unambiguous closure.
- `processed/narratives.parquet` — Complaint ID + narrative text for
  narrative-bearing complaints, deduplicated on ID.
- `processed/test_predictions.parquet`, `processed/narrative_embeddings.npz`
  — model artifacts of the pipeline runs.
- The embedding encoder (all-MiniLM-L6-v2 ONNX weights, served by
  fastembed) is fetched from its publisher on first use and cached
  outside the repository (in the user cache directory); it is a model
  artifact, not project data, and is never committed.
