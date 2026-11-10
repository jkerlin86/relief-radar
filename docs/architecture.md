# Architecture

```
raw downloads (gitignored)
  complaints.csv.zip            FOIA CCDB_Export_*.zip      [SEC tickers]
        |                              |                          |
scripts/build_dataset.py  (streamed, chunk-wise; nothing big in RAM)
        v                              v
data/processed/complaints.parquet   narratives.parquet
        |                              |
        +------+----------------+------+--------+
        v      v                v               v
  run_modelling  run_selection_audit  run_narrative_arm  run_crosswalk
        |      |                |               |
        +------+-------+--------+-------+-------+
                       v
        reports/parts/*.json + *.csv (measured facts only)
                       v
        scripts/make_figures.py -> figures/*.png
        scripts/make_reports.py -> reports/results.json,
                                   executive_summary.md, model_card.md,
                                   selection_audit.md, alignment_report.md
```

## Design rules

- **Streaming first.** The bulk CSV is 5.8 GB uncompressed; every stage
  that touches it reads in chunks and writes parquet. Peak memory stays
  bounded on a small machine (see the methods notes in the QA report).
- **Facts flow one way.** Scripts that measure write JSON/CSV parts;
  documents are rendered from those parts by make_reports.py. No number
  in a document is typed by hand.
- **Point-in-time discipline.** Feature construction lives in
  src/relief_radar/features.py and is shared by the structured model
  and the narrative arm, so both comparisons use identical features and
  identical history windows.
- **Tests never touch the downloads.** The suite runs on small
  synthetic fixtures with hand-computable expectations.

## Compute and memory discipline (measured on the build machine)

The build machine has about 1.5 shared CPU cores and a 6 GB memory
limit; three design choices follow from measurements, not preference:

- **Population aggregation runs in Arrow, not pandas.** Converting the
  18.4M-row primary population to pandas peaks above the memory limit
  even with Arrow-backed dtypes (measured 3.2 GB for 8 light columns,
  before any modelling copies). All population-level facts (era base
  rates, monthly history aggregates, audit tables) are computed with
  pyarrow group-bys over a dictionary-encoded table; only small
  aggregates and sampled rows cross into pandas
  (`relief_radar.data.load_primary_arrow`, `fetch_rows_by_ids`).
- **Heavy phases run as subprocesses.** The narrative arm's population
  pass and its embedding pass each run in a child process that exits,
  so allocator-held pages cannot accumulate into the modelling phase.
- **Embedding is checkpointed, length-sorted, and small-batch.**
  60,000 sampled narratives are embedded with the all-MiniLM-L6-v2
  weights under ONNX Runtime (fastembed). Texts are truncated to
  2,000 characters (the encoder itself truncates at 256 tokens, which
  2,000 characters always cover), sorted by length so ONNX batches pad
  uniformly, encoded at batch size 32 (batch 256 spikes activation
  memory past the limit), and checkpointed every 2,000 texts so an
  interrupted run resumes. Every one of these choices is recorded in
  the run outputs (reports/parts/narrative_results.json) and the QA
  report.
