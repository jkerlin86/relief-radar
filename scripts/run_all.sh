#!/usr/bin/env bash
# End-to-end reproduction: download -> gate inspection -> dataset build
# -> structured modelling -> selection audit -> crosswalk -> narrative
# arm -> figures -> rendered reports -> tests.
# Run from the project root:  bash scripts/run_all.sh
#
# Compute notes (see docs/architecture.md): the full pipeline is heavy
# on a small machine. The narrative embedding phase is checkpointed per
# 2,000 texts and resumes if interrupted. EDGAR steps degrade to a
# documented "blocked" status when SEC fair access is unavailable;
# nothing is substituted from third-party mirrors.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

echo "== 1/9 download data (CFPB bulk + FOIA narratives; SEC tickers attempted) =="
$PY scripts/download_data.py
echo "== 2/9 snapshot gate inspection =="
$PY scripts/inspect_snapshot.py
echo "== 3/9 build processed datasets =="
$PY scripts/build_dataset.py all
echo "== 4/9 structured outcome modelling (temporal split) =="
$PY scripts/run_modelling.py
echo "== 5/9 narrative-selection audit =="
$PY scripts/run_selection_audit.py
echo "== 6/9 name-to-CIK crosswalk (may record EDGAR blocked) =="
$PY scripts/run_crosswalk.py
echo "== 7/9 narrative arm (pass 1 + checkpointed embedding + comparison) =="
$PY scripts/run_narrative_arm.py
echo "== 8/9 figures + reports =="
$PY scripts/make_figures.py
$PY scripts/make_reports.py
echo "== 9/9 tests =="
$PY -m pytest -q
echo "All stages complete. See reports/results.json, reports/executive_summary.md, reports/model_card.md"
