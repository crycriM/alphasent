#!/usr/bin/env bash
# Refresh the study after a batch extraction: features -> IC study -> diff.
#
#   ./scripts/refresh_study.sh            # full refresh
#   ./scripts/refresh_study.sh --diff-only
#
# Extraction itself is a separate, resumable job:
#   .venv/bin/python scripts/batch_etl_parallel.py --workers 4
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [ "${1:-}" != "--diff-only" ]; then
  echo "== 1/3  rebuilding feature store from cached extractions  ($(date -Is))"
  $PY scripts/run_text_pipeline.py
  echo "== 2/3  IC study (grid + CAR + figures)                   ($(date -Is))"
  $PY scripts/ic_study.py all
fi

echo "== 3/3  diff vs baseline                                   ($(date -Is))"
$PY scripts/diff_study_results.py
echo "done ($(date -Is))"
