#!/usr/bin/env bash
# Rebuild the llama-3 arm after its batch extraction, then refresh the study.
#
# Extraction (separate, resumable, ~8 h):
#   MODEL_VERSION=llama3-8b-q4km-v1 LLM_MODEL_NAME=llama3-8b \
#     .venv/bin/python scripts/batch_etl_parallel.py --workers 4
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=llama3-8b-q4km-v1
STAMP=$(date +%Y%m%d)

echo "== 1/6  snapshot results + July llama features            ($(date -Is))"
[ -d data/results_v2_phi4only_$STAMP ] || cp -a data/results data/results_v2_phi4only_$STAMP
[ -d data/features_llama3_frozen_20260717 ] || mv data/features_llama3_backup data/features_llama3_frozen_20260717

echo "== 2/6  llama-3 feature store                             ($(date -Is))"
MODEL_VERSION=$L FEATURES_DIR=data/features_llama3_backup $PY scripts/run_text_pipeline.py > /dev/null

echo "== 3/6  IC study                                          ($(date -Is))"
$PY scripts/ic_study.py all

echo "== 4/6  era decomposition, both models                    ($(date -Is))"
for m in phi4-q6k-v1 $L; do
  for f in composite_full composite_rare polarity_sum mag_weighted_polarity \
           regulation_flag hack_flag depeg_flag listing_flag; do
    for h in 72 120; do $PY scripts/era_decomposition.py --model $m --feature $f --horizon $h; done
  done
done

echo "== 5/6  redaction audit, both models                      ($(date -Is))"
for m in phi4-q6k-v1 $L; do $PY scripts/contamination_audit.py --model-version $m; done

echo "== 6/6  report facts + grid diff                          ($(date -Is))"
$PY scripts/refresh_report_facts.py
$PY scripts/diff_study_results.py
echo "done ($(date -Is))"
