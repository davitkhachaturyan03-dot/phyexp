#!/bin/zsh
# Summarizer re-run (2026-10-11): the report's "Reproduce it" commands, run now on HEAD, against stored inputs.
# Usage (any cwd): zsh tracker/features/2026-10-01-evaluation-layer/artifacts/summarizer_rerun.sh
# Output: artifacts/summarizer_rerun_stdout.txt (tee'd by the caller).
# Writes: only this feature's artifacts/ (the verify scripts rewrite their own artifacts; the caller checks
# `git status` afterwards) and data/eval/returns-* via verify_returns_s6 --real's temp dirs. Reads the real lineage
# store (mlflow.db sha256 printed before/after); records no MLflow run. No simulator, no network.
set -u
cd "${0:A:h}/../../../.."   # repo root
export HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache LIBERO_CONFIG_PATH=$PWD/data/libero_config HF_HUB_OFFLINE=1 \
       MUJOCO_GL=cgl MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/bin/python
quiet() { grep -v -e '^objc\[' -e 'Downloading artifacts' ; }
SP=data/eval/eval-smolvla-harness-spatial0-20261004T150110Z
L8=data/eval/eval-smolvla-harness-l10t8-20261004T153306Z
echo "# summarizer re-run $(date -u +%Y-%m-%dT%H:%M:%SZ), git HEAD $(git rev-parse --short HEAD)"
echo "mlflow.db sha256 before: $(shasum -a 256 data/lineage/mlflow.db | cut -c1-16)"
echo

echo '$ .venv/bin/python -m vlapoc.lineage version check-blind v1   # S0 (then v0)'
$PY -m vlapoc.lineage version check-blind v1 2>&1 | quiet | tail -1
$PY -m vlapoc.lineage version check-blind v0 2>&1 | quiet | tail -1
echo

echo '$ .venv/bin/python scripts/verify_metrics_s1.py   # S1'
$PY scripts/verify_metrics_s1.py 2>&1 | quiet
echo

echo '$ .venv/bin/python scripts/phail_validate.py   # S2 (stored sorted export, data/phail/cohort_mixed.parquet)'
echo "cohort_mixed.parquet sha256: $(shasum -a 256 data/phail/cohort_mixed.parquet | cut -c1-16)"
$PY scripts/phail_validate.py 2>&1 | quiet | grep -E '^AC2\.(1 max|1 macro-KS D|2 cohort|3 OpenPI|3 tol|3 macro|4|5|6 (teleop|croiss))'
echo

echo '$ .venv/bin/python scripts/human_reference.py --summary   # S3'
$PY scripts/human_reference.py --summary 2>&1 | quiet | grep -E '^AC3\.[34]'
echo

echo '$ .venv/bin/python scripts/verify_operations_s4.py   # S4'
$PY scripts/verify_operations_s4.py 2>&1 | quiet | grep -E '7/7|AC4\.5'
echo

echo "\$ .venv/bin/python scripts/verify_rollouts_s5.py --summary $SP $L8   # S5 (stored AC5.4 rollouts)"
$PY scripts/verify_rollouts_s5.py --summary $SP $L8 2>&1 | quiet | grep -E '^AC5\.4'
echo

echo "\$ .venv/bin/python scripts/verify_returns_s6.py --real $SP --real $L8 --against-verdict   # S6"
$PY scripts/verify_returns_s6.py --real $SP --real $L8 --against-verdict 2>&1 | quiet | grep -E '^AC6\.(1 REAL \(|10 max|11 REAL)'
echo

echo '$ .venv/bin/python -m vlapoc.verdict --phail GR00T ACT   # S7, seed 0'
$PY -m vlapoc.verdict --phail GR00T ACT 2>&1 | quiet
echo

echo '$ .venv/bin/python scripts/verify_verdict_s7.py --phail-seeds 0 1 2 3 4   # S7 AC7.1 under the r3 criterion'
$PY scripts/verify_verdict_s7.py --phail-seeds 0 1 2 3 4 2>&1 | quiet | grep -E 'GR00T vs ACT seed|OpenPI vs ACT seed|AC7\.1 (every|\(r3)'
echo

echo "\$ .venv/bin/python scripts/verify_verdict_s7.py --family-consistency $SP $L8   # S7 AC7.7"
$PY scripts/verify_verdict_s7.py --family-consistency $SP $L8 2>&1 | quiet | grep -E 'max'
echo

echo '$ .venv/bin/python scripts/verify_subset_stats_s8.py --check v0   # S8'
$PY scripts/verify_subset_stats_s8.py --check v0 2>&1 | quiet | grep -E '^AC8\.1'
echo

echo '$ .venv/bin/python -m vlapoc.lineage trace 85f7d7d95884483480996e97c3cccb19   # S9 (the r2 eval-smoke verify run)'
$PY -m vlapoc.lineage trace 85f7d7d95884483480996e97c3cccb19 2>/dev/null | $PY -c "
import json, sys
d = json.load(sys.stdin)
runs = list(d['runs'].values())
print('runs in trace:', len(runs), '| stages:', sorted({r['stage'] for r in runs}))
print('stand-in runs:', [r['name'] for r in runs if r['stand_in']])
print('entity kinds:', sorted({e['kind'] for e in d['entities'].values()}))
print('raw files reached:', len(d['raw_files']), '| first:', d['raw_files'][0]['source_file'].split('/')[-1][:48], d['raw_files'][0]['sha256'][:16])
"
echo

echo "mlflow.db sha256 after: $(shasum -a 256 data/lineage/mlflow.db | cut -c1-16)"
