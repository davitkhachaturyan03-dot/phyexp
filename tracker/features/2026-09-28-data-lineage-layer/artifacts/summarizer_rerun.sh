#!/bin/zsh
# Summarizer re-run (2026-10-01): read-only lineage queries + tamper refusal, against the real store.
# Usage: zsh tracker/features/2026-09-28-data-lineage-layer/artifacts/summarizer_rerun.sh
# Writes nothing under data/lineage (mlflow.db sha256 printed before/after); the tamper clone lives in
# data/scratch/ and is deleted. Output saved as summarizer_rerun_stdout.txt.
set -u
cd "${0:A:h}/../../../.."   # repo root (this file sits in tracker/features/<feature>/artifacts/)
export HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/bin/python
q() { "$PY" -m vlapoc.lineage "$@" 2>/dev/null; }
echo "# summarizer re-run $(date -u +%Y-%m-%dT%H:%M:%SZ), git HEAD $(git rev-parse --short HEAD)"
echo "mlflow.db sha256 before: $(shasum -a 256 data/lineage/mlflow.db | cut -c1-16)"
echo
echo '$ .venv/bin/python -m vlapoc.lineage version diff v0 v1'
q version diff v0 v1 | tail -1
echo
echo '$ .venv/bin/python -m vlapoc.lineage version verify v0   (then v1)'
q version verify v0 | tr -d '\n' ; echo
q version verify v1 | tr -d '\n' ; echo
echo
echo '$ .venv/bin/python -m vlapoc.lineage diff ff64512b40974587a9d711cd8e4f8b04 9339d4685a8e4d009c191fa4641ccf3a   # train-full-s0 vs train-drop-top-20'
q diff ff64512b40974587a9d711cd8e4f8b04 9339d4685a8e4d009c191fa4641ccf3a | "$PY" -c "
import json,sys; d=json.load(sys.stdin)
print('runs:', [r['name'] for r in d['runs']])
print('data: removed', d['data']['n_removed'], '| added', d['data']['n_added'], '| changed_version', d['data']['n_changed_version'], '| same dataset version:', d['data']['dataset_version_a']==d['data']['dataset_version_b'])
print('params:', d['params'], '| config:', d['config'], '| git same:', d['git']['same'])"
echo
echo '$ .venv/bin/python -m vlapoc.lineage diff ff64512b40974587a9d711cd8e4f8b04 dc6b5f18ed544c6cb535d90754ebbf0a   # train-full-s0 vs train-full-s1'
q diff ff64512b40974587a9d711cd8e4f8b04 dc6b5f18ed544c6cb535d90754ebbf0a | "$PY" -c "
import json,sys; d=json.load(sys.stdin)
print('data: removed', d['data']['n_removed'], '| added', d['data']['n_added'], '| changed_version', d['data']['n_changed_version'])
print('params:', d['params'], '| config:', d['config'], '| git same:', d['git']['same'])"
echo
echo '$ .venv/bin/python -m vlapoc.lineage trace 9339d4685a8e4d009c191fa4641ccf3a   # train-drop-top-20'
q trace 9339d4685a8e4d009c191fa4641ccf3a | "$PY" -c "
import json,sys; t=json.load(sys.stdin)
print('runs reached:', len(t['runs']), sorted(r['stage']+':'+r['name'] for r in t['runs'].values()))
print('edges:', len(t['edges']), '| each with exactly one producer:', all(len(e['produced_by'])==1 for e in t['edges']))
print('raw files reached:', len(t['raw_files']))
ref=json.load(open('tracker/features/2026-09-28-data-lineage-layer/artifacts/s6_trace_train_drop_top.json'))
print('identical to artifacts/s6_trace_train_drop_top.json:', t==ref)"
echo
P=libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_17
echo "\$ .venv/bin/python -m vlapoc.lineage consumers $P@6cfad44e5243...   # its v1 (corrupted) version"
q consumers "$P@6cfad44e524360005fc747954450b6b4f3e6f438fad4cb6585ec51ed31b88bce" | "$PY" -c "import json,sys; print([r['name'] for r in json.load(sys.stdin)])"
echo "\$ .venv/bin/python -m vlapoc.lineage consumers $P@3456348a0125...   # its clean v0 version"
q consumers "$P@3456348a01251f652d20e1b030c1821c22da6e04c5a51b6013231187e4d477fc" | "$PY" -c "import json,sys; print([r['name'] for r in json.load(sys.stdin)])"
echo
echo '$ tamper_check(v1) from scripts/lineage_dry_run.py (driver; the committed artifact is not overwritten)'
"$PY" -c "
import sys; sys.path[:0]=['scripts','.']
import lineage_dry_run as d
from vlapoc.lineage import resolve_version
d.tamper_check(resolve_version('v1'))" 2>&1 | grep '^AC6.5'
echo
echo "mlflow.db sha256 after:  $(shasum -a 256 data/lineage/mlflow.db | cut -c1-16)"
