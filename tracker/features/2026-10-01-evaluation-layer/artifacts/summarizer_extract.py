"""Extract the data the report's figures need from the feature's real outputs, using the feature's own code.

Run by the summarizer (2026-10-11). Every number written here is computed now, from:
- ``data/phail/cohort_mixed.parquet`` (S2's sorted export of PhAIL's released cohort, sha256 35b7181b...), through
  ``vlapoc.evaluation.phail_cohort_arms`` / ``pooled`` / ``km``: the pooled Kaplan-Meier CDFs of every arm;
- ``data/eval/human_reference/human_reference.parquet`` (S3's goal-check replay of the 1000 v0 demos): per-demo
  ``T_success_s`` and ``duration_s``, both suites;
- ``data/eval/eval-smolvla-harness-*/rollouts.parquet`` (S5, AC5.4): the 20 real SmolVLA rollouts, through
  ``vlapoc.evaluation.libero_arm`` (family W and O) and ``libero_human_arm``;
- ``data/eval/returns-e138fec97361/`` and ``returns-e92604a8b995/`` (S6): the per-rollout family-W returns and the
  libero_10:8 family-O operation returns, as the current ``vlapoc.returns`` writes them.

Output: ``artifacts/summarizer_data.json`` (read by ``summarizer_figures.py``) and a printed summary, saved as
``artifacts/summarizer_extract_stdout.txt``. Nothing is written outside ``artifacts/``; no MLflow access.

Usage (repo root)::

    .venv/bin/python tracker/features/2026-10-01-evaluation-layer/artifacts/summarizer_extract.py \
        | tee tracker/features/2026-10-01-evaluation-layer/artifacts/summarizer_extract_stdout.txt
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
from vlapoc import evaluation as ev  # noqa: E402

TAU_PHAIL = 240.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def km_steps(eps, t_max: float) -> list[list[float]]:
    """Exact KM step points (t, F) of the pooled pairs, up to t_max (F right-continuous)."""
    tg, S, _ = ev.km(eps)
    out = [[0.0, round(float(ev.F_curve(eps, np.array([0.0]))[0]), 6)]]  # F(0): T = 0 pairs (PhAIL multi-grip extras)
    for t, s in zip(tg, S):
        if np.isfinite(t) and 0.0 < t <= t_max:
            out.append([round(float(t), 4), round(float(1.0 - s), 6)])
    return out


def main() -> int:
    data: dict = {"sources": {}}

    # 1. PhAIL pooled KM CDFs (all five arms), and F at a few probe times for the captions
    cohort = ROOT / "data" / "phail" / "cohort_mixed.parquet"
    data["sources"]["cohort_mixed.parquet"] = sha(cohort)
    arms = ev.phail_cohort_arms(cohort)
    data["phail_km"] = {}
    probes = [2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 240.0]
    print(f"PhAIL cohort {cohort.relative_to(ROOT)} sha256 {data['sources']['cohort_mixed.parquet'][:16]}")
    for m in ["OpenPI", "GR00T", "ACT", "SmolVLA", "Teleop"]:
        eps = ev.pooled(arms[m])
        steps = km_steps(eps, TAU_PHAIL)
        Fp = {str(t): round(float(x), 4) for t, x in zip(probes, ev.F_curve(eps, np.array(probes)))}
        data["phail_km"][m] = {"n_episodes": len(eps), "n_pairs": int(sum(len(e.pairs) for e in eps)), "steps": steps,
                               "F_probe": Fp, "rmst240": round(ev.rmst(eps, TAU_PHAIL), 4)}
        print(f"  {m:8s} episodes {len(eps):3d} pairs {data['phail_km'][m]['n_pairs']:4d} RMST(240) "
              f"{data['phail_km'][m]['rmst240']:8.3f} s  F at {probes}: {list(Fp.values())}")
    # where the GR00T and ACT pooled CDFs cross (on a fine grid)
    grid = np.round(np.arange(0.0, TAU_PHAIL + 1e-9, 0.05), 2)
    fg = ev.F_curve(ev.pooled(arms["GR00T"]), grid)
    fa = ev.F_curve(ev.pooled(arms["ACT"]), grid)
    d = fa - fg
    sign_changes = [float(grid[i]) for i in range(1, len(grid)) if np.sign(d[i]) != np.sign(d[i - 1]) and d[i] != 0]
    act_ahead = [float(grid[i]) for i in range(len(grid)) if d[i] > 0]
    data["phail_cross"] = {"sign_changes_s": sign_changes[:20], "act_ahead_first_s": act_ahead[:1],
                           "act_ahead_last_s": act_ahead[-1:], "max_act_lead": round(float(d.max()), 4),
                           "max_act_lead_at_s": float(grid[int(d.argmax())]), "max_groot_lead": round(float(-d.min()), 4),
                           "max_groot_lead_at_s": float(grid[int(d.argmin())])}
    print(f"  GR00T vs ACT pooled CDF: sign changes of F_ACT - F_GR00T at {sign_changes[:20]} s; "
          f"ACT ahead at most by {data['phail_cross']['max_act_lead']} (t = {data['phail_cross']['max_act_lead_at_s']} s); "
          f"GR00T ahead at most by {data['phail_cross']['max_groot_lead']} (t = {data['phail_cross']['max_groot_lead_at_s']} s)")

    # 2. human reference: T_success_s vs duration_s per suite (every demo an event, no censoring)
    hr = ROOT / "data" / "eval" / "human_reference" / "human_reference.parquet"
    data["sources"]["human_reference.parquet"] = sha(hr)
    h = pd.read_parquet(hr)
    data["human"] = {}
    print(f"human reference {hr.relative_to(ROOT)} sha256 {data['sources']['human_reference.parquet'][:16]}: "
          f"{len(h)} demos, success_any {int(h['success_any'].sum())}")
    for s in ["libero_spatial", "libero_10"]:
        g = h[h["suite"] == s]
        T = np.sort(g["T_success_s"].to_numpy(float))
        D = np.sort(g["duration_s"].to_numpy(float))
        data["human"][s] = {"T_success_s": [round(x, 3) for x in T.tolist()], "duration_s": [round(x, 3) for x in D.tolist()],
                            "median_T": float(np.median(T)), "median_D": float(np.median(D)),
                            "mean_minT60": float(np.mean(np.minimum(T, 60))), "mean_minD60": float(np.mean(np.minimum(D, 60)))}
        print(f"  {s:15s} n {len(g)} median T_success {np.median(T):.2f} s vs duration {np.median(D):.2f} s; "
              f"RMST(60) {np.mean(np.minimum(T, 60)):.4f} vs {np.mean(np.minimum(D, 60)):.4f} s; "
              f"ratio {np.mean(np.minimum(D, 60)) / np.mean(np.minimum(T, 60)):.4f}")
    # per-task human T for the two tasks the SmolVLA harness ran
    data["human_task"] = {}
    for s, t in [("libero_spatial", 0), ("libero_10", 8)]:
        g = h[(h["suite"] == s) & (h["libero_task_id"] == t)]
        data["human_task"][f"{s}/{t}"] = sorted(round(x, 3) for x in g["T_success_s"].astype(float))
        print(f"  human {s}/{t}: n {len(g)}, T_success median {g['T_success_s'].median():.2f} s, max {g['T_success_s'].max():.2f} s")

    # 3. the 20 SmolVLA harness rollouts (external checkpoint; harness validation only)
    data["smolvla"] = {}
    for name in ["eval-smolvla-harness-spatial0-20261004T150110Z", "eval-smolvla-harness-l10t8-20261004T153306Z"]:
        rp = ROOT / "data" / "eval" / name / "rollouts.parquet"
        r = pd.read_parquet(rp)
        data["sources"][f"{name}/rollouts.parquet"] = sha(rp)
        cell = f"{r['suite'].iloc[0]}/{int(r['libero_task_id'].iloc[0])}"
        armW = ev.libero_arm(r, "W")[cell]
        armO = ev.libero_arm(r, "O")[cell]
        rows = [{"rollout_id": x.rollout_id, "init": int(x.init_state_index), "T_s": None if pd.isna(x.T_s) else float(x.T_s),
                 "event": int(x.event), "n_ops_done": int(x.n_ops_done), "n_ops": int(x.n_ops),
                 "pairs": json.loads(x.pairs), "batch_size": int(x.batch_size)} for x in r.itertuples()]
        data["smolvla"][cell] = {"eval_run": name, "rows": rows, "km_W": km_steps(armW, 60.0), "km_O": km_steps(armO, 60.0),
                                 "F30_W": ev.F_at(armW, 30.0), "rmst60_W": ev.rmst(armW, 60.0), "median_W": ev.median_T(armW),
                                 "F30_O": ev.F_at(armO, 30.0), "rmst60_O": ev.rmst(armO, 60.0),
                                 "censored_rate": ev.censored_rate(armW)}
        print(f"  SmolVLA {cell} ({name}): {sum(x['event'] for x in rows)}/{len(rows)} succeeded; family W F(30) "
              f"{ev.F_at(armW, 30.0):.3f}, RMST(60) {ev.rmst(armW, 60.0):.3f} s, median {ev.median_T(armW)}; family O pairs "
              f"{sum(len(e.pairs) for e in armO)}, KM RMST(60) {ev.rmst(armO, 60.0):.3f} s")

    # 4. per-rollout returns (family W) and operation returns (family O) as stored by the current vlapoc.returns
    data["returns"] = []
    ret_cols = ["ret_f30", "ret_rmst60", "ret_auc_expected", "ret_auc_sampled", "ret_hrt60_lin", "ret_censored60"]
    for d_ in ["returns-e138fec97361", "returns-e92604a8b995"]:
        p = ROOT / "data" / "eval" / d_ / "returns.parquet"
        data["sources"][f"{d_}/returns.parquet"] = sha(p)
        r = pd.read_parquet(p)
        for x in r.itertuples():
            data["returns"].append({"rollout_id": x.rollout_id, "cell": x.cell, "T_s": None if pd.isna(x.T_s) else float(x.T_s),
                                    "failure_type": x.failure_type, **{c: float(getattr(x, c)) for c in ret_cols}})
        print(f"  {d_}/returns.parquet: {len(r)} rows; cell means " +
              ", ".join(f"{c} {r[c].mean():.4f}" for c in ret_cols))
    op = ROOT / "data" / "eval" / "returns-e92604a8b995" / "operation_returns.parquet"
    data["sources"]["returns-e92604a8b995/operation_returns.parquet"] = sha(op)
    o = pd.read_parquet(op)
    data["op_returns"] = [{"rollout_id": x.rollout_id, "op_k": int(x.op_k), "t_start": int(x.t_start_steps),
                           "t_end": int(x.t_end_steps), "T_op_s": float(x.T_op_s), "event": int(x.event),
                           "ret": float(x.ret_rmst60_op_cap)} for x in o.itertuples()]
    print(f"  returns-e92604a8b995/operation_returns.parquet: {len(o)} operation samples; mean ret_rmst60_op_cap "
          f"{o['ret_rmst60_op_cap'].mean():.6f}; unfinished attempts {int((o['event'] == 0).sum())}, all -60: "
          f"{bool((o.loc[o['event'] == 0, 'ret_rmst60_op_cap'] == -60).all())}")

    out = ART / "summarizer_data.json"
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
