"""Verify evaluation-layer S6 (returns): identities on SYNTHETIC rollouts and on real rollouts (AC6.1), the HRT
linearization (AC6.2), the attribution join contract (AC6.3), registry coverage (AC6.4), failure-type subsets
(AC6.5) and column naming (AC6.6).

Grounding: PoC §4.3 p. 10, §9.1 Step 3 p. 19, §3.4 p. 7, p. 1 (episode × metric matrix); CUPID §5.1 p. 5.

    .venv/bin/python scripts/verify_returns_s6.py --synthetic                    # AC6.1 (synthetic), AC6.2, AC6.4, AC6.6
    .venv/bin/python scripts/verify_returns_s6.py --join data/eval/returns-<eval_run>/returns.parquet   # AC6.3
    .venv/bin/python scripts/verify_returns_s6.py --real data/eval/<run>         # AC6.1 (real), AC6.5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import evaluation as ev  # noqa: E402
from vlapoc import returns as rt  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
OUT: dict = {}


def synthetic_rollouts(rng: np.random.Generator, cells: dict[str, float], n: int, p_censor: float, tau_steps: int = 1200) -> pd.DataFrame:
    """SYNTHETIC rollout records: per cell, T ~ scale * Gamma(3, 1) seconds on the 20 Hz grid, a share censored
    at τ_episode; only the columns ``vlapoc.returns`` reads."""
    rows = []
    for cell, scale in cells.items():
        suite, tid = cell.split("/")
        for j in range(n):
            T = round(float(scale * rng.gamma(3.0, 1.0)) * 20) / 20 + 0.05 * (j % 7) / 7  # off-grid jitter: no ties
            cens = rng.random() < p_censor or T >= tau_steps / 20
            rows.append({"eval_run_id": "SYNTHETIC", "rollout_id": f"{cell}/r{j}", "train_run_id": "", "checkpoint_sha256": "",
                         "external_checkpoint": "", "policy_type": "synthetic", "suite": suite, "libero_task_id": int(tid),
                         "init_state_index": j, "seed": j, "event": int(not cens), "censored": bool(cens),
                         "T_s": None if cens else T, "n_ops": 1, "n_ops_done": int(not cens), "tau_episode_steps": tau_steps,
                         "pairs": json.dumps([[tau_steps / 20, 0]] if cens else [[T, 1]])})
    return pd.DataFrame(rows)


def km_eps(df: pd.DataFrame) -> list:
    return [ev.libero_episode(r.rollout_id, [] if r.censored else [r.T_s], (r.tau_episode_steps / 20) if r.censored else r.T_s, 1)
            for r in df.itertuples()]


def synthetic() -> None:
    rng = np.random.default_rng(0)
    cells = {"syn/0": 4.0, "syn/1": 7.0, "syn/2": 10.0}
    human = {("syn", i): np.round(rng.gamma(3.0, 1.0, 50) * s * 0.4, 6) + 1e-7 * np.arange(50) for i, s in enumerate(cells.values())}
    res = {}
    for label, p_c in (("censoring only at tau_episode (10%)", 0.10), ("no censoring", 0.0)):
        ro = synthetic_rollouts(rng, cells, 60, p_c)
        ret, meta = rt.rollout_returns(ro, human)
        d = {"rmst": [], "f30": [], "auc": []}
        for cell, g in ret.groupby("cell"):
            eps = km_eps(ro[ro["rollout_id"].isin(g["rollout_id"])])
            d["rmst"].append(abs(g["ret_rmst60"].mean() + ev.rmst(eps, 60.0)))
            d["f30"].append(abs(g["ret_f30"].mean() - ev.F_at(eps, 30.0)))
            s, t = cell.split("/")
            heps = [ev.libero_episode(f"h{i}", [x], x, 1) for i, x in enumerate(human[(s, int(t))])]
            if p_c == 0:
                d["auc"].append(abs(g["ret_auc_expected"].mean() - ev.auc(eps, heps)))
        res[label] = {k: (max(v) if v else None) for k, v in d.items()}
        print(f"AC6.1 SYNTHETIC ({label}, 3 cells x 60 rollouts): max |mean(ret_rmst60) + RMST60_KM| = {res[label]['rmst']:.2e}; "
              f"max |mean(ret_f30) - F30_KM| = {res[label]['f30']:.2e}"
              + (f"; max |mean(ret_auc_expected) - AUC_censored| = {res[label]['auc']:.2e}" if p_c == 0 else ""))
        if p_c == 0:
            sampled = ret.groupby("cell")["ret_auc_sampled"].mean()
            expected = ret.groupby("cell")["ret_auc_expected"].mean()
            print(f"AC6.1 SYNTHETIC ret_auc_sampled vs ret_auc_expected per cell (sampling noise): "
                  f"{ {c: (round(sampled[c], 3), round(expected[c], 4)) for c in sampled.index} }")
    OUT["AC6.1_synthetic"] = res
    # ---- AC6.2: HRT linearization -------------------------------------------------------------------------
    ro = synthetic_rollouts(rng, cells, 60, 0.0)
    base, meta = rt.rollout_returns(ro, human)
    w = {c: v["w"] for c, v in meta["hrt_weights"].items()}

    def hrt_exact(df):
        return ev.hrt({c: km_eps(df[df["rollout_id"].str.startswith(c + "/")]) for c in cells},
                      {c: [ev.libero_episode(f"h{i}", [x], x, 1) for i, x in enumerate(human[(c.split('/')[0], int(c.split('/')[1]))])]
                       for c in cells}, 60.0)

    h0 = hrt_exact(ro)
    curve = []
    for delta in (-0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10):
        for mode in ("uniform", "one cell"):
            pert = ro.copy()
            factor = np.where((mode == "uniform") | pert["rollout_id"].str.startswith("syn/1/"), 1 + delta, 1.0)
            pert["T_s"] = pert["T_s"] * factor
            h1 = hrt_exact(pert)
            cap0 = np.minimum(ro["T_s"].astype(float).fillna(np.inf), 60.0).to_numpy()  # censored (T >= 60) -> 60
            cap1 = np.minimum(pert["T_s"].astype(float).fillna(np.inf), 60.0).to_numpy()
            cell = ro["suite"] + "/" + ro["libero_task_id"].astype(str)
            pred = sum(np.mean(-w[c] * cap1[cell == c]) - np.mean(-w[c] * cap0[cell == c]) for c in cells)
            exact = h1 - h0
            curve.append({"delta": delta, "mode": mode, "rel_change_hrt": exact / h0, "exact": exact, "pred": pred,
                          "rel_err": abs(pred - exact) / abs(exact)})
    in_scope = [c for c in curve if abs(c["rel_change_hrt"]) <= 0.05]
    print(f"AC6.2 SYNTHETIC HRT = {h0:.4f} (3 cells); sum_cells mean(ret_hrt60_lin) = {base.groupby('cell')['ret_hrt60_lin'].mean().sum():.4f} "
          f"(= -HRT at the expansion point)")
    for c in curve:
        print(f"AC6.2   {c['mode']:8s} delta {c['delta']:+.2f}: dHRT/HRT {c['rel_change_hrt']:+.4f}, exact {c['exact']:+.5f}, "
              f"linearized {c['pred']:+.5f}, rel. error {c['rel_err']:.2%}")
    worst = max(c["rel_err"] for c in in_scope)
    print(f"AC6.2 worst relative error over perturbations with |dHRT|/HRT <= 5%: {worst:.2%} ({len(in_scope)} cases; criterion <= 10%)")
    OUT["AC6.2"] = {"hrt0": h0, "curve": curve, "worst_in_scope": worst, "pass": bool(worst <= 0.10)}
    pd.DataFrame(curve).to_csv(ART / "s6_hrt_linearization_curve.csv", index=False)
    # ---- AC6.4 / AC6.6: registry coverage and naming -------------------------------------------------------
    reg = rt.metrics_registry()
    names = " | ".join(r["metric"] for r in reg)
    # §3.2 p. 6 primitives, PhAIL Table 1 p. 5 scalars, §4.1 p. 9 / §10 targets (F(30 s), HRT, RMST, AUC, ghost-event rate)
    required = ["time-to-success T", "ghost-event rate", "right-censoring", "Kaplan–Meier CDF", "Success@τ", "median", "RMST",
                "UPH", "completion", "MTBF/A", "HRT", "bootstrap", "macro-KS", "P-P", "F(30 s)", "AUC"]
    missing = [m for m in required if m not in names]
    no_status = [r["metric"] for r in reg if not r["transfer_status"] or not r["reason"]]
    print(f"AC6.4 registry: {len(reg)} entries; required (§3.2 primitives, Table 1, §4/§10 targets) missing: {missing}; "
          f"entries without status/reason: {no_status}")
    for r in reg:
        print(f"AC6.4   {r['metric']:62s} -> {r['transfer_status']:38s} return: {r['return_column']}")
    # AC6.6: every return column named in the registry carries its metric and, where the functional has a τ in
    # seconds, that τ; the per-rollout ones are exactly RETURN_COLUMNS
    naming = {}
    for r in reg:
        if not r["return_column"] or r["return_column"] == "failure_type":
            continue
        tau = re.match(r"(\d+) s", r["tau"])
        for col in (c.split(" (")[0].strip() for c in r["return_column"].split(", ")):
            naming[col] = {"tau_s": tau.group(1) if tau else None, "ok": tau is None or tau.group(1) in col}
    per_rollout = [c for c in naming if not c.endswith("_op")]  # operation-level returns live in operation_returns.parquet
    same_set = sorted(per_rollout) == sorted(rt.RETURN_COLUMNS)
    print(f"AC6.6 return columns vs their registry τ: {naming}; per-rollout registry columns == RETURN_COLUMNS: {same_set}")
    OUT["AC6.4"] = {"n": len(reg), "missing": missing, "no_status": no_status, "pass": not missing and not no_status}
    OUT["AC6.6"] = {"columns": naming, "same_set": same_set, "pass": same_set and all(v["ok"] for v in naming.values())}
    (ART / "s6_metrics_registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))


def join(returns_path: Path) -> None:
    """AC6.3: returns.parquet + its train run's training subset -> the empty episode x metric frame (no influence)."""
    from vlapoc.lineage import LineageError
    from vlapoc.runs import check_scores, client

    ret = pd.read_parquet(returns_path)
    train_ids = sorted(set(ret["train_run_id"]))
    if len(train_ids) != 1 or not train_ids[0]:
        raise SystemExit(f"returns must come from one tracked train run, got {train_ids}")
    train_run_id = train_ids[0]
    ckpt_ok = set(ret["checkpoint_sha256"]) == {client().get_run(train_run_id).data.tags["checkpoint.weights.sha256"]}
    frame = rt.attribution_frame(train_run_id)
    metric_cols = [c for c in frame.columns if c not in ("episode_id", "content_hash")]
    ok = check_scores(frame.fillna(0.0), train_run_id)
    bad = frame.copy()
    bad.loc[0, "content_hash"] = "0" * 64
    try:
        check_scores(bad, train_run_id)
        rejected = None
    except LineageError as e:
        rejected = str(e)[:160]
    print(f"AC6.3 {returns_path.relative_to(REPO_ROOT)}: {len(ret)} rollouts of eval run(s) {sorted(set(ret['eval_run_id']))}; "
          f"train run {train_run_id[:12]}; returns' checkpoint_sha256 == train run's weights sha: {ckpt_ok}")
    print(f"AC6.3 attribution frame: shape {frame.shape} (rows = training subset's (episode_id, content_hash); metric columns "
          f"{metric_cols} == RETURN_COLUMNS: {metric_cols == rt.RETURN_COLUMNS}); check_scores: {ok}; "
          f"a stale content_hash is rejected: {rejected is not None}")
    print(f"AC6.3 rejection message: {rejected}")
    OUT["AC6.3"] = {"returns": str(returns_path.relative_to(REPO_ROOT)), "train_run_id": train_run_id, "checkpoint_matches": ckpt_ok,
                    "shape": list(frame.shape), "metric_columns": metric_cols, "check_scores": ok, "stale_rejected": rejected is not None}


def real(path: Path) -> None:
    ro = pd.read_parquet(path / "rollouts.parquet")
    ret, meta = rt.rollout_returns(ro)
    ops = rt.operation_returns(ro)
    diffs = {"rmst": [], "f30": [], "censored": []}
    for cell, g in ret.groupby("cell"):
        eps = km_eps(ro[ro["rollout_id"].isin(g["rollout_id"])])
        diffs["rmst"].append(abs(g["ret_rmst60"].mean() + ev.rmst(eps, 60.0)))
        diffs["f30"].append(abs(g["ret_f30"].mean() - ev.F_at(eps, 30.0)))
        diffs["censored"].append(abs(g["ret_censored60"].mean() - ev.censored_rate(eps)))
    print(f"AC6.1 REAL ({path.name}, {len(ret)} rollouts, {ret['cell'].nunique()} cells): max |mean(ret_rmst60) + RMST60| = "
          f"{max(diffs['rmst']):.2e}; max |mean(ret_f30) - F30| = {max(diffs['f30']):.2e}; max |mean(ret_censored60) - censored_rate| = "
          f"{max(diffs['censored']):.2e}")
    by = ret.groupby(["cell", "failure_type"]).size().unstack(fill_value=0)
    print("AC6.5 rollouts per cell x failure_type:\n" + by.to_string())
    for ft, g in ret.groupby("failure_type"):
        eps = km_eps(ro[ro["rollout_id"].isin(g["rollout_id"])])
        print(f"AC6.5 subset failure_type={ft}: n {len(g)}, F(30 s) {ev.F_at(eps, 30.0):.3f}, RMST(60) {ev.rmst(eps, 60.0):.2f} s, "
              f"mean ret_auc_expected {g['ret_auc_expected'].mean():.3f}")
    print(f"AC6.1 REAL HRT linearization weights {json.dumps({c: round(v['w'], 6) for c, v in meta['hrt_weights'].items()})}; "
          f"macro HRT at policy {meta['hrt_macro_at_policy']:.4f}; operation rows {len(ops)}")
    ret.to_parquet(path / "returns.parquet", index=False)
    ops.to_parquet(path / "operation_returns.parquet", index=False)
    OUT[f"AC6.1_real_{path.name}"] = {k: max(v) for k, v in diffs.items()}
    OUT[f"AC6.5_{path.name}"] = by.to_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--join", type=Path, help="a returns.parquet from python -m vlapoc.returns")
    p.add_argument("--real", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.synthetic:
        synthetic()
    if a.join:
        join((REPO_ROOT / a.join).resolve())
    for r in a.real:
        real(r)
    path = ART / "s6_verify_returns.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**old, **OUT}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
