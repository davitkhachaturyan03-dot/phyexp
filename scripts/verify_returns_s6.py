"""Verify evaluation-layer S6 (returns): family-W identities on SYNTHETIC rollouts and on real rollouts (AC6.1), the
HRT linearization (AC6.2), the attribution join contract (AC6.3), registry coverage (AC6.4), failure-type subsets
(AC6.5), column naming (AC6.6), one registry (AC6.8), returns == verdict on the declared functional (AC6.10) and
the operation-return ranking (AC6.11).

Grounding: PoC §4.3 p. 10, §9.1 Step 3 p. 19, §3.4 p. 7 ("Declared targets"), §4.4 p. 10, p. 1 (episode × metric
matrix); CUPID §5.1 p. 5; plan r2 "Declared functionals (r2)".

    .venv/bin/python scripts/verify_returns_s6.py --synthetic                    # AC6.1 (synthetic), AC6.2, AC6.4, AC6.6, AC6.8, AC6.11 (synthetic)
    .venv/bin/python scripts/verify_returns_s6.py --join data/eval/returns-<eval_run>/returns.parquet   # AC6.3
    .venv/bin/python scripts/verify_returns_s6.py --real data/eval/<run> [--real ...] --against-verdict   # AC6.1 real, AC6.5, AC6.10, AC6.11
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


def synthetic_rollouts(rng: np.random.Generator, cells: dict[str, float], n: int, p_censor: float, tau_steps: int = 1200,
                      jitter: bool = True) -> pd.DataFrame:
    """SYNTHETIC rollout records: per cell, T ~ scale * Gamma(3, 1) seconds on the 20 Hz grid, a share censored
    at τ_episode; only the columns ``vlapoc.returns`` reads."""
    rows = []
    for cell, scale in cells.items():
        suite, tid = cell.split("/")
        for j in range(n):
            T = round(float(scale * rng.gamma(3.0, 1.0)) * 20) / 20 + (0.05 * (j % 7) / 7 if jitter else 0.0)  # jitter: no ties
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


def w_identities(ro: pd.DataFrame, human: dict) -> dict:
    """Family-W identities per cell, against the whole-task KM (``evaluation.libero_arm(ro, "W")``) and the whole-task
    human arm: max |difference| over cells for f30, rmst60, censored60 and auc_h (ties 1/2)."""
    ret, _ = rt.rollout_returns(ro, human)
    arm = ev.libero_arm(ro.assign(T_s=ro["T_s"].astype(float)), "W")
    d = {"f30": [], "rmst60": [], "censored60": [], "auc_h": []}
    for cell, g in ret.groupby("cell"):
        eps = arm[cell]
        s_, t_ = cell.split("/")
        heps = [ev.Episode(f"h{i}", ((float(x), 1),)) for i, x in enumerate(human[(s_, int(t_))])]
        d["f30"].append(abs(g["ret_f30"].mean() - ev.F_at(eps, 30.0)))
        d["rmst60"].append(abs(g["ret_rmst60"].mean() + ev.rmst(eps, 60.0)))
        d["censored60"].append(abs(g["ret_censored60"].mean() - (1.0 - ev.F_at(eps, 60.0))))
        d["auc_h"].append(abs(g["ret_auc_expected"].mean() - ev.auc_ties_half(eps, heps)))
    return {k: float(max(v)) for k, v in d.items()}


def synthetic() -> None:
    rng = np.random.default_rng(0)
    cells = {"syn/0": 4.0, "syn/1": 7.0, "syn/2": 10.0}
    human = {("syn", i): np.round(rng.gamma(3.0, 1.0, 50) * s * 0.4, 6) + 1e-7 * np.arange(50) for i, s in enumerate(cells.values())}
    res = {}
    for label, p_c in (("censoring only at tau_episode (10%)", 0.10), ("no censoring", 0.0)):
        ro = synthetic_rollouts(rng, cells, 60, p_c)
        res[label] = w_identities(ro, human)
        r = res[label]
        print(f"AC6.1 SYNTHETIC ({label}, 3 cells x 60 rollouts), family W vs whole-task KM: max |mean(ret_rmst60) + RMST60| = "
              f"{r['rmst60']:.2e}; max |mean(ret_f30) - F30| = {r['f30']:.2e}; max |mean(ret_censored60) - (1 - F60)| = "
              f"{r['censored60']:.2e}; max |mean(ret_auc_expected) - AUC_h (ties 1/2)| = {r['auc_h']:.2e}")
    # ties: model and human times on the same 20 Hz grid (separate RNG, so AC6.2's draws are unchanged)
    rng_t = np.random.default_rng(7)
    ro_t = synthetic_rollouts(rng_t, cells, 60, 0.10, jitter=False)
    human_t = {("syn", i): np.round(rng_t.gamma(3.0, 1.0, 50) * s_ * 0.4 * 20) / 20 for i, s_ in enumerate(cells.values())}
    # the family-W premise (registry auc_h): every human time is below τ_episode, so a censored rollout is comparable to all
    assert max(h.max() for h in human_t.values()) < 60.0
    r = w_identities(ro_t, human_t)
    n_ties = sum(int(np.sum(np.isclose(np.asarray(g["T_s"].dropna())[:, None], human_t[("syn", int(c.split("/")[1]))][None, :])))
                 for c, g in ro_t.assign(cell=ro_t["suite"] + "/" + ro_t["libero_task_id"].astype(str)).groupby("cell"))
    print(f"AC6.1 SYNTHETIC WITH TIES (shared 20 Hz grid, {n_ties} tied model-human pairs, 10% censored at τ): max |mean(ret_auc_expected) - "
          f"AUC_h (ties 1/2)| = {r['auc_h']:.2e}; rmst60 {r['rmst60']:.2e}; f30 {r['f30']:.2e}; censored60 {r['censored60']:.2e}")
    res["with ties"] = {**r, "n_tied_pairs": n_ties}
    OUT["AC6.1_synthetic"] = res
    # AC6.11 synthetic: an unfinished attempt with a short tail never outranks a completed operation
    syn = pd.DataFrame([{"eval_run_id": "SYNTHETIC", "rollout_id": rid, "train_run_id": "", "checkpoint_sha256": "", "suite": "syn",
                         "libero_task_id": 9, "n_ops": 2, "pairs": json.dumps(p)} for rid, p in
                        (("slow_complete", [[20.0, 1], [39.9, 1]]), ("stall_late", [[59.0, 1], [1.0, 0]]), ("stall_early", [[2.0, 1], [58.0, 0]]),
                         ("none", [[60.0, 0]]))])
    ops = rt.operation_returns(syn)
    worst_done = ops.loc[ops["event"] == 1, "ret_rmst60_op_cap"].min()
    best_open = ops.loc[ops["event"] == 0, "ret_rmst60_op_cap"].max()
    print(f"AC6.11 SYNTHETIC: unfinished attempts (tails 1.0 s, 58.0 s, 60.0 s) score {sorted(ops.loc[ops['event'] == 0, 'ret_rmst60_op_cap'])}; "
          f"worst completed operation {worst_done}; max unfinished <= min completed: {best_open <= worst_done}")
    OUT["AC6.11_synthetic"] = {"unfinished": ops.loc[ops["event"] == 0, "ret_rmst60_op_cap"].tolist(), "worst_completed": float(worst_done),
                               "pass": bool(best_open <= worst_done)}
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
    # ---- AC6.4 / AC6.6 / AC6.8: the single registry ----------------------------------------------------------
    from vlapoc import metrics_registry as mr

    reg = rt.metrics_registry()
    names = " | ".join(r["metric"] for r in reg)
    # §3.2 p. 6 primitives (right-censoring = the censored-rate entry), PhAIL Table 1 p. 5 scalars, §4.1 p. 9 / §10 targets
    required = ["time-to-success T", "ghost-event rate", "censored rate", "Kaplan–Meier CDF", "Success@τ", "median", "RMST",
                "UPH", "completion", "MTBF/A", "HRT", "bootstrap", "macro-KS", "P-P", "F(30 s)", "AUC"]
    missing = [m for m in required if m not in names]
    no_status = [r["metric"] for r in reg if not r["transfer_status"] or not r["reason"]]
    print(f"AC6.4 registry: {len(reg)} entries; required (§3.2 primitives, Table 1, §4/§10 targets) missing: {missing}; "
          f"entries without status/reason: {no_status}")
    for r in reg:
        print(f"AC6.4   {r['metric']:62s} [{r['family']:16s}] -> {r['transfer_status'][:40]:40s} returns: {r['return_columns']}")
    # AC6.6: every return column carries its metric and, where the functional has a τ in seconds, that τ
    naming = {}
    for r in reg:
        tau = re.match(r"(\d+) s", r["tau"])
        for col in r["return_columns"]:
            if col != "failure_type":
                naming[col] = {"tau_s": tau.group(1) if tau else None, "ok": tau is None or tau.group(1) in col}
    same_set = sorted(mr.return_columns("W")) == sorted(rt.RETURN_COLUMNS)
    print(f"AC6.6 return columns vs their registry τ: {naming}; family-W registry columns == returns.RETURN_COLUMNS: {same_set}")
    # AC6.8: one registry
    src = {f: (REPO_ROOT / "vlapoc" / f).read_text() for f in ("evaluation.py", "returns.py", "verdict.py")}
    tables = {f: bool(re.search(r"^(METRICS|REGISTRY)\s*=", t, re.M)) for f, t in src.items()}
    imports = {f: "metrics_registry" in t for f, t in src.items()}
    uph = mr.entry("uph")["functional"]
    fam_ok = {r["key"]: bool(r["family"]) and bool(r["T"]) for r in reg if r["return_columns"] or r["verdict"]}
    declared = {"f30": "W", "rmst60": "W", "auc_h": "W", "hrt60": "W", "censored60": "W", "rmst60_op_cap": "O"}
    fam_match = {k: mr.entry(k)["family"] == f and set(mr.entry(k)["return_columns"]) >= set() for k, f in declared.items()}
    arms = ev.phail_cohort_arms()
    uph_dev = max(abs(ev.uph(ev.pooled(arms[m]), 240.0) - 3600.0 / ev.rmst(ev.pooled(arms[m]), 240.0)) for m in arms)
    print(f"AC6.8 metric tables (METRICS = / REGISTRY =) in evaluation/returns/verdict: {tables}; import metrics_registry: {imports}")
    print(f"AC6.8 registry UPH functional starts: {uph[:40]!r}; entries with a return or verdict metric that have family and T: "
          f"{sum(fam_ok.values())}/{len(fam_ok)}; declared families as in plan r2: {fam_match}")
    print(f"AC6.8 uph() vs 3600 / rmst(τ_episode = 240 s) on the S2 cohort (5 arms, pooled): max |diff| = {uph_dev:.2e}")
    OUT["AC6.4"] = {"n": len(reg), "missing": missing, "no_status": no_status, "pass": not missing and not no_status}
    OUT["AC6.6"] = {"columns": naming, "same_set": same_set, "pass": same_set and all(v["ok"] for v in naming.values())}
    OUT["AC6.8"] = {"tables": tables, "imports": imports, "uph": uph, "family_T": fam_ok, "declared": fam_match, "uph_max_diff": uph_dev,
                    "pass": not any(tables.values()) and all(imports.values()) and uph.startswith("3600 / RMST(τ_episode)")
                    and all(fam_ok.values()) and all(fam_match.values()) and uph_dev <= 1e-9}
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
    """AC6.1 (real, family W vs whole-task KM) and AC6.5 on one eval output dir."""
    ro = pd.read_parquet(path / "rollouts.parquet")
    ret, meta = rt.rollout_returns(ro)
    human = {(c.split("/")[0], int(c.split("/")[1])): np.array([e.pairs[0][0] for e in eps])
             for c, eps in ev.libero_human_arm("W").items()}
    r = w_identities(ro, human)
    print(f"AC6.1 REAL ({path.name}, {len(ret)} rollouts, {ret['cell'].nunique()} cells), family W vs whole-task KM: "
          f"max |mean(ret_rmst60) + RMST60| = {r['rmst60']:.2e}; max |mean(ret_f30) - F30| = {r['f30']:.2e}; "
          f"max |mean(ret_censored60) - (1 - F60)| = {r['censored60']:.2e}; max |mean(ret_auc_expected) - AUC_h| = {r['auc_h']:.2e}")
    arm = ev.libero_arm(ro, "W")
    for cell, eps in arm.items():
        print(f"AC6.1 REAL {path.name} {cell} whole-task values: RMST(60) {ev.rmst(eps, 60.0):.3f} s, F(30) {ev.F_at(eps, 30.0):.3f}, "
              f"1-F(60) {1 - ev.F_at(eps, 60.0):.3f}")
    by = ret.groupby(["cell", "failure_type"]).size().unstack(fill_value=0)
    print("AC6.5 rollouts per cell x failure_type:\n" + by.to_string())
    for ft, g in ret.groupby("failure_type"):
        eps = km_eps(ro[ro["rollout_id"].isin(g["rollout_id"])])
        print(f"AC6.5 subset failure_type={ft}: n {len(g)}, F(30 s) {ev.F_at(eps, 30.0):.3f}, RMST(60) {ev.rmst(eps, 60.0):.2f} s, "
              f"mean ret_auc_expected {g['ret_auc_expected'].mean():.3f}")
    OUT[f"AC6.1_real_{path.name}"] = r
    OUT[f"AC6.5_{path.name}"] = by.to_dict()


def against_verdict(paths: list[Path]) -> None:
    """AC6.10 (returns == the verdict's declared functional, per arm and cell, both families) and AC6.11 (operation
    ranking) on eval output dirs; also writes each run's returns/operation_returns next to its rollouts."""
    from vlapoc import verdict as vd

    rows, ok, rank = [], True, {}
    for path in paths:
        ro = pd.read_parquet(path / "rollouts.parquet")
        ret, meta = rt.rollout_returns(ro)
        ops = rt.operation_returns(ro)
        ret.to_parquet(path / "returns.parquet", index=False)
        ops.to_parquet(path / "operation_returns.parquet", index=False)
        for fam in ("W", "O"):
            arm = ev.libero_arm(ro, fam)
            human = ev.libero_human_arm(fam, sorted(arm))
            v = vd.compare_policies(arm, arm, human, fam, n_boot=20, n_boot_ks=20, seed=0, names=("A", "A2"))
            for key, m in v["metrics"].items():
                for cell, pt in m["cells"].items():
                    if fam == "O":
                        g = ops[(ops["suite"] + "/" + ops["libero_task_id"].astype(str)) == cell]
                        rmean, rel = float(g["ret_rmst60_op_cap"].mean()), "-mean(ret_rmst60_op_cap) vs rmst60_op_cap"
                        diff = abs(-rmean - pt["A"])
                    else:
                        g = ret[ret["cell"] == cell]
                        col, sign = {"f30": ("ret_f30", 1), "rmst60": ("ret_rmst60", -1), "auc_h": ("ret_auc_expected", 1),
                                     "censored60": ("ret_censored60", 1), "hrt60": ("ret_hrt60_lin", None)}[key]
                        if sign is None:
                            continue  # macro: checked once per arm below
                        rmean, rel = float(g[col].mean()), f"{'-' if sign < 0 else ''}mean({col}) vs {key}"
                        diff = abs(sign * rmean - pt["A"])
                    rows.append({"run": path.name, "family": fam, "metric": key, "cell": cell, "relation": rel,
                                 "returns_side": (-rmean if (fam == "O" or key == "rmst60") else rmean), "verdict_side": pt["A"], "abs_diff": diff})
                    ok &= diff <= 1e-9
            if fam == "W":
                s_ = float(ret.groupby("cell")["ret_hrt60_lin"].mean().sum())
                d = abs(-s_ - v["metrics"]["hrt60"]["A"])
                rows.append({"run": path.name, "family": "W", "metric": "hrt60", "cell": "macro", "relation": "-Σ_cells mean(ret_hrt60_lin) vs HRT(60)",
                             "returns_side": -s_, "verdict_side": v["metrics"]["hrt60"]["A"], "abs_diff": d})
                ok &= d <= 1e-9
            if not v["label"].startswith(f"family {fam}") or any(f"family {fam}" not in m["label"] for m in v["metrics"].values()):
                ok = False
        # AC6.11 on real data: unfinished attempts score -60 and never exceed a completed operation in the same cell
        for cell, g in ops.groupby(ops["suite"] + "/" + ops["libero_task_id"].astype(str)):
            open_, done = g[g["event"] == 0], g[g["event"] == 1]
            rank[f"{path.name}|{cell}"] = {"unfinished_all_minus_60": bool((open_["ret_rmst60_op_cap"] == -60.0).all()),
                                          "max_unfinished": float(open_["ret_rmst60_op_cap"].max()) if len(open_) else None,
                                          "min_completed": float(done["ret_rmst60_op_cap"].min()) if len(done) else None}
    tab = pd.DataFrame(rows)
    tab.to_csv(ART / "s6_ac6_10_returns_vs_verdict.csv", index=False)
    print("AC6.10 returns vs verdict (family W per cell; HRT as the documented macro identity; family O per cell):\n"
          + tab.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"AC6.10 max |diff| = {tab['abs_diff'].max():.2e}; all <= 1e-9 and every verdict label carries its family: {ok}")
    for k, r in rank.items():
        good = r["unfinished_all_minus_60"] and (r["max_unfinished"] is None or r["min_completed"] is None or r["max_unfinished"] <= r["min_completed"])
        r["pass"] = bool(good)
        print(f"AC6.11 REAL {k}: unfinished attempts all -60: {r['unfinished_all_minus_60']}; max unfinished {r['max_unfinished']}, "
              f"min completed {r['min_completed']}; ranking correct: {good}")
    OUT["AC6.10"] = {"max_abs_diff": float(tab["abs_diff"].max()), "pass": bool(ok)}
    OUT["AC6.11_real"] = rank


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--join", type=Path, help="a returns.parquet from python -m vlapoc.returns")
    p.add_argument("--real", type=Path, action="append", default=[])
    p.add_argument("--against-verdict", action="store_true", help="AC6.10/AC6.11 on the --real dirs")
    a = p.parse_args()
    if a.synthetic:
        synthetic()
    if a.join:
        join((REPO_ROOT / a.join).resolve())
    for r in a.real:
        real(r)
    if a.against_verdict and a.real:
        against_verdict(a.real)
    path = ART / "s6_verify_returns.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**old, **OUT}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
