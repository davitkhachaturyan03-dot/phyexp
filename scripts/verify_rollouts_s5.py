"""Verify evaluation-layer S5 harness properties: determinism across runs and resets (AC5.2), the horizon cap
of LeRobot's LiberoEnv, init-state pairing independent of batch size (AC5.6), and the trajectory store (AC5.5).

Grounding: PoC §2.1 item 3 p. 3 ("fixed seeds and initial states"), §9.1 Step 2 p. 18; CUPID Prop. 1 p. 5.

    .venv/bin/python scripts/verify_rollouts_s5.py --rerun data/eval/eval-noop-smoke-a   # AC5.2 (a): identical record
    .venv/bin/python scripts/verify_rollouts_s5.py --resets       # AC5.2 (b): one env, repeated seeded/unseeded resets
    .venv/bin/python scripts/verify_rollouts_s5.py --horizon      # LeRobot LiberoEnv unpatched: where does it stop?
    .venv/bin/python scripts/verify_rollouts_s5.py --pairing      # AC5.6: 4 rollouts, batch 1 vs batch 4
    .venv/bin/python scripts/verify_rollouts_s5.py --trajectories data/eval/<run>   # AC5.5
    .venv/bin/python scripts/verify_rollouts_s5.py --summary data/eval/<run> [...]  # AC5.4: outcomes + per-task KM (S1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl")

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
VOLATILE = {"wall_seconds", "eval_run_id"}


def _compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    cols = [c for c in a.columns if c not in VOLATILE]
    a, b = a.sort_values("rollout_id").reset_index(drop=True), b.sort_values("rollout_id").reset_index(drop=True)
    norm = lambda v: json.dumps(v.tolist() if hasattr(v, "tolist") else v, default=str)  # noqa: E731 (parquet lists -> numpy)
    diff = {c: int(sum(norm(x) != norm(y) for x, y in zip(a[c], b[c]))) for c in cols}
    return {"rows": len(a), "columns_compared": len(cols), "differing": {c: n for c, n in diff.items() if n}}


def rerun(path: Path) -> None:
    """AC5.2 (a): re-run a stored noop eval in this process and compare the records column by column."""
    from vlapoc import rollouts as ro

    ref = pd.read_parquet(path / "rollouts.parquet")
    tasks = sorted({(r.suite, int(r.libero_task_id)) for r in ref.itertuples()})
    n = int(ref.groupby(["suite", "libero_task_id"]).size().max())
    df, _ = ro.run_rollouts(ro.noop_policy(), ro.plan_rollouts(tasks, n), int(ref["tau_episode_steps"].iloc[0]),
                            int(ref["render_hw"].iloc[0]), int(ref["batch_size"].iloc[0]))
    c = _compare(ref, df)
    print(f"AC5.2 (a) re-run of {path.name} in a new process: {c['rows']} rollouts, {c['columns_compared']} columns compared "
          f"(all but {sorted(VOLATILE)}); differing columns: {c['differing'] or 'none'}")
    json.dump(c, open(ART / "s5_ac5_2a_rerun.json", "w"), indent=1)


def _scene_digest(env) -> dict:
    sim = env._env.sim
    return {"state": hashlib.sha256(np.asarray(env._env.get_sim_state(), np.float64).tobytes()).hexdigest()[:16],
            "body_pos": hashlib.sha256(np.asarray(sim.model.body_pos, np.float64).tobytes()).hexdigest()[:16],
            "xpos": hashlib.sha256(np.asarray(sim.data.xpos, np.float64).tobytes()).hexdigest()[:16]}


def resets() -> None:
    """One ProbedLiberoEnv (libero_10 task 8, init state 0): post-settle digests over repeated resets."""
    import h5py

    from vlapoc import libero_goals as lg
    from vlapoc import rollouts as ro

    Env = ro._probed_env_class()
    from libero.libero import benchmark

    ts = benchmark.get_benchmark_dict()["libero_10"]()
    env = Env(ts, 8, "libero_10", 0, 1200, 128)
    out = {}
    for label, seeds in (("seed 0 x3", [0, 0, 0]), ("seed 1 x2", [1, 1]), ("unseeded x3", [None, None, None])):
        d = []
        for s in seeds:
            env.reset(seed=s)
            d.append(_scene_digest(env))
        out[label] = d
        print(f"AC5.2 (b) {label}: post-settle sim-state sha {[x['state'] for x in d]}; model body_pos sha "
              f"{[x['body_pos'] for x in d]}; world xpos sha {[x['xpos'] for x in d]}")
    # the S3 case: demo_19 of libero_10 task 8 replayed through the goal check in the BDDL-built scene, seeded
    from vlapoc.lineage import version_episodes

    e = version_episodes("v0").set_index("episode_id").loc["libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/demo_19"]
    with h5py.File(REPO_ROOT / e["source_file"]) as f:
        states = f[f"data/{e['source_demo']}/states"][()]
    renv = lg.make_replay_env(lg.bddl_path("libero_10", e["task_name"]))
    rep = {}
    for label, seeds in (("seed 0 x3", [0, 0, 0]), ("seed 1 x3", [1, 1, 1]), ("unseeded x3", [None, None, None])):
        T = []
        for s in seeds:
            renv.seed(s)
            T.append(lg.first_true(lg.replay_goal_trace(renv, states, None)["success"]))
        rep[label] = T
        print(f"AC5.2 (b) S3 case demo_19 (task 8), BDDL scene, {label}: first-success stored state {T}")
    json.dump({"env_resets": out, "s3_case_replay": rep}, open(ART / "s5_ac5_2b_resets.json", "w"), indent=1)


def horizon() -> None:
    """LeRobot's LiberoEnv as shipped (episode_length 1200), noop actions: at which step does it terminate?"""
    from vlapoc import libero_goals as lg

    lg.setup_libero()
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    ts = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(ts, 0, "libero_spatial", episode_length=1200, observation_width=128, observation_height=128)
    env.reset(seed=0)
    t0 = time.time()
    term, err, last_ok = False, None, 0
    for k in range(1, 1201):
        try:
            _, _, term, trunc, info = env.step(np.array(get_libero_dummy_action(), dtype=np.float32))
        except ValueError as e:  # LIBERO's step replaces robosuite's done by _check_success (bddl_base_domain.py:808-809)
            err = f"{type(e).__name__}: {e}"
            break
        last_ok = k
        if term or trunc:
            break
    print(f"horizon: unpatched LeRobot LiberoEnv(episode_length=1200), noop: {last_ok} policy steps completed, then "
          f"{'exception ' + repr(err) if err else 'terminated=' + str(term)} at step {k}; inner horizon = "
          f"{env._env.env.horizon}, settle steps = {env.num_steps_wait}; {time.time() - t0:.1f}s")
    json.dump({"steps_completed": last_ok, "failed_at_step": k if err else None, "error": err, "terminated": bool(term),
               "horizon": env._env.env.horizon, "settle_steps": env.num_steps_wait}, open(ART / "s5_horizon_check.json", "w"), indent=1)


def pairing() -> None:
    """AC5.6: the same 4 rollouts at batch 1 and batch 4 start from the same init state and t0 sim state."""
    from vlapoc import rollouts as ro

    plan = ro.plan_rollouts([("libero_spatial", 3)], 4)
    res = {}
    for b in (1, 4):
        df, _ = ro.run_rollouts(ro.noop_policy(), plan, 40, 128, batch_size=b)
        res[b] = df.set_index("rollout_id")[["init_state_index", "seed", "t0_state_sha256"]]
        print(f"AC5.6 batch_size={b}: " + "; ".join(f"{i}: init {r.init_state_index} t0 {r.t0_state_sha256[:12]}"
                                                    for i, r in res[b].iterrows()))
    same = res[1].equals(res[4])
    print(f"AC5.6 identical (init_state_index, seed, t0 sim-state sha256) for batch 1 vs 4: {same}")
    json.dump({"batch1": res[1].reset_index().to_dict("records"), "batch4": res[4].reset_index().to_dict("records"),
               "identical": same}, open(ART / "s5_ac5_6_pairing.json", "w"), indent=1)


def trajectories(path: Path) -> None:
    """AC5.5: the lossless trajectory store loads with LeRobotDataset; frames == sum(sim_steps); every frame maps to
    (rollout_id, policy_step) through ``traj_episode_index`` (episode) and ``frame_index`` (policy step)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vlapoc.lineage import isolated_hf_datasets_cache

    rec = pd.read_parquet(path / "rollouts.parquet")
    with isolated_hf_datasets_cache():
        ds = LeRobotDataset("local/trajectories", root=path / "trajectories")
        frames = ds.hf_dataset.select_columns(["episode_index", "frame_index"]).to_pandas()
        f0 = ds[0]
    by_ep = frames.groupby("episode_index")["frame_index"].apply(list).to_dict()
    ep_of = {int(r.traj_episode_index): r for r in rec.itertuples()}
    lengths_ok = all(len(by_ep[e]) == int(ep_of[e].sim_steps) for e in by_ep) and set(by_ep) == set(ep_of)
    steps_ok = all(v == list(range(len(v))) for v in by_ep.values())
    img = tuple(f0["observation.images.image"].shape)
    print(f"AC5.5 {path.name}: LeRobotDataset loads: {ds.num_episodes} episodes, {ds.num_frames} frames; sum(sim_steps) = "
          f"{int(rec['sim_steps'].sum())}; episodes == rollouts' traj_episode_index and per-episode length == sim_steps: {lengths_ok}; "
          f"frame_index == policy step 0..sim_steps-1 in every episode: {steps_ok}; video keys: {ds.meta.video_keys}; "
          f"first frame image {img}, state {tuple(f0['observation.state'].shape)}, action {tuple(f0['action'].shape)}")
    json.dump({"run_dir": path.name, "episodes": ds.num_episodes, "frames": ds.num_frames, "sum_sim_steps": int(rec["sim_steps"].sum()),
               "lengths_match": lengths_ok, "frame_index_is_policy_step": steps_ok, "video_keys": list(ds.meta.video_keys), "image_shape": img},
              open(ART / f"s5_ac5_5_trajectories_{path.name}.json", "w"), indent=1)


LEROBOT_BUDGET = {"libero_spatial": 280, "libero_10": 520}  # LeRobot's per-suite max steps (docs/lerobot/libero_plus table)


def summary(paths: list[Path]) -> None:
    """AC5.4: real outcomes of an external checkpoint per task: T, ops, censoring, per-task KM through S1."""
    from vlapoc import evaluation as ev

    rec = pd.concat([pd.read_parquet(p / "rollouts.parquet") for p in paths], ignore_index=True)
    out = {"runs": sorted(set(rec["eval_run_id"])), "external_checkpoint": sorted(set(rec["external_checkpoint"])),
           "label": "external checkpoint (not trained on our data), harness validation, not attribution-grade", "tasks": {}}
    for (suite, tid), g in rec.groupby(["suite", "libero_task_id"]):
        eps = [ev.libero_episode(r.rollout_id, [t / 20 for t in [o["t_steps"] for o in json.loads(r.ops)]],
                                 r.T_s if r.event else r.tau_episode_steps / 20, int(r.n_ops)) for r in g.itertuples()]
        whole = [ev.libero_episode(r.rollout_id, [r.T_s] if r.event else [], r.T_s if r.event else r.tau_episode_steps / 20, 1)
                 for r in g.itertuples()]
        b = LEROBOT_BUDGET[suite]
        t = {"n": len(g), "successes": int(g["event"].sum()), "censored": int(g["censored"].sum()),
             f"success_within_lerobot_budget_{b}_steps": int(((g["event"] == 1) & (g["T_steps"] <= b)).sum()),
             "T_steps_successes": sorted(int(x) for x in g.loc[g["event"] == 1, "T_steps"]),
             "n_ops": int(g["n_ops"].iloc[0]), "n_ops_done": g["n_ops_done"].value_counts().sort_index().to_dict(),
             "whole_task_KM": {"F30": ev.F_at(whole, 30.0), "F60": ev.F_at(whole, 60.0), "median_s": ev.median_T(whole),
                               "RMST60_s": ev.rmst(whole, 60.0)},
             "operation_KM": {"n_pairs": sum(len(e.pairs) for e in eps), "F30": ev.F_at(eps, 30.0), "RMST60_s": ev.rmst(eps, 60.0)},
             "wall_s_per_rollout_mean": round(float(g["wall_seconds"].mean()), 1), "sim_steps_total": int(g["sim_steps"].sum())}
        out["tasks"][f"{suite}:{tid}"] = t
        print(f"AC5.4 {suite}:{tid} ({g['task_name'].iloc[0]}): {t['successes']}/{t['n']} succeeded within 1200 steps "
              f"({t[f'success_within_lerobot_budget_{b}_steps']}/{t['n']} within LeRobot's {b}-step budget), censored {t['censored']}; "
              f"T (steps) of successes {t['T_steps_successes']}; ops done {t['n_ops_done']} of {t['n_ops']}; KM whole-task "
              f"F(30 s) {t['whole_task_KM']['F30']:.2f}, F(60 s) {t['whole_task_KM']['F60']:.2f}, median {t['whole_task_KM']['median_s']}, "
              f"RMST(60) {t['whole_task_KM']['RMST60_s']:.2f} s; operation pairs {t['operation_KM']['n_pairs']}; "
              f"wall {t['wall_s_per_rollout_mean']} s/rollout")
    json.dump(out, open(ART / "s5_ac5_4_summary.json", "w"), indent=1, default=str)


def grade_tests(lineage_root: Path) -> None:
    """AC5.9 (r2): attribution grade is asserted by the caller and protocol-checked, in a scratch lineage store."""
    from vlapoc import runs
    from vlapoc.lineage import LineageError

    c = runs.client(lineage_root)
    exp = runs.experiment_id(lineage_root)
    n0 = len(c.search_runs([exp], max_results=5000))
    refusals = {}
    for label, args in (("2 init states", dict(tasks=[("libero_spatial", 0)], n_init=2)),
                        ("full protocol, smoke checkpoint", dict(tasks=runs.CANONICAL_TASKS, n_init=50))):
        try:
            runs.eval_run("train:ff64512b40974587a9d711cd8e4f8b04", name=f"eval-grade-refused [smoke]", tau_episode_steps=1200,
                          batch_size=10, attribution_grade=True, lineage_root=lineage_root, **args)
            refusals[label] = None
        except LineageError as e:
            refusals[label] = str(e)
    n1 = len(c.search_runs([exp], max_results=5000))
    out = runs.eval_run("noop", [("libero_spatial", 0)], 1, "eval-grade-default [smoke]", tau_episode_steps=20, smoke=True,
                        lineage_root=lineage_root)
    tags = c.get_run(out["run_id"]).data.tags
    params = c.get_run(out["run_id"]).data.params
    for k, v in refusals.items():
        print(f"AC5.9 (a) attribution_grade=True, {k}: refused: {v is not None} -> {v}")
    print(f"AC5.9 runs in the store before/after the refusals: {n0}/{n1} (no run created)")
    print(f"AC5.9 (b) default eval run {out['run_id'][:8]}: eval.attribution_grade={tags.get('eval.attribution_grade')}, "
          f"eval.attribution_grade.asserted={tags.get('eval.attribution_grade.asserted', 'absent')}, vlapoc.smoke={tags.get('vlapoc.smoke')}, "
          f"eval.batch_plan={params.get('eval.batch_plan')}, eval.batch_plan.sha256={tags.get('eval.batch_plan.sha256', '')[:12]}")
    json.dump({"refusals": refusals, "runs_before_after": [n0, n1], "default_run": out["run_id"],
               "default_tags": {k: tags.get(k) for k in ("eval.attribution_grade", "eval.attribution_grade.asserted", "vlapoc.smoke", "eval.batch_plan.sha256")},
               "pass": all(v is not None for v in refusals.values()) and n0 == n1 and tags.get("eval.attribution_grade") == "false"
               and "eval.attribution_grade.asserted" not in tags}, open(ART / "s5_ac5_9_grade_tests.json", "w"), indent=1)


def t0_model(compare_dir: Path | None) -> None:
    """AC5.10 (r2): t0_model_sha256 is a function of (task, init state): equal across batch sizes and repeated runs,
    and (optionally) equal to another eval run's record; a reused env object re-reset changes it (negative test)."""
    from vlapoc import rollouts as ro

    plan = ro.plan_rollouts([("libero_10", 8), ("libero_spatial", 0)], 2)
    dfs = {b: ro.run_rollouts(ro.noop_policy(), plan, 20, 128, batch_size=b, log=lambda m: None)[0] for b in (1, 2)}
    rep = ro.run_rollouts(ro.noop_policy(), plan, 20, 128, batch_size=2, log=lambda m: None)[0]
    key = lambda d: dict(zip(d["suite"] + "/" + d["libero_task_id"].astype(str) + "/init" + d["init_state_index"].astype(str), d["t0_model_sha256"]))  # noqa: E731
    k1, k2, k3 = key(dfs[1]), key(dfs[2]), key(rep)
    same_batch = k1 == k2
    same_rep = k2 == k3
    distinct_inits = len(set(k1.values())) == len(k1)
    print(f"AC5.10 t0_model_sha256 per (task, init) at batch 1: { {k: v[:12] for k, v in k1.items()} }")
    print(f"AC5.10 equal at batch 2: {same_batch}; equal in a repeated run (batch 2): {same_rep}; distinct across (task, init): {distinct_inits}")
    cross = None
    if compare_dir is not None:
        other = key(pd.read_parquet(compare_dir / "rollouts.parquet"))
        common = sorted(set(other) & set(k1))
        cross = all(other[k] == k1[k] for k in common) if common else None
        print(f"AC5.10 vs the records of {compare_dir.name} (another eval run, another process) on {common}: equal: {cross}")
    Env = ro._probed_env_class()
    from libero.libero import benchmark

    env = Env(benchmark.get_benchmark_dict()["libero_10"](), 8, "libero_10", 0, 1200, 128)
    for _ in range(3):
        env.reset(seed=0)
    env.close()
    reused = [h[:12] for h in env.t0_model_sha256]
    st = [h[:12] for h in env.t0_state_sha256]
    print(f"AC5.10 negative test, one env reset(seed=0) x3 on libero_10:8: t0_model_sha256 {reused} (distinct: {len(set(reused)) == 3}); "
          f"t0_state_sha256 {st} (blind to it: {len(set(st)) == 1})")
    errs = ro.validate_record(dfs[1].drop(columns=["t0_model_sha256"]))
    print(f"AC5.10 validate_record without the column: {errs}")
    json.dump({"batch1": k1, "batch2": k2, "repeat": k3, "cross_run_equal": cross, "reused_env_model": reused, "reused_env_state": st,
               "validate_without_column": errs,
               "pass": same_batch and same_rep and distinct_inits and cross is not False and len(set(reused)) == 3 and bool(errs)},
              open(ART / "s5_ac5_10_t0_model.json", "w"), indent=1)


def main() -> int:
    """Run the selected mode (usage in the module docstring)."""
    p = argparse.ArgumentParser()
    p.add_argument("--rerun", type=Path)
    p.add_argument("--resets", action="store_true")
    p.add_argument("--horizon", action="store_true")
    p.add_argument("--pairing", action="store_true")
    p.add_argument("--trajectories", type=Path, nargs="+")
    p.add_argument("--summary", type=Path, nargs="+", help="AC5.4: eval output dirs (external checkpoint)")
    p.add_argument("--grade-tests", type=Path, help="AC5.9 (r2): a scratch lineage store")
    p.add_argument("--t0-model", action="store_true", help="AC5.10 (r2)")
    p.add_argument("--compare", type=Path, help="with --t0-model: another eval output dir to compare t0_model_sha256 with")
    a = p.parse_args()
    if a.rerun:
        rerun(a.rerun)
    if a.resets:
        resets()
    if a.horizon:
        horizon()
    if a.pairing:
        pairing()
    if a.trajectories:
        for t in a.trajectories:
            trajectories((REPO_ROOT / t).resolve())
    if a.summary:
        summary([(REPO_ROOT / x).resolve() for x in a.summary])
    if a.grade_tests:
        grade_tests((REPO_ROOT / a.grade_tests).resolve())
    if a.t0_model:
        t0_model((REPO_ROOT / a.compare).resolve() if a.compare else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
