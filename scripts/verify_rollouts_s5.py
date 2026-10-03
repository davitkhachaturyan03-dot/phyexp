"""Verify evaluation-layer S5 harness properties: determinism across runs and resets (AC5.2), the horizon cap
of LeRobot's LiberoEnv, init-state pairing independent of batch size (AC5.6), and the trajectory store (AC5.5).

Grounding: PoC §2.1 item 3 p. 3 ("fixed seeds and initial states"), §9.1 Step 2 p. 18; CUPID Prop. 1 p. 5.

    .venv/bin/python scripts/verify_rollouts_s5.py --rerun data/eval/eval-noop-smoke-a   # AC5.2 (a): identical record
    .venv/bin/python scripts/verify_rollouts_s5.py --resets       # AC5.2 (b): one env, repeated seeded/unseeded resets
    .venv/bin/python scripts/verify_rollouts_s5.py --horizon      # LeRobot LiberoEnv unpatched: where does it stop?
    .venv/bin/python scripts/verify_rollouts_s5.py --pairing      # AC5.6: 4 rollouts, batch 1 vs batch 4
    .venv/bin/python scripts/verify_rollouts_s5.py --trajectories data/eval/<run>   # AC5.5
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
    diff = {c: int(sum(json.dumps(x, default=str) != json.dumps(y, default=str) for x, y in zip(a[c], b[c]))) for c in cols}
    return {"rows": len(a), "columns_compared": len(cols), "differing": {c: n for c, n in diff.items() if n}}


def rerun(path: Path) -> None:
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
    for k in range(1, 1201):
        _, _, term, trunc, info = env.step(np.array(get_libero_dummy_action(), dtype=np.float32))
        if term or trunc:
            break
    print(f"horizon: unpatched LeRobot LiberoEnv(episode_length=1200) terminated after {k} policy steps "
          f"(terminated={term}, is_success={info['is_success']}, robosuite done={info.get('done')}); "
          f"inner horizon = {env._env.env.horizon}; {time.time() - t0:.1f}s")
    json.dump({"terminated_after_steps": k, "is_success": bool(info["is_success"]), "robosuite_done": bool(info.get("done")),
               "horizon": env._env.env.horizon}, open(ART / "s5_horizon_check.json", "w"), indent=1)


def pairing() -> None:
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
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from vlapoc.lineage import isolated_hf_datasets_cache

    rec = pd.read_parquet(path / "rollouts.parquet")
    with isolated_hf_datasets_cache():
        ds = LeRobotDataset("local/trajectories", root=path / "trajectories")
        meta = ds.meta.episodes.to_pandas() if hasattr(ds.meta.episodes, "to_pandas") else pd.DataFrame(ds.meta.episodes)
        n = ds.num_frames
        f0 = ds[0]
    lens = dict(zip(meta["episode_index"], meta["length"]))
    m = rec.dropna(subset=["traj_episode_index"])
    ok_len = all(int(lens[int(r.traj_episode_index)]) == int(r.sim_steps) for r in m.itertuples())
    print(f"AC5.5 trajectories: LeRobotDataset loads: {ds.num_episodes} episodes, {n} frames; sum(sim_steps) = "
          f"{int(rec['sim_steps'].sum())}; per-rollout length == sim_steps: {ok_len}; each frame = (episode_index -> rollout_id via "
          f"traj_episode_index, frame_index = policy step); first frame keys {sorted(k for k in f0 if not k.startswith('task'))}, "
          f"image {tuple(f0['observation.images.image'].shape)}")
    json.dump({"episodes": ds.num_episodes, "frames": n, "sum_sim_steps": int(rec["sim_steps"].sum()), "lengths_match": ok_len},
              open(ART / "s5_ac5_5_trajectories.json", "w"), indent=1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rerun", type=Path)
    p.add_argument("--resets", action="store_true")
    p.add_argument("--horizon", action="store_true")
    p.add_argument("--pairing", action="store_true")
    p.add_argument("--trajectories", type=Path)
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
        trajectories(a.trajectories)
    return 0


if __name__ == "__main__":
    sys.exit(main())
