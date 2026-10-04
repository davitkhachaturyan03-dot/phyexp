"""AC5.2 (b), determinism across resets: is the scene a rollout starts from a function of (task, init state, seed)?

The batch-1 finding (S3) was that the same stored demo states gave different goal-check results across resets of
one BDDL-built LIBERO scene. ``verify_rollouts_s5.py --resets`` (round 1) measured that the flattened sim state
after ``set_init_state`` is identical across resets while the *model* (``body_pos``) is not, even with the same
seed. This script separates the two ways a scene could vary, for a task with a LIBERO property initializer
(libero_10 task 8, ``(turnon flat_stove_1)`` in ``:init``) and one without (libero_spatial task 0):

1. the k-th reset of one env object, same seed (what NEXT_STEP autoreset or env reuse would see);
2. the first seeded reset of fresh env objects (what ``vlapoc.rollouts.run_rollouts`` does: one env per rollout);
3. (AC5.6, strengthened) the harness's ``SyncVectorEnv`` with batch 4 vs batch 1 on the property-initializer
   task, comparing the model's ``body_pos``/``site_pos`` as well as the flattened state.

Mechanism checked against LIBERO's source (``bddl_base_domain.py``): ``_load_model`` (run on every hard reset)
calls ``_setup_placement_initializer``, which *appends* an OpenClose/TurnOnOff sampler to
``self.object_property_initializers`` (lines 653, 674); ``_reset_internal`` samples every one of them from the
global ``np.random`` (line 743) before the placement sampler. So the list, and the random draws consumed before
placement, grow with the reset count.

Grounding: PoC §2.1 item 3 p. 3 ("fixed seeds and initial states"), §9.1 Step 2 p. 18 ("fixed initial states").

    .venv/bin/python scripts/verify_resets_s5.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl")
ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:12]


def snapshot(env) -> dict:
    dom = env._env.env
    m = dom.sim.model
    names = [m.body_id2name(i) for i in range(m.nbody)]
    return {"state": _sha(env._env.get_sim_state()), "body_pos": _sha(m.body_pos), "site_pos": _sha(m.site_pos),
            "n_property_initializers": len(dom.object_property_initializers),
            "_body_pos": np.array(m.body_pos), "_names": names}


def diff(a: dict, b: dict) -> dict:
    if a["_names"] != b["_names"]:
        return {"same_body_names": False}
    d = np.abs(a["_body_pos"] - b["_body_pos"]).max(axis=1)
    moved = [(a["_names"][i], round(float(d[i]), 4)) for i in np.argsort(-d) if d[i] > 0]
    return {"same_body_names": True, "n_bodies": len(d), "n_moved": len(moved), "max_abs_m": round(float(d.max()), 4),
            "moved_top": moved[:6]}


def main() -> int:
    from vlapoc import rollouts as ro

    Env = ro._probed_env_class()
    from libero.libero import benchmark

    out = {}
    for suite, tid in (("libero_spatial", 0), ("libero_10", 8)):
        ts = benchmark.get_benchmark_dict()[suite]()
        key = f"{suite}:{tid}"
        # (1) one env object, reset(seed=0) three times
        env = Env(ts, tid, suite, 0, 1200, 128)
        same_obj = []
        for _ in range(3):
            env.reset(seed=0)
            same_obj.append(snapshot(env))
        env.close()
        # (2) three fresh env objects, one reset(seed=0) each (the harness's pattern)
        fresh = []
        for _ in range(3):
            e = Env(ts, tid, suite, 0, 1200, 128)
            e.reset(seed=0)
            fresh.append(snapshot(e))
            e.close()
        pub = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}  # noqa: E731
        out[key] = {
            "same_env_resets_seed0": [pub(s) for s in same_obj],
            "same_env_reset2_vs_reset1": diff(same_obj[0], same_obj[1]),
            "fresh_envs_seed0": [pub(s) for s in fresh],
            "fresh_env2_vs_env1": diff(fresh[0], fresh[1]),
            "fresh_env1_vs_same_env_reset1": diff(fresh[0], same_obj[0]),
        }
        r = out[key]
        print(f"{key} one env, reset(seed=0) x3: state {[s['state'] for s in r['same_env_resets_seed0']]} body_pos "
              f"{[s['body_pos'] for s in r['same_env_resets_seed0']]} property initializers "
              f"{[s['n_property_initializers'] for s in r['same_env_resets_seed0']]}; reset2 vs reset1: {r['same_env_reset2_vs_reset1']}")
        print(f"{key} fresh envs, first reset(seed=0) each x3: state {[s['state'] for s in r['fresh_envs_seed0']]} body_pos "
              f"{[s['body_pos'] for s in r['fresh_envs_seed0']]} property initializers "
              f"{[s['n_property_initializers'] for s in r['fresh_envs_seed0']]}; env2 vs env1: {r['fresh_env2_vs_env1']}")
    # (3) AC5.6 strengthened for a property-initializer task: the harness's SyncVectorEnv with batch 4 vs batch 1,
    # comparing the model (fixture placement) as well as the flattened state, which does not contain body_pos.
    import gymnasium as gym

    suite, tid = "libero_10", 8
    ts = benchmark.get_benchmark_dict()[suite]()
    by_batch = {}
    for b in (1, 4):
        snaps = []
        for b0 in range(0, 4, b):
            ids = list(range(b0, b0 + b))
            venv = gym.vector.SyncVectorEnv([(lambda j=j: Env(ts, tid, suite, j, 1200, 128)) for j in ids],
                                            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            venv.reset(seed=ids)
            snaps += [{k: v for k, v in snapshot(e.unwrapped).items() if not k.startswith("_")} for e in venv.envs]
            venv.close()
        by_batch[b] = snaps
    same = by_batch[1] == by_batch[4]
    out["pairing_libero_10_8_batch1_vs_batch4"] = {"batch1": by_batch[1], "batch4": by_batch[4], "identical": same}
    print(f"AC5.6 strengthened, libero_10:8 init/seed 0-3: batch 1 body_pos {[s['body_pos'] for s in by_batch[1]]}; "
          f"batch 4 body_pos {[s['body_pos'] for s in by_batch[4]]}; (state, body_pos, site_pos) identical: {same}")
    json.dump(out, open(ART / "s5_ac5_2b_resets_r2.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
