"""Rollout harness: LeRobot's LIBERO env and eval loop, plus a goal-check probe, fixed init states and records.

Grounding
---------
- PoC §9.1 Step 2 (p. 18): "Run many rollouts per task (e.g. 50) with fixed initial states"; "For each
  rollout, record whether and when it succeeded, using the benchmark's goal check"; failure classification
  "with rules over simulator state" (the raw signals are recorded here; no rule is applied, plan S6).
- PoC §2.1 item 1 (p. 3): "Every eval rollout should be tied back to that checkpoint" (record keys,
  ``vlapoc.runs.eval_run``). §2.1 item 3 (p. 3): "fixed seeds and initial states".
- PoC §3.5 A (pp. 7-8): T = time (or steps) to task success, per sub-task for libero_10; timeouts censored.
  The per-operation pairs come from ``vlapoc.libero_goals.operations_from_trace`` (the single rule, S4).
- CUPID Prop. 1 (p. 5): influence on returns needs every (s', a') a rollout visited, so trajectories can be
  stored losslessly (``record_trajectories``).

What is LeRobot's and what is added (plan: reuse the eval loop, add only the probe and init-state control)
--------------------------------------------------------------------------------------------------------
- Unchanged: ``lerobot.scripts.lerobot_eval.rollout`` (``preprocess_observation`` -> env preprocessor
  ``LiberoProcessorStep`` -> policy preprocessor -> ``select_action`` -> postprocessors -> ``env.step``),
  LeRobot's ``LiberoEnv`` reset (hard reset, ``set_init_state``, 10 no-op settle steps) and step
  (``terminated = done or check_success()``), and the env processors
  (``PolicyProcessorPipeline([LiberoProcessorStep()])``, ``PolicyProcessorPipeline([])``,
  ``lerobot/envs/configs.py:446-450``), in a ``SyncVectorEnv`` with NEXT_STEP autoreset as LeRobot builds it.
- ``ProbedLiberoEnv`` (subclass) adds three things:
  1. **Fixed init state per env**: ``init_state_index`` is used on every reset (``_reset_stride = 0``).
     LeRobot's own assignment is ``init_state_id = episode_index`` then ``+= n_envs`` per reset, so which
     init state a rollout gets depends on batching (``lerobot/envs/libero.py``, plan P16).
  2. **robosuite horizon >= τ_episode**: robosuite ends an episode when ``timestep >= horizon``
     (``robosuite/environments/base.py:429``) and LIBERO's ``ControlEnv`` defaults to ``horizon=1000``,
     which LeRobot's ``OffScreenRenderEnv(...)`` call does not override. With the 10 settle steps a policy
     could act at most 990 steps, so τ_episode = 1200 (60 s) needs ``horizon = τ + settle + 1``. LeRobot's
     own budgets (280 / 520 steps) never reach it.
  3. **A per-step probe** after reset and after every step: each goal conjunct through LIBERO's
     ``_eval_predicate``, ``_check_success``, and the world positions of the task's objects of interest
     (raw signals for later failure rules), plus the sha256 of the flattened sim state at t0.
- Seeds: ``LiberoEnv.reset(seed)`` calls LIBERO's ``seed()`` = ``np.random.seed(seed)``
  (``bddl_base_domain.py:164-165``), and robosuite's placement sampler draws from the global ``np.random``
  (``placement_samplers.py:167,183,196``). Rollout j gets seed ``seed_base + j``. An *unseeded* reset
  samples a different scene each time (S3 finding).
- **One env object per rollout, by design (measured, S5 AC5.2 (b)).** LIBERO's ``_load_model`` runs on every
  hard reset and *appends* an OpenClose/TurnOnOff sampler to ``object_property_initializers``
  (``bddl_base_domain.py:653,674``); ``_reset_internal`` draws from every one of them (l. 743) before the
  placement sampler. So in the 4/20 tasks whose BDDL ``:init`` has open/close/turnon/turnoff, the k-th reset of
  one env object places a fixture differently from the first, even with the same seed (libero_10 task 8:
  ``flat_stove_1_main`` moves up to 7.2 mm per reset; ``set_init_state`` cannot undo it, because a fixture's
  position is in the model, not in the flattened state). The first seeded reset of a fresh env is reproducible.
  ``run_rollouts`` builds a fresh env per rollout and keeps only its first episode (``first_trace``); never
  reuse an env object across rollouts. ``t0_state_sha256`` does not cover fixture placement; ``t0_model_sha256``
  (``model_placement_sha256``) does, so a violation is visible in the record (plan r2 AC5.10).

T convention: index 0 of a trace is the post-settle state, so ``T_steps`` = number of policy actions until
LIBERO's goal check first holds, ``T_s = T_steps / 20``. A rollout that has not succeeded after
``tau_episode_steps`` actions is right-censored (PhAIL §3.1 p. 4). LIBERO has no unrecoverable-state rule
(plan P13, owner decision 6), so there are no ghost events and no ghost column.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from vlapoc.libero_convert import REPO_ROOT

SMOLVLM_DIR = REPO_ROOT / "data" / "hf_models" / "SmolVLM2-500M-Instruct"  # config + tokenizer @ 7b375e1b, no weights
TAU_EPISODE_STEPS = 1200  # 60 s at 20 Hz (plan: τ_episode = 60 s, owner default 2026-10-02)
FPS = 20
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # lerobot.envs.libero.get_libero_dummy_action

# Rollout record schema (one row per rollout; AC5.1 validates against it)
RECORD_COLUMNS = {
    "eval_run_id": "string", "rollout_id": "string", "suite": "string", "libero_task_id": "int64", "task_name": "string",
    "init_state_index": "int64", "seed": "int64", "checkpoint_sha256": "string", "train_run_id": "string",
    "external_checkpoint": "string", "policy_type": "string", "tau_episode_steps": "int64", "render_hw": "int64",
    "control_mode": "string", "T_steps": "float64", "T_s": "float64", "event": "int64", "censored": "bool",
    "n_ops": "int64", "n_ops_done": "int64", "ops": "string", "pairs": "string", "conjuncts": "string",
    "conjunct_trace_rle": "string", "success_trace_rle": "string", "arm_action_norm": "object", "obj_pos": "string",
    "t0_state_sha256": "string", "t0_model_sha256": "string", "sim_steps": "int64", "wall_seconds": "float64", "batch_size": "int64",
    "traj_episode_index": "float64",
}


def validate_record(df: pd.DataFrame) -> list[str]:
    """Problems with a rollout record (missing/extra columns, nulls where not allowed, broken invariants)."""
    errs = []
    missing, extra = set(RECORD_COLUMNS) - set(df.columns), set(df.columns) - set(RECORD_COLUMNS)
    if missing:
        errs.append(f"missing columns {sorted(missing)}")
    if extra:
        errs.append(f"extra columns {sorted(extra)}")
    if errs:
        return errs
    if df["rollout_id"].duplicated().any():
        errs.append("duplicate rollout_id")
    ok_event = ((df["event"] == 1) & df["T_steps"].notna() & ~df["censored"]) | ((df["event"] == 0) & df["T_steps"].isna())
    if not ok_event.all():
        errs.append(f"{int((~ok_event).sum())} rows with inconsistent event/T_steps/censored")
    if ((df["event"] == 0) & (df["sim_steps"] != df["tau_episode_steps"])).any():
        errs.append("a non-successful rollout stopped before tau_episode_steps")
    if (df["n_ops_done"] > df["n_ops"]).any():
        errs.append("n_ops_done > n_ops")
    return errs


# ---------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------


def model_placement_sha256(model) -> str:
    """sha256 of the MuJoCo model's placement after reset (plan r2 AC5.10): ``body_pos`` then ``site_pos``, each as
    C-contiguous float64 bytes, concatenated. ``t0_state_sha256`` hashes the flattened sim state, which cannot see a
    fixture's position (a fixture has no free joint: its position is in the model); LIBERO re-places fixtures on a
    re-reset of the same env object (module docstring), and this hash is what makes that visible per rollout."""
    h = hashlib.sha256()
    for arr in (model.body_pos, model.site_pos):
        h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


def _probed_env_class():
    """``ProbedLiberoEnv`` (defined lazily: LIBERO must be configured by ``setup_libero`` before import)."""
    from vlapoc import libero_goals as lg

    lg.setup_libero()
    from lerobot.envs.libero import LiberoEnv

    class ProbedLiberoEnv(LiberoEnv):
        """LeRobot ``LiberoEnv`` + fixed init state, horizon >= τ_episode, per-step goal probe (module docstring)."""

        def __init__(self, task_suite, task_id: int, task_suite_name: str, init_state_index: int,
                     tau_episode_steps: int, render_hw: int, control_mode: str = "relative"):
            super().__init__(task_suite=task_suite, task_id=task_id, task_suite_name=task_suite_name,
                             episode_length=tau_episode_steps, obs_type="pixels_agent_pos",
                             observation_width=render_hw, observation_height=render_hw, init_states=True,
                             episode_index=init_state_index, n_envs=1, control_mode=control_mode)
            self._reset_stride = 0  # every reset of this env uses init_state_index (explicit assignment)
            self.init_state_index = init_state_index
            self.trace: list[dict] = []
            self.finished: list[list[dict]] = []  # traces of episodes ended before an autoreset
            self.t0_state_sha256: list[str] = []
            self.t0_model_sha256: list[str] = []

        def _ensure_env(self) -> None:
            super()._ensure_env()
            self._env.env.horizon = self._max_episode_steps + self.num_steps_wait + 1

        def probe(self) -> dict:
            dom = self._env.env
            goal = dom.parsed_problem["goal_state"]
            pos = {n: [round(float(x), 5) for x in dom.sim.data.body_xpos[dom.obj_body_id[n]]]
                   for n in dom.obj_of_interest if n in dom.obj_body_id}
            return {"truth": [bool(dom._eval_predicate(list(g))) for g in goal], "success": bool(dom._check_success()),
                    "obj_pos": pos}

        def reset(self, seed=None, **kwargs):
            if self.trace:
                self.finished.append(self.trace)
            obs, info = super().reset(seed=seed, **kwargs)
            self.t0_state_sha256.append(hashlib.sha256(np.asarray(self._env.get_sim_state(), dtype=np.float64).tobytes()).hexdigest())
            self.t0_model_sha256.append(model_placement_sha256(self._env.env.sim.model))
            self.trace = [self.probe()]
            return obs, info

        def step(self, action):
            out = super().step(action)
            self.trace.append(self.probe())
            return out

        def first_trace(self) -> list[dict]:
            """The trace of the rollout's first episode (later ones are NEXT_STEP autoresets)."""
            return self.finished[0] if self.finished else self.trace

        def conjunct_labels(self) -> list[str]:
            return ["(" + " ".join(c) + ")" for c in self._env.env.parsed_problem["goal_state"]]

    return ProbedLiberoEnv


# ---------------------------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------------------------


@dataclass
class PolicySpec:
    """A policy plus its processors and the lineage keys of where it came from."""

    policy: Any
    preprocessor: Callable
    postprocessor: Callable
    policy_type: str
    checkpoint_sha256: str = ""
    train_run_id: str = ""
    external_checkpoint: str = ""
    device: str = "cpu"
    meta: dict = field(default_factory=dict)


def noop_policy() -> PolicySpec:
    """LIBERO's dummy action every step (arm still, gripper open); identity processors. Smoke tests only."""
    import torch
    from torch import nn

    class NoopPolicy(nn.Module):
        def reset(self):
            pass

        def select_action(self, observation):
            b = observation["observation.state"].shape[0]
            return torch.tensor([LIBERO_DUMMY_ACTION], dtype=torch.float32).repeat(b, 1)

    return PolicySpec(NoopPolicy(), lambda o: o, lambda a: a, "noop")


def checkpoint_policy(pretrained: Path, device: str = "cpu", train_run_id: str = "", external: str = "") -> PolicySpec:
    """A LeRobot checkpoint dir (``pretrained_model``): policy via ``vlapoc.runs.load_checkpoint`` (the draccus
    path that works on Python 3.14) and LeRobot's own saved processors via ``make_pre_post_processors``."""
    from lerobot.policies.factory import make_pre_post_processors

    from vlapoc.catalog import sha256_file
    from vlapoc.runs import load_checkpoint

    pretrained = Path(pretrained)
    cfg_type = json.loads((pretrained / "config.json").read_text())["type"]
    cfg_over, pre_over = {}, {"device_processor": {"device": device}}
    if cfg_type == "smolvla":
        # The checkpoint holds every weight (strict load); only the VLM's config and the tokenizer are needed
        # from HuggingFaceTB/SmolVLM2-500M-Instruct, pinned and repo-local (SMOLVLM_DIR), so nothing is
        # downloaded at load time (smolvlm_with_expert.py:90-101).
        cfg_over = {"load_vlm_weights": False, "vlm_model_name": str(SMOLVLM_DIR)}
        pre_over["tokenizer_processor"] = {"tokenizer_name": str(SMOLVLM_DIR)}
    policy = load_checkpoint(pretrained, cfg_over)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(pretrained), preprocessor_overrides=pre_over)
    return PolicySpec(policy, pre, post, policy.config.type, sha256_file(pretrained / "model.safetensors"),
                      train_run_id, external, device, {"pretrained": str(pretrained)})


# ---------------------------------------------------------------------------------------------
# trajectory capture (lossless, training schema)
# ---------------------------------------------------------------------------------------------


class _Capture:
    """Wraps the env preprocessor: stores what the policy saw at each step (images uint8 HWC after
    ``LiberoProcessorStep``, i.e. in the training orientation, and the 8-d state), per env."""

    def __init__(self, inner: Callable):
        self.inner = inner
        self.frames: list[dict] = []

    def __call__(self, observation):
        out = self.inner(observation)
        frame = {}
        for k in ("observation.images.image", "observation.images.image2"):
            img = out[k].detach().to("cpu").float()  # (B, 3, H, W) in [0, 1]
            frame[k] = (img * 255.0).round().clamp(0, 255).to(dtype=__import__("torch").uint8).permute(0, 2, 3, 1).numpy()
        frame["observation.state"] = out["observation.state"].detach().to("cpu").float().numpy()
        self.frames.append(frame)
        return out


def write_trajectories(root: Path, episodes: list[dict], fps: int = FPS) -> dict:
    """Lossless LeRobot v3 dataset (PNG images, ``use_videos=False``), one episode per rollout, in the training
    schema (``observation.images.image``/``image2``, ``observation.state`` (8), ``action`` (7)). Frame t of
    episode e = observation the policy acted on at step t and the action executed (``frame_index`` =
    policy step). Returns ``{"root", "episodes", "frames"}``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hw = episodes[0]["images"].shape[1:3]
    feats = {"observation.images.image": {"dtype": "image", "shape": (*hw, 3), "names": ["height", "width", "channel"]},
             "observation.images.image2": {"dtype": "image", "shape": (*hw, 3), "names": ["height", "width", "channel"]},
             "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
             "action": {"dtype": "float32", "shape": (7,), "names": None}}
    ds = LeRobotDataset.create(repo_id=f"local/{Path(root).name}", fps=fps, features=feats, root=root, robot_type="panda",
                               use_videos=False, image_writer_threads=4)
    n = 0
    for ep in episodes:
        for t in range(len(ep["action"])):
            ds.add_frame({"observation.images.image": ep["images"][t], "observation.images.image2": ep["images2"][t],
                          "observation.state": ep["state"][t].astype(np.float32), "action": ep["action"][t].astype(np.float32),
                          "task": ep["task"]})
            n += 1
        ds.save_episode()
    ds.finalize()
    return {"root": str(root), "episodes": len(episodes), "frames": n}


# ---------------------------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutSpec:
    """One rollout to run: task and the explicit (init state, seed) pair."""

    suite: str
    libero_task_id: int
    init_state_index: int
    seed: int


def plan_rollouts(tasks: list[tuple[str, int]], n_init: int, seed_base: int = 0, init_offset: int = 0) -> list[RolloutSpec]:
    """Rollout j of a task uses init state ``init_offset + j`` and seed ``seed_base + init_offset + j``
    (PoC §2.1 item 3, fixed seeds and initial states; plan S5 (i))."""
    return [RolloutSpec(s, t, init_offset + j, seed_base + init_offset + j) for s, t in tasks for j in range(n_init)]


def run_rollouts(spec: PolicySpec, rollouts: list[RolloutSpec], tau_episode_steps: int = TAU_EPISODE_STEPS,
                 render_hw: int = 128, batch_size: int = 1, control_mode: str = "relative",
                 record_trajectories: bool = False, traj_root: Path | None = None, eval_run_id: str = "",
                 log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict | None]:
    """Run ``rollouts`` with ``spec`` through LeRobot's ``rollout()``, in batches of ``batch_size`` envs of the
    same task; return the rollout record (``RECORD_COLUMNS``) and, if asked, the trajectory dataset summary.

    torch is seeded with the first rollout's seed of each batch (``lerobot.utils.random_utils.set_seed``), as
    ``lerobot-eval`` seeds once per run; a stochastic policy's outcome can therefore depend on batch
    composition, while the env's init state and scene do not (AC5.6 checks the latter).
    """
    from vlapoc import libero_goals as lg

    lg.setup_libero()  # before any libero import: otherwise LIBERO creates ~/.libero and prompts on stdin
    import gymnasium as gym
    import torch
    from libero.libero import benchmark
    from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.random_utils import set_seed

    from vlapoc.libero_goals import operations_from_trace, rle

    Env = _probed_env_class()
    rows, traj_eps = [], []
    by_task: dict[tuple[str, int], list[RolloutSpec]] = {}
    for r in rollouts:
        by_task.setdefault((r.suite, r.libero_task_id), []).append(r)
    suites = {}
    for (suite, tid), rs in by_task.items():
        ts = suites.setdefault(suite, benchmark.get_benchmark_dict()[suite]())
        task = ts.get_task(tid)
        for b0 in range(0, len(rs), batch_size):
            batch = rs[b0:b0 + batch_size]
            env_fns = [(lambda r=r: Env(ts, tid, suite, r.init_state_index, tau_episode_steps, render_hw, control_mode)) for r in batch]
            venv = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
            env_pre = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
            env_post = PolicyProcessorPipeline(steps=[])
            cap = _Capture(env_pre) if record_trajectories else None
            set_seed(batch[0].seed)
            t0 = time.time()
            with torch.no_grad():
                ret = rollout(venv, spec.policy, cap or env_pre, env_post, spec.preprocessor, spec.postprocessor,
                              seeds=[r.seed for r in batch])
            wall = time.time() - t0
            done = ret["done"].numpy()
            actions = ret["action"].numpy()
            for i, r in enumerate(batch):
                e = venv.envs[i].unwrapped
                n_steps = int(np.argmax(done[i]) + 1) if done[i].any() else int(done.shape[1])
                tr = e.first_trace()[: n_steps + 1]
                truth = np.array([p["truth"] for p in tr], dtype=bool)
                succ = np.array([p["success"] for p in tr], dtype=bool)
                rid = f"{suite}/{tid}/init{r.init_state_index:03d}/seed{r.seed}"
                o = operations_from_trace(truth, succ, n_steps, labels=e.conjunct_labels(), key=rid)
                t_s = o["success_step"]
                a = actions[i, :n_steps]
                row = {
                    "eval_run_id": eval_run_id, "rollout_id": rid, "suite": suite, "libero_task_id": tid, "task_name": task.name,
                    "init_state_index": r.init_state_index, "seed": r.seed, "checkpoint_sha256": spec.checkpoint_sha256,
                    "train_run_id": spec.train_run_id, "external_checkpoint": spec.external_checkpoint,
                    "policy_type": spec.policy_type, "tau_episode_steps": tau_episode_steps, "render_hw": render_hw,
                    "control_mode": control_mode, "T_steps": float(t_s) if t_s is not None else None,
                    "T_s": t_s / FPS if t_s is not None else None, "event": int(t_s is not None), "censored": bool(o["censored"]),
                    "n_ops": o["n_ops"], "n_ops_done": o["n_ops_done"],
                    "ops": json.dumps([{"k": k + 1, "t_steps": t, "conjuncts": c} for k, (t, c) in enumerate(zip(o["t_ops"], o["op_conjuncts"]))]),
                    "pairs": json.dumps(o["episode"].pairs), "conjuncts": json.dumps(e.conjunct_labels()),
                    "conjunct_trace_rle": json.dumps([rle(truth[:, c]) for c in range(truth.shape[1])]),
                    "success_trace_rle": json.dumps(rle(succ)),
                    "arm_action_norm": [round(float(x), 6) for x in np.linalg.norm(a[:, :6], axis=1)],
                    "obj_pos": json.dumps({n: [p["obj_pos"][n] for p in tr] for n in tr[0]["obj_pos"]}),
                    "t0_state_sha256": e.t0_state_sha256[0], "t0_model_sha256": e.t0_model_sha256[0], "sim_steps": n_steps, "wall_seconds": round(wall / len(batch), 3),
                    "batch_size": len(batch), "traj_episode_index": None,
                }
                lerobot_success = bool(ret["success"][i, :n_steps].any())
                if lerobot_success != (t_s is not None) or (t_s is not None and t_s != n_steps):
                    raise AssertionError(f"{rid}: probe success step {t_s} disagrees with LeRobot (success {lerobot_success}, "
                                         f"steps {n_steps})")
                if cap is not None:
                    row["traj_episode_index"] = float(len(traj_eps))
                    traj_eps.append({"images": np.stack([f["observation.images.image"][i] for f in cap.frames[:n_steps]]),
                                     "images2": np.stack([f["observation.images.image2"][i] for f in cap.frames[:n_steps]]),
                                     "state": np.stack([f["observation.state"][i] for f in cap.frames[:n_steps]]),
                                     "action": a, "task": task.language})
                rows.append(row)
                log(f"[rollouts] {rid}: {'success at step ' + str(t_s) if t_s is not None else 'censored at ' + str(n_steps)}"
                    f" ({o['n_ops_done']}/{o['n_ops']} ops), {wall / len(batch):.1f}s")
            if cap is not None:
                cap.frames.clear()  # the batch's frames are copied into traj_eps above; free them before the next batch
            venv.close()
    df = pd.DataFrame(rows)
    for c in ("T_steps", "T_s", "traj_episode_index"):
        df[c] = df[c].astype(float)
    traj = write_trajectories(Path(traj_root), traj_eps) if record_trajectories and traj_eps else None
    return df, traj


__all__ = ["TAU_EPISODE_STEPS", "RECORD_COLUMNS", "validate_record", "model_placement_sha256", "PolicySpec", "noop_policy", "checkpoint_policy",
           "RolloutSpec", "plan_rollouts", "run_rollouts", "write_trajectories"]
