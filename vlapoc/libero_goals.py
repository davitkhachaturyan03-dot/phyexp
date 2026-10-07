"""LIBERO goal predicates, repo-local LIBERO setup, and demo replay through LIBERO's own goal check.

Grounding
---------
- PoC §3.5 A (pp. 7-8): "Human reference = durations of LIBERO's original human-teleoperated
  demonstrations on the same tasks. This is approximate, since demos and rollouts don't share initial
  states." PoC §10 (p. 20): "HRT against LIBERO demo durations". PoC §9.1 Step 2.2 (p. 18): "record whether
  and when it succeeded, using the benchmark's goal check".
- Why replay instead of the stored duration (evaluation-layer plan P6-P8): LIBERO's
  ``scripts/create_dataset.py`` (GitHub 8f1084e, l. 228-231) sets ``rewards[-1] = 1`` and ``dones[-1] = 1``
  unconditionally, so the stored reward is not a goal check; and human collection only stops after the
  goal has held for 10 consecutive steps (``collect_demonstration.py``: ``task_completion_hold_count =
  10``). The time-to-success of a demo is therefore the first stored state at which LIBERO's own goal
  check holds, not its length.
- LIBERO's goal check (hf-libero 0.1.4 = LIBERO source): ``_check_success`` is the conjunction of
  ``_eval_predicate(state)`` over ``self.parsed_problem["goal_state"]`` (e.g.
  ``envs/problems/libero_tabletop_manipulation.py:135-162``; one ``_check_success`` per problem class,
  all the same conjunction), with ``parsed_problem = BDDLUtils.robosuite_parse_problem(bddl_file)``
  (``envs/bddl_base_domain.py:127``). This module calls those functions; it does not re-implement a
  predicate.
- Replay recipe = LIBERO's own (``create_dataset.py`` l. 130-153): ``reset_from_xml_string(
  postprocess_model_xml(model_file))``, ``sim.reset()``, then ``sim.set_state_from_flattened(states[t])`` +
  ``sim.forward()`` for every stored state. The demo's own XML is required: building the scene from the
  BDDL instead does not reproduce success for some demos (plan P9; reproduced by
  ``scripts/human_reference.py --bddl-check``).

Alignment conventions (AC3.5)
-----------------------------
- Human T counts from **stored state 0**, which is original collection step 5: ``create_dataset.py``
  keeps ``states[valid_index]`` with ``j >= cap_index = 5`` ("the force sensor is not stable in the
  beginning"). ``T_success_steps`` = index of the first stored state where ``_check_success()`` holds;
  ``T_success_s`` = steps / 20 (control_freq 20 Hz).
- Rollout T (S5) counts from the **post-settle state**: LeRobot's ``LiberoEnv.reset`` steps
  ``num_steps_wait = 10`` no-op actions after setting the init state (``lerobot/envs/libero.py``).
- Residual misalignment: up to 5 steps = 0.25 s (inferred: 5 dropped collection steps vs 10 settle
  steps, both before the operator or policy acts). On top of that, demos and rollouts do not share
  initial states (PoC p. 8: "This is approximate"), and the human reference is replayed in the demo XML
  while rollouts run in the BDDL-built scene (P9). The report states all three.

Repo-local LIBERO configuration (never ``~/.libero``)
-----------------------------------------------------
``import libero.libero`` writes ``$LIBERO_CONFIG_PATH/config.yaml`` and, if it is missing, *prompts on
stdin* (``libero/libero/__init__.py``); the default path is ``~/.libero``. ``setup_libero()`` therefore
sets ``LIBERO_CONFIG_PATH=data/libero_config``, writes the config itself before the first import, and
pins the assets: ``get_assets_path()`` returns the module-level ``_assets_path_cache`` when that path
exists, so the assets come from ``data/libero_assets`` (HF dataset ``lerobot/libero-assets`` at
``ASSETS_REVISION``) instead of an on-demand download to ``~/.cache/libero/assets``. Call it before
anything imports ``libero`` (``lerobot.envs.libero`` included).

Where it sits (docs/dataflow.md)
--------------------------------
Step 2, item 2 ("success + when, via the benchmark's goal check") and the human reference that HRT and AUC are
measured against. ``scripts/human_reference.py`` replays the 1000 v0 demos through ``replay_goal_trace`` once
(S3) and turns the stored traces into per-operation T with ``operations_from_trace`` (S4). The rollout harness
``vlapoc.rollouts`` calls the same ``operations_from_trace`` on every rollout (S5), so a demo and a rollout get T
by one rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from itertools import groupby
from pathlib import Path

import numpy as np

from vlapoc.libero_convert import REPO_ROOT

LIBERO_CONFIG_DIR = REPO_ROOT / "data" / "libero_config"
LIBERO_ASSETS_DIR = REPO_ROOT / "data" / "libero_assets"
ASSETS_REPO = "lerobot/libero-assets"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
FPS = 20
# Absolute asset roots baked into the released demos' model XML (measured over all 1000 demos).
AUTHOR_ASSET_PREFIXES = ("/Users/yifengz/workspace/libero-dev/chiliocosm/assets/",
                         "/home/yifengz/workspace/libero-dev/chiliocosm/assets/")


def _libero_package_root() -> Path:
    """``.../site-packages/libero/libero`` located without importing it (importing would read the config)."""
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("libero (hf-libero) is not installed in this interpreter")
    return Path(list(spec.submodule_search_locations)[0]) / "libero"


def setup_libero():
    """Configure LIBERO for this repo and import it; returns the ``libero.libero`` module.

    Writes ``data/libero_config/config.yaml`` (bddl_files and init_states from the installed package,
    assets = ``data/libero_assets``) if missing, sets ``LIBERO_CONFIG_PATH`` and pins the assets path.
    Raises if the pinned assets are not present (``scripts/human_reference.py --fetch-assets`` gets them).
    """
    import sys

    if "libero.libero" in sys.modules and os.environ.get("LIBERO_CONFIG_PATH") != str(LIBERO_CONFIG_DIR):
        raise RuntimeError("libero was imported before setup_libero(); its config path is already fixed (~/.libero?)")
    os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG_DIR)
    cfg = LIBERO_CONFIG_DIR / "config.yaml"
    if not cfg.is_file():
        import yaml

        root = _libero_package_root()
        LIBERO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        cfg.write_text(yaml.dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
                                  "init_states": str(root / "init_files"),
                                  "datasets": str(REPO_ROOT / "data" / "raw" / "libero"),
                                  "assets": str(LIBERO_ASSETS_DIR)}))
    if not (LIBERO_ASSETS_DIR / "scenes").is_dir():
        raise FileNotFoundError(f"LIBERO assets missing at {LIBERO_ASSETS_DIR}; fetch {ASSETS_REPO}@{ASSETS_REVISION}")
    import libero.libero as ll

    ll._assets_path_cache = str(LIBERO_ASSETS_DIR)
    return ll


def fetch_assets() -> str:
    """Download the pinned LIBERO assets (HF dataset ``lerobot/libero-assets`` @ ``ASSETS_REVISION``)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(ASSETS_REPO, repo_type="dataset", revision=ASSETS_REVISION, local_dir=str(LIBERO_ASSETS_DIR))


def assets_digest(root: Path = LIBERO_ASSETS_DIR) -> dict:
    """sha256 over the sorted (relative path, file sha256) list of the assets dir (``.cache`` excluded)."""
    files = sorted(p for p in Path(root).rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts)
    h = hashlib.sha256()
    total = 0
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
        total += p.stat().st_size
    return {"files": len(files), "bytes": total, "sha256": h.hexdigest()}


def bddl_path(suite: str, task_name: str) -> Path:
    """The installed BDDL file of a task (``<bddl_files>/<suite>/<task_name>.bddl``)."""
    from libero.libero import get_libero_path

    return Path(get_libero_path("bddl_files")) / suite / f"{task_name}.bddl"


def goal_conjuncts(bddl_file: Path) -> list[tuple[str, ...]]:
    """The task's goal conjuncts as LIBERO parses them (``robosuite_parse_problem(...)["goal_state"]``),
    e.g. ``[("on", "akita_black_bowl_1", "plate_1")]``."""
    from libero.libero.envs import bddl_utils

    return [tuple(c) for c in bddl_utils.robosuite_parse_problem(str(bddl_file))["goal_state"]]


def conjunct_label(c: tuple) -> str:
    """``("on", "a", "b")`` -> ``"(on a b)"``."""
    return "(" + " ".join(c) + ")"


def make_replay_env(bddl_file: Path):
    """A LIBERO ``ControlEnv`` for the task, without renderers or camera observations (predicates only)."""
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(bddl_file_name=str(bddl_file), has_renderer=False, has_offscreen_renderer=False,
                      use_camera_obs=False)


def demo_model_xml(model_file: str) -> str:
    """The demo's stored MuJoCo XML with the authors' absolute asset roots mapped to ``data/libero_assets``,
    then LIBERO's own ``postprocess_model_xml`` (which remaps the robosuite paths)."""
    from libero.libero.utils.utils import postprocess_model_xml

    xml = model_file
    for prefix in AUTHOR_ASSET_PREFIXES:
        xml = xml.replace(prefix, str(LIBERO_ASSETS_DIR) + "/")
    left = sorted({m.group(1) for m in re.finditer(r'file="(/[^"]*?/assets/)', xml)
                   if not m.group(1).startswith(str(LIBERO_ASSETS_DIR)) and "robosuite" not in m.group(1)})
    if left:
        raise ValueError(f"unmapped absolute asset roots in demo XML: {left}")
    return postprocess_model_xml(xml, {})


def replay_goal_trace(env, states: np.ndarray, model_xml: str | None) -> dict:
    """Evaluate every goal conjunct and ``_check_success`` at every stored state of one demo.

    ``model_xml`` = the demo's processed XML (``demo_model_xml``); ``None`` replays in the scene built from
    the BDDL (the P9 comparison). Returns ``{"conjuncts", "truth" (T, C) bool, "success" (T,) bool,
    "consistent"}``, where ``consistent`` says ``_check_success`` equals the AND of the conjuncts at every
    state (it must, by LIBERO's definition).
    """
    env.reset()
    if model_xml is not None:
        env.reset_from_xml_string(model_xml)
    env.sim.reset()
    dom = env.env
    goal = [tuple(c) for c in dom.parsed_problem["goal_state"]]
    n = len(states)
    truth = np.zeros((n, len(goal)), dtype=bool)
    success = np.zeros(n, dtype=bool)
    for t in range(n):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        for c, g in enumerate(goal):
            truth[t, c] = bool(dom._eval_predicate(list(g)))
        success[t] = bool(dom._check_success())
    return {"conjuncts": goal, "truth": truth, "success": success,
            "consistent": bool((success == truth.all(axis=1)).all())}


def first_true(x: np.ndarray) -> int:
    """Index of the first True in a 1-D bool array, or -1."""
    idx = np.flatnonzero(x)
    return int(idx[0]) if len(idx) else -1


def rle(x: np.ndarray) -> list[list[int]]:
    """Run-length encoding of a 1-D bool array: ``[[value, length], ...]`` with value 0/1 (compact trace storage
    in the rollout record and the human-reference parquet; ``rle_decode`` inverts it)."""
    return [[int(v), sum(1 for _ in run)] for v, run in groupby(np.asarray(x, dtype=bool))]


def rle_decode(runs: list[list[int]]) -> np.ndarray:
    """Inverse of ``rle``: ``[[value, length], ...]`` -> 1-D bool array."""
    return np.concatenate([np.full(n, bool(v)) for v, n in runs]) if runs else np.zeros(0, dtype=bool)


# ---------------------------------------------------------------------------------------------
# operations: per-sub-task time-to-success (evaluation-layer S4)
# ---------------------------------------------------------------------------------------------


def operations_table(bddl_file: Path) -> dict:
    """Static operations of a task from its BDDL: goal conjuncts, the *maintained* ones (goal conjuncts that
    are also in ``:init``, i.e. true by construction at t0, e.g. libero_10 task 8's ``(turnon flat_stove_1)``)
    and the *operations* (the rest). ``operations_from_trace`` uses each trace's own t0 truth; this table
    is what it is expected to agree with (S4 AC4.1)."""
    from libero.libero.envs import bddl_utils

    p = bddl_utils.robosuite_parse_problem(str(bddl_file))
    goal = [tuple(c) for c in p["goal_state"]]
    init = {tuple(x.lower() for x in c) for c in p["initial_state"]}
    maintained = [c for c in goal if tuple(x.lower() for x in c) in init]
    return {"goal": [conjunct_label(c) for c in goal], "maintained": [conjunct_label(c) for c in maintained],
            "operations": [conjunct_label(c) for c in goal if c not in maintained], "n_ops": len(goal) - len(maintained)}


def operations_from_trace(conjunct_truth: np.ndarray, success: np.ndarray, tau_end_steps: int | None = None,
                          fps: int = FPS, labels: list[str] | None = None, key: str = "") -> dict:
    """Operation-level (T, event) pairs of one LIBERO trace (human demo replay or rollout). THE single
    implementation: ``scripts/human_reference.py --operations`` and the rollout harness (``vlapoc.rollouts``)
    both call it (S4 AC4.5).

    Grounding: PoC §3.5 A (p. 7) "T = time (or steps) to task success per episode; for long-horizon LIBERO-10
    tasks, per sub-task"; PoC §3.4 (p. 7) "Operation-level (T, event) pairs give several outcome samples per
    rollout"; PoC §3.2 (p. 6) "each rollout yields several"; PhAIL §3.1 (p. 4) the T-pair extraction
    (time since the previous success, censored tail for the unfinished attempt). LIBERO success is the
    conjunction over ``parsed_problem["goal_state"]`` (``_check_success``,
    ``envs/problems/libero_tabletop_manipulation.py:135-143``, ``libero_kitchen_tabletop_manipulation.py:136-144``).

    The rule (declared; evaluation-layer plan S4):

    - **Operations** = goal conjuncts that are false at t0 (index 0 of the trace: the post-settle state for a
      rollout, stored state 0 for a demo). Conjuncts already true at t0 are **maintained conditions**, not
      operations (libero_10 task 8's ``(turnon flat_stove_1)``).
    - **Operation k < n completes** at the first step where at least k operation-conjuncts hold
      *simultaneously* (counting, not identity: BDDL conjunct order is arbitrary, and a conjunct that
      toggles off and on is not double-counted).
    - **Operation n completes** at the first step where LIBERO's own ``_check_success`` holds (``success``),
      so the last operation time equals whole-task T by construction, and a broken maintained condition
      delays it until restored.
    - **T_k = t_k - t_{k-1}** (t_0 = 0), in seconds = steps / ``fps``. If the trace ends without success,
      the unfinished operation adds one right-censored pair ``(τ_end - t_last, 0)``; with no operation done,
      ``(τ_end, 0)``. This is PhAIL's ``episode_to_T_pairs`` (applied through
      ``vlapoc.evaluation.libero_episode``) with ``multi_grip_threshold_s = 0``: sim timestamps are exact,
      and PhAIL's 0.3 s clustering exists for annotation jitter (a recorded divergence).

    Args:
        conjunct_truth: (T, C) bool, goal-conjunct truth at trace indices 0..T-1 (index t = after t actions).
        success: (T,) bool, ``_check_success()`` at the same indices.
        tau_end_steps: the trace's observed length in steps (a rollout's executed actions); default T - 1.
            Used only for the censored tail of an unfinished trace.
        labels: conjunct labels for diagnostics.

    Returns ``{"n_ops", "maintained", "operations", "t_ops" (steps of completed operations), "op_conjuncts"
    (operation-conjuncts that became true at each t_k; diagnostics), "success_step" (or None), "n_ops_done",
    "censored", "trivial" (n_ops == 0), "episode" (``vlapoc.evaluation.Episode`` with the (T, E) pairs)}``.
    """
    from vlapoc.evaluation import libero_episode

    truth = np.asarray(conjunct_truth, dtype=bool)
    succ = np.asarray(success, dtype=bool)
    if truth.ndim != 2 or len(truth) != len(succ) or len(succ) == 0:
        raise ValueError(f"truth {truth.shape} and success {succ.shape} must be (T, C) and (T,), T > 0")
    labels = labels or [f"c{c}" for c in range(truth.shape[1])]
    tau_end = int(tau_end_steps) if tau_end_steps is not None else len(succ) - 1
    ops = [c for c in range(truth.shape[1]) if not truth[0, c]]
    maintained = [c for c in range(truth.shape[1]) if truth[0, c]]
    n = len(ops)
    t_succ = first_true(succ)
    count = truth[:, ops].sum(axis=1) if ops else np.zeros(len(succ), dtype=int)
    t_ops: list[int] = []
    for k in range(1, n):
        t_k = first_true(count >= k)
        if t_k < 0 or (t_succ >= 0 and t_k > t_succ):
            break
        t_ops.append(t_k)
    if n and t_succ >= 0:
        t_ops = t_ops[: n - 1] + [t_succ]
    if any(b < a for a, b in zip(t_ops, t_ops[1:])):
        raise AssertionError(f"operation times not monotone: {t_ops}")
    op_conj = [[labels[c] for c in ops if truth[t, c] and (t == 0 or not truth[t - 1, c])] for t in t_ops]
    done = t_succ >= 0
    duration_steps = t_succ if done else tau_end
    ep = libero_episode(key, [t / fps for t in t_ops], duration_steps / fps, n)
    return {"n_ops": n, "maintained": [labels[c] for c in maintained], "operations": [labels[c] for c in ops],
            "t_ops": t_ops, "op_conjuncts": op_conj, "success_step": t_succ if done else None, "n_ops_done": len(t_ops),
            "censored": any(e == 0 for _, e in ep.pairs), "trivial": n == 0, "episode": ep}


def env_versions() -> dict:
    """Interpreter and sim-stack versions for run records (AC3.1)."""
    import importlib.metadata as md
    import platform
    import sys

    pk = {}
    for name in ("hf-libero", "robosuite", "mujoco", "bddl", "numba", "numpy", "lerobot", "torch"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk,
            "libero_config": str(LIBERO_CONFIG_DIR.relative_to(REPO_ROOT)), "assets_repo": ASSETS_REPO,
            "assets_revision": ASSETS_REVISION, "mujoco_gl": os.environ.get("MUJOCO_GL")}


__all__ = ["setup_libero", "fetch_assets", "assets_digest", "bddl_path", "goal_conjuncts", "conjunct_label",
           "make_replay_env", "demo_model_xml", "replay_goal_trace", "first_true", "rle", "rle_decode", "env_versions",
           "operations_table", "operations_from_trace",
           "LIBERO_CONFIG_DIR", "LIBERO_ASSETS_DIR", "ASSETS_REPO", "ASSETS_REVISION", "FPS"]

if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(env_versions(), indent=2))
