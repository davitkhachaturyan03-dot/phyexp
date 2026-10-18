"""The seven injected corruption types, as pure ``EpisodeEdit`` transforms (ground-truth feature, S1).

Where this sits: ``vlapoc.groundtruth`` picks which episodes get which type and calls ``make_edit`` for each; the
resulting ``EpisodeEdit`` list goes to ``derive.derive_version``, which applies each transform while it writes the new
dataset version and stores the type, ``params`` and seed in ``labels.parquet`` (the answer key). Nothing here touches
disk. A newcomer should read ``make_edit``, then one constructor (``time_shift`` is the shortest that changes length,
``action_noise`` the simplest), then ``clip_actions``.

Grounding
---------
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data: noisy or shifted actions,
  mismatched language instructions, altered camera viewpoints, or low-quality teleoperation demos. Then check
  whether your attribution method flags exactly those."
- PoC §2.5 (p. 5), table "Corruption types for the injection test": "Noisy or time-shifted actions" (teleoperation
  lag, timestamp misalignment); "Swapped or wrong language instructions" (annotation errors); "Altered camera
  viewpoint or dropped frames" (sensor drift, hardware faults); "Low-skill or hesitant demonstrations"
  (inexperienced operators, idle periods, corrections). "Inject 5-10% corrupted episodes".
- PoC §8.10 (p. 17): "Corrupt episodes two ways: pad some with idle or hesitant segments (should hurt speed,
  not reliability) and add gripper-release noise to others (should hurt reliability, not speed)."
- PoC §4.5 (p. 11): "Metric-targeted injections ... idle or hesitant segments aimed at time-based metrics, and
  noisy gripper release near placement aimed at drops ... These mappings are hypotheses the experiment should
  confirm." So ``hypothesis_for`` returns a hypothesis only for those two types and "none stated" otherwise;
  it is a hypothesis, never an outcome.
- RoboDrop App. A.1 (p. 13, docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf): "Action
  corruption adds zero-mean Gaussian noise with standard deviation 0.25 sigma_a to the demonstrated actions,
  where sigma_a is the empirical action standard deviation. Temporal corruption shifts the action sequence by
  two seconds relative to the observations." Defines ``action_noise`` (0.25 sigma_a) and ``time_shift``
  (2 s = 40 steps at the dataset's 20 fps, ``meta/info.json``). It does not state the shift direction, the tail
  fill, whether the gripper is noised, or any clipping.
- Owner decisions (brief.md, 2026-10-15): (1) every corrupted action is clipped to [-1, 1] and the clipped
  fraction recorded; (2) all types except altered viewpoint (needs simulator re-rendering; owner-approved out
  of scope).

Parameter provenance. Every label's ``params`` has ``source`` and ``parameters``, a map
``name -> {"value": ..., "basis": "quoted" | "measured" | "inferred", "from": ...}``. What is *inferred*:

- ``action_noise``: noise on action dims 0-5 only. DIVERGES from v1 (``derive.robodrop_action_noise``, all 7
  dims, literal reading): the gripper is exactly {-1, +1} in 200,340/200,340 v0 frames (measured), so noise
  would make a value that clean data never has, and 50.1% of noised gripper values would fall outside
  [-1, 1] and be clipped. The gripper's own corruption is ``gripper_release``.
- ``time_shift``: direction (actions lead observations: the logged command a[t] is the one the operator issued
  2 s later, a'[t] = a[t + 40]); trim instead of fill (owner ruling A, 2026-10-17): the child is T - 40 frames,
  obs[0:T-40] paired with a[40:T]; round 1 padded an idle tail, which was only an artifact.
- ``instruction_swap``: same-suite target, uniform.
- ``dropped_frames``: one camera per burst, bursts until about 20% of frames are held. The burst length cap of
  10 frames is quoted from LeRobot's ``read_latest(max_age_ms=500)`` at 20 fps.
- ``detour``: window m = 20 steps (1 s), negated arm deltas as the reverse motion (OSC_POSE delta control);
  prior art Bedi 2026 "detour", SCIZOR RQ4 "Move Back and Forth".
- ``idle_padding``: two pauses of 20 frames (each at most the 21-step maximum identical-action run in clean
  data, measured), placed in [0.2T, 0.8T] where clean data has no pauses (measured).
- ``gripper_release``: premature release of 20 steps; prior art Bedi 2026 "Early release".

Contract with ``derive.derive_version``. Each constructor returns an ``EpisodeEdit``. ``params`` is *completed
by the transform when it runs* (``corrupted_frames`` in child frame indices, clip counts, per-episode draws),
because those depend on the episode's length and the per-episode generator; ``derive_version`` serialises it
after its writer pass. Each type draws from ``derive._episode_rng(seed, "<type>:<episode_id>")``, so an
episode's corruption depends on neither the other edits nor the other types.
"""

from __future__ import annotations

import hashlib

import numpy as np

from vlapoc.derive import EpisodeEdit, _episode_rng

FPS = 20  # meta/info.json of v0 (measured)
CLIP_LOW, CLIP_HIGH = -1.0, 1.0  # LIBERO's valid command range (owner decision 1)
CLEAN_TRANSLATION_MAX = 0.9375  # |a| on dims 0-2 never exceeds this in v0 (measured, meta/stats.json min/max)
NOISE_SCALE = 0.25  # RoboDrop App. A.1 p.13 (quoted)
NOISE_DIMS = slice(0, 6)  # inferred: the gripper stays binary
SHIFT_STEPS = 2 * FPS  # RoboDrop App. A.1 p.13 "two seconds" x 20 fps
BURST_MAX = 10  # LeRobot read_latest(max_age_ms=500) at 20 fps
HELD_FRACTION = 0.20  # inferred
DETOUR_STEPS = 20  # inferred; the window has DETOUR_STEPS + 1 frames
PAUSE_STEPS = 20
N_PAUSES = 2
RELEASE_STEPS = 20
RELEASE_MIN_CARRY = 30  # eligibility: closed span before the final release (measured P7: >= 40 in 95%)
MIDDLE = (0.2, 0.8)  # fraction of the episode where detours and pauses go

TYPES = ("action_noise", "time_shift", "instruction_swap", "dropped_frames", "detour", "idle_padding",
         "gripper_release")

#: PoC §4.5 p.11 / §8.10 p.17: the only two metric hypotheses the PoC states.
_HYPOTHESES = {
    "idle_padding": "speed (time-based metrics), not reliability (PoC §8.10 p.17; §4.5 p.11)",
    "gripper_release": "reliability (drops), not speed (PoC §8.10 p.17; §4.5 p.11)",
}


def hypothesis_for(corruption_type: str) -> str:
    """The §4.5 metric hypothesis for a type, or ``"none stated"`` (a hypothesis, never an outcome)."""
    return _HYPOTHESES.get(corruption_type, "none stated")


def _p(value, basis: str, src: str) -> dict:
    """One ``parameters`` entry: its value, whether it is quoted / measured / inferred, and from where."""
    return {"value": value, "basis": basis, "from": src}


# ---------------------------------------------------------------------------------------------
# clipping (owner decision 1)
# ---------------------------------------------------------------------------------------------


def clip_actions(a: np.ndarray) -> tuple[np.ndarray, dict]:
    """Clip actions to [-1, 1] (owner decision 1) and account for how much the clip bit.

    ``a`` is the float32/float64 (T, 7) action array to clip. Returns ``(clipped float32, stats)``:
    ``n_clipped`` (values outside [-1, 1] that the clip changed), ``n_values`` (a.size), ``frac_clipped``
    (their ratio), ``n_clipped_dims0_2`` / ``frac_clipped_dims0_2`` (the translation dims, where P3 measured
    0.5-1.5% for 0.25 sigma_a noise), and ``n_beyond_clean_range`` / ``frac_beyond_clean_range``: values on dims 0-2
    with 0.9375 < |a| <= 1 *after* clipping. Clean v0 data has none (``CLEAN_TRANSLATION_MAX``, measured), so
    these are values valid for the environment but impossible in this dataset. The stats describe ``a``, the
    array before clipping.
    """
    a64 = np.asarray(a, dtype=np.float64)
    out = np.clip(a64, CLIP_LOW, CLIP_HIGH)
    changed = out != a64
    t = a64.shape[0]
    n_beyond = int((np.abs(out[:, :3]) > CLEAN_TRANSLATION_MAX).sum())
    stats = {
        "clip_bounds": [CLIP_LOW, CLIP_HIGH],
        "n_values": int(a64.size),
        "n_clipped": int(changed.sum()),
        "frac_clipped": float(changed.sum() / a64.size),
        "n_clipped_dims0_2": int(changed[:, :3].sum()),
        "frac_clipped_dims0_2": float(changed[:, :3].sum() / (3 * t)),
        "n_beyond_clean_range": n_beyond,
        "frac_beyond_clean_range": float(n_beyond / (3 * t)),
    }
    return out.astype(np.float32), stats


#: Clip accounting of a type that never touches the actions (same keys, same order, as ``clip_actions`` writes).
NO_CLIP_STATS = {"n_clipped": 0, "n_clipped_dims0_2": 0, "n_beyond_clean_range": 0,
                 "frac_clipped": 0.0, "frac_clipped_dims0_2": 0.0, "frac_beyond_clean_range": 0.0}


def _ranges(idx: list[tuple[int, int]]) -> list[list[int]]:
    """``[[start, end), ...]`` as JSON-friendly lists, sorted."""
    return [[int(a), int(b)] for a, b in sorted(idx)]


def _finish(params: dict, ep: dict, action: np.ndarray, frames: list[tuple[int, int]], **extra) -> dict:
    """Clip ``action``, complete ``params`` in place (clip counts, ``corrupted_frames``, extras) and return
    the edited episode. ``frames`` are [start, end) child indices."""
    clipped, stats = clip_actions(action)
    params.update(stats)
    params["corrupted_frames"] = _ranges(frames)
    params.update(extra)
    return {**ep, "action": clipped}


def _rng(seed: int, ctype: str, episode_id: str) -> np.random.Generator:
    """The per-(type, episode) generator: ``derive._episode_rng`` keyed ``"<type>:<episode_id>"``."""
    return _episode_rng(seed, f"{ctype}:{episode_id}")


_RNG_DESC = "derive._episode_rng(seed, '<type>:<episode_id>') = default_rng(SeedSequence([seed, int(sha256(key)[:16], 16)]))"


# ---------------------------------------------------------------------------------------------
# eligibility
# ---------------------------------------------------------------------------------------------


def constant_gripper_windows(gripper: np.ndarray, length: int, lo: int, hi: int) -> list[int]:
    """Start indices t in [lo, hi - length] such that ``gripper[t : t + length + 1]`` is constant (the detour
    window is ``length + 1`` frames)."""
    out = []
    for t in range(lo, hi - length + 1):
        w = gripper[t:t + length + 1]
        if (w == w[0]).all():
            out.append(t)
    return out


def final_release(gripper: np.ndarray) -> tuple[int, int] | None:
    """``(r, carry)``: ``r`` is the index of the episode's final +1 -> -1 gripper transition (gripper[r] = -1,
    gripper[r-1] = +1) and ``carry`` the length of the closed (+1) run ending at ``r - 1``; ``None`` without one."""
    g = np.asarray(gripper)
    idx = np.where((g[1:] == -1) & (g[:-1] == 1))[0]
    if len(idx) == 0:
        return None
    r = int(idx[-1]) + 1
    k = r
    while k > 0 and g[k - 1] == 1:
        k -= 1
    return r, r - k


def eligible(ctype: str, episode: dict) -> bool:
    """Can ``ctype`` be applied to this episode? ``episode`` needs ``action`` (T, 7); the other keys are not read.

    ``action_noise``, ``time_shift`` (T > 40), ``instruction_swap`` and ``dropped_frames``: always (given the
    length). ``detour``: a window of 21 frames with constant gripper inside [0.2T, 0.8T]. ``idle_padding``:
    two pause points at least 5 apart inside [0.2T, 0.8T]. ``gripper_release``: a final release with a closed
    carry span of at least ``RELEASE_MIN_CARRY`` steps (measured premise P7).
    """
    a = np.asarray(episode["action"])
    t = len(a)
    lo, hi = int(MIDDLE[0] * t), int(MIDDLE[1] * t)
    if ctype in ("action_noise", "instruction_swap"):
        return True
    if ctype == "time_shift":
        return t > SHIFT_STEPS
    if ctype == "dropped_frames":
        return t >= 3 * BURST_MAX
    if ctype == "detour":
        return len(constant_gripper_windows(a[:, 6], DETOUR_STEPS, lo, hi)) > 0
    if ctype == "idle_padding":
        return hi - lo >= 5
    if ctype == "gripper_release":
        fr = final_release(a[:, 6])
        return fr is not None and fr[1] >= RELEASE_MIN_CARRY and fr[0] - RELEASE_STEPS >= 0
    raise ValueError(f"unknown corruption type {ctype!r}")


# ---------------------------------------------------------------------------------------------
# the seven constructors
# ---------------------------------------------------------------------------------------------


def action_noise(episode_id: str, sigma_a: np.ndarray, seed: int, scale: float = NOISE_SCALE) -> EpisodeEdit:
    """``action_noise``: a[:, 0:6] += N(0, (scale * sigma_a)^2), then clip to [-1, 1]; the gripper is untouched.

    PoC §2.5 p.5 "Noisy ... actions"; RoboDrop App. A.1 p.13 (0.25 sigma_a, quoted). ``sigma_a`` is the 7-vector
    of per-dim action std (v0 ``meta/stats.json``, as v1). Dims 0-5 only: inferred and a divergence from v1 (module
    docstring). Length unchanged; ``corrupted_frames`` = the whole episode.
    """
    sigma_a = np.asarray(sigma_a, dtype=np.float64)
    params = {
        "source": "PoC §2.5 p.5 'Noisy or time-shifted actions'; RoboDrop App. A.1 p.13 'zero-mean Gaussian noise with "
                  "standard deviation 0.25 sigma_a ... sigma_a is the empirical action standard deviation'",
        "parameters": {
            "std_scale": _p(scale, "quoted", "RoboDrop App. A.1 p.13"),
            "sigma_a": _p([float(x) for x in sigma_a], "measured", "v0 meta/stats.json action.std"),
            "noised_dims": _p([0, 1, 2, 3, 4, 5], "inferred",
                              "gripper is exactly {-1,+1} in v0 (measured); differs from v1's 7-dim reading"),
            "clip": _p([CLIP_LOW, CLIP_HIGH], "quoted", "owner decision 1; LIBERO Box(-1, 1, (7,))"),
            "rng": _p(_RNG_DESC, "inferred", "same scheme as derive.robodrop_action_noise"),
        },
    }

    def f(ep: dict) -> dict:
        rng = _rng(seed, "action_noise", episode_id)
        a = ep["action"].astype(np.float64)
        noise = rng.normal(0.0, scale * sigma_a[NOISE_DIMS], size=(len(a), 6))
        a[:, NOISE_DIMS] += noise
        return _finish(params, ep, a, [(0, len(a))])

    return EpisodeEdit(episode_id, "action_noise", params, seed, f, hypothesis=hypothesis_for("action_noise"))


def time_shift(episode_id: str, seed: int, steps: int = SHIFT_STEPS) -> EpisodeEdit:
    """``time_shift``: a TRIM. The child is frames 0 .. T-steps-1 with each frame's state and images unchanged and
    its action replaced by the action ``steps`` (40) steps later: a'[t] = a[t + 40], so the child has T - 40 frames.

    PoC §2.5 p.5 "time-shifted actions" (teleoperation lag); RoboDrop App. A.1 p.13 "shifts the action sequence
    by two seconds relative to the observations" (40 steps = 2 s x 20 fps). Direction (actions lead the
    observations) is inferred: RoboDrop states none.

    Round-2 change (owner ruling A, 2026-10-17, dispute-r1): round 1 filled the vacated last 40 steps with idle
    (arm action 0, gripper held). That tail was detectable only as an artifact (an ``idle_steps`` of at least 40,
    beyond the clean maximum of 21, gave AUROC 1.0) and put two seconds of idle into a type whose PoC row says nothing
    about idle. The trim has no fill: every frame and every action is a real parent value. The cost, reported
    openly (plan AC5.8): the episode is 40 steps shorter, a residual ``length_steps`` signal. ``params`` records
    ``length_change = -40``. Deterministic (no random draw): ``seed`` is recorded but unused. Every parent has
    T >= 75, so the shortest child has 35 frames.
    """
    params = {
        "source": "PoC §2.5 p.5 'time-shifted actions'; RoboDrop App. A.1 p.13 'shifts the action sequence by two "
                  "seconds relative to the observations'",
        "length_change": -steps,
        "parameters": {
            "shift_steps": _p(steps, "quoted", "RoboDrop App. A.1 p.13 two seconds x 20 fps (meta/info.json)"),
            "direction": _p("a'[t] = a[t + shift]: actions lead the observations", "inferred", "RoboDrop does not state it"),
            "vacated_steps": _p("trimmed: the child keeps frames 0 .. T-shift-1, no fill", "inferred",
                                "owner ruling A (2026-10-17): the round-1 idle tail was an artifact"),
            "clip": _p([CLIP_LOW, CLIP_HIGH], "quoted", "owner decision 1"),
            "rng": _p("none: deterministic", "inferred", "no random draw"),
        },
    }

    def f(ep: dict) -> dict:
        a = ep["action"].astype(np.float64)
        t = len(a)
        if t <= steps:
            raise ValueError(f"{episode_id}: length {t} <= shift {steps}")
        edited = _finish(params, ep, a[steps:], [(0, t - steps)])
        keep = slice(0, t - steps)
        return {**edited, "state": ep["state"][keep], "image": ep["image"][keep], "image2": ep["image2"][keep]}

    return EpisodeEdit(episode_id, "time_shift", params, seed, f, hypothesis=hypothesis_for("time_shift"))


def instruction_swap(episode_id: str, original_task: str, same_suite_tasks: list[str], seed: int) -> EpisodeEdit:
    """``instruction_swap``: the episode's task string becomes another task of the same suite (uniform, seeded).

    PoC §2.5 p.5 "Swapped or wrong language instructions" (annotation errors). ``same_suite_tasks`` are task
    strings of the parent's ``tasks.parquet`` in the episode's suite; the target is drawn from those other than
    ``original_task`` at construction, so ``EpisodeEdit.new_task`` is known before any write (``derive_version``
    refuses a target that is not in the parent's tasks). Same-suite rule: inferred. Provenance (``episode_id``,
    ``task_name``, physical position) stays true (design D1): a provenance-vs-instruction cross-check finds
    every swap; the ground-truth feature measures that. Only the task string changes; arrays are identical.
    """
    alts = sorted(t for t in set(same_suite_tasks) if t != original_task)
    if not alts:
        raise ValueError(f"{episode_id}: no same-suite alternative to {original_task!r}")
    rng = _rng(seed, "instruction_swap", episode_id)
    new_task = alts[int(rng.integers(len(alts)))]
    params = {
        "source": "PoC §2.5 p.5 'Swapped or wrong language instructions' (annotation errors)",
        "parameters": {
            "original_task": _p(original_task, "measured", "stored task string"),
            "new_task": _p(new_task, "inferred", f"uniform over {len(alts)} other same-suite tasks"),
            "same_suite": _p(True, "inferred", "keeps the instruction valid for the benchmark"),
            "provenance": _p("episode_id, task_name, libero_task_id, scene, objects, position unchanged", "inferred",
                             "design D1: a real annotation error leaves the collection record true"),
            "rng": _p(_RNG_DESC, "inferred", "target drawn at construction"),
        },
        "corrupted_frames": [],  # completed with [[0, T]] by the transform
    }
    params.update(NO_CLIP_STATS)

    def f(ep: dict) -> dict:
        if ep["task"] != original_task:
            raise ValueError(f"{episode_id}: stored task {ep['task']!r} != declared original {original_task!r}")
        params["corrupted_frames"] = [[0, len(ep["action"])]]
        return {**ep, "task": new_task}

    return EpisodeEdit(episode_id, "instruction_swap", params, seed, f, new_task=new_task,
                       hypothesis=hypothesis_for("instruction_swap"))


def dropped_frames(episode_id: str, seed: int, held_fraction: float = HELD_FRACTION, burst_max: int = BURST_MAX) -> EpisodeEdit:
    """``dropped_frames``: sample-and-hold of one camera. Within a burst, frame t shows that camera's image from
    (burst start - 1); state and action are untouched.

    PoC §2.5 p.5 "dropped frames" (sensor drift, hardware faults). Burst length 1-10 (quoted from LeRobot
    ``read_latest(max_age_ms=500)``, which fills a late camera frame with the previous one, at 20 fps = 10 frames).
    One camera per burst (uniform), bursts (non-adjacent, at least one clean frame apart) until at least
    ``held_fraction`` of the frames are held: inferred. Every output image is byte-identical to a parent image.
    """
    params = {
        "source": "PoC §2.5 p.5 'dropped frames' (hardware faults); LeRobot cameras/opencv read_latest: stale frame "
                  "up to max_age_ms=500",
        "parameters": {
            "burst_length": _p([1, burst_max], "quoted", "read_latest(max_age_ms=500) at 20 fps = 10 frames"),
            "held_fraction": _p(held_fraction, "inferred", "target share of held frames"),
            "camera": _p("one per burst, uniform over {image, image2}", "inferred", "no spec"),
            "hold": _p("frame t in the burst shows the image of burst start - 1", "quoted", "read_latest returns the previous frame"),
            "rng": _p(_RNG_DESC, "inferred", "per-episode generator"),
        },
    }
    params.update(NO_CLIP_STATS)

    def f(ep: dict) -> dict:
        rng = _rng(seed, "dropped_frames", episode_id)
        t = len(ep["action"])
        target = int(np.ceil(held_fraction * t))
        taken = np.zeros(t + 1, dtype=bool)  # a held frame or the clean frame before/after it
        bursts, held = [], 0
        for _ in range(10_000):
            if held >= target:
                break
            n = int(rng.integers(1, burst_max + 1))
            s = int(rng.integers(1, t - n + 1))  # start >= 1: needs a previous frame
            lo, hi = max(s - 1, 0), min(s + n + 1, t)  # keep one clean frame each side
            if taken[lo:hi].any():
                continue
            taken[lo:hi] = True
            cam = int(rng.integers(2))
            bursts.append((s, s + n, cam))
            held += n
        if held < target:
            raise ValueError(f"{episode_id}: could not place bursts holding {target} of {t} frames")
        out = {"image": ep["image"].copy(), "image2": ep["image2"].copy()}
        for s, e, cam in bursts:
            k = "image" if cam == 0 else "image2"
            out[k][s:e] = ep[k][s - 1]
        params["corrupted_frames"] = _ranges([(s, e) for s, e, _ in bursts])
        params["bursts"] = [{"start": s, "end": e, "camera": ["image", "image2"][c]} for s, e, c in sorted(bursts)]
        params["n_held_frames"] = held
        params["frac_held"] = held / t
        return {**ep, **out}

    return EpisodeEdit(episode_id, "dropped_frames", params, seed, f, hypothesis=hypothesis_for("dropped_frames"))


def detour(episode_id: str, seed: int, steps: int = DETOUR_STEPS) -> EpisodeEdit:
    """``detour``: a hesitant correction. Pick a window of ``steps + 1`` frames [t, t + steps] with constant gripper
    inside [0.2T, 0.8T], play it forward, then backward, then forward again, then continue (T + 2 * steps frames).

    Child frame order (parent indices): 0 .. t+steps-1, then t+steps, t+steps-1, ..., t (going back), then
    t+1 .. t+steps, then t+steps+1 onward. Every frame is a real parent frame, and no two consecutive frames are
    the same parent frame (the turn points are not repeated). Arm actions on the backward leg are the negated
    forward deltas, a'[frame j] = -a[j-1] for the step from j to j-1 (inferred: OSC_POSE is delta control);
    the gripper is constant over the window so it is unchanged. PoC §2.5 p.5 "Low-skill or hesitant
    demonstrations ... corrections"; prior art SCIZOR RQ4 "Move Back and Forth", Bedi 2026 "detour".
    ``corrupted_frames`` = [[t, t + 3 * steps + 1)] in child indices.
    """
    params = {
        "source": "PoC §2.5 p.5 'Low-skill or hesitant demonstrations' (corrections); SCIZOR RQ4 'Move Back and Forth'",
        "parameters": {
            "window_steps": _p(steps, "inferred", "1 s at 20 fps"),
            "placement": _p("window of steps+1 frames with constant gripper inside [0.2T, 0.8T]", "inferred", "idle/pauses never occur mid-episode in v0 (measured)"),
            "reverse_actions": _p("arm dims 0-5 negated, gripper unchanged", "inferred", "OSC_POSE delta control"),
            "rng": _p(_RNG_DESC, "inferred", "start drawn uniformly among eligible windows"),
        },
    }

    def f(ep: dict) -> dict:
        a = ep["action"].astype(np.float64)
        t = len(a)
        lo, hi = int(MIDDLE[0] * t), int(MIDDLE[1] * t)
        starts = constant_gripper_windows(a[:, 6], steps, lo, hi)
        if not starts:
            raise ValueError(f"{episode_id}: no constant-gripper window of {steps + 1} frames in [{lo}, {hi}]")
        rng = _rng(seed, "detour", episode_id)
        s = int(starts[int(rng.integers(len(starts)))])
        e = s + steps
        order = (list(range(0, e)) + list(range(e, s - 1, -1)) + list(range(s + 1, e + 1)) + list(range(e + 1, t)))
        # action per output position: forward = a[j]; backward leg at frame j -> -a[j-1] (j = e .. s+1), then a[s] at j = s
        acts = []
        back_pos = range(e, e + steps + 1)
        for pos, j in enumerate(order):
            if pos in back_pos and j > s:
                v = a[j - 1].copy()
                v[:6] *= -1.0
                acts.append(v)
            else:
                acts.append(a[j].copy())
        act = np.stack(acts)
        o = np.asarray(order)
        out = {"state": ep["state"][o], "image": ep["image"][o], "image2": ep["image2"][o]}
        edited = _finish(params, ep, act, [(s, s + 3 * steps + 1)], window_start_parent=s,
                         window_end_parent=e)
        return {**edited, **out}

    return EpisodeEdit(episode_id, "detour", params, seed, f, hypothesis=hypothesis_for("detour"))


def idle_padding(episode_id: str, seed: int, pause_steps: int = PAUSE_STEPS, n_pauses: int = N_PAUSES) -> EpisodeEdit:
    """``idle_padding``: ``n_pauses`` pauses of ``pause_steps`` frames, at distinct random points in [0.2T, 0.8T].

    A pause before parent frame p inserts ``pause_steps`` copies of frame p's observation and state, with arm
    action 0 and the gripper held at a[p-1, 6] (as ``derive.idle_padding_edit``). T + 40 frames. PoC §8.10
    p.17 "pad some with idle or hesitant segments (should hurt speed, not reliability)"; §2.5 p.5 "idle
    periods". Each pause is at most 21 steps, the longest clean identical-action run (measured P5). Frames
    are exact duplicates (clean data has none, measured P4): the repeat is findable by a duplicate detector.
    The §4.5 hypothesis (time-based metrics, not reliability) is recorded in the label's ``hypothesis`` column.
    """
    params = {
        "source": "PoC §8.10 p.17 'pad some with idle or hesitant segments'; §2.5 p.5 'idle periods'; §4.5 p.11",
        "parameters": {
            "pause_steps": _p(pause_steps, "inferred", "<= 21, the clean maximum identical-action run (measured)"),
            "n_pauses": _p(n_pauses, "inferred", "+40 steps = +23% of the median length (measured 172.5)"),
            "placement": _p("distinct points at least 5 apart in [0.2T, 0.8T]", "inferred", "clean idle frames sit at the start (measured)"),
            "idle_frame": _p("observation and state repeated, arm action 0, gripper held", "inferred", "as derive.idle_padding_edit"),
            "rng": _p(_RNG_DESC, "inferred", "per-episode generator"),
        },
    }
    params.update(NO_CLIP_STATS)

    def f(ep: dict) -> dict:
        a = ep["action"].astype(np.float64)
        t = len(a)
        lo, hi = int(MIDDLE[0] * t), int(MIDDLE[1] * t)
        rng = _rng(seed, "idle_padding", episode_id)
        for _ in range(1000):
            pts = sorted(int(x) for x in rng.choice(np.arange(max(lo, 1), hi + 1), size=n_pauses, replace=False))
            if all(b - a_ >= 5 for a_, b in zip(pts, pts[1:])):
                break
        else:
            raise ValueError(f"{episode_id}: could not place {n_pauses} pauses")
        order, act, frames = [], [], []
        for p in range(t):
            if p in pts:
                frames.append((len(order), len(order) + pause_steps))
                pad = np.repeat(a[p - 1:p], pause_steps, axis=0).copy()
                pad[:, :6] = 0.0
                order += [p] * pause_steps
                act += list(pad)
            order.append(p)
            act.append(a[p])
        o = np.asarray(order)
        out = {"state": ep["state"][o], "image": ep["image"][o], "image2": ep["image2"][o]}
        edited = _finish(params, ep, np.stack(act), frames, pause_points_parent=pts)
        return {**edited, **out}

    return EpisodeEdit(episode_id, "idle_padding", params, seed, f, hypothesis=hypothesis_for("idle_padding"))


def gripper_release(episode_id: str, seed: int, steps: int = RELEASE_STEPS) -> EpisodeEdit:
    """``gripper_release``: premature release. The gripper command is -1 (open) on the ``steps`` steps before the
    episode's final +1 -> -1 transition, so the object would be let go ``steps`` steps early.

    PoC §8.10 p.17 "add gripper-release noise to others (should hurt reliability, not speed)"; §4.5 p.11 "noisy
    gripper release near placement aimed at drops". Eligibility: a final release preceded by a closed span of at
    least 30 steps (measured P7: >= 40 in 95% of v0). ``steps`` = 20 (1 s) is inferred; prior art Bedi 2026
    "Early release". State and images are not re-rendered (out of scope), so they still show the unreleased
    object: the corruption is in the logged command only. Deterministic. The gripper stays in {-1, +1}.
    """
    params = {
        "source": "PoC §8.10 p.17 'gripper-release noise'; §4.5 p.11 'noisy gripper release near placement aimed at drops'",
        "parameters": {
            "release_steps": _p(steps, "inferred", "1 s at 20 fps; stays inside the carry phase (carry span >= 30)"),
            "min_carry_steps": _p(RELEASE_MIN_CARRY, "measured", "P7: closed span before the final release >= 40 in 95% of v0"),
            "rng": _p("none: deterministic", "inferred", "no random draw"),
        },
    }

    def f(ep: dict) -> dict:
        a = ep["action"].astype(np.float64)
        fr = final_release(a[:, 6])
        if fr is None or fr[1] < RELEASE_MIN_CARRY or fr[0] - steps < 0:
            raise ValueError(f"{episode_id}: not eligible for gripper_release")
        r, carry = fr
        a[r - steps:r, 6] = -1.0
        return _finish(params, ep, a, [(r - steps, r)], final_release_index=r, carry_steps=carry,
                       final_release_rel_position=r / len(a))

    return EpisodeEdit(episode_id, "gripper_release", params, seed, f, hypothesis=hypothesis_for("gripper_release"))


def make_edit(ctype: str, episode_id: str, seed: int, *, sigma_a=None, original_task: str | None = None,
              same_suite_tasks: list[str] | None = None) -> EpisodeEdit:
    """Construct the edit for ``ctype`` (one of ``TYPES``); extra keyword arguments are the ones that type needs
    (``sigma_a`` for ``action_noise``; ``original_task`` and ``same_suite_tasks`` for ``instruction_swap``)."""
    if ctype == "action_noise":
        return action_noise(episode_id, sigma_a, seed)
    if ctype == "time_shift":
        return time_shift(episode_id, seed)
    if ctype == "instruction_swap":
        return instruction_swap(episode_id, original_task, same_suite_tasks, seed)
    if ctype == "dropped_frames":
        return dropped_frames(episode_id, seed)
    if ctype == "detour":
        return detour(episode_id, seed)
    if ctype == "idle_padding":
        return idle_padding(episode_id, seed)
    if ctype == "gripper_release":
        return gripper_release(episode_id, seed)
    raise ValueError(f"unknown corruption type {ctype!r}")


def frame_digests(images: np.ndarray) -> list[str]:
    """sha256 of each frame's bytes (for the 'every output image is a parent image' invariant)."""
    return [hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest() for f in images]


__all__ = ["TYPES", "eligible", "make_edit", "hypothesis_for", "clip_actions", "action_noise", "time_shift",
           "instruction_swap", "dropped_frames", "detour", "idle_padding", "gripper_release", "final_release",
           "constant_gripper_windows", "frame_digests", "CLEAN_TRANSLATION_MAX"]
