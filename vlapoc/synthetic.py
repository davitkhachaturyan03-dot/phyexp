"""SYNTHETIC rollout x episode influence matrices for the dashboard demo (dashboard S6). Method side: reads training data only through
``open_for_method`` and never an answer key. Everything built from this is labelled SYNTHETIC wherever it appears.

Where this sits: ``scripts/dashboard_demo.py build`` calls ``demo_inputs`` and ``synthetic_matrix`` to give the real 960 training episodes of
``v3/candidate`` and the 20 real SmolVLA harness rollouts a *made-up* influence matrix with planted structure, then runs the real ``attribution.score_table``,
``column_declarations``, ``ensemble_run`` and the dashboard's own stages on it. It exists so the whole demonstration flow can be shown before any server
result exists (owner answer 2026-10-22: "Synthetic scores, labelled"). It is not an attribution result about any policy.

Grounding
---------
- PoC §2.7 (p. 5): the dashboard views need a rollout x episode matrix; §4.2 (pp. 9-10) / §4.5 (p. 11): an episode x metric table whose structure is
  recoverable ("Check whether attribution assigns each corrupted group to the expected metric"). The structure planted here is stated in plain words
  (``PLANTED_TEXT``) and recorded in the run params.
- PoC §2.1 item 4 (p. 3) and owner decision 3: the generator is method code. It opens ``v3/candidate`` through ``vlapoc.method_data.open_for_method`` (a fenced,
  leaky, ground-truth-derived or tampered subset is refused), and the planted structure depends only on method-facing descriptors (``vlapoc.clusters``'s
  catalog and heuristic columns), never on the planted corruption labels. ``scripts/verify_dashboard.py --demo`` scans this file's imports.

How the structure is planted (so that it is recoverable per metric)
-----------------------------------------------------------------
``score_table`` computes, per return column R, the influence of episode xi as (1/m) sum_tau w_R(tau) M[tau, xi] with w_R = 2R-1 (CUPID's +-1) for yes/no
returns and R - mean(R) for continuous ones. M is Gaussian noise plus ``s_k(xi) * u_k(tau)``, where u_f30 is w_f30 with its projection on w_hrt removed and u_hrt is w_hrt
with its projection on w_f30 removed. Then the F(30) influence of an episode gets only its F(30) term and the HRT influence only its HRT term (cross terms are
exactly zero); the other return columns pick up a share, as correlated metrics do. ``s_k`` is scaled so a planted episode sits ``shift`` noise s.d. from an
unplanted one on its metric (harm: negative influence for a higher-is-better return).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PLANTED_TEXT = ("SYNTHETIC: episodes in the top fifth of length_vs_task (long for their task) are harmful on HRT only; episodes of one scene are harmful on F(30) only; "
                "a small slice (rough actions, 3-4 gripper toggles) is helpful on F(30). Noise elsewhere. Planted through the rollouts' returns, from method-facing descriptors; "
                "no answer key is read.")


def demo_inputs(subset_ref: str, lineage_root: Path | None = None, cache_dir: Path | None = None):
    """``(view, desc, codes, specs)``: the subset opened through ``open_for_method`` (refusals apply), its descriptor table
    (``clusters.descriptor_table``), and the binned codes and specs (``clusters.bin_descriptors``, min bin 19 = the stage's min_sup)."""
    from vlapoc import clusters as cl
    from vlapoc.method_data import open_for_method

    view = open_for_method(subset_ref, lineage_root)
    desc = cl.descriptor_table(view, True, cache_dir if cache_dir is not None else cl.DEFAULT_CACHE)
    codes, specs, _ = cl.bin_descriptors(desc, 5, max(10, int(round(0.02 * len(desc)))))
    return view, desc, codes, specs


def plant_cells(desc: pd.DataFrame, codes: pd.DataFrame, specs: dict) -> dict:
    """The planted groups as boolean masks over ``desc``'s rows (deterministic): ``hrt_harm`` = ``length_vs_task`` top bin; ``f30_harm`` = the scene (not the suite-wide
    one) whose size is closest to 60 episodes; ``f30_boost`` = ``smoothness`` top bin AND the ``gripper_toggles`` bin whose cell is closest to 40. Returns
    ``{"masks": {...}, "text": {...}}``."""
    n = len(desc)
    top = int(codes["length_vs_task"].max())
    hrt = (codes["length_vs_task"] == top).to_numpy()
    sizes = desc["scene"].value_counts()
    sizes = sizes[sizes < n * 0.25]  # not the scene that covers a whole suite
    scene = min(sizes.index, key=lambda s: abs(int(sizes[s]) - 60))
    f30h = (desc["scene"] == scene).to_numpy()
    stop = int(codes["smoothness"].max())
    cells = {c: int(((codes["smoothness"] == stop) & (codes["gripper_toggles"] == c)).sum()) for c in sorted(codes["gripper_toggles"].unique())}
    gb = min(cells, key=lambda c: abs(cells[c] - 40))
    f30b = ((codes["smoothness"] == stop) & (codes["gripper_toggles"] == gb)).to_numpy()
    lab = lambda f, c: f"{f} {specs[f]['labels'][c]}"  # noqa: E731
    return {"masks": {"hrt_harm": hrt, "f30_harm": f30h, "f30_boost": f30b},
            "text": {"hrt_harm": lab("length_vs_task", top), "f30_harm": f"scene = {scene}", "f30_boost": f"{lab('smoothness', stop)} AND {lab('gripper_toggles', gb)}"},
            "n": {"hrt_harm": int(hrt.sum()), "f30_harm": int(f30h.sum()), "f30_boost": int(f30b.sum())}}


def weight_vectors(ret: pd.DataFrame) -> dict[str, np.ndarray]:
    """The per-rollout weight ``w_R`` of every return column exactly as ``attribution.score_table`` applies it (+-1 for yes/no returns, centred otherwise)."""
    from vlapoc.attribution import column_plan, returns_of

    returns = returns_of(ret)
    plan = column_plan(returns, True, "pm1")
    return {k: (2.0 * np.asarray(r, float) - 1.0 if plan[k]["encoding"] == "pm1" else np.asarray(r, float) - np.mean(r)) for k, r in returns.items()}


def synthetic_matrix(ret: pd.DataFrame, cells: dict, seed: int, n_episodes: int, shift: float = 1.5) -> tuple[np.ndarray, dict]:
    """A SYNTHETIC ``(n_rollouts, n_episodes)`` matrix (module docstring). ``ret`` is the returns table of the rollouts in row order, ``cells`` the masks of
    ``plant_cells`` (rows = episodes in catalog order). Returns ``(M, truth)``; ``truth`` records the scaling and the cross-term check."""
    m = len(ret)
    rng = np.random.default_rng(seed)
    w = weight_vectors(ret)
    wf, wh = w["ret_f30"], w["ret_hrt60_lin"]
    u_f = wf - (wf @ wh) / (wh @ wh) * wh
    u_h = wh - (wh @ wf) / (wf @ wf) * wf
    noise = rng.normal(size=(m, n_episodes))
    sd = {"f30": float(((wf @ noise) / m).std()), "hrt": float(((wh @ noise) / m).std())}
    gain = {"f30": float((wf @ u_f) / m), "hrt": float((wh @ u_h) / m)}
    # harm = -good_sign * influence and both returns are higher-is-better, so a harmful episode needs negative influence
    s_f = np.where(cells["f30_harm"], -1.0, 0.0) + np.where(cells["f30_boost"], 1.0, 0.0)
    s_h = np.where(cells["hrt_harm"], -1.0, 0.0)
    mat = noise + np.outer(u_f, s_f * shift * sd["f30"] / gain["f30"]) + np.outer(u_h, s_h * shift * sd["hrt"] / gain["hrt"])
    truth = {"shift_sd": shift, "noise_sd": sd, "cross_term_f30_uh": float((wf @ u_h) / m), "cross_term_hrt_uf": float((wh @ u_f) / m), "seed": seed, "n_rollouts": m}
    return mat, truth
