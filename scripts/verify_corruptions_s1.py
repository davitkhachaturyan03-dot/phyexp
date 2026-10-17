"""Verify ground-truth S1 (the seven corruption transforms) against AC1.1-AC1.6, in memory on real v0 episodes.

Grounding: PoC §2.1 item 4 p.3, §2.5 p.5, §4.5 p.11, §8.10 p.17; RoboDrop App. A.1 p.13; owner decisions 1-2
(brief.md 2026-10-15).

    .venv/bin/python scripts/verify_corruptions_s1.py --version v0 --per-type 100 --seed 0 \
        --dump-hashes data/scratch/gt_s1_hashes.json
    .venv/bin/python scripts/verify_corruptions_s1.py --version v0 --per-type 100 --seed 0 \
        --check-hashes data/scratch/gt_s1_hashes.json          # second process: AC1.5

Nothing is written to a dataset or to the lineage store. Outputs: ``artifacts/s1_invariants.json`` (all numbers),
``artifacts/s1_invariants.txt`` (the printed table, only when ``--out`` is given), and with ``--figure`` the two PNGs
``artifacts/s1_traces.png`` and ``artifacts/s1_filmstrip.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import corruptions as C  # noqa: E402
from vlapoc.catalog import episode_content_hash, read_episodes_meta, read_tasks  # noqa: E402
from vlapoc.derive import _canonical_episode  # noqa: E402
from vlapoc.lineage import Subset, resolve_version, version_episodes  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"
BASES = {"quoted", "measured", "inferred"}


def load_episodes(root: Path, indices: list[int]):
    """Yield ``(episode_index, canonical episode)`` for the given indices, reading each data file once."""
    meta = read_episodes_meta(root)
    tasks = read_tasks(root)
    sel = meta[meta["episode_index"].isin(indices)]
    for (ci, fi), g in sel.groupby(["data/chunk_index", "data/file_index"]):
        path = root / "data" / f"chunk-{int(ci):03d}" / f"file-{int(fi):03d}.parquet"
        table = pq.read_table(path, filters=[("episode_index", "in", [int(i) for i in g["episode_index"]])]).to_pandas()
        for idx, frames in table.groupby("episode_index", sort=True):
            yield int(idx), _canonical_episode(frames.sort_values("frame_index"), tasks[int(frames["task_index"].iloc[0])])


def all_actions(root: Path) -> dict[int, np.ndarray]:
    """``{episode_index: action (T, 7)}`` for every stored episode, reading only the ``action`` column."""
    meta = read_episodes_meta(root)
    out = {}
    for (ci, fi), _ in meta.groupby(["data/chunk_index", "data/file_index"]):
        path = root / "data" / f"chunk-{int(ci):03d}" / f"file-{int(fi):03d}.parquet"
        t = pq.read_table(path, columns=["episode_index", "frame_index", "action"]).to_pandas()
        for idx, g in t.groupby("episode_index"):
            out[int(idx)] = np.stack(g.sort_values("frame_index")["action"].to_numpy()).astype(np.float32)
    return out


def check_params(edit, T_child: int) -> list[str]:
    """AC1.1: ``source``, every parameter tagged quoted/measured/inferred, ``corrupted_frames`` as [start, end) ranges."""
    p, errs = edit.params, []
    if not p.get("source"):
        errs.append("no source")
    for k, v in p.get("parameters", {}).items():
        if not (isinstance(v, dict) and v.get("basis") in BASES and "value" in v):
            errs.append(f"parameter {k} lacks a quoted/measured/inferred tag")
    cf = p.get("corrupted_frames")
    if not isinstance(cf, list) or not cf:
        errs.append("corrupted_frames missing")
    else:
        for s, e in cf:
            if not (0 <= s < e <= T_child):
                errs.append(f"range [{s}, {e}) outside [0, {T_child}]")
    for k in ("n_clipped", "frac_clipped", "frac_beyond_clean_range"):
        if k not in p:
            errs.append(f"missing {k}")
    json.dumps(p)  # must serialise
    return errs


idle_dist: dict[str, int] = {}  # AC1.8: time_shift idle_steps over the test episodes (seed-0 pass only is recorded below)


def run_episode(ctype: str, ep: dict, eid: str, seed: int, sigma, suite_tasks: dict, parent_tasks: set, suite_of_task: dict):
    """Apply one type to one episode and check its invariants. Returns (output, edit, failures list)."""
    kw = {}
    if ctype == "action_noise":
        kw["sigma_a"] = sigma
    if ctype == "instruction_swap":
        kw.update(original_task=ep["task"], same_suite_tasks=suite_tasks[suite_of_task[ep["task"]]])
    edit = C.make_edit(ctype, eid, seed, **kw)
    out = edit.transform(dict(ep))
    T, T2 = len(ep["action"]), len(out["action"])
    fails = []
    a, b = ep["action"], out["action"]
    if out["action"].dtype != np.float32 or out["state"].dtype != ep["state"].dtype:
        fails.append("dtype")
    if not ((b >= -1).all() and (b <= 1).all()):
        fails.append("range")
    if not np.isin(b[:, 6], [-1.0, 1.0]).all():
        fails.append("gripper not binary")
    exp_len = T + 40 if ctype in ("detour", "idle_padding") else T - 40 if ctype == "time_shift" else T
    if T2 != exp_len:
        fails.append(f"length {T2} != {exp_len}")
    same = lambda k: len(out[k]) == len(ep[k]) and np.array_equal(out[k], ep[k])  # noqa: E731
    if ctype == "action_noise":
        if not (same("state") and same("image") and same("image2") and out["task"] == ep["task"]):
            fails.append("non-action array changed")
        if not np.array_equal(a[:, 6], b[:, 6]):
            fails.append("gripper changed")
    elif ctype == "time_shift":
        if not (np.array_equal(out["state"], ep["state"][:T - 40]) and np.array_equal(out["image"], ep["image"][:T - 40])
                and np.array_equal(out["image2"], ep["image2"][:T - 40]) and out["task"] == ep["task"]):
            fails.append("state/images are not parent frames 0..T-41")
        if not np.array_equal(b, a[40:]):
            fails.append("actions are not parent actions 40..T-1")
        if edit.params.get("length_change") != -40:
            fails.append("length_change != -40")
        idle_child = int((b[:, :6] == 0).all(axis=1).sum())
        idle_exp = int((a[40:, :6] == 0).all(axis=1).sum())
        idle_dist[eid] = idle_child
        if idle_child != idle_exp:  # AC1.8
            fails.append(f"idle_steps {idle_child} != parent's idle in a[40:T] {idle_exp}")
    elif ctype == "instruction_swap":
        if not (same("action") and same("state") and same("image") and same("image2")):
            fails.append("arrays changed")
        if out["task"] == ep["task"] or out["task"] not in parent_tasks or suite_of_task[out["task"]] != suite_of_task[ep["task"]]:
            fails.append("task not a different same-suite parent task")
        if edit.new_task != out["task"]:
            fails.append("new_task mismatch")
    elif ctype == "dropped_frames":
        if not (same("action") and same("state")):
            fails.append("action/state changed")
        pd_ = {k: set(C.frame_digests(ep[k])) for k in ("image", "image2")}
        for k in ("image", "image2"):
            if not set(C.frame_digests(out[k])) <= pd_[k]:
                fails.append(f"{k} contains a non-parent image")
        n_held = sum(int((out[k][1:] == out[k][:-1]).all(axis=(1, 2, 3)).sum()) for k in ("image", "image2"))
        if n_held < edit.params["n_held_frames"]:
            fails.append("held frames are not duplicates")
    elif ctype in ("detour", "idle_padding"):
        for k in ("image", "image2"):
            if not set(C.frame_digests(out[k])) <= set(C.frame_digests(ep[k])):
                fails.append(f"{k} contains a non-parent image")
        if not np.isin(b[:, 6], a[:, 6]).all():
            fails.append("gripper value outside parent")
        if ctype == "detour":
            dup = sum(int((out[k][1:] == out[k][:-1]).all(axis=(1, 2, 3)).sum()) for k in ("image", "image2"))
            if dup:
                fails.append(f"{dup} consecutive identical frame pairs")
        else:
            pts = edit.params["pause_points_parent"]
            for s, e in edit.params["corrupted_frames"]:
                if not ((b[s:e, :6] == 0).all() and (out["image"][s:e] == out["image"][s]).all()):
                    fails.append("pause not idle")
    elif ctype == "gripper_release":
        if not (same("state") and same("image") and same("image2")):
            fails.append("non-action array changed")
        diff = np.where((a != b).any(axis=1))[0]
        (s, e), = edit.params["corrupted_frames"]
        if not (np.array_equal(a[:, :6], b[:, :6]) and set(diff) <= set(range(s, e)) and (b[s:e, 6] == -1).all()):
            fails.append("release edit outside declared range")
    fails += check_params(edit, T2)
    if ctype != "action_noise" and (edit.params["n_clipped"] or edit.params["frac_beyond_clean_range"]):
        fails.append("non-noise type records clipping")
    return out, edit, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0")
    ap.add_argument("--per-type", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-hashes", type=Path)
    ap.add_argument("--check-hashes", type=Path)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--artifacts", type=Path, default=FEATURE_DIR / "artifacts")
    ap.add_argument("--no-seed-compare", action="store_true", help="skip the second-seed hash comparison")
    a = ap.parse_args()
    a.artifacts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v = resolve_version(a.version)
    root = REPO_ROOT / v["lerobot_root"]
    cand = Subset.load(f"{v['name']}/candidate")
    eps = cand.episodes().set_index("episode_index")
    stats = json.loads((root / "meta" / "stats.json").read_text())
    sigma = np.array(stats["action"]["std"], dtype=np.float64)
    parent_tasks = set(read_tasks(root).values())
    vt = version_episodes(v["dataset_version_id"])
    suite_of_task = dict(zip(vt["language_instruction"], vt["suite"]))
    suite_tasks = {s: sorted(g["language_instruction"].unique()) for s, g in vt.groupby("suite")}
    assert set(suite_of_task) == parent_tasks, "catalog instructions != tasks.parquet strings"
    print(f"version {v['name']} {v['dataset_version_id'][:12]}: {len(vt)} episodes, {len(cand.episode_ids)} candidates, "
          f"{len(parent_tasks)} tasks, suites {sorted(suite_tasks)}")

    # ---- AC1.6: eligibility over all candidates (actions only) -----------------------------------
    acts = all_actions(root)
    elig = {t: [] for t in C.TYPES}
    for idx in eps.index:
        for t in C.TYPES:
            if C.eligible(t, {"action": acts[int(idx)]}):
                elig[t].append(int(idx))
    elig_rows = []
    for t in C.TYPES:
        tasks_ = eps.loc[elig[t], "task_name"].nunique()
        suites_ = eps.loc[elig[t], "suite"].nunique()
        elig_rows.append({"type": t, "eligible": len(elig[t]), "of": len(eps), "tasks": int(tasks_), "suites": int(suites_)})
    print("\nAC1.6 eligibility over the 960 candidates")
    print(pd.DataFrame(elig_rows).to_string(index=False))
    # measured premise P7 for gripper_release
    n_final = sum(1 for i in eps.index if C.final_release(acts[int(i)][:, 6]) is not None)
    car = [C.final_release(acts[int(i)][:, 6])[1] for i in eps.index if C.final_release(acts[int(i)][:, 6]) is not None]
    print(f"P7 recheck on candidates: final release in {n_final}/{len(eps)}; carry span median {int(np.median(car))}, "
          f">=40 in {np.mean(np.array(car) >= 40):.1%}, >=30 in {np.mean(np.array(car) >= 30):.1%}")

    # ---- the test sample: the same seeded candidates for every type ------------------------------
    rng = np.random.default_rng(1000 + a.seed)
    n_sample = int(np.ceil(a.per_type / 0.85))
    pool = sorted(set.intersection(*[set(e) for e in elig.values()]))
    sample = sorted(int(x) for x in rng.choice(pool, size=min(n_sample, len(pool)), replace=False))
    print(f"\nsample: {len(sample)} candidates eligible for all seven types (seeded), over "
          f"{eps.loc[sample, 'task_name'].nunique()} tasks")

    fails_by = {t: [] for t in C.TYPES}
    n_by = {t: 0 for t in C.TYPES}
    hashes, hashes2 = {t: {} for t in C.TYPES}, {t: {} for t in C.TYPES}
    tot = {k: 0 for k in ("n_values", "n_clipped", "n_clipped_dims0_2", "n_beyond_clean_range", "n_dims0_2_values")}
    ranges_summary = {t: [] for t in C.TYPES}
    example_params = {}
    for k, (idx, ep) in enumerate(load_episodes(root, sample)):
        eid = eps.loc[idx, "episode_id"]
        for t in C.TYPES:
            out, edit, fails = run_episode(t, ep, eid, a.seed, sigma, suite_tasks, parent_tasks, suite_of_task)
            n_by[t] += 1
            if fails:
                fails_by[t].append((eid, fails))
            h = episode_content_hash(out["task"], out["action"], out["state"], out["image"], out["image2"])
            hashes[t][eid] = h
            if t == "action_noise":
                tot["n_values"] += edit.params["n_values"]
                tot["n_clipped"] += edit.params["n_clipped"]
                tot["n_clipped_dims0_2"] += edit.params["n_clipped_dims0_2"]
                tot["n_beyond_clean_range"] += edit.params["n_beyond_clean_range"]
                tot["n_dims0_2_values"] += 3 * len(out["action"])
            ranges_summary[t].append(sum(e - s for s, e in edit.params["corrupted_frames"]) / len(out["action"]))
            if t not in example_params:
                example_params[t] = json.loads(json.dumps(edit.params))
            if not a.no_seed_compare:
                out2, _, _ = run_episode(t, ep, eid, a.seed + 1, sigma, suite_tasks, parent_tasks, suite_of_task)
                hashes2[t][eid] = episode_content_hash(out2["task"], out2["action"], out2["state"], out2["image"], out2["image2"])
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{len(sample)} episodes ({time.time() - t0:.0f}s)")

    rows = []
    for t in C.TYPES:
        rows.append({"type": t, "episodes": n_by[t], "invariant_failures": len(fails_by[t]),
                     "mean_corrupted_share": round(float(np.mean(ranges_summary[t])), 3), "hypothesis": C.hypothesis_for(t)})
    print("\nAC1.2 / AC1.4 / AC1.1 invariants per type (range, gripper binary, length, content, params)")
    print(pd.DataFrame(rows).to_string(index=False))
    for t in C.TYPES:
        for eid, f in fails_by[t][:3]:
            print(f"  FAIL {t} {eid}: {f}")

    print("\nAC1.3 clip accounting, action_noise totals over the sample")
    clip = {"n_values": tot["n_values"], "n_clipped": tot["n_clipped"],
            "frac_clipped": tot["n_clipped"] / tot["n_values"],
            "n_clipped_dims0_2": tot["n_clipped_dims0_2"],
            "frac_clipped_dims0_2": tot["n_clipped_dims0_2"] / tot["n_dims0_2_values"],
            "n_beyond_clean_range": tot["n_beyond_clean_range"],
            "frac_beyond_clean_range": tot["n_beyond_clean_range"] / tot["n_dims0_2_values"]}
    print(json.dumps(clip, indent=1))
    print("other six types: n_clipped asserted 0 and frac_beyond_clean_range 0 inside run_episode "
          f"({sum(len(fails_by[t]) for t in C.TYPES if t != 'action_noise')} failures)")

    # ---- AC1.5 determinism -----------------------------------------------------------------------
    det = {"same_seed_cross_process": None, "different_seed_differs": None}
    if a.dump_hashes:
        a.dump_hashes.parent.mkdir(parents=True, exist_ok=True)
        a.dump_hashes.write_text(json.dumps(hashes, sort_keys=True))
        print(f"\nAC1.5 hashes dumped to {a.dump_hashes}")
    if a.check_hashes:
        prev = json.loads(a.check_hashes.read_text())
        det["same_seed_cross_process"] = {t: prev[t] == hashes[t] for t in C.TYPES}
        print("\nAC1.5 same (seed, episode_id) vs hashes of an earlier process, per type:", det["same_seed_cross_process"])
    if not a.no_seed_compare:
        det["different_seed_differs"] = {
            t: {"n": len(hashes[t]), "differing": sum(hashes[t][e] != hashes2[t][e] for e in hashes[t])} for t in C.TYPES}
        print("AC1.5 episodes whose hash differs between seed "
              f"{a.seed} and {a.seed + 1} (of {len(sample)}):", {t: d["differing"] for t, d in det["different_seed_differs"].items()})

    # ---- AC1.5 amended thresholds (plan r2) and AC1.8 distribution --------------------------------------
    ac15_ok = True
    if det["different_seed_differs"]:
        d_ = det["different_seed_differs"]
        n_ = len(sample)
        ac15_ok = (all(d_[t]["differing"] == n_ for t in ("action_noise", "dropped_frames", "idle_padding"))
                   and all(d_[t]["differing"] >= 0.8 * n_ for t in ("detour", "instruction_swap"))
                   and all(d_[t]["differing"] == 0 for t in ("time_shift", "gripper_release")))
        print(f"AC1.5 amended: noise/dropped/idle differ for all {n_}: {all(d_[t]['differing'] == n_ for t in ('action_noise', 'dropped_frames', 'idle_padding'))}; "
              f"detour {d_['detour']['differing'] / n_:.1%} and swap {d_['instruction_swap']['differing'] / n_:.1%} >= 80%: "
              f"{all(d_[t]['differing'] >= 0.8 * n_ for t in ('detour', 'instruction_swap'))}; time_shift and gripper_release seed-independent: "
              f"{all(d_[t]['differing'] == 0 for t in ('time_shift', 'gripper_release'))} -> {ac15_ok}")
    iv = np.array(list(idle_dist.values()))
    print(f"AC1.8 time_shift child idle_steps == parent's idle in a[40:T] for every episode (checked in run_episode); distribution over "
          f"{len(iv)} episodes: max {iv.max()}, p99 {np.percentile(iv, 99):.0f}, >0 in {(iv > 0).sum()}; clean maximum run is 21 (measured earlier)")
    # ---- AC1.1 an example label per type -----------------------------------------------------------
    print("\nAC1.1 example label params (keys), first sampled episode per type")
    for t in C.TYPES:
        p = example_params[t]
        print(f"  {t}: source={p['source'][:70]!r}; parameters={ {k: v['basis'] for k, v in p['parameters'].items()} }; "
              f"corrupted_frames={p['corrupted_frames'][:3]}; n_clipped={p['n_clipped']}")

    summary = {"version": v["name"], "seed": a.seed, "n_sample": len(sample), "eligibility": elig_rows,
               "p7": {"final_release": n_final, "carry_median": int(np.median(car)),
                      "carry_ge_40": float(np.mean(np.array(car) >= 40)), "carry_ge_30": float(np.mean(np.array(car) >= 30))},
               "invariants": rows, "clip_action_noise": clip, "determinism": det, "ac15_amended_ok": ac15_ok, "ac18_idle_steps": {"max": int(iv.max()), "p99": float(np.percentile(iv, 99)), "n_gt0": int((iv > 0).sum()), "n": int(len(iv))},
               "example_params": example_params, "failures": {t: fails_by[t][:5] for t in C.TYPES},
               "seconds": round(time.time() - t0, 1)}
    (a.artifacts / "s1_invariants.json").write_text(json.dumps(summary, indent=2, default=str))
    ok = all(not fails_by[t] for t in C.TYPES) and all(r["episodes"] >= a.per_type for r in rows)
    ok = ok and ac15_ok
    if det["same_seed_cross_process"]:
        ok = ok and all(det["same_seed_cross_process"].values())
    print(f"\nall invariants hold on >= {a.per_type} episodes per type: {ok} ({time.time() - t0:.0f}s)")

    if a.figure:
        figure(root, eps, sample, sigma, suite_tasks, suite_of_task, a)
    return 0 if ok else 1


def figure(root, eps, sample, sigma, suite_tasks, suite_of_task, a) -> None:
    """Seven-panel trace figure and a filmstrip (dropped_frames, detour, idle_padding) for one episode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, ep = next(iter(load_episodes(root, [sample[3]])))
    eid = eps.loc[idx, "episode_id"]
    outs, edits = {}, {}
    for t in C.TYPES:
        outs[t], edits[t], _ = run_episode(t, ep, eid, a.seed, sigma, suite_tasks, set(suite_of_task), suite_of_task)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=False)
    for ax, t in zip(axes.flat, C.TYPES):
        o = outs[t]
        ax.plot(ep["action"][:, 0], color="0.6", lw=1, label="clean dim 0 (x)")
        ax.plot(o["action"][:, 0], color="C0", lw=1, label="corrupted dim 0")
        ax.plot(ep["action"][:, 6] * 0.5 - 1.6, color="0.6", lw=1, ls="--")
        ax.plot(o["action"][:, 6] * 0.5 - 1.6, color="C3", lw=1, ls="--", label="gripper (shifted down)")
        for s, e in edits[t].params["corrupted_frames"]:
            if (e - s) < len(o["action"]):
                ax.axvspan(s, e, color="C1", alpha=0.18)
        ax.set_title(f"{t}  (T {len(ep['action'])} -> {len(o['action'])})", fontsize=10)
        ax.set_ylim(-2.2, 1.2)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"S1: clean vs corrupted action traces, {eid.split('/')[-1]} of {eps.loc[idx, 'task_name'][:50]} (orange = corrupted frames)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.artifacts / "s1_traces.png", dpi=110)
    plt.close(fig)
    rows = [("dropped_frames", "image"), ("detour", "image"), ("idle_padding", "image")]
    fig, axes = plt.subplots(len(rows) * 2, 8, figsize=(16, 4.2 * len(rows)))
    for r, (t, k) in enumerate(rows):
        s, e = edits[t].params["corrupted_frames"][0]
        pick = np.linspace(max(s - 2, 0), min(e + 2, len(outs[t]["image"]) - 1), 8).astype(int)
        for c, j in enumerate(pick):
            axes[2 * r, c].imshow(outs[t][k][j]); axes[2 * r, c].set_title(f"child {j}", fontsize=7)
            axes[2 * r, c].axis("off")
        base = np.linspace(max(s - 2, 0), min(s + 8, len(ep["image"]) - 1), 8).astype(int)
        for c, j in enumerate(base):
            axes[2 * r + 1, c].imshow(ep[k][j]); axes[2 * r + 1, c].set_title(f"clean {j}", fontsize=7)
            axes[2 * r + 1, c].axis("off")
        axes[2 * r, 0].text(-0.05, 0.5, t, transform=axes[2 * r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("S1 filmstrip: corrupted child frames (top row of each pair) vs clean parent frames (bottom)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.artifacts / "s1_filmstrip.png", dpi=90)
    plt.close(fig)
    print(f"figures -> {a.artifacts / 's1_traces.png'}, {a.artifacts / 's1_filmstrip.png'}")


if __name__ == "__main__":
    sys.exit(main())
