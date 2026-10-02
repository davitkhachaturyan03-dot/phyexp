"""Evaluation-layer S3: the LIBERO human reference, by replaying every v0 demo through LIBERO's goal check.

Grounding: PoC §3.5 A (pp. 7-8) "Human reference = durations of LIBERO's original human-teleoperated
demonstrations on the same tasks"; §10 (p. 20) "HRT against LIBERO demo durations"; §9.1 Step 2.2 (p. 18)
"using the benchmark's goal check". Mechanism and alignment conventions: ``vlapoc.libero_goals``.

    .venv/bin/python scripts/human_reference.py --suites libero_spatial,libero_10   # replay (~minutes) -> parquet + sim_env.json
    .venv/bin/python scripts/human_reference.py --bddl-check     # AC3.6: the same states in the BDDL-built scene
    .venv/bin/python scripts/human_reference.py --init-states    # AC3.7: 50 fixed init states per task (LeRobot)
    .venv/bin/python scripts/human_reference.py --summary        # AC3.3, AC3.4 tables + figure (metric layer, S1)

Output: ``data/eval/human_reference/human_reference.parquet``, one row per v0 episode (keyed by
``episode_id`` + ``content_hash``): ``T_success_steps`` (first stored state where ``_check_success()``
holds), ``T_success_s``, ``duration_s``, ``gap_steps`` = ``length_steps - T_success_steps``, per-conjunct
first-true steps and truth at t0, and the run-length-encoded conjunct and success traces (S4 builds the
per-operation T from them without replaying again). The HRT anchor is all 1000 v0 originals, fixed
across every compared policy (plan: a ratio's denominator must not move between arms; inferred).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "eval" / "human_reference"
ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
TAU_EPISODE = 60.0  # plan: τ_episode = 60 s = RMST cap (owner default, brief 2026-10-02)


def _replay_task(args: tuple[str, list[dict], bool]) -> list[dict]:
    """Replay every demo of one raw HDF5 file (one task). ``use_demo_xml=False`` is the P9 comparison."""
    source_file, eps, use_demo_xml = args
    import h5py

    from vlapoc import libero_goals as lg

    lg.setup_libero()
    suite, task_name = eps[0]["suite"], eps[0]["task_name"]
    bddl = lg.bddl_path(suite, task_name)
    env = lg.make_replay_env(bddl)
    rows = []
    with h5py.File(REPO_ROOT / source_file, "r") as f:
        for ep in eps:
            g = f[f"data/{ep['source_demo']}"]
            states = g["states"][()]
            t0 = time.time()
            try:
                xml = lg.demo_model_xml(g.attrs["model_file"]) if use_demo_xml else None
                tr = lg.replay_goal_trace(env, states, xml)
                err = None
            except Exception as e:  # recorded, never dropped (AC3.2)
                tr, err = None, f"{type(e).__name__}: {e}"[:300]
            row = {"episode_id": ep["episode_id"], "content_hash": ep["content_hash"], "suite": suite, "task_name": task_name,
                   "libero_task_id": int(ep["libero_task_id"]), "source_demo": ep["source_demo"],
                   "length_steps": int(ep["length_steps"]), "duration_s": float(ep["duration_s"]), "n_states": int(len(states)),
                   "replay": "demo_xml" if use_demo_xml else "bddl_scene", "error": err, "replay_s": round(time.time() - t0, 3)}
            if tr is not None:
                t_succ = lg.first_true(tr["success"])
                tail = 0
                for v in tr["success"][::-1]:
                    if not v:
                        break
                    tail += 1
                row.update({
                    "success_any": bool(t_succ >= 0), "T_success_steps": t_succ if t_succ >= 0 else None,
                    "T_success_s": t_succ / lg.FPS if t_succ >= 0 else None,
                    "gap_steps": int(ep["length_steps"]) - t_succ if t_succ >= 0 else None,
                    "success_at_last_state": bool(tr["success"][-1]), "success_tail_run": int(tail),
                    "consistent": tr["consistent"], "conjuncts": [lg.conjunct_label(c) for c in tr["conjuncts"]],
                    "conjunct_first_true": [lg.first_true(tr["truth"][:, c]) for c in range(tr["truth"].shape[1])],
                    "conjunct_true_at_t0": [bool(x) for x in tr["truth"][0]],
                    "conjunct_trace_rle": json.dumps([lg.rle(tr["truth"][:, c]) for c in range(tr["truth"].shape[1])]),
                    "success_trace_rle": json.dumps(lg.rle(tr["success"])),
                })
            rows.append(row)
    env.env.close() if hasattr(env.env, "close") else None
    return rows


def _tasks(suites: list[str]) -> list[tuple[str, list[dict]]]:
    from vlapoc.lineage import version_episodes

    e = version_episodes("v0")
    e = e[e["suite"].isin(suites)]
    return [(sf, g.sort_values("source_demo", key=lambda s: s.str.split("_").str[1].astype(int)).to_dict("records"))
            for sf, g in e.groupby("source_file", sort=True)]


def replay(suites: list[str], workers: int, use_demo_xml: bool = True) -> pd.DataFrame:
    t0 = time.time()
    jobs = [(sf, eps, use_demo_xml) for sf, eps in _tasks(suites)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_replay_task, jobs) for r in part]
    df = pd.DataFrame(rows)
    print(f"replayed {len(df)} demos from {len(jobs)} task files ({'demo XML' if use_demo_xml else 'BDDL scene'}) in "
          f"{time.time() - t0:.1f}s with {workers} workers")
    return df


def cmd_replay(a) -> None:
    from vlapoc import libero_goals as lg

    home_libero_before = Path.home().joinpath(".libero").exists()
    df = replay(a.suites.split(","), a.workers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "human_reference.parquet", index=False)
    lg.setup_libero()
    env = {**lg.env_versions(), "assets_digest": lg.assets_digest(), "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
           "home_libero_exists_before": home_libero_before, "home_libero_exists_after": Path.home().joinpath(".libero").exists(),
           "home_cache_libero_exists": Path.home().joinpath(".cache", "libero").exists(),
           "n_demos": len(df), "errors": int(df["error"].notna().sum())}
    (ART / "sim_env.json").write_text(json.dumps(env, indent=2))
    print(json.dumps({k: env[k] for k in ("python", "packages", "assets_revision", "assets_digest", "home_libero_exists_after",
                                          "home_cache_libero_exists")}, indent=1))
    ok = df["success_any"].fillna(False)
    print(f"AC3.2 goal reached at some stored state: {int(ok.sum())}/{len(df)}; replay errors: {int(df['error'].notna().sum())}; "
          f"_check_success == AND(conjuncts) at every state: {int(df['consistent'].fillna(False).sum())}/{len(df)}")
    fail = df[~ok]
    if len(fail):
        print("AC3.2 demos without success (listed, not dropped):")
        print(fail[["episode_id", "error"]].to_string(index=False))
    print(f"wrote {(OUT_DIR / 'human_reference.parquet').relative_to(REPO_ROOT)} and {(ART / 'sim_env.json').relative_to(REPO_ROOT)}")


def cmd_bddl_check(a) -> None:
    df = replay(a.suites.split(","), a.workers, use_demo_xml=False)
    ref = pd.read_parquet(OUT_DIR / "human_reference.parquet").set_index("episode_id")
    df = df.set_index("episode_id")
    both = df.join(ref[["T_success_steps", "success_any"]], rsuffix="_demo_xml")
    no_succ = both[~both["success_any"].fillna(False).astype(bool)]
    differ = both[both["success_any"].fillna(False).astype(bool) & (both["T_success_steps"] != both["T_success_steps_demo_xml"])]
    print(f"AC3.6 BDDL-scene replay: errors {int(df['error'].notna().sum())}; no success at any state: {len(no_succ)}/{len(df)}; "
          f"success but different T than demo-XML replay: {len(differ)}")
    for eid, r in no_succ.iterrows():
        print(f"AC3.6   no success in BDDL scene: {eid} (demo-XML T = {r['T_success_steps_demo_xml']}, error {r['error']})")
    for eid, r in differ.head(10).iterrows():
        print(f"AC3.6   T differs: {eid}: BDDL {r['T_success_steps']} vs demo XML {r['T_success_steps_demo_xml']}")
    out = both.reset_index()[["episode_id", "error", "success_any", "T_success_steps", "T_success_steps_demo_xml"]]
    out.to_csv(ART / "s3_bddl_vs_demo_xml.csv", index=False)


def cmd_init_states(a) -> None:
    from vlapoc import libero_goals as lg

    lg.setup_libero()
    from libero.libero import benchmark
    from lerobot.envs.libero import get_task_init_states

    res = {}
    for suite in a.suites.split(","):
        ts = benchmark.get_benchmark_dict()[suite]()
        res[suite] = {ts.get_task(i).name: list(get_task_init_states(ts, i).shape) for i in range(ts.get_num_tasks())}
        print(f"AC3.7 {suite}: {ts.get_num_tasks()} tasks, init-state array shapes {sorted({tuple(v) for v in res[suite].values()})}")
    n50 = sum(v[0] == 50 for s in res.values() for v in s.values())
    print(f"AC3.7 tasks with exactly 50 fixed init states: {n50}/{sum(len(s) for s in res.values())}")
    (ART / "s3_init_states.json").write_text(json.dumps(res, indent=1))


def cmd_summary(a) -> None:
    from vlapoc import evaluation as ev

    df = pd.read_parquet(OUT_DIR / "human_reference.parquet")
    df = df[df["success_any"].fillna(False).astype(bool)]
    print(f"AC3.3 T_success_s < duration_s: {int((df['T_success_s'] < df['duration_s']).sum())}/{len(df)}")
    g = df.groupby(["suite", "libero_task_id"])["gap_steps"].agg(["min", "median", "max"]).reset_index()
    g.to_csv(ART / "s3_gap_steps_per_task.csv", index=False)
    print("AC3.3 gap = length_steps - T_success_steps per task (min/median/max): range over tasks: "
          f"min {int(g['min'].min())}-{int(g['min'].max())}, median {g['median'].min():.1f}-{g['median'].max():.1f}, "
          f"max {int(g['max'].min())}-{int(g['max'].max())} (planning: min 8-10, median 10-18, max 81)")
    print(f"AC3.3 success holds over the final stored states: tail run min {int(df['success_tail_run'].min())}, "
          f"median {df['success_tail_run'].median():.0f}; success at the last stored state: {int(df['success_at_last_state'].sum())}/{len(df)}")
    rows = []
    for (suite, tid), gg in [*df.groupby(["suite", "libero_task_id"]), *[((s, "all"), x) for s, x in df.groupby("suite")]]:
        hT = [ev.libero_episode(r.episode_id, [r.T_success_s], r.T_success_s, 1) for r in gg.itertuples()]
        hD = [ev.libero_episode(r.episode_id, [r.duration_s], r.duration_s, 1) for r in gg.itertuples()]
        rT, rD = ev.rmst(hT, TAU_EPISODE), ev.rmst(hD, TAU_EPISODE)
        rows.append({"suite": suite, "libero_task_id": tid, "n": len(gg), "median_T_success_s": ev.median_T(hT),
                     "median_duration_s": ev.median_T(hD), "RMST60_T_success": rT, "RMST60_duration": rD,
                     "F30_T_success": ev.F_at(hT, 30.0), "F30_duration": ev.F_at(hD, 30.0),
                     "HRT_bias_factor_duration_over_T": rD / rT,
                     "task_name": gg["task_name"].iloc[0] if tid != "all" else ""})
    t = pd.DataFrame(rows)
    t.to_csv(ART / "s3_human_reference_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.drop(columns=["task_name"]).round(3).to_string(index=False))
    for s in ("libero_spatial", "libero_10"):
        r = t[(t["suite"] == s) & (t["libero_task_id"] == "all")].iloc[0]
        print(f"AC3.4 {s}: median T_success {r['median_T_success_s']:.2f} s vs duration {r['median_duration_s']:.2f} s; RMST60 "
              f"{r['RMST60_T_success']:.3f} vs {r['RMST60_duration']:.3f} s -> an HRT anchored on duration_s is {r['HRT_bias_factor_duration_over_T']:.3f}x too high")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, s, tmax in zip(axes, ("libero_spatial", "libero_10"), (12, 30)):
        grid = np.linspace(0, tmax, 600)
        gs = df[df["suite"] == s]
        for tid, gg in gs.groupby("libero_task_id"):
            ax.step(grid, ev.F_curve([ev.libero_episode(r.episode_id, [r.T_success_s], r.T_success_s, 1) for r in gg.itertuples()], grid),
                    where="post", color="#4c72b0", alpha=0.35, linewidth=0.9)
        ax.step(grid, ev.F_curve([ev.libero_episode(r.episode_id, [r.T_success_s], r.T_success_s, 1) for r in gs.itertuples()], grid),
                where="post", color="#1f3b73", linewidth=2.2, label="T_success (goal check, replay)")
        ax.step(grid, ev.F_curve([ev.libero_episode(r.episode_id, [r.duration_s], r.duration_s, 1) for r in gs.itertuples()], grid),
                where="post", color="#c44e52", linewidth=1.8, linestyle="--", label="duration_s (demo length)")
        ax.set_title(f"{s}: 10 tasks x 50 human demos", fontsize=10, loc="left")
        ax.set_xlabel("time (s of simulated time, 20 Hz)")
    axes[0].set_ylabel("F(t), KM over demos")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "s3_fig_human_reference_cdfs.png", dpi=130)
    print("wrote artifacts/s3_human_reference_table.csv, s3_gap_steps_per_task.csv, s3_fig_human_reference_cdfs.png")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_spatial,libero_10")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bddl-check", action="store_true")
    p.add_argument("--init-states", action="store_true")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.bddl_check:
        cmd_bddl_check(a)
    elif a.init_states:
        cmd_init_states(a)
    elif a.summary:
        cmd_summary(a)
    else:
        cmd_replay(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
