"""Evaluation-layer S3: the LIBERO human reference, by replaying every v0 demo through LIBERO's goal check.

Grounding: PoC §3.5 A (pp. 7-8) "Human reference = durations of LIBERO's original human-teleoperated
demonstrations on the same tasks"; §10 (p. 20) "HRT against LIBERO demo durations"; §9.1 Step 2.2 (p. 18)
"using the benchmark's goal check". Mechanism and alignment conventions: ``vlapoc.libero_goals``.

    .venv/bin/python scripts/human_reference.py --suites libero_spatial,libero_10   # replay (~minutes) -> parquet + sim_env.json
    .venv/bin/python scripts/human_reference.py --bddl-check --only libero_10:8   # AC3.6: same states, BDDL-built scene
    .venv/bin/python scripts/human_reference.py --init-states    # AC3.7: 50 fixed init states per task (LeRobot)
    .venv/bin/python scripts/human_reference.py --summary        # AC3.3, AC3.4 tables + figure (metric layer, S1)
    .venv/bin/python scripts/human_reference.py --operations     # S4: per-operation T (operations_from_trace) from stored traces

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


def _tasks(suites: list[str], only: str | None = None) -> list[tuple[str, list[dict]]]:
    """Episodes grouped by raw file; ``only`` = "suite:task_id,..." restricts to those tasks."""
    from vlapoc.lineage import version_episodes

    e = version_episodes("v0")
    e = e[e["suite"].isin(suites)]
    if only:
        keep = {(x.split(":")[0], int(x.split(":")[1])) for x in only.split(",")}
        e = e[[(s, int(t)) in keep for s, t in zip(e["suite"], e["libero_task_id"])]]
    return [(sf, g.sort_values("source_demo", key=lambda s: s.str.split("_").str[1].astype(int)).to_dict("records"))
            for sf, g in e.groupby("source_file", sort=True)]


def replay(suites: list[str], workers: int, use_demo_xml: bool = True, only: str | None = None) -> pd.DataFrame:
    """Replay every demo of ``suites`` (one worker process per task file) and return one row per demo."""
    t0 = time.time()
    jobs = [(sf, eps, use_demo_xml) for sf, eps in _tasks(suites, only)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_replay_task, jobs) for r in part]
    df = pd.DataFrame(rows)
    print(f"replayed {len(df)} demos from {len(jobs)} task files ({'demo XML' if use_demo_xml else 'BDDL scene'}) in "
          f"{time.time() - t0:.1f}s with {workers} workers")
    return df


def cmd_replay(a) -> None:
    """S3 / AC3.1-AC3.2: replay all v0 demos through the demo XML, write ``human_reference.parquet`` and ``sim_env.json``."""
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
    """AC3.6: replay the same stored states in the BDDL-built scene and compare first-success steps with the demo-XML replay."""
    df = replay(a.suites.split(","), a.workers, use_demo_xml=False, only=a.only)
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
    """AC3.7: count each task's fixed init states through LeRobot's ``get_task_init_states``."""
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
    """AC3.3-AC3.4: the gap between first success and demo length, and the per-task human reference table and figure."""
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


def cmd_operations(a) -> None:
    """S4: per-operation human T from the stored S3 traces (no second replay), through the single rule
    ``vlapoc.libero_goals.operations_from_trace``. AC4.1 operations table, AC4.3 consistency, AC4.4 CDFs."""
    from vlapoc import evaluation as ev
    from vlapoc import libero_goals as lg

    lg.setup_libero()
    df = pd.read_parquet(OUT_DIR / "human_reference.parquet")
    # AC4.1: the static table from the installed BDDL files vs each demo's own t0 truth
    table = []
    for (suite, tid), g in df.groupby(["suite", "libero_task_id"]):
        t = lg.operations_table(lg.bddl_path(suite, g["task_name"].iloc[0]))
        dyn = [sum(not x for x in r) for r in g["conjunct_true_at_t0"]]
        table.append({"suite": suite, "libero_task_id": int(tid), "task_name": g["task_name"].iloc[0], **t,
                      "n_ops_from_t0_truth": sorted(set(dyn)), "demos_matching_bddl": int(sum(d == t["n_ops"] for d in dyn)),
                      "n_demos": len(g)})
    tab = pd.DataFrame(table)
    tab.to_csv(ART / "s4_operations_table.csv", index=False)
    for suite in ("libero_10", "libero_spatial"):
        ts = tab[tab["suite"] == suite].sort_values("libero_task_id")
        print(f"AC4.1 {suite} n_ops per task (BDDL) = {ts['n_ops'].tolist()}; maintained = "
              f"{ {int(r.libero_task_id): r.maintained for r in ts.itertuples() if r.maintained} }; demos whose t0 truth "
              f"gives the same n_ops: {int(ts['demos_matching_bddl'].sum())}/{int(ts['n_demos'].sum())}")
    # operations per demo
    rows, check = [], []
    for r in df.itertuples():
        traces = json.loads(r.conjunct_trace_rle)
        truth = np.stack([lg.rle_decode(c) for c in traces], axis=1)
        succ = lg.rle_decode(json.loads(r.success_trace_rle))
        o = lg.operations_from_trace(truth, succ, len(succ) - 1, labels=list(r.conjuncts), key=r.episode_id)
        T = [t for t, _ in o["episode"].pairs]
        check.append({"episode_id": r.episode_id, "suite": r.suite, "libero_task_id": r.libero_task_id, "n_ops": o["n_ops"],
                      "events": sum(e for _, e in o["episode"].pairs), "censored": o["censored"],
                      "sum_T": float(np.sum(T)), "T_success_s": r.T_success_s})
        prev = 0
        for k, (t_k, conj) in enumerate(zip(o["t_ops"], o["op_conjuncts"]), start=1):
            rows.append({"episode_id": r.episode_id, "content_hash": r.content_hash, "suite": r.suite,
                         "libero_task_id": r.libero_task_id, "op_k": k, "n_ops": o["n_ops"], "t_start_steps": prev,
                         "t_end_steps": t_k, "T_op_s": (t_k - prev) / lg.FPS, "event": 1, "conjuncts_completed": conj})
            prev = t_k
    ops = pd.DataFrame(rows)
    ops.to_parquet(OUT_DIR / "human_operations.parquet", index=False)
    ck = pd.DataFrame(check)
    l10 = ck[ck["suite"] == "libero_10"]
    ok = (l10["sum_T"] - l10["T_success_s"]).abs() <= 1e-9
    print(f"AC4.3 libero_10: sum_k T_k == T_success_s for {int(ok.sum())}/{len(l10)}; demos with events == n_ops: "
          f"{int((l10['events'] == l10['n_ops']).sum())}/{len(l10)}; censored pairs: {int(l10['censored'].sum())}; "
          f"operation samples: {int(l10['events'].sum())} from {len(l10)} demos")
    sp = ck[ck["suite"] == "libero_spatial"]
    print(f"AC4.3 libero_spatial: 1 op each: {int((sp['n_ops'] == 1).sum())}/{len(sp)}; T_op == T_success_s: "
          f"{int(((sp['sum_T'] - sp['T_success_s']).abs() <= 1e-9).sum())}/{len(sp)}")
    # AC4.4: per-operation human CDFs per libero_10 task, per op index, clustered bootstrap
    stats = []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    grid = np.linspace(0, 20, 400)
    for ax, (tid, g) in zip(axes.flat, ops[ops["suite"] == "libero_10"].groupby("libero_task_id")):
        eps_all = [ev.Episode(eid, tuple((float(x), 1) for x in gg["T_op_s"])) for eid, gg in g.groupby("episode_id")]
        ci_all = ev.cluster_bootstrap({"h": {"t": eps_all}}, lambda a: ev.median_T(a["h"]["t"]), 1000, np.random.default_rng(0))
        per_op = ev.cluster_bootstrap({"h": {"t": eps_all}}, lambda a: ev.median_T(a["h"]["t"]), 1000, np.random.default_rng(0),
                                      unit="operation")
        row = {"libero_task_id": int(tid), "n_demos": len(eps_all), "n_op_samples": int(len(g)),
               "median_T_op_all": ci_all["point"], "median_T_op_all_ci": [ci_all["lo"], ci_all["hi"]],
               "rmst60_all": ev.rmst(eps_all, TAU_EPISODE)}
        for k, gk in g.groupby("op_k"):
            ek = [ev.Episode(r.episode_id, ((float(r.T_op_s), 1),)) for r in gk.itertuples()]
            c = ev.cluster_bootstrap({"h": {"t": ek}}, lambda a: ev.median_T(a["h"]["t"]), 1000, np.random.default_rng(0))
            row[f"median_T_op{k}"] = c["point"]
            row[f"median_T_op{k}_ci"] = [c["lo"], c["hi"]]
            ax.step(grid, ev.F_curve(ek, grid), where="post", label=f"op {k} (n={len(ek)})")
        # clustered (whole demos) vs per-operation CI width for the pooled median (within-demo structure)
        row["median_T_op_all_ci_per_operation"] = [per_op["lo"], per_op["hi"]]
        row["ci_width_ratio_clustered_vs_per_op"] = (ci_all["hi"] - ci_all["lo"]) / max(per_op["hi"] - per_op["lo"], 1e-12)
        stats.append(row)
        ax.set_title(f"libero_10 task {tid}", fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("per-operation T (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F(t)")
    fig.suptitle("Human per-operation time-to-success, libero_10 (KM over 50 demos per task; replay of LIBERO's goal check)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ART / "s4_fig_human_operation_cdfs.png", dpi=120)
    st = pd.DataFrame(stats)
    st.to_csv(ART / "s4_human_operation_stats.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(st.round(3).to_string(index=False))
    print(f"wrote {(OUT_DIR / 'human_operations.parquet').relative_to(REPO_ROOT)} ({len(ops)} operation rows), "
          "artifacts/s4_operations_table.csv, s4_human_operation_stats.csv, s4_fig_human_operation_cdfs.png")


def main() -> int:
    """Dispatch to one command (usage in the module docstring); with no flag, the full replay."""
    p = argparse.ArgumentParser()
    p.add_argument("--suites", default="libero_spatial,libero_10")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--bddl-check", action="store_true")
    p.add_argument("--init-states", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--operations", action="store_true", help="S4: per-operation T from the stored traces")
    p.add_argument("--only", default=None, help='restrict --bddl-check to tasks, e.g. "libero_10:8"')
    a = p.parse_args()
    if a.bddl_check:
        cmd_bddl_check(a)
    elif a.init_states:
        cmd_init_states(a)
    elif a.summary:
        cmd_summary(a)
    elif a.operations:
        cmd_operations(a)
    else:
        cmd_replay(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
