"""Rollouts page (PoC §2.7 view 1; PhAIL's run explorer is the design reference): pick a rollout, see its outcome and video, then the training episodes that pushed it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import common
from vlapoc import results as res


def episode_rows(order: np.ndarray, row: np.ndarray, idx: dict, cat: pd.DataFrame) -> pd.DataFrame:
    """The catalog rows of the episodes at matrix columns ``order`` (in that order) with their influence on the rollout."""
    c = cat.set_index("episode_id")
    ids = [idx["episode_id"][j] for j in order]
    out = c.loc[ids, ["task_name", "scene", "source", "source_file"]].reset_index()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["influence on this rollout"] = [float(row[j]) for j in order]
    return out


def render(ctx: dict) -> None:
    st.subheader("Rollouts: which training episodes pushed this rollout?")
    lr = ctx["lr"]
    seeds = res.seed_runs(ctx["run_id"], common._r(lr))
    seed_run = seeds[0]
    if len(seeds) > 1:
        st.caption("This is an ensemble (a z-scored average over seeds), which has no rollout x episode matrix of its own; choose a seed run.")
        info = {s: common.run_info(lr, s)["name"] for s in seeds}
        seed_run = st.selectbox("Seed run", seeds, format_func=lambda s: f"{info[s]} ({s[:8]})", key="seed_run")
    try:
        ro = common.rollouts_of(lr, seed_run)
        m, idx = common.influence_of(lr, seed_run)
    except res.NotRecordedError as e:
        st.info(f"{e}")
        return
    ro = ro.copy()
    if "task_name" not in ro:  # an eval record without task names (e.g. a test fixture): label by suite and task id
        ro["task_name"] = ro["suite"].astype(str) + "/" + ro["libero_task_id"].astype(str)
    ro["failed"] = ro["failure_type"] != "success"
    f1, f2, f3 = st.columns(3)
    only_failed = f1.checkbox("Failed rollouts only", value=True, key="only_failed")
    types = sorted(ro["failure_type"].dropna().unique())
    ftypes = f2.multiselect("Failure type", types, default=types, key="ftypes")
    tasks = sorted(ro["task_name"].dropna().unique())
    ftasks = f3.multiselect("Task", tasks, default=tasks, key="ftasks")
    view = ro[(ro["failed"] | (not only_failed)) & ro["failure_type"].isin(ftypes) & ro["task_name"].isin(ftasks)]
    cols = [c for c in ("rollout_id", "task_name", "failure_type", "T_s", "event", "censored") if c in view]  # an eval record may lack the timing columns
    st.caption(f"{len(view)} of {len(ro)} rollouts")
    ev = st.dataframe(view[cols], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="roll_table")
    ids = list(view["rollout_id"])
    if not ids:
        st.info("No rollout matches the filters.")
        return
    clicked = ev.selection.rows[0] if ev is not None and ev.selection.rows else None
    rid = st.selectbox("Rollout (or click a row above)", ids, index=clicked if clicked is not None and clicked < len(ids) else 0, key="rollout_pick")
    r = ro[ro["rollout_id"] == rid].iloc[0]
    st.markdown(f"**{rid}**")
    a, b, c, d = st.columns(4)
    a.metric("Failure type", r["failure_type"])
    b.metric("T (s)", "not recorded" if "T_s" not in r else ("none (did not succeed)" if pd.isna(r["T_s"]) else f"{r['T_s']:.1f}"))
    c.metric("Event", "not recorded" if "event" not in r else int(r["event"]))
    d.metric("Censored", "not recorded" if "censored" not in r else str(bool(r["censored"])))
    if pd.notna(r.get("traj_episode_index")):
        vid = common.rollout_video(lr, r["eval_run_id"], int(r["traj_episode_index"]))
        if vid is None:
            st.info("Rollout video not on this machine (the eval's recorded trajectories were not copied back).")
        else:
            st.video(vid, width=420)
    if pd.isna(r["matrix_row"]):
        st.info("This rollout has no row in the influence matrix.")
        return
    row = m[int(r["matrix_row"])]
    pos, neg = np.argsort(-row, kind="stable")[:10], np.argsort(row, kind="stable")[:10]
    st.markdown(f"Influence of each training episode on this rollout (matrix cell). Positive = pushed the policy toward this rollout, negative = pushed it away. "
                f"The matrix and its index come from run `{common.run_info(lr, seed_run)['name']}`.")
    for title, order, key in (("Pushed toward this rollout (top 10)", pos, "toward"), ("Pushed away from this rollout (top 10)", neg, "away")):
        st.markdown(f"**{title}**")
        tbl = episode_rows(order, row, idx, ctx["cat"])
        st.dataframe(tbl, hide_index=True, width="stretch", key=f"tbl_{key}")
        pick = st.selectbox("Play episode (rank)", list(tbl["rank"]), key=f"play_{key}")
        eid = tbl.loc[tbl["rank"] == pick, "episode_id"].iloc[0] if "episode_id" in tbl else idx["episode_id"][order[pick - 1]]
        st.video(common.training_video(lr, seed_run, eid), width=420)
        st.caption(f"{eid}: source file {tbl.loc[tbl['rank'] == pick, 'source_file'].iloc[0]}")
