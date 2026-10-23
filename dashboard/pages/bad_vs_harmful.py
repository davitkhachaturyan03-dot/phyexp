"""Bad vs harmful page (PoC §9.2 p. 19: "episodes that look bad but do not hurt the policy, and episodes that look clean but do"). Planted-data versions only.

Everything shown is read from the ``discrepancy`` run (estimated, from Round A scores) and the ``discrepancy-verdict`` run (Round B, causal); the status and sentence of a
Round B arm are displayed verbatim, never re-worded here.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common
from vlapoc import results as res
from vlapoc.discrepancy import QUADRANTS


def render(ctx: dict) -> None:
    st.subheader("Bad vs harmful: do the episodes that look bad hurt the policy?")
    lr = ctx["lr"]
    runs = [r for r in common.runs_from(lr, "score", "score_run_id", ctx["run_id"]) if "discrepancy_episodes" in r["outputs"]]
    if not runs:
        st.info("No discrepancy run for this score run. It needs a version with an answer key (planted corruptions, e.g. v3): "
                "`scripts/stage.py discrepancy --scores <run> --version v3 --subset candidate --name <name>`.")
        return
    drun = runs[0]
    st.markdown("#### ground-truth check (answer key read after attribution)")
    st.caption(f"Discrepancy run `{drun['name']}` ({drun['run_id'][:8]}). \"Looks bad\" is the planted corruption label, read after the attribution (primary), or a flag of the label-free "
               "heuristics H1-H8 (secondary: what a practical filter would have dropped). \"Harmful\" is a harm z-score of at least 1. These are estimates from first-order influence (CUPID p. 11).")
    ep, ty = common.table_of(lr, drun["run_id"], "discrepancy_episodes"), common.table_of(lr, drun["run_id"], "discrepancy_types")
    ex = common.json_of(lr, drun["run_id"], "examples")
    summ = common.json_of(lr, drun["run_id"], "discrepancy_summary")
    metrics = [c.removeprefix("z__") for c in ep.columns if c.startswith("z__")]
    m = st.selectbox("Metric", metrics, format_func=lambda k: common.label(f"harm_{k}", ctx["decls"]), key="bvh_metric")
    lab = common.label(f"harm_{m}", ctx["decls"])
    use_filter = st.toggle("Define \"looks bad\" by the H1-H8 filter instead of the planted labels", value=False, key="bvh_filter")
    qcol = f"quadrant_filter__{m}" if use_filter else f"quadrant__{m}"
    counts = ep[qcol].value_counts().reindex(QUADRANTS, fill_value=0)
    st.markdown(f"Quadrant counts for **{lab}** ({'H1-H8 filter' if use_filter else 'planted labels'}):")
    st.dataframe(pd.DataFrame({"quadrant": counts.index, "episodes": counts.values}), hide_index=True, width="stretch", key="bvh_quadrants")
    bad = ep["looks_bad_filter"] if use_filter else ep["planted_type"] != "clean"
    d = ep.assign(group=[("looks bad: " + (", ".join(f.split(",")[:1]) if f else "flagged")) if (use_filter and b) else (f"looks bad: {t}" if b else "looks clean")
                          for b, t, f in zip(bad, ep["planted_type"], ep["flagged_by"])])
    syn = "SYNTHETIC DATA - " if ctx["prov"]["label"] == "SYNTHETIC" else ""
    chart = alt.Chart(d).mark_circle(size=35, opacity=0.6).encode(
        x=alt.X("group:N", title="looks bad (planted type or H-flag) vs clean"), y=alt.Y(f"z__{m}:Q", title="harm z-score: " + lab),
        color=alt.Color(f"{qcol}:N", scale=alt.Scale(domain=list(QUADRANTS)), title="quadrant"), tooltip=["episode_id", "planted_type", f"z__{m}"]).properties(
        title=f"{syn}Harm z-score by whether the episode looks bad ({lab})", height=360)
    st.altair_chart(chart, width="stretch")
    st.markdown("**Per planted type (estimated):** mean harm z against the clean episodes, with the bounded reading.")
    tt = ty[ty["metric"] == m][["planted_type", "n", "mean_z", "clean_mean_z", "diff", "diff_lo", "diff_hi", "estimated_reading", "best_heuristic", "best_heuristic_auroc", "hypothesis"]]
    tt = tt.rename(columns={"mean_z": f"mean z, {lab}", "clean_mean_z": f"clean mean z, {lab}", "best_heuristic_auroc": "best label-free heuristic AUROC (visibly bad)"})
    st.dataframe(tt, hide_index=True, width="stretch", key="bvh_types")
    vr = [r for r in common.runs_from(lr, "summarize", "kind", "discrepancy-verdict (Round B planted, PoC §9.2)") if "discrepancy_verdict" in r["outputs"]]
    if vr:
        st.markdown(f"**Round B (retrained without the arm; verdict run `{vr[0]['name']}`, {vr[0]['run_id'][:8]}).** Statuses and sentences are shown as recorded.")
        vt = common.table_of(lr, vr[0]["run_id"], "discrepancy_verdict")
        st.dataframe(vt[["arm", "metric", "status", "sentence", "calls", "claim_clean_but_harmful"]], hide_index=True, width="stretch", key="bvh_verdict")
    else:
        st.caption("No Round B planted verdicts yet (`server_pipeline.py --round-b-planted`, then `discrepancy-verdict`).")
    st.markdown("**Example: the corrupted episode next to its clean original**")
    ty_pick = st.selectbox("Planted type", sorted(ex["types"]), key="bvh_type")
    e = ex["types"][ty_pick]
    which = st.radio("Member", ["most_harmful", "least_harmful"], format_func=lambda k: e["by_metric"][m][k]["which"], horizontal=True, key="bvh_which")
    pick = e["by_metric"][m][which]
    ev = e["visibly_bad_evidence"]
    st.caption(f"{ty_pick}: {e['hypothesis']}. " + (f"Visibly bad: the label-free heuristic {ev['best_heuristic']} separates it from clean data with AUROC {ev['auroc']:.3f}." if ev else ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"Corrupted ({pick['episode_id']}, z = {pick['z']:.2f})")
        st.video(common.training_video(lr, ctx["run_id"], pick["episode_id"]), width=420)
    with c2:
        st.markdown("Clean original")
        orig = common.original_video(lr, ctx["run_id"], pick["original_episode_id"], pick["original_content_hash"])
        if orig is None:
            st.info("The clean original is not available in the parent version on this machine.")
        else:
            st.video(orig, width=420)
