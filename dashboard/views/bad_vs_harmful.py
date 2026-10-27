"""Bad vs harmful page (PoC §9.2 p. 19: "episodes that look bad but do not hurt the policy, and episodes that look clean but do"). Planted-data versions only.

Everything shown is read from the ``discrepancy`` run (estimated, from Round A scores) and the ``discrepancy-verdict`` run (Round B, causal); the status and sentence of a
Round B arm are displayed verbatim, never re-worded here (the recorded text of a synthetic run begins ``SYNTHETIC:``).
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common, words
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
               "heuristics H1-H8 (secondary: what a practical filter would have dropped). \"Hurts\" means a harm z-score of at least 1. These are estimates from first-order influence (CUPID p. 11); "
               "readings are uncorrected across the 7 types x 5 metrics.")
    ep, ty = common.table_of(lr, drun["run_id"], "discrepancy_episodes"), common.table_of(lr, drun["run_id"], "discrepancy_types")
    ex = common.json_of(lr, drun["run_id"], "examples")
    metrics = [c.removeprefix("z__") for c in ep.columns if c.startswith("z__")]
    m = st.selectbox("Metric", metrics, format_func=lambda k: words.metric_name(k), key="bvh_metric", help="; ".join(common.declared(f"harm_{k}", ctx["decls"]) for k in metrics))
    common.declared_caption([f"harm_{m}"], ctx["decls"])
    lab = words.metric_name(m)
    use_filter = st.toggle("Define \"looks bad\" by the H1-H8 filter instead of the planted labels", value=False, key="bvh_filter")
    qcol = f"quadrant_filter__{m}" if use_filter else f"quadrant__{m}"
    counts = ep[qcol].value_counts().reindex(QUADRANTS, fill_value=0)
    st.markdown(f"Episodes by quadrant for **{lab}** ({'H1-H8 filter' if use_filter else 'planted labels'}):")
    st.dataframe(pd.DataFrame({"quadrant": [words.QUADRANTS[q] for q in counts.index], "episodes": counts.values}), hide_index=True, width="stretch", key="bvh_quadrants")
    bad = ep["looks_bad_filter"] if use_filter else ep["planted_type"] != "clean"
    d = ep.assign(group=[("flagged by a filter" if b else "looks clean") if use_filter else (words.PLANTED.get(t, t) if b else "clean") for b, t in zip(bad, ep["planted_type"])],
                  quadrant=[words.QUADRANTS[q] for q in ep[qcol]], z=ep[f"z__{m}"])
    syn = "SYNTHETIC - " if ctx["prov"]["label"] == "SYNTHETIC" else ""
    chart = alt.Chart(d).mark_circle(size=35, opacity=0.6).encode(
        y=alt.Y("group:N", title=None), x=alt.X("z:Q", title="Harm score (z; positive = harmful)"),
        color=alt.Color("quadrant:N", scale=alt.Scale(domain=[words.QUADRANTS[q] for q in QUADRANTS], range=["#c1121f", "#f4a261", "#7b2cbf", "#8d99ae"]), legend=alt.Legend(orient="bottom", title=None, columns=2)),
        tooltip=["episode_id", "group", "z"]).properties(title=f"{syn}Harm score of each episode on {lab}, by whether it looks bad", height=alt.Step(30))
    st.altair_chart(chart, width="stretch")
    st.markdown("**Per planted type (estimated):** mean harm score against the clean episodes, with the bounded reading (Welch interval; Holm across the metrics in the sentence).")
    tt = ty[ty["metric"] == m][["planted_type", "n", "mean_z", "clean_mean_z", "diff", "diff_lo", "diff_hi", "p_holm_metrics", "estimated_reading", "best_heuristic", "best_heuristic_auroc", "hypothesis"]].copy()
    tt["planted_type"] = [words.PLANTED.get(x, x) for x in tt["planted_type"]]
    tt["best_heuristic"] = [words.plain(x) for x in tt["best_heuristic"]]
    tt = tt.rename(columns={"planted_type": "type", "n": "episodes", "mean_z": "mean harm", "clean_mean_z": "clean mean harm", "diff": "difference", "diff_lo": "95% CI low (Welch)",
                            "diff_hi": "95% CI high (Welch)", "p_holm_metrics": "Holm p (across metrics)", "estimated_reading": "reading (estimated)",
                            "best_heuristic": "best label-free heuristic", "best_heuristic_auroc": "its AUROC (visibly bad)", "hypothesis": "label's stated hypothesis"})
    st.dataframe(tt, hide_index=True, width="stretch", key="bvh_types")
    vr = [r for r in common.runs_from(lr, "summarize", "kind", "discrepancy-verdict (Round B planted, PoC §9.2)") if "discrepancy_verdict" in r["outputs"]]
    if vr:
        st.markdown(f"**Round B (retrained without the arm; verdict run `{vr[0]['name']}`, {vr[0]['run_id'][:8]}).** Statuses and sentences are shown as recorded.")
        common.declared_caption([f"harm_{k}" for k in metrics], ctx["decls"])
        vt = common.table_of(lr, vr[0]["run_id"], "discrepancy_verdict")
        vv = vt[["arm", "metric", "status", "sentence", "calls", "claim_clean_but_harmful"]].copy()
        def arm_name(a: str) -> str:
            if a.startswith("type-"):
                return words.PLANTED.get(a.removeprefix("type-"), a) + " removed"
            return {"oracle": "all planted episodes removed (positive control)"}.get(a, "the most harmful clean episodes removed" if a.startswith("cleanharm") else words.plain(a))

        vv["arm"] = [arm_name(a) for a in vv["arm"]]
        vv["metric"] = [words.metric_name(x) for x in vv["metric"]]
        st.dataframe(vv.rename(columns={"calls": "calls by run", "claim_clean_but_harmful": "clean but harmful (claimed)"}), hide_index=True, width="stretch", key="bvh_verdict")
    else:
        st.caption("No Round B planted verdicts yet (`server_pipeline.py --round-b-planted`, then `discrepancy-verdict`).")
    st.markdown("**Example: the corrupted episode next to its clean original**")
    ty_pick = st.selectbox("Planted type", sorted(ex["types"]), key="bvh_type", format_func=lambda k: words.PLANTED.get(k, k))
    e = ex["types"][ty_pick]
    which = st.radio("Member", ["most_harmful", "least_harmful"], format_func=lambda k: e["by_metric"][m][k]["which"], horizontal=True, key="bvh_which")
    pick = e["by_metric"][m][which]
    ev = e["visibly_bad_evidence"]
    st.caption(f"{words.PLANTED.get(ty_pick, ty_pick)}: {e['hypothesis']}. " + (f"Visibly bad: the label-free heuristic {words.plain(ev['best_heuristic'])} separates it from clean data with AUROC {ev['auroc']:.3f}." if ev else ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"Corrupted ({words.plain(pick['episode_id'])[-50:]}, harm z = {pick['z']:.2f})")
        st.video(common.training_video(lr, ctx["run_id"], pick["episode_id"]), width=360)
    with c2:
        st.markdown("Clean original")
        orig = common.original_video(lr, ctx["run_id"], pick["original_episode_id"], pick["original_content_hash"])
        if orig is None:
            st.info("The clean original is not available in the parent version on this machine.")
        else:
            st.video(orig, width=360)
