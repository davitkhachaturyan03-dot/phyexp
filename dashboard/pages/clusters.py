"""Clusters page (the owner's "pinpoint exact clusters and the patterns inside them"; PoC §1 p. 3, §4.2, §4.5, §9.2): heatmap and trade-off scatter, then the clusters of
``vlapoc.clusters`` with pattern, estimated effect, status and lineage. Status is ``estimated`` until a Round B ``confirm`` run says otherwise."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common
from vlapoc import results as res
from vlapoc.analysis import GROUP_COLORS, GROUPS, metric_groups, plot_heatmap

DEFAULT_FIG = common.REPO_ROOT / "data" / "dashboard_cache" / "figs"


@st.cache_data(show_spinner=False)
def heatmap_png(lr: str, run_id: str, cols: tuple[str, ...], synthetic: bool) -> bytes:
    t, decls = common.scores_of(lr, run_id)
    DEFAULT_FIG.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_FIG / f"{run_id}-heatmap.png"
    plot_heatmap(t, list(cols), metric_groups(t, list(cols)), decls, path, synthetic)
    return path.read_bytes()


def scatter(ctx: dict, cols: list[str]) -> alt.Chart:
    """Throughput influence (x) against reliability influence (y) per episode, coloured by the four groups, quadrants labelled (PoC §4.5 p. 11); clickable."""
    t, decls = ctx["table"], ctx["decls"]
    x = "harm_rmst60" if "harm_rmst60" in cols else cols[0]
    y = "harm_f30" if "harm_f30" in cols else cols[-1]
    df = t[["episode_id", x, y]].copy()
    df["group"] = metric_groups(t, cols).to_numpy()
    xl, yl = common.label(x, decls), common.label(y, decls)
    df = df.rename(columns={x: "throughput", y: "reliability"})
    pick = alt.selection_point(name="pick", fields=["episode_id"])
    syn = "SYNTHETIC DATA - " if ctx["prov"]["label"] == "SYNTHETIC" else ""
    pts = alt.Chart(df).mark_circle(size=40, opacity=0.8).encode(
        x=alt.X("throughput:Q", title=xl + " (throughput)"), y=alt.Y("reliability:Q", title=yl + " (reliability)"),
        color=alt.Color("group:N", scale=alt.Scale(domain=list(GROUPS), range=[GROUP_COLORS[g] for g in GROUPS])),
        opacity=alt.condition(pick, alt.value(0.9), alt.value(0.35)), tooltip=["episode_id", "group"]).add_params(pick)
    lo, hi = float(df[["throughput", "reliability"]].min().min()), float(df[["throughput", "reliability"]].max().max())
    q = pd.DataFrame({"throughput": [hi * 0.7, lo * 0.7, lo * 0.7, hi * 0.7], "reliability": [hi * 0.9, lo * 0.9, hi * 0.9, lo * 0.9],
                      "name": ["consistent harmers", "consistent helpers", "trade-off", "trade-off"]})
    txt = alt.Chart(q).mark_text(fontSize=11, opacity=0.7).encode(x="throughput:Q", y="reliability:Q", text="name:N")
    return (pts + txt).properties(title=f"{syn}Throughput vs reliability influence, positive = harmful (quadrants labelled; group labels are descriptive, unverified)", height=380)


def status_of(ctx: dict, cluster_run: str) -> tuple[dict, pd.DataFrame | None, str | None]:
    """``({cluster_id: status}, cluster_status table, confirm run id)`` from the newest ``confirm`` run of this cluster run, else all ``estimated``."""
    runs = [r for r in common.runs_from(ctx["lr"], "summarize", "cluster_run_id", cluster_run) if "cluster_status" in r["outputs"]]
    if not runs:
        return {}, None, None
    rid = runs[0]["run_id"]
    return json.loads(runs[0]["params"].get("overall_vs_base", "{}")), common.table_of(ctx["lr"], rid, "cluster_status"), rid


def render(ctx: dict) -> None:
    st.subheader("Clusters: where in the data, and what is the pattern?")
    lr, t, decls = ctx["lr"], ctx["table"], ctx["decls"]
    cols = common.harm_cols(ctx)
    synthetic = ctx["prov"]["label"] == "SYNTHETIC"
    common.functional_note(ctx, cols)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Episode x metric heatmap** (z-scored harm; red = harmful)")
        st.image(heatmap_png(lr, ctx["run_id"], tuple(cols), synthetic), width="stretch")
    with c2:
        st.markdown("**Throughput vs reliability** (click an episode)")
        ev = st.altair_chart(scatter(ctx, cols), width="stretch", on_select="rerun", key="scatter")
        sel = (ev.selection.get("pick") or []) if ev is not None else []
        if sel:
            eid = sel[0]["episode_id"]
            st.caption(f"Selected episode {eid}")
            st.video(common.training_video(lr, ctx["run_id"], eid), width=420)
    runs = common.runs_from(lr, "cluster", "score_run_id", ctx["run_id"])
    cfg_pick = next((r for r in runs if ctx["cfg"].clusters in (r["name"], r["run_id"])), None)
    if not runs:
        st.info("No cluster run for this score run yet. Run `scripts/stage.py clusters --scores <run> --name <name>`.")
        return
    crun = (cfg_pick or runs[0])
    st.markdown(f"**Cluster run** `{crun['name']}` ({crun['run_id'][:8]}). Every cluster is **estimated** (first-order sum of individual influence, CUPID p. 11) until a Round B `confirm` run marks it.")
    clu, pats = common.table_of(lr, crun["run_id"], "clusters"), {p["cluster_id"]: p for p in common.json_of(lr, crun["run_id"], "patterns")}
    overall, cstatus, confirm_id = status_of(ctx, crun["run_id"])
    lens = st.radio("Lens", ["slice", "label", "metric_group"], format_func={"slice": "Rule-like slices", "label": "Label groups", "metric_group": "Four metric groups"}.get, horizontal=True, key="lens")
    sub = clu[clu["lens"] == lens].copy()
    metric = st.selectbox("Metric", cols, format_func=lambda c: common.label(c, decls), key="cl_metric")
    if lens == "slice":
        only_match = st.checkbox("Only slices whose estimated reading on their target metric matches their direction", value=True, key="only_match",
                                 help="SliceLine also returns large near-population slices; this keeps the ones whose interval excludes the subset mean in the stated direction.")
        sub = sub[sub["target_metric"] == metric]
        if only_match:
            want = sub["direction"].map({"harm": "harmful", "boost": "helpful"})
            sub = sub[sub[f"{metric}__reading"] == want]
        sub = sub.sort_values(["direction", "rank"])
    elif lens == "label":
        sub = sub.sort_values(f"harm_rank__{metric}")
    lab = common.label(metric, decls)
    show = pd.DataFrame({"cluster": sub["cluster_id"], "definition": sub["definition"], "direction": sub["direction"], "n": sub["n"].astype(int), "share of subset": sub["share_of_subset"].round(3),
                         f"mean {lab}": sub[f"{metric}__mean"], f"95% CI low: {lab}": sub[f"{metric}__lo"], f"95% CI high: {lab}": sub[f"{metric}__hi"],
                         "reading": sub[f"{metric}__reading"], "status": [overall.get(c, "estimated") for c in sub["cluster_id"]]})
    st.caption(f"{len(show)} clusters. Status `estimated` unless a confirm run exists for the cluster; the effect is the members' mean with a percentile-bootstrap interval, uncorrected for metrics and clusters.")
    st.dataframe(show, hide_index=True, width="stretch", key="cluster_table")
    if show.empty:
        return
    cid = st.selectbox("Cluster", list(show["cluster"]), key="cl_pick")
    p = pats[cid]
    badge = overall.get(cid, "estimated")
    st.markdown(f"**{cid}**  status: `{badge}`")
    st.write(p["text"])
    if confirm_id and cid in overall:
        cs = cstatus[cstatus["cluster_id"] == cid][["comparison", "metric", "expected", "status", "calls_by_seed", "met", "contradicted", "verify_run_ids"]]
        st.markdown(f"Round B verdicts (confirm run `{confirm_id[:8]}`; rule: confirmed if the expectation is met in at least 2 of 3 seeds and contradicted in none):")
        st.dataframe(cs, hide_index=True, width="stretch", key="cl_status")
    d = pd.DataFrame(p["descriptor_contrasts"]).head(6)
    if len(d):
        st.markdown("Descriptors that differ most (inside vs outside, in s.d. of the whole subset)")
        st.dataframe(d, hide_index=True, width="stretch", key="cl_contrast")
    st.markdown("Declarations of the metrics reported: " + "; ".join(f"`{c}` explains {v['functional']} [tau {v['tau']}, {v['rollout_subset']}]" for c, v in p["declarations"].items()))
    mem = common.table_of(lr, crun["run_id"], "members")
    ex = mem[mem["cluster_id"] == cid].head(3)
    st.markdown("Most influential member episodes")
    cols3 = st.columns(3)
    for col_, (_, r) in zip(cols3, ex.iterrows()):
        with col_:
            st.video(common.training_video(lr, ctx["run_id"], r["episode_id"]), width=420)
            st.caption(f"{r['episode_id']}: rank {int(r['rank'])}")
    st.markdown("**Lineage.** " + f"[cluster run in the MLflow UI]({common.mlflow_run_url(ctx, crun['run_id'])})")
    st.code(common.trace_command(ctx, crun["run_id"]), language="bash")
    rec = [r for r in common.runs_from(lr, "score", "cluster_run_id", crun["run_id"]) if "cluster_recovery" in r["outputs"]]
    if rec:
        st.markdown("#### ground-truth check (answer key read after discovery)")
        tb = common.table_of(lr, rec[0]["run_id"], "cluster_recovery")
        one = tb[(tb["cluster_id"] == cid) & (tb["planted_type"] != "any_planted")][["planted_type", "n_cluster", "n_overlap", "precision", "recall", "lift", "p_fisher"]]
        st.caption("Overlap of this cluster with each planted corruption type (precision = share of the cluster that is of the type; lift over the base rate; one-sided Fisher p).")
        st.dataframe(one.sort_values("p_fisher"), hide_index=True, width="stretch", key="cl_recovery")
