"""Clusters page (the owner's "pinpoint exact clusters and the patterns inside them"; PoC §1 p. 3, §4.2, §4.5, §9.2): heatmap and trade-off scatter, then the clusters of
``vlapoc.clusters`` with pattern, estimated effect, status and lineage. Status is ``estimated`` until a Round B ``confirm`` run says otherwise."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common, words
from vlapoc.analysis import GROUP_COLORS, GROUPS, metric_groups, plot_heatmap

DEFAULT_FIG = common.REPO_ROOT / "data" / "dashboard_cache" / "figs"
PLOT_HEIGHT = 560  # the plot area is measured in a real browser by verify_dashboard.py --render (>= 200 px)


@st.cache_data(show_spinner=False)
def heatmap_png(lr: str, run_id: str, cols: tuple[str, ...], synthetic: bool) -> bytes:
    t, decls = common.scores_of(lr, run_id)
    DEFAULT_FIG.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_FIG / f"{run_id}-heatmap-v2.png"
    plot_heatmap(t, list(cols), metric_groups(t, list(cols)), decls, path, synthetic, tick_labels={c: words.metric_name(c) for c in cols})
    return path.read_bytes()


def scatter(ctx: dict, cols: list[str]) -> alt.Chart:
    """Throughput harm (x) against reliability harm (y) per episode, coloured by the four groups, quadrants labelled (PoC §4.5 p. 11); clickable."""
    t = ctx["table"]
    x = "harm_rmst60" if "harm_rmst60" in cols else cols[0]
    y = "harm_f30" if "harm_f30" in cols else cols[-1]
    df = t[["episode_id", x, y]].copy()
    df["group"] = [words.GROUPS[g] for g in metric_groups(t, cols)]
    df = df.rename(columns={x: "throughput", y: "reliability"})
    pick = alt.selection_point(name="pick", fields=["episode_id"])
    syn = "SYNTHETIC - " if ctx["prov"]["label"] == "SYNTHETIC" else ""
    names = [words.GROUPS[g] for g in GROUPS]
    pts = alt.Chart(df).mark_circle(size=40, opacity=0.8).encode(
        x=alt.X("throughput:Q", title=f"Harm on {words.metric_name(x)}"), y=alt.Y("reliability:Q", title=f"Harm on {words.metric_name(y)}"),
        color=alt.Color("group:N", scale=alt.Scale(domain=names, range=[GROUP_COLORS[g] for g in GROUPS]), legend=alt.Legend(orient="bottom", title=None, columns=2)),
        opacity=alt.condition(pick, alt.value(0.9), alt.value(0.35)), tooltip=["episode_id", "group"]).add_params(pick)
    lo, hi = float(df[["throughput", "reliability"]].min().min()), float(df[["throughput", "reliability"]].max().max())
    q = [{"throughput": hi * 0.7, "reliability": hi * 0.9, "name": "consistent harmers"}, {"throughput": lo * 0.7, "reliability": lo * 0.9, "name": "consistent helpers"},
         {"throughput": lo * 0.7, "reliability": hi * 0.9, "name": "trade-off"}, {"throughput": hi * 0.7, "reliability": lo * 0.9, "name": "trade-off"}]
    txt = alt.Chart(alt.Data(values=q)).mark_text(fontSize=11, opacity=0.7).encode(x="throughput:Q", y="reliability:Q", text="name:N")
    return (pts + txt).properties(title=f"{syn}Each dot is a training episode (higher = more harmful)", height=PLOT_HEIGHT)


def status_of(ctx: dict, cluster_run: str) -> tuple[dict, pd.DataFrame | None, str | None]:
    """``({cluster_id: status text}, cluster_status table, confirm run id)`` from the newest ``confirm`` run of this cluster run, else all ``estimated``.
    The text is the recorded display text (it begins ``SYNTHETIC:`` for a synthetic run)."""
    runs = [r for r in common.runs_from(ctx["lr"], "summarize", "cluster_run_id", cluster_run) if "cluster_status" in r["outputs"]]
    if not runs:
        return {}, None, None
    prm = runs[0]["params"]
    return json.loads(prm.get("overall_display", prm.get("overall_vs_base", "{}"))), common.table_of(ctx["lr"], runs[0]["run_id"], "cluster_status"), runs[0]["run_id"]


def side_for(row: pd.Series, metric: str) -> str | None:
    """The direction (``harm`` / ``boost``) in which this slice was found for ``metric`` (its own target, or an equivalent definition's), else ``None``."""
    col = f"harm_{metric}" if not metric.startswith("harm_") else metric
    if row["lens"] != "slice":
        return row["direction"]
    if row["target_metric"] == col:
        return row["direction"]
    for e in json.loads(row.get("equivalent_definitions") or "[]"):
        if e["lens"] == "slice" and e["target_metric"] == col:
            return e["direction"]
    return None


def render(ctx: dict) -> None:
    st.subheader("Clusters: where in the data, and what is the pattern?")
    lr, t, decls = ctx["lr"], ctx["table"], ctx["decls"]
    cols = common.harm_cols(ctx)
    synthetic = ctx["prov"]["label"] == "SYNTHETIC"
    common.functional_note(ctx, cols)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Episode x metric heatmap** (red = more harmful than average, blue = more helpful)")
        st.image(heatmap_png(lr, ctx["run_id"], tuple(cols), synthetic), width="stretch")
    with c2:
        st.markdown("**Throughput vs reliability** (click an episode)")
        ev = st.altair_chart(scatter(ctx, cols), width="stretch", on_select="rerun", key="scatter")
        sel = (ev.selection.get("pick") or []) if ev is not None else []
        if sel:
            eid = sel[0]["episode_id"]
            st.caption(f"Selected episode {words.plain(eid)}")
            st.video(common.training_video(lr, ctx["run_id"], eid), width=420)
    runs = common.runs_from(lr, "cluster", "score_run_id", ctx["run_id"])
    cfg_pick = next((r for r in runs if ctx["cfg"].clusters in (r["name"], r["run_id"])), None)
    if not runs:
        st.info("No cluster run for this score run yet. Run `scripts/stage.py clusters --scores <run> --name <name>`.")
        return
    crun = cfg_pick or runs[0]
    prm = json.loads(crun["params"]["slice_finder"])
    st.markdown(f"**Cluster run** `{crun['name']}` ({crun['run_id'][:8]}). Every cluster is **estimated** (first-order sum of individual influence, CUPID p. 11) until a Round B `confirm` run marks it. "
                f"Slice minimum size: {prm['min_sup']} episodes (a planted corruption type has 13); bins are quintiles of about 192 episodes, so a slice isolates a type only through conjunctions of several conditions.")
    clu, pats = common.table_of(lr, crun["run_id"], "clusters"), {p["cluster_id"]: p for p in common.json_of(lr, crun["run_id"], "patterns")}
    overall, cstatus, confirm_id = status_of(ctx, crun["run_id"])
    lens = st.radio("Lens", ["slice", "label", "metric_group"], format_func=words.LENS.get, horizontal=True, key="lens")
    metric = st.selectbox("Metric", cols, format_func=lambda c: common.label(c, decls), key="cl_metric")
    key = metric.removeprefix("harm_")
    sub = clu[clu["lens"] == lens].copy()
    if lens == "slice":
        only_match = st.checkbox("Only slices whose estimated reading on this metric matches their direction", value=True, key="only_match",
                                 help="SliceLine also returns large near-population slices; this keeps the ones whose interval excludes the subset mean in the stated direction.")
        sub["side"] = [side_for(r, key) for _, r in sub.iterrows()]
        sub = sub[sub["side"].notna()]
        if only_match:
            sub = sub[sub[f"{metric}__reading"] == sub["side"].map({"harm": "harmful", "boost": "helpful"})]
        sub = sub.sort_values(["side", "rank"])
    elif lens == "label":
        sub = sub.sort_values(f"harm_rank__{metric}")
    lab = common.label(metric, decls)
    show = pd.DataFrame({"cluster": [words.plain(d) for d in sub["definition"]], "effect on the metric": [words.DIRECTION.get(d, d) for d in (sub["side"] if "side" in sub else sub["direction"])],
                         "episodes": sub["n"].astype(int), "share of data": sub["share_of_subset"].round(3), f"mean harm: {lab}": sub[f"{metric}__mean"],
                         "95% CI low": sub[f"{metric}__lo"], "95% CI high": sub[f"{metric}__hi"], "reading": sub[f"{metric}__reading"],
                         "status": [overall.get(c, "estimated") for c in sub["cluster_id"]], "same episodes as": [json.loads(e or "[]") and f"{len(json.loads(e))} other definition(s)" or "" for e in sub["equivalent_definitions"]]})
    st.caption(f"{len(show)} clusters. Status is `estimated` unless a confirm run exists for the cluster; the effect is the members' mean harm with a percentile-bootstrap interval, uncorrected for metrics and clusters. Clusters with identical members are listed once.")
    st.dataframe(show, hide_index=True, width="stretch", key="cluster_table")
    if show.empty:
        return
    ids = list(sub["cluster_id"])
    cid = st.selectbox("Cluster", ids, key="cl_pick", format_func=lambda i: words.plain(sub.set_index("cluster_id").loc[i, "definition"]))
    p = pats[cid]
    badge = overall.get(cid, "estimated")
    st.markdown(f"**{words.plain(p['definition'])}**  status: `{badge}`")
    st.write(words.plain(p["text"]))
    eq = json.loads(sub.set_index("cluster_id").loc[cid, "equivalent_definitions"] or "[]")
    if eq:
        st.caption("The same episodes are also described as: " + "; ".join(words.plain(e["definition"]) for e in eq))
    if confirm_id and cid in overall:
        cs = cstatus[cstatus["cluster_id"] == cid][["comparison", "metric", "expected", "status_display", "calls_by_seed", "met", "contradicted"]].copy()
        cs["expected"] = cs["expected"].map({"treated_better": "removal should help", "treated_worse": "removal should hurt", "-": "none stated"})
        cs["metric"] = [words.metric_name(m) for m in cs["metric"]]
        cs["comparison"] = cs["comparison"].map({"vs_base": "against the original policy", "vs_matched_random": "against a matched random removal"})
        st.markdown(f"Round B verdicts (confirm run `{confirm_id[:8]}`; rule: confirmed if the expectation is met in at least 2 of 3 seeds and contradicted in none; T = removal helped, C = it hurt, 0 = no significant change, g = the macro-KS gate did not fire):")
        st.dataframe(cs.rename(columns={"status_display": "status", "calls_by_seed": "calls by seed", "met": "seeds as expected", "contradicted": "seeds contradicting"}), hide_index=True, width="stretch", key="cl_status")
    d = pd.DataFrame(p["descriptor_contrasts"]).head(6)
    if len(d):
        d["descriptor"] = [words.plain(x) for x in d["descriptor"]]
        st.markdown("Descriptors that differ most (inside vs outside, difference in s.d. of the whole subset)")
        st.dataframe(d.rename(columns={"mean_in": "mean inside", "mean_out": "mean outside", "std_diff": "difference (s.d.)"}), hide_index=True, width="stretch", key="cl_contrast")
    st.caption("Metrics reported for this cluster: " + " | ".join(common.declared(c, decls) for c in p["declarations"]))
    mem = common.table_of(lr, crun["run_id"], "members")
    ex = mem[mem["cluster_id"] == cid].head(3)
    st.markdown("Most influential member episodes")
    cols3 = st.columns(3)
    for col_, (_, r) in zip(cols3, ex.iterrows()):
        with col_:
            st.video(common.training_video(lr, ctx["run_id"], r["episode_id"]), width=300)
            st.caption(f"{words.plain(r['episode_id'])[-60:]}: rank {int(r['rank'])}")
    with st.expander("Lineage of this cluster (ids and commands)"):
        st.markdown(f"[cluster run in the MLflow UI]({common.mlflow_run_url(ctx, crun['run_id'])}); cluster id `{cid}`")
        st.code(common.trace_command(ctx, crun["run_id"]), language="bash")
    rec = [r for r in common.runs_from(lr, "score", "cluster_run_id", crun["run_id"]) if "cluster_recovery" in r["outputs"]]
    if rec:
        st.markdown("#### ground-truth check (answer key read after discovery)")
        tb = common.table_of(lr, rec[0]["run_id"], "cluster_recovery")
        one = tb[(tb["cluster_id"] == cid) & (tb["planted_type"] != "any_planted")][["planted_type", "n_cluster", "n_overlap", "precision", "recall", "lift", "p_fisher"]].copy()
        one["planted_type"] = [words.PLANTED.get(x, x) for x in one["planted_type"]]
        st.caption(f"Overlap of this cluster with each planted corruption type (precision = share of the cluster that is of the type; lift over the base rate; one-sided Fisher p). "
                   f"Slice minimum size {prm['min_sup']}; a planted type has 13 episodes.")
        st.dataframe(one.rename(columns={"planted_type": "planted type", "n_cluster": "cluster size", "n_overlap": "overlap", "p_fisher": "Fisher p"}).sort_values("Fisher p"),
                     hide_index=True, width="stretch", key="cl_recovery")
