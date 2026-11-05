"""Curation results page (the owner's "add the curation headline results to the dashboard"; PoC §4.5 p. 11, §2.6 p. 5, §9.1 Step 4 p. 19, §4.4 p. 10, §3.4 p. 7).

What it shows, from the Round A ``step4-summary`` run of the chosen score run (curation-results S4):

1. the PoC §4.5 headline, "same dataset, opposite curations": did the policy retrained without the episodes ranked most harmful for speed (HRT) win on speed, and the one
   retrained without those ranked most harmful for reliability (F(30 s)) win on reliability? The status and sentence are the recorded ones, shown verbatim; "inconclusive"
   where the macro-KS gate fired in fewer than ceil(2n/3) seeds; "not shown" where it fired in at least ceil(2n/3) seeds but neither curated policy's win on its own metric
   replicated; a forest plot per seed oriented so that "wins its own metric" is positive;
2. every curated arm judged against the original policy and against random removal (§2.6 "more than random removal"), per metric, with the recorded status, the calls per seed
   (T/C/0/g) and the sentence;
3. an operation-level (family O) section, recorded alongside (vs the original policy only, outside the headline), or a caption saying none is recorded;
4. a per-arm forest plot, the episodes each arm removed (read from the child subset, with a training video), the lineage (MLflow links, trace command), and the declared
   functionals: what each arm was ranked by and what each verdict judges, with the point that the HRT arm was ranked by the linearized return and judged on the ratio (§4.3 p. 10).

The words are not written here: they come from ``vlapoc.step4`` (``annotate`` / ``headline``), recorded in the summary run. A summary run recorded before those columns existed is
annotated at display time by the same functions from the recorded per-seed calls, and the page says so. A SYNTHETIC input makes every recorded status begin ``SYNTHETIC:``.
Nothing is corrected across arms or seeds: the across-seed rule is a replication rule, and the Holm correction is across the 5 metrics inside each verdict only.
"""

from __future__ import annotations

import re

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import common, words
from vlapoc import results as res
from vlapoc import step4
from vlapoc.metrics_registry import entry, verdict_key_of

PLOT_HEIGHT_MIN = 220  # the plot area is measured in a real browser by verify_dashboard.py --render (>= 200 px)
COMPARISONS = {"vs_base": 0, "curated_vs_random": 1, "head_to_head": 2}
CALL_WORDS = {"T": "better", "C": "worse", "0": "not shown", "g": "gate not fired"}


def arm_label(row: dict) -> str:
    """Plain-language name of an arm (``5 most harmful per task for Success within 30 s removed``)."""
    arm, k = str(row["arm"]), row.get("k")
    n = f"{int(float(k))}" if k not in (None, "") and str(k) != "nan" else "the top"
    op = str(row.get("op", ""))
    if op.startswith("drop_top_k"):
        name = words.metric_name(str(row.get("metric_column", "")))
        who = "most harmful" if row.get("direction") == "highest" else "most helpful"
        return f"{n} {who}{' per task' if op.endswith('per_task') else ''} for {name} removed"
    m = re.fullmatch(r"(g?)random(\d+)-[kn](\d+)", arm)
    if m:
        return f"{m.group(3)} random{' per task' if not m.group(1) else ''} removed (draw {int(m.group(2)) + 1})"
    if arm.startswith("drop-random-"):
        return f"{arm.removeprefix('drop-random-')} random episodes removed (overall)"
    return words.plain(arm)


def control_label(row: dict) -> str:
    """Plain-language name of what a verdict compares the arm with."""
    c, kind = str(row["control"]), row["comparison"]
    if kind == "vs_base":
        return "the original policy"
    if kind == "head_to_head":
        return "the policy curated for reliability (Success within 30 s)"
    m = re.fullmatch(r"(g?)random(\d+)-[kn]\d+", c)
    return f"random removal (draw {int(m.group(2)) + 1})" if m else f"random removal ({words.plain(c)})"


def metric_caption() -> str:
    """What each verdict metric explains: ``<plain name>: <functional> [τ ...]`` for the five family-W metrics (PoC §3.4 p. 7, §4.4 p. 10), so no number is bare."""
    parts = []
    for key in ("f30", "rmst60", "auc_h", "hrt60", "censored60"):
        e = entry(key)
        tau = str(e["tau"])
        parts.append(f"{words.metric_name(key)}: {e['functional']} [τ {'none' if tau.startswith('none') else tau.split(' =')[0]}]")
    return "What each verdict judges (family W, whole-task time): " + " | ".join(parts)


def ranking_caption(sel_params: dict) -> str | None:
    """What an arm was ranked by: the declared functional recorded in its ``select`` run (``metric_declaration``), or ``None`` for a random arm."""
    import json

    d = sel_params.get("metric_declaration")
    if not d or not d.startswith("{"):
        return None
    decl = json.loads(d)
    return (f"{words.metric_name(decl['column'])}: {decl['functional']} [τ {decl['tau']}, rollouts: {words.plain(decl.get('rollout_subset', 'all'))}, "
            f"good sign {decl.get('good_sign', '')}]").replace("good sign +", "good sign ")


def forest(df: pd.DataFrame, title: str, xtitle: str) -> alt.Chart:
    """A forest plot (estimate and 95% interval per row, a dashed line at 0, gate-not-fired rows marked): Altair ``mark_rule`` on pre-computed ``lo`` / ``hi`` (the ``mark_errorbar`` composite was measured drawing a 22 px plot in the browser)
    plus ``mark_point``, a rule at 0; no new dependency."""
    base = alt.Chart(df).encode(y=alt.Y("row:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)))
    color = alt.Color("gate:N", scale=alt.Scale(domain=["fired", "did not fire"], range=["#1d6fa5", "#9aa0a6"]), legend=alt.Legend(orient="bottom", title="macro-KS gate"))
    bars = base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title=xtitle), x2="hi:Q", color=color)
    pts = base.mark_point(filled=True, size=80, opacity=1).encode(x="est:Q", color=color, shape=alt.Shape("gate:N", scale=alt.Scale(domain=["fired", "did not fire"], range=["circle", "cross"]), legend=None),
                                                                tooltip=["row", "est", "lo", "hi", "gate"])
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 3], color="#555").encode(x="x:Q")
    return (zero + bars + pts).properties(title=title, height=alt.Step(max(34, -(-PLOT_HEIGHT_MIN // max(len(df), 1)))))


def row_keys(row: dict) -> list[str]:
    """The metric keys a summary row actually holds (``<m>__by_seed`` is a text): the recorded JSON gives every row every column, empty (NaN) for the other family's metrics."""
    return [c[:-len("__by_seed")] for c, v in row.items() if c.endswith("__by_seed") and isinstance(v, str) and v]


def verdict_index(records: list[dict]) -> dict:
    """``{(kind, treated, control, seed): verdict.json}`` of the family-W Step 4 verdicts (the newest run when a comparison was recorded twice)."""
    out: dict = {}
    for r in records:
        if r["kind"] and r.get("family", "W") == "W" and r["seed"] is not None:
            out.setdefault((r["kind"], r["treated"], r["control"], int(r["seed"])), r["verdict"])
    return out


def render(ctx: dict) -> None:
    """Draw the Curation results page for the chosen score run (module docstring)."""
    st.subheader("Curation results: does removing the data the ranking called harmful improve the policy?")
    lr = ctx["lr"]
    cur = common.curation_of(lr, ctx["run_id"])
    if cur is None:
        st.info("No curation results (Round A retraining) are recorded for this score run. They come from the removal tests of the server chain: "
                "`.venv/bin/python scripts/server_pipeline.py --tag <tag> --subset v3/candidate --curation per_task --metrics f30 hrt60_lin --from select` (see the server-run runbook), "
                "which ends in the `step4-summary` run this page reads.")
        return
    sj, summ, sel = cur["summary_json"], cur["summary"], cur["selects"]
    prov = common.provenance_of(lr, summ["run_id"])
    common.banner(prov, condense=True)
    syn = prov["label"] == "SYNTHETIC"
    prefix = "SYNTHETIC - " if syn else ""
    all_rows = [r for r in sj["rows"] if r.get("op") in res.CURATION_OPS or r["comparison"] == "head_to_head"]
    rows = [r for r in all_rows if r.get("family", "W") == "W"]
    orows = [r for r in all_rows if r.get("family", "W") == "O"]  # family O (operation level): recorded alongside, only vs the original policy; never in the headline, the W table or the plots
    keys = row_keys(rows[0])
    computed = any(f"{m}__status" not in r for r in rows for m in row_keys(r))
    if computed:  # a summary recorded before the status columns existed: the same function, from the recorded per-seed calls
        for r in rows:
            r.update(step4.annotate(r, row_keys(r), syn))
        for r in orows:
            r.update(step4.annotate(r, row_keys(r), syn))
    heads = sj.get("headline") if not computed and "headline" in sj else step4.headline(rows, syn)
    st.caption(f"Summary run `{summ['name']}` ({summ['run_id'][:8]}), chain `{cur['chain']}`: {len(rows)} comparisons over {max(int(r['n_seeds']) for r in rows)} training seeds. "
               + ("The statuses and sentences were computed at display time from the recorded per-seed calls (this run predates the recorded columns); otherwise they are shown as recorded. "
                  if computed else "Statuses and sentences are shown as recorded, never re-worded here. ")
               + "Every verdict is the two-step test: the macro-KS gate (did the time-to-success distribution change at all?), then each metric's bootstrap interval for the direction, Holm-corrected across the 5 metrics.")
    verd = verdict_index(common.curation_verdicts(lr, cur["chain"]))
    common.functional_note_text(metric_caption())

    # ---- 1. the headline ---------------------------------------------------------------------------------------------------
    st.markdown("#### Same dataset, opposite curations (the PoC's headline result)")
    if not heads:
        st.info("This chain recorded no head-to-head comparison between a speed-curated and a reliability-curated arm (needs both `--metrics f30 hrt60_lin`).")
    for h in heads:
        st.markdown(f"**{h['status']}**")
        st.markdown(h["sentence"])
        tab = pd.DataFrame([{"seed": p["seed"], "speed (HRT-curated vs reliability-curated)": p["speed"], "reliability (Success within 30 s)": p["reliability"]} for p in h["per_seed"]])
        st.dataframe(tab, hide_index=True, width="stretch", key=f"head_seeds_{h['treated']}")
        hv = [(p["seed"], verd.get(("head_to_head", h["treated"], h["control"], int(p["seed"])))) for p in h["per_seed"]]
        frows = []
        for seed, v in hv:
            if v is None:
                continue
            gate = "fired" if v["gate"]["fires"] else "did not fire"
            a, lo, hi = step4.treated_advantage(v["metrics"]["hrt60"])  # treated = the HRT-curated arm
            frows.append({"row": f"seed {seed}: speed-curated policy, advantage on Speed vs human (HRT)", "est": a, "lo": lo, "hi": hi, "gate": gate})
            a, lo, hi = step4.treated_advantage(v["metrics"]["f30"])  # the F30 arm is the control: flip the sign so "wins its own metric" is positive
            frows.append({"row": f"seed {seed}: reliability-curated policy, advantage on Success within 30 s", "est": -a, "lo": -hi, "hi": -lo, "gate": gate})
        if frows:
            st.altair_chart(forest(pd.DataFrame(frows), f"{prefix}Does each curated policy win its own metric? (positive = yes; 95% interval)",
                                   "Advantage of the curated policy over the other curated policy, in the metric's own units"), width="stretch")
            st.caption("A cross marks a seed whose macro-KS gate did not fire: the two policies were not shown to differ at all, so that seed claims no direction. "
                       "The speed-curated arm was ranked by the linearized HRT return (a first-order approximation, since HRT is a ratio, PoC §4.3) and is judged here on the HRT ratio itself.")

    # ---- 2. the arm table ------------------------------------------------------------------------------------------------
    st.markdown("#### Every curated arm, judged against the original policy and against random removal")
    first = next((verdict_key_of(r["metric_column"]) for r in rows if r.get("metric_column")), keys[0])
    m = st.selectbox("Metric", keys, index=keys.index(first) if first in keys else 0, format_func=words.metric_name, key="cur_metric",
                     help="; ".join(f"{words.metric_name(k)}: {entry(k)['functional']} [τ {str(entry(k)['tau']).split(' =')[0]}]" for k in keys))
    ordered = sorted(rows, key=lambda r: (words.metric_name(m) not in {words.metric_name(r.get("metric_column", ""))}, COMPARISONS.get(r["comparison"], 9), r["arm"], r["control"]))
    show = pd.DataFrame({"arm": [arm_label(r) for r in ordered], "compared with": [control_label(r) for r in ordered], "status": [r[f"{m}__status"] for r in ordered],
                         "calls by seed": [", ".join(CALL_WORDS[c] for c in str(r[f"{m}__by_seed"]).split(",")) for r in ordered], "gate fired": [r["gate_fires"] for r in ordered],
                         "sentence": [r[f"{m}__sentence"] for r in ordered]})
    st.caption(f"Metric shown: {words.metric_name(m)}. Arms ranked by this metric come first. Calls by seed: better = the curated policy beat the comparison, worse = the comparison beat it, "
               "not shown = the difference is not shown at this sample size, gate not fired = the macro-KS gate did not fire so no direction is claimed. "
               "A random-removal arm has no expectation; it is the control. Nothing is corrected across arms or seeds: the status is a replication rule "
               "(the expectation met in at least 2 of 3 seeds and contradicted in none), not a further test.")
    st.dataframe(show, hide_index=True, width="stretch", key="cur_table")
    cur_arms = sorted({r["arm"] for r in rows if str(r.get("op", "")).startswith("drop_top_k")})
    sel_arm = st.session_state.get("cur_arm") if st.session_state.get("cur_arm") in cur_arms else (cur_arms[0] if cur_arms else None)
    if sel_arm:  # the table cannot wrap long text, so the selected arm's recorded sentences are also given as text
        st.markdown(f"**Sentences for the arm selected under \"One arm in detail\" ({words.metric_name(m)}):**")
        for r in sorted((x for x in rows if x["arm"] == sel_arm), key=lambda x: (COMPARISONS.get(x["comparison"], 9), x["control"])):
            st.markdown(f"- *{control_label(r)}* ({r[f'{m}__status']}): {r[f'{m}__sentence']}")

    # ---- 2b. family O (operation level), alongside --------------------------------------------------------------------------
    st.markdown("#### Operation-level check (family O), alongside")
    if not orows:
        st.caption("No operation-level (family O) verdicts are recorded for this chain.")
    else:
        okeys = row_keys(orows[0])
        orec = [(r, k) for r in sorted(orows, key=lambda r: (r["arm"], r["control"])) for k in okeys]
        otab = pd.DataFrame({"arm": [arm_label(r) for r, _ in orec], "metric": [entry(k)["metric"] for _, k in orec], "compared with": [control_label(r) for r, _ in orec],
                             "status": [r[f"{k}__status"] for r, k in orec], "calls by seed": [", ".join(CALL_WORDS[c] for c in str(r[f"{k}__by_seed"]).split(",")) for r, k in orec],
                             "gate fired": [r["gate_fires"] for r, _ in orec], "sentence": [r[f"{k}__sentence"] for r, k in orec]})
        st.caption("Operation-level verdicts: " + " | ".join(f"{entry(k)['metric']}: {entry(k)['functional']} [τ {str(entry(k)['tau']).split(' =')[0]}]" for k in okeys)
                   + ". The arms were ranked by a family-W return; family W (whole-task time) is the declared verdict family and carries the headline; family O is recorded alongside, "
                   "only against the original policy, and is not part of the headline.")
        st.dataframe(otab, hide_index=True, width="stretch", key="cur_table_o")

    # ---- 3. per-arm detail -----------------------------------------------------------------------------------------------
    curated = [r for r in rows if str(r.get("op", "")).startswith("drop_top_k")]
    arms = sorted({r["arm"] for r in curated})
    if arms:
        st.markdown("#### One arm in detail")
        pick = st.selectbox("Arm", arms, format_func=lambda a: arm_label(next(r for r in curated if r["arm"] == a)), key="cur_arm")
        prow = next(r for r in curated if r["arm"] == pick)
        sp = sel.get(pick, {}).get("params", {})
        rc = ranking_caption(sp)
        if rc:
            st.caption(f"This arm was ranked by: {rc}")
        drows = []
        for r in sorted((x for x in rows if x["arm"] == pick and x["comparison"] in ("vs_base", "curated_vs_random")), key=lambda r: (COMPARISONS[r["comparison"]], r["control"])):
            for seed in str(r["seeds"]).split(","):
                v = verd.get((r["comparison"], pick, r["control"], int(seed)))
                if v is None:
                    continue
                a, lo, hi = step4.treated_advantage(v["metrics"][m])
                drows.append({"row": f"seed {seed}: against {control_label(r)}", "est": a, "lo": lo, "hi": hi, "gate": "fired" if v["gate"]["fires"] else "did not fire"})
        if drows:
            st.altair_chart(forest(pd.DataFrame(drows), f"{prefix}{arm_label(prow)}: advantage on {words.metric_name(m)} (positive = the curated policy is better; 95% interval)",
                                   "Advantage of the curated policy, in the metric's own units"), width="stretch")

        # ---- 4. removed episodes -----------------------------------------------------------------------------------------
        st.markdown("#### Which training episodes this arm removed")
        sr = sel.get(pick)
        if sr is None:
            st.caption("The select run of this arm is not in the store.")
        else:
            rem = common.removed_of(lr, sr["run_id"])
            per_task = rem.groupby(["suite", "libero_task_id"]).size()
            total = int(sp.get("child_n_episodes", 0)) + len(rem)
            span = f"{per_task.min()} in each of {len(per_task)} tasks" if per_task.min() == per_task.max() else f"{per_task.min()} to {per_task.max()} in each of {len(per_task)} tasks"
            st.markdown(f"**{arm_label(prow)}**: {len(rem)} of {total} episodes removed ({span}).")
            tbl = pd.DataFrame({"episode": [words.plain(e.split("/")[-1]) for e in rem["episode_id"]], "task": [words.plain(t) for t in rem["task_name"]]})
            if "rank" in rem:
                tbl["rank in its task (1 = removed first)"], tbl["score (higher = more harmful)" if prow.get("direction") == "highest" else "score (lower = more helpful)"] = rem["rank"].to_numpy(), rem["score"].to_numpy()
            st.dataframe(tbl, hide_index=True, width="stretch", key="cur_removed")
            vid = st.selectbox("Watch a removed episode", list(rem["episode_id"]), format_func=lambda e: words.plain(e.split("/")[-2])[:40] + " / " + words.plain(e.split("/")[-1]), key="cur_video")
            try:
                st.video(common.training_video(lr, ctx["run_id"], vid), width=360)
            except Exception as e:  # noqa: BLE001 - the dataset may not be on this machine
                st.caption(f"Video not available on this machine ({type(e).__name__}).")

        # ---- 5. lineage --------------------------------------------------------------------------------------------------
        with st.expander("Lineage of these results (ids and commands)"):
            st.markdown(f"[summary run in the MLflow UI]({common.mlflow_run_url(ctx, summ['run_id'])})" + (f"; [select run of this arm]({common.mlflow_run_url(ctx, sr['run_id'])})" if sr else ""))
            links = [f"[{r['name']}]({common.mlflow_run_url(ctx, r['run_id'])})" for r in common.curation_verdicts(lr, cur["chain"]) if r["treated"] == pick and r.get("family", "W") == "W"]
            if links:
                st.markdown("Verdict runs of this arm: " + ", ".join(links))
            st.code(common.trace_command(ctx, summ["run_id"]), language="bash")
            if sr:
                st.code(common.trace_command(ctx, sr["run_id"]), language="bash")
