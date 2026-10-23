"""Overview page: what the numbers are, how the provenance banner was decided, the declared functionals and the lineage of the chosen run."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import common


def render(ctx: dict) -> None:
    st.subheader("Overview")
    prov, t = ctx["prov"], ctx["table"]
    st.markdown(
        "This dashboard shows which **training episodes** helped or hurt a policy, **per metric** (PoC §4.2): success within 30 s, HRT, RMST, AUC, timeouts. "
        "Per-episode influence comes from the CUPID estimator on the policy's rollouts; it is an *estimate*. A cluster of episodes is called **confirmed** only after "
        "removing it, retraining and passing the two-step verdict (macro-KS gate, then per-metric intervals); until then it is **estimated**.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Provenance", prov["label"])
    c2.metric("Training episodes", len(t))
    c3.metric("Score columns", len([c for c in t.columns if c.startswith(("harm_", "infl_"))]))
    c4.metric("Recorded runs checked", prov["n_runs"])
    st.markdown("**How the banner is decided.** From the recorded runs behind the score table (`vlapoc.results.provenance`), fail-closed: *SYNTHETIC* if any run is tagged synthetic; "
                "*REAL* only if no run is a stand-in, every train run is tagged `vlapoc.smoke=false` and every eval run `eval.attribution_grade=true`; *SMOKE* otherwise. "
                "A missing tag counts against REAL. The label cannot be switched off in the app.")
    if prov["reasons"]:
        st.dataframe(pd.DataFrame({"reason this run is not REAL": prov["reasons"]}), hide_index=True, width="stretch")
    st.markdown("**Declared functionals** (PoC §3.4: every attribution result states which functional it explains)")
    rows = [{"column": c, "functional": d["functional"], "tau": d["tau"], "rollout subset": d["rollout_subset"], "good sign": d["good_sign"], "weighting": d.get("weighting", "")}
            for c, d in ctx["decls"].items() if c.startswith("harm_") and "__" not in c]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    info = common.run_info(ctx["lr"], ctx["run_id"])
    st.markdown("**Lineage of this run**")
    st.code(common.trace_command(ctx, ctx["run_id"]), language="bash")
    st.markdown(f"[Open the run in the MLflow UI]({common.mlflow_run_url(ctx, ctx['run_id'])}) (start it with `mlflow ui --backend-store-uri sqlite:///<lineage root>/mlflow.db`)")
    st.caption(f"Run `{info['name']}` stage `{info['stage']}`; training subset `{ctx['subset_id'][:12]}`.")
    st.markdown("**Pages.** Rollouts: pick a failed rollout, see its failure type and the training episodes that pushed it. Episodes: ranked harmful and helpful lists, aggregated by any catalog column. "
                "Clusters: label groups, rule-like slices and the four metric groups, each with a pattern, an estimated effect and a status. Bad vs harmful: where episodes that look bad do not hurt "
                "(planted-data versions only).")
