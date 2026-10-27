"""Episodes page (PoC §2.7 view 2, §1 p. 3): a ranked list of harmful episodes, aggregated by any catalog column, with video."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import common, words
from vlapoc.analysis import aggregate_by
from vlapoc.catalog import CATALOG_COLUMNS
from vlapoc.lineage import GROUP_COLUMNS_EXCLUDED

LIST_VALUED = ("objects",)  # list-valued catalog columns cannot be grouped (aggregate_by would fail on an unhashable value)


def groupable() -> list[str]:
    """The catalog columns "aggregate by" offers: ``CATALOG_COLUMNS`` minus the per-episode identifiers (``lineage.GROUP_COLUMNS_EXCLUDED``) and list-valued columns."""
    return [c for c in CATALOG_COLUMNS if c not in GROUP_COLUMNS_EXCLUDED and c not in LIST_VALUED]


def renamed_aggregate(agg: pd.DataFrame, col: str, group: str) -> pd.DataFrame:
    """``aggregate_by``'s table with plain column names (the declared functional is in the caption above the table)."""
    lab = words.metric_name(col)
    return agg.rename(columns={f"{col}__mean": f"mean {lab}", f"{col}__lo": f"95% CI low: {lab}", f"{col}__hi": f"95% CI high: {lab}", group: words.plain(group)})[
        [words.plain(group), "n", f"mean {lab}", f"95% CI low: {lab}", f"95% CI high: {lab}"]]


def render(ctx: dict) -> None:
    """Draw the Episodes page: the ranked harmful episodes and the "aggregate by catalog column" table for the chosen run."""
    st.subheader("Episodes: ranked lists and aggregation by catalog column")
    t, decls, cat, lr = ctx["table"], ctx["decls"], ctx["cat"], ctx["lr"]
    cols = common.harm_cols(ctx)
    col = st.selectbox("Metric (positive score = harmful)", cols, format_func=lambda c: common.label(c, decls), key="ep_col",
                       help="; ".join(common.declared(c, decls) for c in cols))
    common.declared_caption([col], decls)
    lab = common.label(col, decls)
    n_show = st.slider("Episodes shown", 5, 100, 20, key="ep_n")
    extra = cat.set_index("episode_id")[["task_name", "scene", "source", "length_steps"]]
    base = t[["episode_id", col]].merge(extra, left_on="episode_id", right_index=True, how="left")
    for title, asc, key in (("Most harmful episodes", False, "harm"), ("Most helpful episodes", True, "help")):
        st.markdown(f"**{title}**")
        ranked = base.sort_values([col, "episode_id"], ascending=[asc, True]).head(n_show).rename(columns={col: lab}).reset_index(drop=True)
        ranked.insert(0, "rank", ranked.index + 1)
        st.dataframe(ranked.rename(columns={"episode_id": "episode", "task_name": "task", "length_steps": "length (steps)"}), hide_index=True, width="stretch", key=f"rank_{key}")
    st.markdown("**Aggregate by a catalog column** (PoC §1 p. 3: findings are usually per group of metadata)")
    group = st.selectbox("Aggregate by", groupable(), index=groupable().index("task_name"), key="ep_group", format_func=words.plain)
    agg = aggregate_by(t, group, [col], cat)
    st.dataframe(renamed_aggregate(agg, col, group), hide_index=True, width="stretch", key="agg_table")
    if cat[group].isna().all():
        st.caption(f"`{words.plain(group)}` is not recorded in this dataset's catalog (all values empty), so there is one group.")
    st.markdown("**Play an episode**")
    ranked = base.sort_values([col, "episode_id"], ascending=[False, True]).head(n_show)
    eid = st.selectbox("Episode (most harmful first)", list(ranked["episode_id"]), key="ep_play", format_func=words.plain)
    st.video(common.training_video(lr, ctx["run_id"], eid), width=420)
    row = cat[cat["episode_id"] == eid].iloc[0]
    st.caption(f"{words.plain(eid)}: {words.plain(row['task_name'])}; scene {words.plain(row['scene'])}; source file {words.plain(row['source_file'])}; {int(row['length_steps'])} steps")
