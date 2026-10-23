"""Shared pieces of the dashboard: configuration, cached readers over ``vlapoc.results`` (the lineage store, never an ad-hoc path), the provenance banner and the
header every page shows (dashboard S5).

Grounding: PoC §2.7 (p. 5) the surface; §3.4 (p. 7) "Every attribution result states which functional it explains": every score column is shown under its
``axis_label`` (functional and tau from ``functionals.json``); brief.md (owner answer): "Synthetic scores, labelled ... SYNTHETIC everywhere they appear", and the
label comes from the lineage store's recorded provenance (``results.provenance``), never from a switch in this code.

Configuration: ``streamlit run dashboard/app.py -- --lineage-root <store> --scores <run name|id> [--clusters <run name|id>] [--mlflow-ui <url>]``; the environment
variables ``DASH_LINEAGE_ROOT``, ``DASH_SCORES``, ``DASH_CLUSTERS`` and ``DASH_MLFLOW_UI`` take precedence (the headless tests use them). Pointing at another store changes
every page with no edit.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from vlapoc import results as res  # noqa: E402
from vlapoc import video  # noqa: E402
from dashboard import words  # noqa: E402
from vlapoc.analysis import axis_label  # noqa: E402


@dataclass(frozen=True)
class Config:
    lineage_root: str  # "" = the real store data/lineage
    scores: str
    clusters: str
    mlflow_ui: str

    @property
    def root(self) -> Path | None:
        return Path(self.lineage_root) if self.lineage_root else None


def get_config() -> Config:
    """Command line (after ``--``) and environment (which wins) -> ``Config``."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--lineage-root", default="")
    ap.add_argument("--scores", default="")
    ap.add_argument("--clusters", default="")
    ap.add_argument("--mlflow-ui", default="http://localhost:5000")
    a, _ = ap.parse_known_args(sys.argv[1:])
    e = os.environ
    return Config(e.get("DASH_LINEAGE_ROOT", a.lineage_root), e.get("DASH_SCORES", a.scores), e.get("DASH_CLUSTERS", a.clusters), e.get("DASH_MLFLOW_UI", a.mlflow_ui))


# ---------------------------------------------------------------------------------------------
# cached readers (arguments are plain strings so Streamlit can hash them)
# ---------------------------------------------------------------------------------------------


def _r(lr: str) -> Path | None:
    return Path(lr) if lr else None


@st.cache_data(show_spinner=False, ttl=30)
def score_runs(lr: str) -> list[dict]:
    """Attribute and ensemble runs of the store with a score table, newest first."""
    return [r for r in res.find_runs("attribute", _r(lr)) if "attribution_scores" in r["outputs"]]


@st.cache_data(show_spinner=False, ttl=30)
def scores_of(lr: str, run_id: str):
    return res.score_table(run_id, _r(lr)), res.functionals(run_id, _r(lr))


@st.cache_data(show_spinner=False)
def catalog_of(lr: str, run_id: str):
    return res.training_episodes(run_id, _r(lr))


@st.cache_data(show_spinner=False, ttl=15)
def provenance_of(lr: str, run_id: str) -> dict:
    return res.provenance(run_id, _r(lr))


@st.cache_data(show_spinner=False, ttl=30)
def run_info(lr: str, run_id: str) -> dict:
    return res.run_summary(run_id, _r(lr))


@st.cache_data(show_spinner=False, ttl=30)
def runs_from(lr: str, stage: str, key: str, value: str) -> list[dict]:
    return res.runs_with_param(stage, key, value, _r(lr))


@st.cache_data(show_spinner=False)
def table_of(lr: str, run_id: str, name: str) -> pd.DataFrame:
    return res.output_table(run_id, name, _r(lr))


@st.cache_data(show_spinner=False)
def json_of(lr: str, run_id: str, name: str):
    return res.output_json(run_id, name, _r(lr))


@st.cache_data(show_spinner=False)
def influence_of(lr: str, run_id: str):
    return res.influence(run_id, _r(lr))


@st.cache_data(show_spinner=False)
def rollouts_of(lr: str, run_id: str) -> pd.DataFrame:
    return res.rollouts(run_id, _r(lr))


@st.cache_data(show_spinner=False)
def training_video(lr: str, run_id: str, episode_id: str) -> bytes:
    """mp4 of one training episode of the run's subset (disk cache keyed by the episode's content hash)."""
    sid, cat = catalog_of(lr, run_id)
    row = cat[cat["episode_id"] == episode_id].iloc[0]
    return video.episode_video(res.dataset_root(sid, _r(lr)), int(row["episode_index"]), row["content_hash"])


@st.cache_data(show_spinner=False)
def original_video(lr: str, run_id: str, original_episode_id: str, original_content_hash: str) -> bytes | None:
    """mp4 of the clean original (in the parent version) of a corrupted episode, or ``None`` if the parent version has no such episode."""
    sid, _ = catalog_of(lr, run_id)
    pv = res.parent_version(sid, _r(lr))
    if pv is None:
        return None
    root, tab = pv
    row = tab[(tab["episode_id"] == original_episode_id) & (tab["content_hash"] == original_content_hash)]
    return None if row.empty else video.episode_video(root, int(row.iloc[0]["episode_index"]), original_content_hash)


@st.cache_data(show_spinner=False)
def rollout_video(lr: str, eval_run_id: str, traj_episode_index: int) -> bytes | None:
    """mp4 of one recorded rollout, or ``None`` when its trajectories are not on this machine."""
    root = res.trajectory_root(eval_run_id, _r(lr))
    if root is None:
        return None
    return video.episode_video(root, int(traj_episode_index), f"{eval_run_id[:16]}-{int(traj_episode_index)}")


# ---------------------------------------------------------------------------------------------
# labels, banner, header
# ---------------------------------------------------------------------------------------------


def label(col: str, decls: dict) -> str:
    """The short, plain name of a score column (``Speed vs human (HRT)``). The declared functional is reachable next to it: ``declared(col, decls)`` in a caption or
    tooltip (PoC §3.4 p. 7, §4.4 p. 10); ``analysis.axis_label`` is the full text."""
    return words.metric_name(col)


def declared(col: str, decls: dict) -> str:
    """``Speed vs human (HRT): HRT(60) = ... [τ 60 s, all, good_sign +1]``: the short name and the declaration from ``functionals.json``."""
    return f"{words.metric_name(col)}: {axis_label(col, decls).split(': ', 1)[1]}".replace("good_sign ", "good sign ")


def declared_caption(cols: list[str], decls: dict) -> None:
    """One caption listing what each shown metric explains (the declared functional and τ), so no number is bare."""
    st.caption("What the scores explain: " + " | ".join(declared(c, decls) for c in cols))


def banner(prov: dict) -> None:
    """The provenance banner, one line (SYNTHETIC in red, SMOKE in amber, a quiet caption for REAL), with the recorded reasons in an expander below it
    (decided by ``results.provenance`` from the recorded runs; PoC §3.4 and the owner's "labelled everywhere")."""
    if prov["label"] == "SYNTHETIC":
        st.error("**SYNTHETIC scores:** a demonstration, not a result about any policy (videos, catalog and rollouts may be real).")
    elif prov["label"] == "SMOKE":
        st.warning("**SMOKE run:** real code on a toy-scale chain, not attribution-grade; do not read the numbers as results.")
    else:
        st.caption(f"Provenance: REAL ({prov['n_runs']} recorded runs checked: every train run is not smoke, every eval run is attribution-grade, nothing synthetic).")
    if prov["reasons"]:
        with st.expander("Why this label (recorded runs)"):
            for r in prov["reasons"]:
                st.caption(r)


PAGES = ("Overview", "Rollouts", "Episodes", "Clusters", "Bad vs harmful")


def header(cfg: Config) -> dict:
    """Title, run picker and provenance banner (every page). Returns the context the pages use: ``lr``, ``run_id``, ``run_name``, ``table``, ``decls``, ``cat``,
    ``subset_id``, ``prov``, ``cfg``."""
    runs = score_runs(cfg.lineage_root)
    if not runs:
        st.title("VLA training-data attribution")
        st.error(f"No attribute or ensemble run with a score table in the lineage store {cfg.lineage_root or 'data/lineage'}.")
        st.stop()
    names = {f"{r['name']}  ({r['run_id'][:8]})": r["run_id"] for r in runs}
    default = next((i for i, r in enumerate(runs) if cfg.scores in (r["name"], r["run_id"])), 0)
    with st.sidebar:
        st.header("Run")
        pick = st.selectbox("Score run (attribute or ensemble)", list(names), index=default, key="score_run")
    run_id = names[pick]
    st.title("VLA training-data attribution")
    prov = provenance_of(cfg.lineage_root, run_id)
    banner(prov)
    try:
        table, decls = scores_of(cfg.lineage_root, run_id)
    except res.NotRecordedError as e:  # a stand-in or pre-declaration run: nothing may be shown without its declared functional (PoC §3.4 p. 7)
        st.error(f"This run cannot be shown: {e}. Every number here must name the functional it explains, and this run declares none.")
        st.stop()
    sid, cat = catalog_of(cfg.lineage_root, run_id)
    return {"cfg": cfg, "lr": cfg.lineage_root, "run_id": run_id, "run_name": pick.split("  (")[0], "table": table, "decls": decls, "cat": cat, "subset_id": sid, "prov": prov}


def functional_note(ctx: dict, cols: list[str]) -> None:
    """Caption with the declared functional and τ of the columns in view (PoC §3.4 p. 7)."""
    declared_caption(cols, ctx["decls"])


def harm_cols(ctx: dict) -> list[str]:
    """The all-rollout harm columns of the table (not ``__same_task``, not failure types)."""
    return [c for c in ctx["table"].columns if c.startswith("harm_") and "__" not in c and not c.startswith("harm_ft_")]


def mlflow_run_url(ctx: dict, run_id: str) -> str:
    exp = run_info(ctx["lr"], run_id)["experiment_id"]
    return f"{ctx['cfg'].mlflow_ui.rstrip('/')}/#/experiments/{exp}/runs/{run_id}"


def trace_command(ctx: dict, run_id: str) -> str:
    return f".venv/bin/python -m vlapoc.lineage trace {run_id}" + (f" --lineage-root {ctx['cfg'].lineage_root}" if ctx["cfg"].lineage_root else "")
