"""The live local dashboard (dashboard S5): PoC §2.7 (p. 5) "select a failed rollout, see its failure type, the most influential training episodes (with video) and
their source datasets; or browse a ranked list of harmful episodes, aggregated by any catalog column", plus the cluster views (heatmap, scatter, slices with a pattern,
estimated vs confirmed) and the §9.2 "bad vs harmful" page.

Launch (from the repo root)::

    .venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens

Everything is read from the lineage store through ``vlapoc.results`` (MLflow runs and their sha-checked outputs); a SYNTHETIC or SMOKE banner is shown whenever the
recorded provenance of the chosen run says so. See ``dashboard/common.py`` for the options.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="VLA attribution dashboard", layout="wide")

from dashboard import common  # noqa: E402
from dashboard.views import bad_vs_harmful, clusters, episodes, overview, rollouts  # noqa: E402

RENDER = {"Overview": overview.render, "Rollouts": rollouts.render, "Episodes": episodes.render, "Clusters": clusters.render, "Bad vs harmful": bad_vs_harmful.render}

cfg = common.get_config()
ctx = common.header(cfg)
with st.sidebar:
    page = st.radio("Page", common.PAGES, key="page")
RENDER[page](ctx)
