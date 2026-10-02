"""Evaluation-layer S2 (b): build PhAIL's "mixed" cohort with PhAIL's OWN loader, export it, and compute
PhAIL's reference numbers with PhAIL's OWN functions (the other side of the AC2.1 equivalence check).

Runs in the PhAIL analysis env, not in ``.venv`` (it imports the clone's ``build`` package)::

    PHAIL_DATA_ROOT=$PWD/data/phail/dataset .venv-phail/bin/python scripts/phail_export_cohort.py

Grounding: PoC §3.5 B (p. 8) and §3.1 (p. 6) (PhAIL's published rollouts and "analysis pipeline published
on GitHub"); PhAIL §3 pp. 4-6. ``.venv-phail`` = Python 3.12 + the versions pinned in the clone's
``uv.lock`` for the packages these modules import (pos3 0.2.2, configuronic 0.4.0, numpy 2.4.4,
scipy 1.17.1, matplotlib 3.10.8, pyarrow 23.0.1); ``positronic`` is not needed (only named in
comments/strings of ``common.py``/``loader.py``; grep).

Writes ``data/phail/cohort_mixed.parquet`` (one row per episode: what ``episode_to_T_pairs`` consumes,
plus model/object/cohort source) and ``artifacts/s2_phail_reference.json``: per model and per (model,
object) cell, PhAIL's functions' values for F(20/30/60 s), median, RMST(240 s), %-of-human, KS distances,
censored AUC (``explore_pp_plot._auc_censored`` / ``_macro_auc``), the macro-KS p exactly as
``build/dashboard.py`` H1 computes it (``default_rng(42)``, pairs in ``combinations(POLICY_MODELS, 2)``
order, n_boot 2000; ``dashboard.py`` itself cannot be imported: it needs ``build/fig_cdf.py``, which is
not in the repo), pooled logrank (``explore_separation._logrank``) and one P-P curve (``_pp_curve``).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONE = REPO_ROOT / "data" / "external" / "phail-paper"
sys.path.insert(0, str(CLONE))
os.environ.setdefault("PHAIL_DATA_ROOT", str(REPO_ROOT / "data" / "phail" / "dataset"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build import explore_pp_plot as pp  # noqa: E402
from build import explore_separation as sep  # noqa: E402
from build import stats, tab_leaderboard  # noqa: E402
from build.markup import loader  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
TAU = 240.0
POLICY_MODELS = ("Physical Intelligence Open π₀.₅", "NVIDIA GR00T N1.6", "Action Chunking Transformer", "Hugging Face SmolVLA")
TELEOP = "Robot teleoperated by Human"
SHORT = {"Physical Intelligence Open π₀.₅": "OpenPI", "NVIDIA GR00T N1.6": "GR00T", "Action Chunking Transformer": "ACT",
         "Hugging Face SmolVLA": "SmolVLA", TELEOP: "Teleop"}


def by_obj(eps):
    out = defaultdict(list)
    for e in eps:
        out[e.eval_object].append(e)
    return out


def main() -> int:
    eps = loader.load(cohort="mixed")
    rows = [{"s3_url": e.s3_url, "cohort_source": e.cohort_source, "place_ts": [float(t) for t in e.place_ts],
             "duration_s": float(e.duration_s), "total_items": int(e.total_items), "n_inf_events": int(e.n_inf_events),
             "n_drops_outside": int(e.n_drops_outside), "had_safety_stop": bool(e.had_safety_stop),
             "eval_object": e.eval_object, "eval_outcome": e.eval_outcome, "model_display": e.model_display,
             "model_short": SHORT.get(e.model_display, e.model_display), "successful_items_eval": int(e.successful_items_eval)}
            for e in eps]
    df = pd.DataFrame(rows)
    out_pq = REPO_ROOT / "data" / "phail" / "cohort_mixed.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"exported {len(df)} episodes -> {out_pq.relative_to(REPO_ROOT)}")
    print(df.groupby(["model_short", "cohort_source"]).size().unstack(fill_value=0).to_string())

    by_model = defaultdict(list)
    for e in eps:
        by_model[e.model_display].append(e)
    ref: dict = {"cohort": "mixed", "tau": TAU, "n_episodes": len(eps),
                 "counts": {SHORT.get(m, m): len(v) for m, v in by_model.items()},
                 "counts_by_object": {SHORT.get(m, m): {o: len(v) for o, v in sorted(by_obj(es).items())} for m, es in by_model.items()}}
    # tab_leaderboard (pooled per model), with its human RMST anchor
    T_h, E_h = sep._pairs(by_model[TELEOP])
    rmst_h = stats.rmst(*stats.kaplan_meier(T_h, E_h)[:2], TAU)
    ref["rmst_human_pooled"] = rmst_h
    ref["leaderboard"] = {SHORT[m]: tab_leaderboard._per_model_row(by_model[m], TAU, rmst_h)
                          for m in (*POLICY_MODELS, TELEOP) if m in by_model}
    # per (model, object) cell with PhAIL's functions
    tele_o = by_obj(by_model[TELEOP])
    cells = {}
    for m in (*POLICY_MODELS, TELEOP):
        for o, ce in sorted(by_obj(by_model[m]).items()):
            T, E = sep._pairs(ce)
            tg, S, _ = stats.kaplan_meier(T, E)
            med = stats.km_median(tg, S)
            r = {"n_eps": len(ce), "rmst240": stats.rmst(tg, S, TAU), "median": float("inf") if med is None else med,
                 **{f"F{th}": float(1.0 - stats.km_step(tg, S, np.array([float(th)]))[0]) for th in (20, 30, 60)}}
            if m != TELEOP and o in tele_o:
                Tt, Et = sep._pairs(tele_o[o])
                r["auc_vs_teleop"] = pp._auc_censored(T, E, Tt, Et)
                r["hrt_pct"] = stats.pct_of_human(r["rmst240"], stats.rmst(*stats.kaplan_meier(Tt, Et)[:2], TAU))
            cells[f"{SHORT[m]}|{o}"] = r
    ref["cells"] = cells
    ref["macro_auc"] = {SHORT[m]: pp._macro_auc(by_obj(by_model[m]), tele_o) for m in POLICY_MODELS}
    # KS distances per object for every model pair
    ref["ks_cells"] = {}
    for a, b in combinations(POLICY_MODELS, 2):
        oa, ob = by_obj(by_model[a]), by_obj(by_model[b])
        for o in sorted(set(oa) & set(ob)):
            ta, Sa, _ = stats.kaplan_meier(*sep._pairs(oa[o]))
            tb, Sb, _ = stats.kaplan_meier(*sep._pairs(ob[o]))
            ref["ks_cells"][f"{SHORT[a]}-{SHORT[b]}|{o}"] = stats.ks_distance(ta, Sa, tb, Sb)
    # macro-KS exactly as dashboard.py H1: one default_rng(42), pairs in combinations order, n_boot 2000
    rng = np.random.default_rng(42)
    ref["macro_ks_dashboard_h1"] = {}
    for a, b in combinations(POLICY_MODELS, 2):
        D, p = stats.per_obj_macro_bootstrap_ks(by_obj(by_model[a]), by_obj(by_model[b]), n_boot=2000, rng=rng)
        ref["macro_ks_dashboard_h1"][f"{SHORT[a]}-{SHORT[b]}"] = {"D": D, "p": p}
        print(f"macro-KS (PhAIL per_obj_macro_bootstrap_ks, dashboard H1 order) {SHORT[a]}-{SHORT[b]}: D={D:.6f} p={p:.6f}")
    # pooled logrank (explore_separation._logrank)
    ref["logrank_pooled"] = {f"{SHORT[a]}-{SHORT[b]}": dict(zip(("chi2", "p"), sep._logrank(by_model[a], by_model[b])))
                             for a, b in combinations(POLICY_MODELS, 2)}
    # one P-P curve: ACT on Batteries (PhAIL Fig. 2's cell)
    act_b = by_obj(by_model["Action Chunking Transformer"])["Batteries"]
    x, y = pp._pp_curve(*sep._pairs(act_b), *sep._pairs(tele_o["Batteries"]), TAU)
    ref["pp_ACT_Batteries"] = {"F_teleop": x.tolist(), "F_model": y.tolist()}
    xq, yq = pp._macro_pp_curve(by_obj(by_model["Action Chunking Transformer"]), tele_o, TAU)
    ref["macro_pp_ACT"] = {"q": xq.tolist(), "F_model": yq.tolist()}
    (ART / "s2_phail_reference.json").write_text(json.dumps(ref, indent=1, default=float))
    print(f"wrote {(ART / 's2_phail_reference.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
