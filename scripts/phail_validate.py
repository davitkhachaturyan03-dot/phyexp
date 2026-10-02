"""Evaluation-layer S2 (c): our metric layer on PhAIL's released real-robot cohort, vs PhAIL's own functions
(same input) and vs the paper's published numbers.

Grounding: PoC §3.1/§3.3 pp. 6-7 (published evidence: "Four principled scalars on the same data produced
three different top-1 models"; the closest pair "stayed unresolved"), §3.5 B and Practical caveats p. 8,
§4.1 p. 9 (the four yardsticks F(30 s), HRT, global RMST, AUC vs human); PhAIL §3 pp. 4-6, §5 pp. 7-10,
Table 2 p. 7, App. L. Input: ``data/phail/cohort_mixed.parquet`` and ``artifacts/s2_phail_reference.json``
from ``scripts/phail_export_cohort.py`` (run first, in ``.venv-phail``).

    .venv/bin/python scripts/phail_validate.py      # ~3-6 min (bootstraps); writes artifacts/phail_validation.json + figures
    .venv/bin/python scripts/phail_validate.py --hrt-ci-seeds 8   # AC2.5 Monte-Carlo spread of both HRT CI variants

AC2.1 implementation equivalence (≤ 1e-9, macro-KS p identical); AC2.2 cohort counts; AC2.3 published
numbers; AC2.4 four yardsticks; AC2.5 HRT CI variant; AC2.6 data access; AC2.7 figures; AC2.8 UPH forms.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import evaluation as ev  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
CLONE = REPO_ROOT / "data" / "external" / "phail-paper"
TAU = 240.0
MODELS = ("OpenPI", "GR00T", "ACT", "SmolVLA")  # = PhAIL's POLICY_MODELS order
TELEOP = "Teleop"
PUBLISHED = {
    "n_eps": {"OpenPI": 165, "GR00T": 165, "ACT": 151, "SmolVLA": 118, "Teleop": 396},  # Table 2 / tab_leaderboard
    "F30": {"GR00T": 0.51, "OpenPI": 0.47, "ACT": 0.44},  # paper §5 p.7 (tab_leaderboard rounds to 2 d.p.)
    "macro_auc": {"ACT": (0.134, 0.108, 0.162), "OpenPI": (0.100, 0.084, 0.116), "GR00T": (0.095, 0.080, 0.110),
                  "SmolVLA": (0.027, 0.016, 0.036)},  # paper §5.3 p.9 (pp_summary.json has full digits)
    "macro_ks_OpenPI_GR00T_p": 0.12,  # paper §5.1; dashboard_per_obj_macro.json 0.11944027986006997
}
OUT: dict = {}


def load_arms() -> dict:
    df = pd.read_parquet(REPO_ROOT / "data" / "phail" / "cohort_mixed.parquet")
    arms: dict = {}
    for r in df.itertuples(index=False):
        ep = ev.episode_from_ops(r.s3_url, list(r.place_ts), r.duration_s, r.total_items, r.n_inf_events,
                                 cohort_source=r.cohort_source)
        arms.setdefault(r.model_short, {}).setdefault(r.eval_object, []).append(ep)
    return {m: {o: arms[m][o] for o in sorted(arms[m])} for m in arms}


def maxdiff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        return float(np.max(np.where(both_inf, 0.0, np.abs(a - b)))) if a.size else 0.0


def ac2_1(arms: dict, ref: dict) -> None:
    diffs = {}
    # pooled per model (tab_leaderboard._per_model_row)
    h_pooled = ev.pooled(arms[TELEOP])
    rmst_h = ev.rmst(h_pooled, TAU)
    lb = ref["leaderboard"]
    d = []
    for m in (*MODELS, TELEOP):
        eps = ev.pooled(arms[m])
        r = lb[m]
        med = r["median_T"] if r["median_T"] is not None else float("inf")
        d.append(maxdiff([ev.rmst(eps, TAU), ev.median_T(eps), *(ev.F_at(eps, t) for t in (20, 30, 60)),
                          100 * rmst_h / ev.rmst(eps, TAU)],
                         [r["rmst_tau"], med, *(r["F_at"][str(t)] for t in (20, 30, 60)), r["pct_of_human"]]))
    diffs["pooled_per_model (rmst240, median, F20/30/60, pct_of_human)"] = max(d)
    # per cell
    d, d_auc, d_hrt = [], [], []
    for key, r in ref["cells"].items():
        m, o = key.split("|")
        eps = arms[m][o]
        d.append(maxdiff([ev.rmst(eps, TAU), ev.median_T(eps), *(ev.F_at(eps, t) for t in (20, 30, 60))],
                         [r["rmst240"], r["median"], r["F20"], r["F30"], r["F60"]]))
        if "auc_vs_teleop" in r:
            d_auc.append(abs(ev.auc(eps, arms[TELEOP][o]) - r["auc_vs_teleop"]))
            d_hrt.append(abs(100 * ev.hrt_cells({o: eps}, {o: arms[TELEOP][o]}, TAU)[o] - r["hrt_pct"]))
    diffs["cells (rmst240, median, F20/30/60)"] = max(d)
    diffs["cells censored AUC vs teleop"] = max(d_auc)
    diffs["cells HRT %"] = max(d_hrt)
    diffs["macro AUC"] = max(abs(ev.macro_auc(arms[m], arms[TELEOP]) - ref["macro_auc"][m]) for m in MODELS)
    d = []
    for key, v in ref["ks_cells"].items():
        pair, o = key.split("|")
        a, b = pair.split("-")
        d.append(abs(ev.ks(arms[a][o], arms[b][o]) - v))
    diffs["KS distance per cell (24 = 6 pairs x 4 objects)"] = max(d)
    # P-P curves
    x, y = ev.pp_curve(arms["ACT"]["Batteries"], arms[TELEOP]["Batteries"], TAU)
    diffs["P-P curve ACT|Batteries"] = max(maxdiff(x, ref["pp_ACT_Batteries"]["F_teleop"]), maxdiff(y, ref["pp_ACT_Batteries"]["F_model"]))
    q, yq = ev.macro_pp_curve(arms["ACT"], arms[TELEOP], TAU)
    diffs["macro P-P curve ACT"] = max(maxdiff(q, ref["macro_pp_ACT"]["q"]), maxdiff(yq, ref["macro_pp_ACT"]["F_model"]))
    # pooled logrank: ours with a single stratum == explore_separation._logrank
    d = []
    for a, b in combinations(MODELS, 2):
        lr = ev.logrank_stratified({"all": ev.pooled(arms[a])}, {"all": ev.pooled(arms[b])})
        d.append(max(abs(lr["chi2"] - ref["logrank_pooled"][f"{a}-{b}"]["chi2"]), abs(lr["p"] - ref["logrank_pooled"][f"{a}-{b}"]["p"])))
    diffs["pooled logrank chi2/p (one stratum)"] = max(d)
    # macro-KS, dashboard H1 order with default_rng(42), n_boot 2000
    rng = np.random.default_rng(42)
    mk = {}
    for a, b in combinations(MODELS, 2):
        r = ev.macro_ks(arms[a], arms[b], 2000, rng)
        mk[f"{a}-{b}"] = {"D": r["D"], "p": r["p"], "D_cells": r["D_cells"]}
    identical = all(mk[k]["D"] == v["D"] and mk[k]["p"] == v["p"] for k, v in ref["macro_ks_dashboard_h1"].items())
    for k in mk:
        print(f"AC2.1 macro-KS {k}: ours D={mk[k]['D']:.6f} p={mk[k]['p']:.6f} | PhAIL D={ref['macro_ks_dashboard_h1'][k]['D']:.6f} "
              f"p={ref['macro_ks_dashboard_h1'][k]['p']:.6f}")
    for k, v in diffs.items():
        print(f"AC2.1 max |ours - PhAIL| {k}: {v:.3e}")
    print(f"AC2.1 macro-KS D and p identical for all 6 pairs (same RNG): {identical}")
    OUT["AC2.1"] = {"max_abs_diff": diffs, "macro_ks_ours": mk, "macro_ks_identical": identical,
                    "pass": identical and all(v <= 1e-9 for v in diffs.values())}
    OUT["_macro_ks"] = mk


def ac2_2(arms: dict, ref: dict) -> None:
    counts = {m: sum(len(v) for v in arms[m].values()) for m in (*MODELS, TELEOP)}
    cand_private = set()
    with open(CLONE / "out" / "release_classifier" / "candidates.jsonl") as f:
        for line in f:
            u = json.loads(line).get("s3_dir", "")
            if u.startswith("s3://inference/"):
                cand_private.add(u)
    ann_private = sorted(p for p in (CLONE / "build/markup/annotations/inference").rglob("*.json"))
    rows = {m: {"exported": counts[m], "table2": PUBLISHED["n_eps"][m], "missing": PUBLISHED["n_eps"][m] - counts[m]} for m in counts}
    print("AC2.2 cohort counts (exported mixed cohort vs Table 2): " +
          ", ".join(f"{m} {r['exported']}/{r['table2']}" for m, r in rows.items()))
    print(f"AC2.2 private-bucket references: {len(cand_private)} s3://inference/ episode dirs in candidates.jsonl, "
          f"{len(ann_private)} annotation JSONs under annotations/inference/ "
          f"({sorted({str(p.relative_to(CLONE / 'build/markup/annotations')).split('/')[1] for p in ann_private})})")
    by_src = {m: {} for m in counts}
    for m in counts:
        for eps in arms[m].values():
            for e in eps:
                by_src[m][e.meta["cohort_source"]] = by_src[m].get(e.meta["cohort_source"], 0) + 1
    print(f"AC2.2 by cohort source: {by_src}")
    OUT["AC2.2"] = {"rows": rows, "by_source": by_src, "complete": all(r["missing"] == 0 for r in rows.values()),
                    "private_candidate_dirs": len(cand_private), "private_annotations": len(ann_private),
                    "counts_by_object": ref["counts_by_object"]}


def _ci(arms_, fn, seed=0, n_boot=1000, **kw):
    r = ev.cluster_bootstrap(arms_, fn, n_boot, np.random.default_rng(seed), **kw)
    return [r["point"], r["lo"], r["hi"]]


def ac2_3_4_5(arms: dict) -> None:
    pub_ci = {r["short"]: r for r in json.loads((CLONE / "out/paper/tab_headline_ci.json").read_text())}
    pp_sum = json.loads((CLONE / "out/explore/pp_summary.json").read_text())
    lb = json.loads((CLONE / "out/paper/tab_leaderboard.json").read_text())["rows"]
    res = {}
    for m in (*MODELS, TELEOP):
        rm = _ci({"m": arms[m]}, lambda a: ev.rmst(ev.pooled(a["m"]), TAU))
        res.setdefault(m, {})["rmst"] = rm
        res[m]["F30"] = _ci({"m": arms[m]}, lambda a: ev.F_at(ev.pooled(a["m"]), 30.0))
        if m != TELEOP:
            res[m]["hrt_both"] = [100 * x for x in _ci({"model": arms[m], "human": arms[TELEOP]},
                                                        lambda a: ev.hrt(a["model"], a["human"], TAU))]
            res[m]["hrt_fixed"] = [100 * x for x in _ci({"model": arms[m], "human": arms[TELEOP]},
                                                         lambda a: ev.hrt(a["model"], a["human"], TAU), fixed=("human",))]
            res[m]["macro_auc"] = _ci({"model": arms[m], "ref": arms[TELEOP]}, lambda a: ev.macro_auc(a["model"], a["ref"]))
    # AC2.3 side by side
    lines = []
    width = lambda c: c[2] - c[1]  # noqa: E731
    checks = {}
    for m in (*MODELS, TELEOP):
        p = pub_ci[m]
        o = res[m]
        dr = o["rmst"][0] - p["rmst"]
        rw = width(o["rmst"]) / (p["rmst_ci"][1] - p["rmst_ci"][0]) - 1
        line = f"{m:8s} RMST ours {o['rmst'][0]:.3f} [{o['rmst'][1]:.2f}, {o['rmst'][2]:.2f}] vs pub {p['rmst']:.3f} [{p['rmst_ci'][0]:.2f}, {p['rmst_ci'][1]:.2f}] (Δ {dr:+.4f} s, CI width {rw:+.1%})"
        checks[f"{m} RMST point ±0.1 s"] = abs(dr) <= 0.1
        checks[f"{m} RMST CI width ±10%"] = abs(rw) <= 0.10
        if m != TELEOP:
            dh = o["hrt_both"][0] - p["hrt"]
            hw_b = width(o["hrt_both"]) / (p["hrt_ci"][1] - p["hrt_ci"][0]) - 1
            hw_f = width(o["hrt_fixed"]) / (p["hrt_ci"][1] - p["hrt_ci"][0]) - 1
            line += (f"\n{'':8s} HRT  ours {o['hrt_both'][0]:.3f}% [{o['hrt_both'][1]:.2f}, {o['hrt_both'][2]:.2f}] (human resampled; "
                     f"width {hw_b:+.1%}) / [{o['hrt_fixed'][1]:.2f}, {o['hrt_fixed'][2]:.2f}] (human fixed; width {hw_f:+.1%}) "
                     f"vs pub {p['hrt']:.3f}% [{p['hrt_ci'][0]:.2f}, {p['hrt_ci'][1]:.2f}] (Δ {dh:+.4f} pp)")
            checks[f"{m} HRT point ±0.1 pp"] = abs(dh) <= 0.1
            pa = pp_sum["macro"][m]
            da = o["macro_auc"][0] - pa["auc"]
            aw = width(o["macro_auc"]) / (pa["auc_hi"] - pa["auc_lo"]) - 1
            line += (f"\n{'':8s} macro-AUC ours {o['macro_auc'][0]:.4f} [{o['macro_auc'][1]:.3f}, {o['macro_auc'][2]:.3f}] vs pub "
                     f"{PUBLISHED['macro_auc'][m][0]:.3f} [{PUBLISHED['macro_auc'][m][1]:.3f}, {PUBLISHED['macro_auc'][m][2]:.3f}] "
                     f"(pp_summary {pa['auc']:.6f}; Δ {da:+.2e}; CI width vs pp_summary {aw:+.1%})")
            checks[f"{m} macro-AUC ±0.005"] = abs(da) <= 0.005
            checks[f"{m} macro-AUC CI width ±10%"] = abs(aw) <= 0.10
            f30_pub = lb[m]["F_at"]["30"]
            line += f"\n{'':8s} F(30 s) ours {o['F30'][0]:.4f} [{o['F30'][1]:.3f}, {o['F30'][2]:.3f}] vs tab_leaderboard {f30_pub:.4f}"
            if m in PUBLISHED["F30"]:
                line += f" / paper {PUBLISHED['F30'][m]:.2f}"
                checks[f"{m} F(30) ±0.005 vs paper"] = abs(o["F30"][0] - PUBLISHED["F30"][m]) <= 0.005
        lines.append(line)
    mk = OUT["_macro_ks"]["OpenPI-GR00T"]
    pub_ks = json.loads((CLONE / "out/explore/ks_power/dashboard_per_obj_macro.json").read_text())["rows"][0]
    checks["OpenPI-GR00T macro-KS p ±0.03"] = abs(mk["p"] - PUBLISHED["macro_ks_OpenPI_GR00T_p"]) <= 0.03
    lines.append(f"macro-KS OpenPI-GR00T: ours D={mk['D']:.6f} p={mk['p']:.6f} vs dashboard_per_obj_macro.json D={pub_ks['D_macro_per_obj']:.6f} "
                 f"p={pub_ks['p']:.6f} / paper p = 0.12")
    for line in lines:
        print("AC2.3 " + line)
    print(f"AC2.3 tolerance checks: {sum(checks.values())}/{len(checks)} within; failing: {[k for k, v in checks.items() if not v]}")
    OUT["AC2.3"] = {"ours": res, "checks": checks, "published_ci": pub_ci}
    # AC2.4 four yardsticks
    yard = {
        "F(30 s)": ({m: res[m]["F30"] for m in MODELS}, max),
        "HRT macro (%)": ({m: res[m]["hrt_both"] for m in MODELS}, max),
        "global RMST(240 s) (s)": ({m: res[m]["rmst"] for m in MODELS}, min),
        "AUC vs human (macro)": ({m: res[m]["macro_auc"] for m in MODELS}, max),
    }
    table = []
    winners = {}
    for name, (vals, best) in yard.items():
        w = best(vals, key=lambda m: vals[m][0])
        winners[name] = w
        table.append({"yardstick": name, **{m: vals[m] for m in MODELS}, "winner": w})
        print(f"AC2.4 {name:24s} " + "  ".join(f"{m} {vals[m][0]:.3f} [{vals[m][1]:.3f}, {vals[m][2]:.3f}]" for m in MODELS) + f"  -> winner {w}")
    expected = {"F(30 s)": "GR00T", "HRT macro (%)": "OpenPI", "global RMST(240 s) (s)": "GR00T", "AUC vs human (macro)": "ACT"}
    n_distinct = len(set(winners.values()))
    print(f"AC2.4 winners {winners}; published GR00T/OpenPI/GR00T/ACT reproduced: {winners == expected}; distinct top-1: {n_distinct}")
    OUT["AC2.4"] = {"table": table, "winners": winners, "reproduced": winners == expected, "n_distinct_top1": n_distinct}
    # AC2.5 HRT CI variant: which matches tab_headline_ci.json
    err = {"human_resampled": [], "human_fixed": []}
    for m in MODELS:
        p = pub_ci[m]["hrt_ci"]
        err["human_resampled"].append(abs(res[m]["hrt_both"][1] - p[0]) + abs(res[m]["hrt_both"][2] - p[1]))
        err["human_fixed"].append(abs(res[m]["hrt_fixed"][1] - p[0]) + abs(res[m]["hrt_fixed"][2] - p[1]))
    tot = {k: float(np.sum(v)) for k, v in err.items()}
    adopted = min(tot, key=tot.get)
    print(f"AC2.5 HRT CI endpoint error vs tab_headline_ci.json, summed over 4 models (pp): {tot}; closer variant: {adopted}")
    OUT["AC2.5"] = {"endpoint_abs_err_pp": {k: [float(x) for x in v] for k, v in err.items()}, "sum": tot, "closer": adopted}


def ac2_6() -> None:
    listing = json.loads((ART / "s2_phail_listing_summary.json").read_text())
    probe = json.loads((ART / "s2_phail_access_probe.json").read_text())
    other = json.loads((ART / "s2_phail_other_prefixes.json").read_text())
    cro = json.loads((CLONE / "dataset/croissant.json").read_text())
    n352 = json.dumps(cro).count("352-episode") + json.dumps(cro).count("(352 episodes)")
    table = [
        {"part": "evaluation rollouts (phail/v1.0/dataset/rollouts/)", "episodes": listing["by_dir"]["rollouts"]["episodes"],
         "GB": round(listing["by_dir"]["rollouts"]["bytes"] / 1e9, 2), "access": "public, anonymous HTTPS", "license": cro["license"]},
        {"part": "teleoperation = fine-tuning set (phail/v1.0/dataset/teleoperation/)", "episodes": listing["by_dir"]["teleoperation"]["episodes"],
         "GB": round(listing["by_dir"]["teleoperation"]["bytes"] / 1e9, 2), "access": "public, anonymous HTTPS (not downloaded: not approved)",
         "license": cro["license"]},
        {"part": "same set at datasets/phail/v1.0/training/", "episodes": other["datasets/phail/v1.0/training/"]["episodes_static_json"],
         "GB": round(other["datasets/phail/v1.0/training/"]["bytes"] / 1e9, 2), "access": "public", "license": cro["license"]},
        {"part": "human (phail/v1.0/dataset/human/)", "episodes": listing["by_dir"]["human"]["episodes"],
         "GB": round(listing["by_dir"]["human"]["bytes"] / 1e9, 2), "access": "public", "license": cro["license"]},
        {"part": "metadata sidecars (no .parquet/.mp4)", "episodes": None, "GB": round(listing["total"]["sidecar_bytes"] / 1e9, 3),
         "access": "public; downloaded to data/phail/dataset/", "license": cro["license"]},
    ] + [{"part": p["url"], "episodes": None, "GB": None, "access": f"HTTP {p['status']} {p.get('reason', '')}".strip(), "license": "n/a"}
         for p in probe if not p["url"].startswith("s3://positronic-public")]
    for r in table:
        print(f"AC2.6 {r['part']}: episodes={r['episodes']} GB={r['GB']} access={r['access']}")
    print(f"AC2.6 croissant.json license {cro['license']}; mentions of a 352-episode fine-tuning set: {n352} (bucket: 449)")
    OUT["AC2.6"] = {"table": table, "croissant_352_mentions": n352}


def ac2_7_8(arms: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ev.plot_km(axes[0], arms[TELEOP]["Batteries"], TAU, "Teleop (human reference)", "#444444")
    ev.plot_km(axes[0], arms["ACT"]["Batteries"], TAU, "ACT", "#7b4fb3")
    axes[0].set_title("ACT on Batteries: KM CDF, 95% episode-clustered bands", fontsize=10, loc="left")
    axes[0].legend(loc="lower right", fontsize=9)
    a = ev.plot_pp(axes[1], arms["ACT"]["Batteries"], arms[TELEOP]["Batteries"], TAU, "#7b4fb3", "ACT vs Teleop")
    axes[1].set_title(f"P-P plot, area = AUC = P(T_ACT < T_human) = {a:.3f}", fontsize=10, loc="left")
    fig.text(0.01, 0.005, "PhAIL v1.0 released data (mixed cohort), computed by vlapoc.evaluation; cf. PhAIL Fig. 2", fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "s2_fig_km_pp_ACT_Batteries.png", dpi=130)
    plt.close(fig)
    taus = np.linspace(30, 240, 85)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = {"OpenPI": "#1f77b4", "GR00T": "#d62728", "ACT": "#9467bd", "SmolVLA": "#ff7f0e"}
    sweep = {}
    for m in MODELS:
        s = ev.uph_mtbf_sweep(arms[m], taus)
        sweep[m] = {"uph": s["macro_uph"].tolist(), "mtbfa": s["macro_mtbfa"].tolist()}
        ax.plot(s["macro_uph"], s["macro_mtbfa"], color=colors[m], label=m)
        ax.annotate("τ=30", (s["macro_uph"][0], s["macro_mtbfa"][0]), fontsize=7, color=colors[m])
    ax.set_xlabel("UPH = 3600 / RMST(τ_episode)  (macro over 4 objects)")
    ax.set_ylabel("MTBF/A = RMST / (1 - F(τ_episode))  (s, macro)")
    ax.set_title("UPH vs MTBF/A as τ_episode sweeps 30 -> 240 s (cf. PhAIL Fig. 10)", fontsize=10, loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "s2_fig_uph_mtbfa_macro.png", dpi=130)
    plt.close(fig)
    mono = {m: {"uph_falls": bool(np.all(np.diff(sweep[m]["uph"]) <= 1e-12)),
                "mtbfa_rises": bool(np.all(np.diff(sweep[m]["mtbfa"]) >= -1e-9))} for m in MODELS}
    print(f"AC2.7 figures: s2_fig_km_pp_ACT_Batteries.png (AUC {a:.4f}), s2_fig_uph_mtbfa_macro.png; "
          f"monotone over τ in [30, 240]: {mono}")
    OUT["AC2.7"] = {"figures": ["s2_fig_km_pp_ACT_Batteries.png", "s2_fig_uph_mtbfa_macro.png"], "auc_ACT_Batteries": a,
                    "sweep_taus": taus.tolist(), "sweep": sweep, "monotone": mono}
    # AC2.8 the three UPH forms, pooled per model at τ_episode = 240 s
    df = pd.read_parquet(REPO_ROOT / "data" / "phail" / "cohort_mixed.parquet")
    forms = {}
    for m in (*MODELS, TELEOP):
        eps = ev.pooled(arms[m])
        sub = df[df["model_short"] == m]
        per_ep = 3600 * sub["successful_items_eval"] / sub["duration_s"]
        forms[m] = {"3600/RMST(240) [adopted; fig_qq_tradeoff]": ev.uph(eps, TAU),
                    "3600/E[T|T<240] [Table 1]": ev.uph_conditional(eps, TAU),
                    "mean 3600*S/T per episode [common.py]": float(per_ep.mean())}
        print(f"AC2.8 UPH {m:8s} " + "; ".join(f"{k} = {v:.1f}" for k, v in forms[m].items()))
    OUT["AC2.8"] = {"forms": forms, "adopted": "3600/RMST(τ_episode), PhAIL build/fig_qq_tradeoff.py (App. L figure)"}


def hrt_ci_seeds(arms: dict, seeds: int) -> None:
    """AC2.5 with Monte-Carlo spread: both HRT CI variants over ``seeds`` RNG seeds (n_boot 1000 each);
    published endpoints (tab_headline_ci.json) as z-scores against each variant's seed distribution."""
    pub = {r["short"]: r for r in json.loads((CLONE / "out/paper/tab_headline_ci.json").read_text())}
    out = {}
    for m in MODELS:
        out[m] = {}
        for name, fixed in (("human_resampled", ()), ("human_fixed", ("human",))):
            ends = np.array([[100 * x for x in _ci({"model": arms[m], "human": arms[TELEOP]},
                                                      lambda a: ev.hrt(a["model"], a["human"], TAU), seed=s, fixed=fixed)[1:]]
                             for s in range(seeds)])
            mu, sd = ends.mean(0), ends.std(0, ddof=1)
            z = (np.array(pub[m]["hrt_ci"]) - mu) / sd
            out[m][name] = {"lo_mean": mu[0], "lo_sd": sd[0], "hi_mean": mu[1], "hi_sd": sd[1], "z_lo": z[0], "z_hi": z[1]}
            print(f"AC2.5 seeds {m:8s} {name:15s} lo {mu[0]:.3f}±{sd[0]:.3f} hi {mu[1]:.3f}±{sd[1]:.3f} | pub "
                  f"[{pub[m]['hrt_ci'][0]:.3f}, {pub[m]['hrt_ci'][1]:.3f}] -> z_lo {z[0]:+.2f} z_hi {z[1]:+.2f}")
    score = {v: float(sum(out[m][v]["z_lo"] ** 2 + out[m][v]["z_hi"] ** 2 for m in MODELS)) for v in ("human_resampled", "human_fixed")}
    print(f"AC2.5 seeds sum of z^2 over 8 endpoints (lower = closer to published): {score}")
    (ART / "s2_hrt_ci_variants_seeds.json").write_text(json.dumps({"seeds": seeds, "n_boot": 1000, "by_model": out, "sum_z2": score}, indent=1))


def main() -> int:
    if "--hrt-ci-seeds" in sys.argv:
        hrt_ci_seeds(load_arms(), int(sys.argv[sys.argv.index("--hrt-ci-seeds") + 1]))
        return 0
    ref = json.loads((ART / "s2_phail_reference.json").read_text())
    arms = load_arms()
    ac2_1(arms, ref)
    ac2_2(arms, ref)
    ac2_3_4_5(arms)
    ac2_6()
    ac2_7_8(arms)
    OUT.pop("_macro_ks", None)
    (ART / "phail_validation.json").write_text(json.dumps(OUT, indent=1, default=float))
    print("wrote artifacts/phail_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
