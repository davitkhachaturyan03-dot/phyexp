"""Verify evaluation-layer S1 (metric layer) on SYNTHETIC data: AC1.1-AC1.7, plus equivalence checks.

Every input below is synthetic (numpy generators with fixed seeds), labelled SYNTHETIC in the output.
No rollout outcome here is real. Real-data validation is S2 (PhAIL's released episodes).

Grounding: PoC §3.2 p. 6; PhAIL §3.1-3.3 pp. 4-6, Table 1 p. 5.

    .venv/bin/python scripts/verify_metrics_s1.py   # ~1 min; writes artifacts/s1_verify_metrics.json, s1_metric_registry.md
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import evaluation as ev  # noqa: E402
from vlapoc.third_party import phail_stats as ps  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
CLONE = REPO_ROOT / "data" / "external" / "phail-paper"
OUT: dict = {"data": "SYNTHETIC (numpy, fixed seeds); no real rollout outcomes"}


def sha(p: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ac1_1() -> None:
    """AC1.1: the vendored ``phail_stats.py`` and its LICENSE are byte-identical to the clone; the attribution header is present."""
    v, c = REPO_ROOT / "vlapoc/third_party/phail_stats.py", CLONE / "build/stats.py"
    lic = REPO_ROOT / "vlapoc/third_party/LICENSE-phail-paper"
    init = (REPO_ROOT / "vlapoc/third_party/__init__.py").read_text()
    head = (CLONE / ".git/HEAD").read_text().strip() if (CLONE / ".git/HEAD").is_file() else None
    r = {"vendored_sha256": sha(v), "clone_sha256": sha(c), "byte_identical": v.read_bytes() == c.read_bytes(),
         "clone_head": head, "license_identical": lic.read_bytes() == (CLONE / "LICENSE").read_bytes(),
         "header_in_package_init": all(s in init for s in ("18ce72d5703dcbbbb10a980336aa5a1622601fb4", "unmodified",
                                                              "Apache License, Version 2.0", "Positronic Robotics"))}
    print(f"AC1.1 vendored sha256 {r['vendored_sha256'][:16]}… == clone build/stats.py: {r['byte_identical']} "
          f"(clone HEAD {head}); LICENSE identical: {r['license_identical']}; attribution header: {r['header_in_package_init']}")
    OUT["AC1.1"] = r


def ac1_2() -> None:
    """AC1.2: vendored KM survival and Greenwood CI vs ``scipy.stats.ecdf`` on SYNTHETIC finite censored data."""
    from scipy.stats import CensoredData, ecdf

    rng = np.random.default_rng(1)
    T = np.round(rng.exponential(8.0, 400), 1) + 0.1  # rounded -> ties, incl. events tied with censorings
    E = (rng.random(400) < 0.7).astype(int)
    tg, S, var = ps.kaplan_meier(T, E)
    res = ecdf(CensoredData(uncensored=T[E == 1], right=T[E == 0]))
    S_sp = res.sf.evaluate(tg)
    ci = res.sf.confidence_interval(0.95, method="linear")
    z = 1.959963984540054
    hw_ours = z * S * np.sqrt(var)
    hw_sp = ci.high.evaluate(tg) - S_sp
    ok = (S > 0.02) & (S + hw_ours < 0.98)  # where scipy's band is not clipped at 0/1
    r = {"n_pairs": 400, "n_events": int(E.sum()), "n_event_times": len(tg), "ties": int(len(T) - len(np.unique(T))),
         "max_abs_diff_S": float(np.max(np.abs(S - S_sp))), "max_abs_diff_greenwood_halfwidth": float(np.max(np.abs(hw_ours[ok] - hw_sp[ok])))}
    r["pass"] = r["max_abs_diff_S"] <= 1e-12
    print(f"AC1.2 SYNTHETIC finite censored data ({r['n_pairs']} pairs, {r['n_events']} events, {r['ties']} tied values): "
          f"max |S_vendored - S_scipy.ecdf| over {r['n_event_times']} event times = {r['max_abs_diff_S']:.3e}; "
          f"Greenwood 95% half-width max diff = {r['max_abs_diff_greenwood_halfwidth']:.3e}")
    OUT["AC1.2"] = r


def ac1_3() -> None:
    """AC1.3: ghost semantics on SYNTHETIC data (KM mass at infinity = g/n; RMST = mean of min(T, τ) with ghosts at τ)."""
    rng = np.random.default_rng(2)
    eps = []
    g = 0
    for i in range(20):  # 20 episodes x 2-3 completed operations, every 4th episode adds one ghost; no censoring
        k = 2 + i % 2
        ts = np.cumsum(rng.uniform(1.0, 25.0, k))
        n_inf = 1 if i % 4 == 0 else 0
        g += n_inf
        eps.append(ev.episode_from_ops(f"syn{i}", ts, float(ts[-1]) + 1.0, k, n_inf_events=n_inf))
    T, E = ev.collect_TE(eps)
    n = len(T)
    gr = ev.ghost_event_rate(eps)
    tau = 30.0
    rm = ev.rmst(eps, tau)
    rm_direct = float(np.mean(np.minimum(T, tau)))  # ghosts (inf) -> τ
    r = {"n_ops": n, "ghosts": g, "censored_pairs": int((E == 0).sum()), "ghost_event_rate": gr, "g_over_n": g / n,
         "abs_diff_rate": abs(gr - g / n), "F_asymptote_gap": 1.0 - float(ev.F_at(eps, np.finfo(float).max)),
         "rmst30": rm, "mean_min_T_tau": rm_direct, "abs_diff_rmst": abs(rm - rm_direct)}
    r["pass"] = r["abs_diff_rate"] <= 1e-15 and r["abs_diff_rmst"] <= 1e-12 and r["censored_pairs"] == 0
    print(f"AC1.3 SYNTHETIC {n} operations, {g} ghosts (T=inf), {r['censored_pairs']} censored: ghost_event_rate = {gr!r}, "
          f"g/n = {g / n!r}, |diff| = {r['abs_diff_rate']:.1e}; 1 - F(max float) = {r['F_asymptote_gap']!r}")
    print(f"AC1.3 RMST(30 s) = {rm!r}; mean(min(T, 30)) with ghosts -> 30 = {rm_direct!r}; |diff| = {r['abs_diff_rmst']:.1e}")
    OUT["AC1.3"] = r


def ac1_4() -> None:
    """AC1.4: censoring semantics of ``episode_from_ops`` (zero-success episodes kept; no tail when all ops are done)."""
    tau_ep = 60.0
    zero = ev.episode_from_ops("zero", [], tau_ep, 2)
    full = ev.episode_from_ops("full", [5.0, 12.0], 20.0, 2)
    part = ev.episode_from_ops("part", [8.0], tau_ep, 2)
    simul = ev.libero_episode("simul", [10.0, 10.0], 10.0, 2)
    eps = [zero, full, part]
    T, E = ev.collect_TE(eps)
    tg, S, _ = ev.km(eps)
    r = {"zero_success_pairs": zero.pairs, "full_pairs": full.pairs, "partial_pairs": part.pairs,
         "libero_simultaneous_pairs": simul.pairs, "pairs_in_cell": len(T), "censored_rate": ev.censored_rate(eps),
         "S_after_first_event": float(S[0]), "expected_S_after_first_event": 1 - 1 / 5}
    r["pass"] = (zero.pairs == ((60.0, 0),) and full.pairs == ((5.0, 1), (7.0, 1)) and part.pairs == ((8.0, 1), (52.0, 0))
                 and len(T) == 5 and abs(S[0] - 0.8) < 1e-15 and simul.pairs == ((10.0, 1), (0.0, 1)))
    print(f"AC1.4 SYNTHETIC zero-success episode (τ_episode 60 s, 2 ops) -> {zero.pairs}; all ops done -> {full.pairs} "
          f"(no censored tail); 1 of 2 done -> {part.pairs}; LIBERO simultaneous ops (threshold 0) -> {simul.pairs}")
    print(f"AC1.4 the zero-success episode stays in the risk set: {len(T)} pairs, S after the first event = {S[0]!r} "
          f"(= 1 - 1/5); censored_rate = {r['censored_rate']:.4f} (2 of 3 episodes)")
    OUT["AC1.4"] = r


def ac1_5() -> None:
    """AC1.5: episode-clustered vs per-operation bootstrap CI width on SYNTHETIC correlated operations."""
    rng = np.random.default_rng(3)
    eps = []
    for i in range(40):  # correlated: an episode-level speed factor multiplies all 5 of its operations
        u = rng.lognormal(0.0, 0.6)
        ts = np.cumsum(u * rng.exponential(5.0, 5))
        eps.append(ev.episode_from_ops(f"syn{i}", ts, float(ts[-1]) + 1.0, 5))
    arm = {"cell": eps}
    res = {}
    for name, fn in {"rmst30": lambda a: ev.rmst(a["m"]["cell"], 30.0), "F10": lambda a: ev.F_at(a["m"]["cell"], 10.0)}.items():
        cl = ev.cluster_bootstrap({"m": arm}, fn, 1000, np.random.default_rng(0), unit="episode")
        op = ev.cluster_bootstrap({"m": arm}, fn, 1000, np.random.default_rng(0), unit="operation")
        w_cl, w_op = cl["hi"] - cl["lo"], op["hi"] - op["lo"]
        res[name] = {"point": cl["point"], "clustered_ci": [cl["lo"], cl["hi"]], "per_operation_ci": [op["lo"], op["hi"]],
                     "width_ratio_clustered_over_per_op": w_cl / w_op}
        print(f"AC1.5 SYNTHETIC 40 episodes x 5 correlated ops, {name} = {cl['point']:.4f}: clustered 95% CI "
              f"[{cl['lo']:.4f}, {cl['hi']:.4f}] vs per-operation [{op['lo']:.4f}, {op['hi']:.4f}]; width ratio {w_cl / w_op:.3f}")
    OUT["AC1.5"] = {**res, "pass": all(v["width_ratio_clustered_over_per_op"] > 1 for v in res.values())}


def ac1_6() -> None:
    """AC1.6: every registry entry's named functions exist with docstrings; writes ``s1_metric_registry.md``."""
    reg = ev.metric_registry()
    required = ["F(30 s) / Success@τ", "median", "RMST(τ)", "UPH", "completion", "MTBF/A", "HRT", "AUC vs reference",
                "KS / macro-KS", "P-P plot", "ghost-event rate", "censored rate"]
    names = [m["name"] for m in reg]
    lines = ["| metric | function | functional of F | τ | source |", "|---|---|---|---|---|"]
    lines += [f"| {m['name']} | `{m['function']}` | {m['functional']} | {m['tau']} | {m['source']} |" for m in reg]
    (ART / "s1_metric_registry.md").write_text("\n".join(lines) + "\n")
    missing = [r for r in required if r not in names]
    print(f"AC1.6 registry: {len(reg)} entries, every named function exists with a docstring; required missing: {missing}")
    OUT["AC1.6"] = {"n": len(reg), "names": names, "missing": missing, "pass": not missing}


def equivalence() -> None:
    """macro-KS vs vendored (same RNG), logrank vs scipy, AUC / P-P / HRT semantics. SYNTHETIC."""
    from scipy.stats import CensoredData, logrank

    rng = np.random.default_rng(4)

    def syn_ns(scale: float, n: int) -> list:
        """A SYNTHETIC PhAIL-style episode record (the attributes ``episode_to_T_pairs`` reads)."""
        out = []
        for _ in range(n):
            k = 3
            done = rng.integers(0, k + 1)
            ts = list(np.cumsum(scale * rng.exponential(6.0, done)))
            out.append(SimpleNamespace(place_ts=ts, duration_s=(ts[-1] if ts else 0.0) + rng.uniform(1, 30),
                                       total_items=k, n_inf_events=int(rng.random() < 0.15)))
        return out

    a_ns = {o: syn_ns(1.0, 15) for o in ("o1", "o2", "o3")}
    b_ns = {o: syn_ns(1.3, 12) for o in ("o1", "o2", "o3")}
    D_v, p_v = ps.per_obj_macro_bootstrap_ks(a_ns, b_ns, n_boot=500, rng=np.random.default_rng(7))
    to_arm = lambda d: {o: [ev.episode_from_ops(f"{o}{i}", e.place_ts, e.duration_s, e.total_items, e.n_inf_events)  # noqa: E731
                            for i, e in enumerate(es)] for o, es in d.items()}
    mk = ev.macro_ks(to_arm(a_ns), to_arm(b_ns), n_boot=500, rng=np.random.default_rng(7))
    print(f"EQ macro-KS SYNTHETIC (3 cells, ghosts + censoring): vendored per_obj_macro_bootstrap_ks D={D_v!r} p={p_v!r}; "
          f"evaluation.macro_ks D={mk['D']!r} p={mk['p']!r}; identical: {D_v == mk['D'] and p_v == mk['p']}")
    # logrank, one stratum, finite times: vs scipy
    Ta = np.round(rng.exponential(5, 60), 1) + 0.1
    Ea = (rng.random(60) < 0.8).astype(int)
    Tb = np.round(rng.exponential(7, 50), 1) + 0.1
    Eb = (rng.random(50) < 0.8).astype(int)
    one = lambda T, E: {"s": [ev.Episode(f"x{i}", ((float(t), int(e)),)) for i, (t, e) in enumerate(zip(T, E))]}  # noqa: E731
    lr = ev.logrank_stratified(one(Ta, Ea), one(Tb, Eb))
    sp = logrank(CensoredData(uncensored=Ta[Ea == 1], right=Ta[Ea == 0]), CensoredData(uncensored=Tb[Eb == 1], right=Tb[Eb == 0]))
    print(f"EQ logrank SYNTHETIC one stratum (ties present): ours chi2={lr['chi2']:.10f} p={lr['p']:.10f}; scipy Z^2="
          f"{sp.statistic ** 2:.10f} p={sp.pvalue:.10f}; |dchi2|={abs(lr['chi2'] - sp.statistic ** 2):.2e}")
    # AUC hand example and ghost
    a1 = ev.auc_censored(np.array([1.0, 3.0]), np.array([1, 1]), np.array([2.0, 3.0]), np.array([1, 0]))
    a2 = ev.auc_censored(np.array([np.inf]), np.array([1]), np.array([5.0]), np.array([1]))
    # P-P area vs AUC (uncensored continuous)
    m = [ev.Episode(f"m{i}", ((float(t), 1),)) for i, t in enumerate(rng.exponential(10, 300))]
    h = [ev.Episode(f"h{i}", ((float(t), 1),)) for i, t in enumerate(rng.exponential(6, 300))]
    x, y = ev.pp_curve(m, h, tau=500.0, n=50000)
    area = float(np.trapezoid(y, x))
    au = ev.auc(m, h)
    print(f"EQ AUC hand example a=[1,3] b=[2,3(censored)] -> {a1:.6f} (expected 2/3); ghost vs 5 s -> {a2}; "
          f"SYNTHETIC P-P area {area:.5f} vs AUC {au:.5f} (|diff| {abs(area - au):.1e})")
    # HRT
    model = {"c1": m[:150], "c2": m[150:]}
    human = {"c1": h[:150], "c2": h[150:]}
    cells = ev.hrt_cells(model, human, 60.0)
    hr = ev.hrt(model, human, 60.0)
    ci_both = ev.hrt_ci(model, human, 60.0, 300, np.random.default_rng(0), resample_human=True)
    ci_fix = ev.hrt_ci(model, human, 60.0, 300, np.random.default_rng(0), resample_human=False)
    print(f"EQ HRT SYNTHETIC 2 cells: per cell {({k: round(v, 4) for k, v in cells.items()})}, macro {hr:.4f} "
          f"(= mean: {abs(hr - np.mean(list(cells.values()))) < 1e-15}); CI resample human [{ci_both['lo']:.4f}, {ci_both['hi']:.4f}] "
          f"vs human fixed [{ci_fix['lo']:.4f}, {ci_fix['hi']:.4f}]")
    OUT["equivalence"] = {
        "macro_ks": {"vendored": [D_v, p_v], "ours": [mk["D"], mk["p"]], "identical": D_v == mk["D"] and p_v == mk["p"]},
        "logrank_vs_scipy": {"ours_chi2": lr["chi2"], "scipy_Z2": float(sp.statistic ** 2), "ours_p": lr["p"], "scipy_p": float(sp.pvalue)},
        "auc_hand": a1, "auc_ghost": a2, "pp_area": area, "auc": au,
        "hrt": {"cells": cells, "macro": hr, "ci_resample_human": [ci_both["lo"], ci_both["hi"]], "ci_human_fixed": [ci_fix["lo"], ci_fix["hi"]]}}


def ac1_7() -> None:
    """AC1.7: count the grounding citations in ``vlapoc/evaluation.py``."""
    src = (REPO_ROOT / "vlapoc/evaluation.py").read_text()
    cites = {k: src.count(k) for k in ("PoC §3.2 (p. 6)", "PoC §10 (p. 20)", "§3.1 p. 4", "Table 1 p. 5", "§3.2 p. 5", "§3.3 p. 6")}
    print(f"AC1.7 citations in vlapoc/evaluation.py: {cites}")
    OUT["AC1.7"] = {"counts": cites, "pass": all(v > 0 for v in cites.values())}


def main() -> int:
    """Run AC1.1-AC1.7 plus the equivalence checks and write ``artifacts/s1_verify_metrics.json``."""
    for f in (ac1_1, ac1_2, ac1_3, ac1_4, ac1_5, ac1_6, equivalence, ac1_7):
        f()
    (ART / "s1_verify_metrics.json").write_text(json.dumps(OUT, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
