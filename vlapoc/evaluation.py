"""Metric layer: PhAIL's evaluation primitives over operation-level (T, E) pairs, per cell and macro.

Grounding
---------
- PoC §3.2 (p. 6), "The evaluation primitives": time-to-success T ("each rollout yields several"),
  ghost events (T = inf: "The gap between the CDF's asymptote and 1 is exactly the hard-failure rate"),
  right-censoring ("Timeouts are slow-but-recoverable, recorded as censored, not failed"), the
  Kaplan-Meier CDF ("Success@τ, median, RMST, UPH, completion, MTBF/A are all projections of F"), HRT
  ("RMST_human / RMST_model per object, macro-averaged"), the clustered bootstrap ("95% CIs resampling
  whole episodes, not operations"), macro-KS ("per object, averaged; p-values from pooled-resample
  bootstrap") and P-P plots ("area = P(T_model < T_human)"). Rule: scoring (a scalar with a CI) and
  significance (macro-KS) are separate jobs; "KS cannot rank on its own".
- PoC §10 (p. 20), Evaluation layer: "Kaplan-Meier time-to-success CDFs with ghost-event and censoring
  rules; HRT against LIBERO demo durations; episode-clustered bootstrap CIs; macro-KS test; P-P plots".
- PhAIL (docs/papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf), the defining formulation:
  §3.1 p. 4 (T pairs, censored tail, ghosts as (inf, 1), zero-success episodes "contribute a single
  right-censored observation of length τ_episode ... we never drop them", ghosts -> τ in RMST);
  Table 1 p. 5 (scalars as functionals of F); §3.2 p. 5 (HRT, τ = 240 s, equal-weight macro; P-P area =
  AUC; "We resample whole episodes"); §3.3 p. 6 (macro-KS D̄ = (1/J) Σ_o D_o with the pooled-resample
  episode-clustered bootstrap; "stratified pooled-across-objects logrank ... as a sanity-check";
  direction "comes from a scalar"); App. L (UPH falls / MTBF/A rises with τ_episode).

What is reused, not re-implemented
----------------------------------
``vlapoc.third_party.phail_stats`` (PhAIL ``build/stats.py`` at 18ce72d, vendored unmodified):
``episode_to_T_pairs``, ``kaplan_meier``, ``km_step``, ``km_median``, ``rmst``, ``ks_distance``. The
scripts' formulas that live outside ``stats.py`` (they import PhAIL's data loader, so they cannot be
imported here) are restated with their source cited and checked for equality on PhAIL's data in S2:
``_auc_censored``, ``_pp_curve``, ``_macro_auc``, ``_macro_pp_curve`` (``build/explore_pp_plot.py``),
UPH = 3600/RMST and MTBF/A = RMST/(1 - F) (``build/fig_qq_tradeoff.py``), the logrank arithmetic
(``build/explore_separation.py:_logrank``). lifelines rejects T = inf and scipy's ``ecdf`` rejects
infinite uncensored values (plan P20), so neither can carry ghost events; scipy is the independent
cross-check on finite data. scipy's ``logrank`` has no strata (signature ``(x, y, alternative)``,
measured), hence ``logrank_stratified`` below.

Data model
----------
An ``Episode`` is one rollout (or one human demo) with its operation-level (T, E) pairs: E = 1 an
operation completed after T seconds (T = inf: a ghost event), E = 0 a right-censored tail. A *cell* is
one stratum (PhAIL: an object; LIBERO: a task), and an *arm* is ``{cell: [Episode, ...]}`` for one
policy (or the human reference). Every scalar is a functional of the Kaplan-Meier F of a cell's pooled
pairs; macro scalars average cells with equal weights; CIs resample whole episodes within each cell.

LIBERO episodes come from ``libero_episode``: ``place_ts`` = operation completion times (seconds of
simulated time, T = steps / 20), ``duration_s`` = the rollout's observed duration (τ_episode when it timed
out), ``total_items`` = number of operations, no ghost events (LIBERO defines no unrecoverable-state rule;
plan P13, owner decision 6), and ``multi_grip_threshold_s = 0`` (sim timestamps are exact; PhAIL's 0.3 s
clustering exists for annotation jitter, ``episode_to_T_pairs`` docstring; a recorded divergence, S4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Iterable

import numpy as np

from vlapoc.third_party import phail_stats as ps

Arm = dict  # {cell: list[Episode]}


# ---------------------------------------------------------------------------------------------
# episodes and their (T, E) pairs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One rollout: its key and its operation-level (T, E) pairs (PhAIL §3.1 p. 4).

    ``meta`` carries free bookkeeping (e.g. ``duration_s``, ``n_ops``) and never enters a metric.
    """

    key: str
    pairs: tuple[tuple[float, int], ...]
    meta: dict = field(default_factory=dict, compare=False)


def episode_from_ops(key: str, place_ts: Iterable[float], duration_s: float, total_items: int,
                     n_inf_events: int = 0, multi_grip_threshold_s: float = 0.3, **meta) -> Episode:
    """Episode from operation completion times through PhAIL's own ``episode_to_T_pairs``.

    PhAIL §3.1 p. 4: T_i = time since the previous success (or episode start), E_i = 1; operations left
    when the episode ended add one censored ``(duration - last placement, 0)``; an episode with no
    success adds ``(duration_s, 0)``; each ghost adds ``(inf, 1)``. ``multi_grip_threshold_s`` is
    PhAIL's default (0.3 s) for PhAIL data; ``libero_episode`` passes 0.
    """
    src = SimpleNamespace(place_ts=list(place_ts), duration_s=float(duration_s), total_items=int(total_items),
                          n_inf_events=int(n_inf_events))
    pairs = ps.episode_to_T_pairs(src, multi_grip_threshold_s=multi_grip_threshold_s)
    return Episode(key, tuple((float(t), int(e)) for t, e in pairs),
                   {"duration_s": float(duration_s), "total_items": int(total_items), "n_inf_events": int(n_inf_events),
                    **meta})


def libero_episode(key: str, place_ts_s: Iterable[float], duration_s: float, n_ops: int, **meta) -> Episode:
    """A LIBERO rollout or human demo: no ghost events, exact timestamps (threshold 0; module docstring)."""
    return episode_from_ops(key, place_ts_s, duration_s, n_ops, n_inf_events=0, multi_grip_threshold_s=0.0, **meta)


def collect_TE(eps: Iterable[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """All (T, E) pairs of ``eps``, concatenated: ``(T float64, E int64)``."""
    pairs = [p for ep in eps for p in ep.pairs]
    return (np.array([t for t, _ in pairs], dtype=float), np.array([e for _, e in pairs], dtype=int))


def km(eps: Iterable[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kaplan-Meier fit of the pooled pairs: ``(t_grid, S, var_logS)`` (``phail_stats.kaplan_meier``,
    Greenwood variance). T = inf events are a jump at the right edge (PhAIL §3.1 p. 4)."""
    T, E = collect_TE(eps)
    return ps.kaplan_meier(T, E)


def F_curve(eps: Iterable[Episode], t: np.ndarray) -> np.ndarray:
    """F(t) = 1 - S(t) at the points ``t`` (right-continuous step function)."""
    tg, S, _ = km(eps)
    return 1.0 - ps.km_step(tg, S, np.asarray(t, dtype=float))


# ---------------------------------------------------------------------------------------------
# scalars: functionals of F (PhAIL Table 1 p. 5)
# ---------------------------------------------------------------------------------------------


def F_at(eps: Iterable[Episode], tau: float) -> float:
    """Success@τ = F(τ) = P(T <= τ) (PhAIL Table 1 "Success rate at τ"; PoC §4.1 F(30 s))."""
    return float(F_curve(eps, np.array([float(tau)]))[0])


def median_T(eps: Iterable[Episode]) -> float:
    """Median time-to-success = inf{t : F(t) >= 0.5} (Table 1); ``inf`` if F never reaches 0.5
    (``phail_stats.km_median`` returns None there). No τ."""
    tg, S, _ = km(eps)
    m = ps.km_median(tg, S)
    return float("inf") if m is None else m


def rmst(eps: Iterable[Episode], tau: float) -> float:
    """RMST(τ) = ∫_0^τ (1 - F(t)) dt (Table 1). Ghosts count as τ (PhAIL §3.1 p. 4: "we set T_ghost -> τ,
    the integration cap"), which the step integral of the KM survival does by construction."""
    tg, S, _ = km(eps)
    return ps.rmst(tg, S, float(tau))


def completion(eps: Iterable[Episode], tau_episode: float) -> float:
    """Completion fraction = F(τ_episode) (Table 1), τ_episode = the per-rollout time budget."""
    return F_at(eps, tau_episode)


def uph(eps: Iterable[Episode], tau_episode: float) -> float:
    """UPH = 3600 / RMST(τ_episode) operations per hour.

    Adopted form: PhAIL's own figure code (``build/fig_qq_tradeoff.py``: ``uph = 3600.0 /
    np.maximum(RMST, 1e-6)``), which Table 1 states as proportional: "3600/E[T | T < τ_episode] ∝
    1/RMST(τ_episode)". The conditional form is ``uph_conditional``; PhAIL's ``common.py`` has a third,
    per-episode ``3600 * S / T`` (plan, prior art). Disclosed in the report (S2 AC2.8).
    """
    r = rmst(eps, tau_episode)
    return 3600.0 / r if r > 0 else float("inf")


def uph_conditional(eps: Iterable[Episode], tau_episode: float) -> float:
    """Table 1's literal UPH: 3600 / E[T | T < τ_episode], from the KM:
    E[T | T <= τ] = (RMST(τ) - τ (1 - F(τ))) / F(τ). ``nan`` when F(τ) = 0."""
    F = F_at(eps, tau_episode)
    if F <= 0:
        return float("nan")
    mean_t = (rmst(eps, tau_episode) - tau_episode * (1.0 - F)) / F
    return 3600.0 / mean_t if mean_t > 0 else float("inf")


def mtbfa(eps: Iterable[Episode], tau_episode: float) -> float:
    """MTBF/A = RMST(τ_episode) / (1 - F(τ_episode)) seconds (Table 1); ``inf`` when every operation
    completes within budget (PhAIL's figure code clips the denominator at 1e-9 instead)."""
    q = 1.0 - F_at(eps, tau_episode)
    return rmst(eps, tau_episode) / q if q > 0 else float("inf")


def ghost_event_rate(eps: Iterable[Episode]) -> float:
    """Hard-failure (ghost-event) rate = the KM probability mass at T = inf, i.e. the gap between the
    CDF's finite-time asymptote and 1 (PoC §3.2 p. 6; PhAIL §3.1 p. 4). With no censoring it is exactly
    g/n (g ghosts among n operations); with censoring, KM redistributes censored mass to later times,
    including inf. 0 when there are no ghosts. No τ."""
    tg, S, _ = km(eps)
    if len(tg) == 0 or not np.isinf(tg[-1]):
        return 0.0
    return float(S[-2]) - float(S[-1]) if len(S) > 1 else 1.0 - float(S[-1])


def ghost_fraction(eps: Iterable[Episode]) -> float:
    """Raw share of operation pairs that are ghost events (T = inf), without KM. Diagnostic."""
    T, _ = collect_TE(eps)
    return float(np.isinf(T).mean()) if len(T) else 0.0


def censored_rate(eps: Iterable[Episode]) -> float:
    """Share of episodes that end with a right-censored pair (timed out or stopped with operations
    left; PoC §3.4 p. 7 "the censored tail"; PhAIL §3.1 p. 4). No τ beyond the protocol's τ_episode.
    It is the mean of S6's per-rollout ``ret_censored60``."""
    eps = list(eps)
    return float(np.mean([any(e == 0 for _, e in ep.pairs) for ep in eps])) if eps else 0.0


# ---------------------------------------------------------------------------------------------
# macro aggregation, HRT, AUC, P-P, KS
# ---------------------------------------------------------------------------------------------


def macro(arm: Arm, fn: Callable[[list], float], cells: Iterable | None = None) -> float:
    """Equal-weight mean of ``fn(episodes)`` over cells (PhAIL §3.2 p. 5 "macro-aggregated across objects
    with equal weights"); ``cells`` defaults to every non-empty cell of ``arm``."""
    cells = sorted(c for c in arm if arm[c]) if cells is None else list(cells)
    return float(np.mean([fn(arm[c]) for c in cells])) if cells else float("nan")


def pooled(arm: Arm) -> list[Episode]:
    """All episodes of an arm, across cells (for global, non-macro scalars such as PoC §4.1 "global RMST")."""
    return [ep for c in sorted(arm) for ep in arm[c]]


def hrt_cells(model: Arm, human: Arm, tau: float) -> dict:
    """HRT per cell = RMST_human,o(τ) / RMST_model,o(τ) (PoC §3.2 p. 6; PhAIL §3.2 p. 5, τ = 240 s there),
    for every cell non-empty in both arms. A ratio (x100 = PhAIL's percentage)."""
    out = {}
    for c in sorted(set(model) & set(human)):
        if model[c] and human[c]:
            rm = rmst(model[c], tau)
            out[c] = rmst(human[c], tau) / rm if rm > 0 else float("inf")
    return out


def hrt(model: Arm, human: Arm, tau: float) -> float:
    """Macro HRT: equal-weight mean of ``hrt_cells`` (PhAIL §3.2 p. 5)."""
    v = hrt_cells(model, human, tau)
    return float(np.mean(list(v.values()))) if v else float("nan")


def auc_censored(T_a: np.ndarray, E_a: np.ndarray, T_b: np.ndarray, E_b: np.ndarray) -> float:
    """P(T_a < T_b) over all comparable pairs, censoring-aware (PhAIL ``explore_pp_plot._auc_censored``).

    A pair counts as ``a < b`` if a is an event and a < b (b's censoring does not matter), as ``a > b`` if
    b is an event and a > b; ties and pairs that cannot be ordered leave the denominator. 0.5 if nothing
    is comparable. PhAIL §3.2 p. 5: the area under the P-P curve "equals P(T_model < T_Human)".
    """
    T_a, E_a, T_b, E_b = map(np.asarray, (T_a, E_a, T_b, E_b))
    if len(T_a) == 0 or len(T_b) == 0:
        return 0.5
    a_wins = int(((T_a[:, None] < T_b[None, :]) & E_a[:, None].astype(bool)).sum())
    b_wins = int(((T_a[:, None] > T_b[None, :]) & E_b[None, :].astype(bool)).sum())
    return a_wins / (a_wins + b_wins) if a_wins + b_wins else 0.5


def auc(model_eps: Iterable[Episode], ref_eps: Iterable[Episode]) -> float:
    """AUC vs reference = P(T_model < T_ref) for one cell (PoC §4.1 "AUC vs human")."""
    return auc_censored(*collect_TE(model_eps), *collect_TE(ref_eps))


def macro_auc(model: Arm, ref: Arm) -> float:
    """Equal-weight mean of per-cell AUC over cells with pairs in both arms (PhAIL ``_macro_auc``)."""
    vals = []
    for c in sorted(set(model) & set(ref)):
        Tm, Em = collect_TE(model[c])
        Tr, Er = collect_TE(ref[c])
        if len(Tm) and len(Tr):
            vals.append(auc_censored(Tm, Em, Tr, Er))
    return float(np.mean(vals)) if vals else 0.5


def pp_curve(model_eps: Iterable[Episode], ref_eps: Iterable[Episode], tau: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """P-P curve (F_ref(t), F_model(t)) for t on ``linspace(0, τ, n)`` (PhAIL ``_pp_curve``; PoC §3.2 p. 6
    "F_human(t) vs F_model(t)"). x = reference, y = model; below the diagonal = slower than reference."""
    grid = np.linspace(0, tau, n)
    return F_curve(ref_eps, grid), F_curve(model_eps, grid)


def macro_pp_curve(model: Arm, ref: Arm, tau: float, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell P-P curves averaged at common F_ref quantiles (PhAIL ``_macro_pp_curve``): for each q,
    τ_q = smallest reference event time with F_ref >= q (τ if none), then F_model(τ_q), averaged."""
    q = np.linspace(0, 1.0, n)
    rows = []
    for c in sorted(set(model) & set(ref)):
        if not model[c] or not ref[c]:
            continue
        tr, Sr, _ = km(ref[c])
        Fr = 1.0 - Sr
        idx = np.searchsorted(Fr, q, side="left")
        tau_q = np.where(idx >= len(tr), tau, tr[np.minimum(idx, len(tr) - 1)])
        rows.append(F_curve(model[c], tau_q))
    return (q, np.mean(np.stack(rows), axis=0)) if rows else (q, q.copy())


def ks(eps_a: Iterable[Episode], eps_b: Iterable[Episode]) -> float:
    """Two-sample KS distance sup_t |F_a(t) - F_b(t)| between KM CDFs (``phail_stats.ks_distance``)."""
    ta, Sa, _ = km(eps_a)
    tb, Sb, _ = km(eps_b)
    return ps.ks_distance(ta, Sa, tb, Sb)


def macro_ks(arm_a: Arm, arm_b: Arm, n_boot: int = 2000, rng: np.random.Generator | None = None) -> dict:
    """Macro-KS significance gate: D̄ = (1/J) Σ_o D_o with a pooled-resample episode-clustered bootstrap p
    (PoC §3.2 p. 6; PhAIL §3.3 p. 6). Significance only: it never gives a direction ("KS cannot rank").

    Same algorithm and the same RNG draw order as ``phail_stats.per_obj_macro_bootstrap_ks`` (cells in
    sorted order; per bootstrap draw and cell, arm A's indices then arm B's, from the cell's pooled
    episodes), so on episodes built with ``episode_to_T_pairs`` and an equally seeded generator it
    returns PhAIL's (D, p) exactly (checked in S1 and on PhAIL's data in S2). It runs on stored pairs,
    so LIBERO's threshold-0 pairs go through unchanged. Returns ``{"D", "p", "D_cells", "n_boot"}``.
    """
    rng = rng or np.random.default_rng(0)
    cells = [c for c in sorted(set(arm_a) & set(arm_b)) if arm_a[c] and arm_b[c]]
    if not cells:
        return {"D": 0.0, "p": 1.0, "D_cells": {}, "n_boot": n_boot}

    def _d_cells(a: Arm, b: Arm) -> dict:
        out = {}
        for c in cells:
            Ta, Ea = collect_TE(a[c])
            Tb, Eb = collect_TE(b[c])
            if len(Ta) and len(Tb):
                out[c] = ps.ks_distance(*ps.kaplan_meier(Ta, Ea)[:2], *ps.kaplan_meier(Tb, Eb)[:2])
        return out

    def _macro(a: Arm, b: Arm) -> float:
        v = _d_cells(a, b)
        return float(np.mean(list(v.values()))) if v else 0.0

    d_cells = _d_cells(arm_a, arm_b)
    obs = float(np.mean(list(d_cells.values()))) if d_cells else 0.0
    pool = {c: list(arm_a[c]) + list(arm_b[c]) for c in cells}
    count = 0
    for _ in range(n_boot):
        ra, rb = {}, {}
        for c in cells:
            ia = rng.integers(0, len(pool[c]), size=len(arm_a[c]))
            ib = rng.integers(0, len(pool[c]), size=len(arm_b[c]))
            ra[c] = [pool[c][i] for i in ia]
            rb[c] = [pool[c][i] for i in ib]
        if _macro(ra, rb) >= obs:
            count += 1
    return {"D": obs, "p": (count + 1) / (n_boot + 1), "D_cells": d_cells, "n_boot": n_boot}


def logrank_stratified(arm_a: Arm, arm_b: Arm) -> dict:
    """Stratified (Mantel-Haenszel) logrank over cells: Σ_o (O - E)_o and Σ_o V_o, chi² = (Σ(O-E))² / ΣV,
    df = 1 (PhAIL §3.3 p. 6: "stratified pooled-across-objects logrank ... as a sanity-check").

    Per stratum the arithmetic is PhAIL's ``build/explore_separation.py:_logrank`` (event times include
    T = inf ghosts; ties at a time share the hypergeometric variance). Events are treated as independent
    within an arm; it is a sanity line, never the gate. With one stratum and finite times it equals
    ``scipy.stats.logrank`` (chi² = Z²; checked in S1). Returns ``{"chi2", "p", "O_minus_E", "V", "strata"}``.
    """
    from scipy.stats import chi2 as _chi2

    O_minus_E, V, strata = 0.0, 0.0, []
    for c in sorted(set(arm_a) & set(arm_b)):
        Ta, Ea = collect_TE(arm_a[c])
        Tb, Eb = collect_TE(arm_b[c])
        if not len(Ta) or not len(Tb):
            continue
        T = np.concatenate([Ta, Tb])
        E = np.concatenate([Ea, Eb])
        G = np.concatenate([np.zeros(len(Ta), int), np.ones(len(Tb), int)])
        for t in np.unique(T[E == 1]):
            at_a = int(((T >= t) & (G == 0)).sum())
            at_b = int(((T >= t) & (G == 1)).sum())
            d_a = int(((T == t) & (E == 1) & (G == 0)).sum())
            d = d_a + int(((T == t) & (E == 1) & (G == 1)).sum())
            n = at_a + at_b
            if n <= 1:
                continue
            O_minus_E += d_a - d * at_a / n
            V += d * at_a * at_b * (n - d) / (n * n * max(n - 1, 1))
        strata.append(c)
    if V <= 0:
        return {"chi2": 0.0, "p": 1.0, "O_minus_E": O_minus_E, "V": V, "strata": strata}
    chi = O_minus_E ** 2 / V
    return {"chi2": float(chi), "p": float(_chi2.sf(chi, df=1)), "O_minus_E": float(O_minus_E), "V": float(V),
            "strata": strata}


# ---------------------------------------------------------------------------------------------
# episode-clustered bootstrap CIs
# ---------------------------------------------------------------------------------------------


def _as_operations(arm: Arm) -> Arm:
    """Every (T, E) pair as its own one-operation episode: the per-operation resampling the PoC warns
    against (§3.2 p. 6 "per-operation resampling gives CIs that are too narrow"); for comparison only."""
    return {c: [Episode(f"{ep.key}#{i}", (p,)) for ep in eps for i, p in enumerate(ep.pairs)] for c, eps in arm.items()}


def cluster_bootstrap(arms: dict[str, Arm], statistic: Callable[[dict[str, Arm]], float], n_boot: int = 1000,
                      rng: np.random.Generator | None = None, level: float = 0.95, unit: str = "episode",
                      fixed: Iterable[str] = ()) -> dict:
    """Bootstrap CI for any scalar of one or more arms, resampling whole episodes within each cell.

    PoC §3.2 p. 6 "95% CIs resampling whole episodes, not operations"; PhAIL §3.2 p. 5 "We resample whole
    episodes". ``arms`` maps a name to an arm; ``statistic(arms)`` returns the scalar. Draw order: arms in
    insertion order, cells sorted, ``rng.integers(0, n, size=n)`` per (arm, cell). Arms named in
    ``fixed`` are not resampled (e.g. a fixed human anchor for HRT). ``unit="operation"`` resamples
    single operations instead (``_as_operations``), to show how much narrower that CI is.
    Returns ``{"point", "lo", "hi", "level", "n_boot", "unit", "boot"}`` (percentile interval).
    """
    rng = rng or np.random.default_rng(0)
    if unit == "operation":
        arms = {k: (_as_operations(v) if k not in set(fixed) else v) for k, v in arms.items()}
    elif unit != "episode":
        raise ValueError("unit must be 'episode' or 'operation'")
    point = float(statistic(arms))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        rs = {}
        for name, arm in arms.items():
            if name in set(fixed):
                rs[name] = arm
                continue
            rs[name] = {c: [arm[c][i] for i in rng.integers(0, len(arm[c]), size=len(arm[c]))] if arm[c] else []
                        for c in sorted(arm)}
        boot[b] = statistic(rs)
    a = (1.0 - level) / 2.0
    return {"point": point, "lo": float(np.quantile(boot, a)), "hi": float(np.quantile(boot, 1.0 - a)),
            "level": level, "n_boot": n_boot, "unit": unit, "boot": boot}


def hrt_ci(model: Arm, human: Arm, tau: float, n_boot: int = 1000, rng: np.random.Generator | None = None,
           resample_human: bool = True) -> dict:
    """Macro HRT with an episode-clustered bootstrap CI. ``resample_human=True`` resamples the human
    anchor's episodes too; ``False`` holds it fixed. PhAIL's Table 2 CI producer is not released (plan
    P19); S2 AC2.5 picks the variant that reproduces ``out/paper/tab_headline_ci.json``."""
    return cluster_bootstrap({"model": model, "human": human}, lambda a: hrt(a["model"], a["human"], tau), n_boot, rng,
                             fixed=() if resample_human else ("human",))


def F_band(eps: list[Episode], grid: np.ndarray, n_boot: int = 1000, rng: np.random.Generator | None = None,
           level: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise episode-clustered bootstrap band for the KM CDF on ``grid``: ``(F, lo, hi)`` (PhAIL §3.2
    p. 5 "All scoring scalars and CDF curves carry 95% episode-clustered bootstrap intervals")."""
    rng = rng or np.random.default_rng(0)
    grid = np.asarray(grid, dtype=float)
    boots = np.stack([F_curve([eps[i] for i in rng.integers(0, len(eps), size=len(eps))], grid) for _ in range(n_boot)])
    a = (1.0 - level) / 2.0
    return F_curve(eps, grid), np.quantile(boots, a, axis=0), np.quantile(boots, 1.0 - a, axis=0)


def uph_mtbf_sweep(arm: Arm, taus: np.ndarray) -> dict:
    """UPH and MTBF/A as τ_episode sweeps ``taus``, per cell and macro (PhAIL App. L; PoC §4.1 p. 9 "as
    the per-rollout time budget grows, throughput-oriented UPH falls while reliability-oriented MTBF/A
    rises"). Exact step-integral RMST at each τ (PhAIL's figure uses a 1500-point Riemann sum)."""
    taus = np.asarray(taus, dtype=float)
    per = {c: {"uph": np.array([uph(arm[c], t) for t in taus]), "mtbfa": np.array([mtbfa(arm[c], t) for t in taus])}
           for c in sorted(arm) if arm[c]}
    return {"taus": taus, "cells": per,
            "macro_uph": np.mean([v["uph"] for v in per.values()], axis=0),
            "macro_mtbfa": np.mean([v["mtbfa"] for v in per.values()], axis=0)}


# ---------------------------------------------------------------------------------------------
# metric registry (AC1.6): every PoC §3.2 primitive and PhAIL Table 1 scalar, with its functional and τ
# ---------------------------------------------------------------------------------------------

METRICS = [
    {"name": "F(τ) / Success@τ", "function": "F_at", "functional": "F(τ) = P(T <= τ)", "tau": "any τ (PoC §4.1: 30 s)",
     "source": "PhAIL Table 1 p.5; PoC §3.2 p.6, §4.1 p.9"},
    {"name": "median", "function": "median_T", "functional": "inf{t : F(t) >= 0.5}", "tau": "none",
     "source": "PhAIL Table 1 p.5"},
    {"name": "RMST(τ)", "function": "rmst", "functional": "∫_0^τ (1 - F(t)) dt, ghosts -> τ", "tau": "integration cap τ",
     "source": "PhAIL Table 1 p.5, §3.1 p.4"},
    {"name": "UPH", "function": "uph", "functional": "3600 / RMST(τ_episode)  (Table 1: 3600/E[T|T<τ_episode] ∝ 1/RMST)",
     "tau": "τ_episode", "source": "PhAIL Table 1 p.5; build/fig_qq_tradeoff.py"},
    {"name": "completion", "function": "completion", "functional": "F(τ_episode)", "tau": "τ_episode",
     "source": "PhAIL Table 1 p.5"},
    {"name": "MTBF/A", "function": "mtbfa", "functional": "RMST(τ_episode) / (1 - F(τ_episode))", "tau": "τ_episode",
     "source": "PhAIL Table 1 p.5"},
    {"name": "HRT", "function": "hrt", "functional": "mean_o RMST_human,o(τ) / RMST_model,o(τ)", "tau": "integration cap τ",
     "source": "PoC §3.2 p.6; PhAIL §3.2 p.5"},
    {"name": "AUC vs reference", "function": "auc / macro_auc", "functional": "P(T_model < T_ref) = area under P-P",
     "tau": "none (censoring-aware pairs)", "source": "PoC §3.2 p.6, §4.1 p.9; PhAIL §3.2 p.5"},
    {"name": "KS / macro-KS", "function": "ks / macro_ks", "functional": "sup_t |F_a - F_b|; mean over cells; pooled-resample p",
     "tau": "none", "source": "PoC §3.2 p.6; PhAIL §3.3 p.6"},
    {"name": "P-P plot", "function": "pp_curve / macro_pp_curve", "functional": "(F_ref(t), F_model(t)), t in [0, τ]",
     "tau": "plot range τ", "source": "PoC §3.2 p.6; PhAIL §3.2 p.5"},
    {"name": "ghost-event rate", "function": "ghost_event_rate", "functional": "KM mass at T = inf = 1 - F(t -> inf)",
     "tau": "none", "source": "PoC §3.2 p.6; PhAIL §3.1 p.4"},
    {"name": "censored rate", "function": "censored_rate", "functional": "share of episodes ending in a censored pair",
     "tau": "τ_episode (protocol)", "source": "PoC §3.2 p.6, §3.4 p.7; PhAIL §3.1 p.4"},
    {"name": "clustered bootstrap CI", "function": "cluster_bootstrap / hrt_ci / F_band", "functional": "percentile CI, episodes resampled within cells",
     "tau": "as the scalar", "source": "PoC §3.2 p.6; PhAIL §3.2 p.5"},
    {"name": "stratified logrank (sanity)", "function": "logrank_stratified", "functional": "Mantel-Haenszel over cells",
     "tau": "none", "source": "PhAIL §3.3 p.6"},
]


def metric_registry() -> list[dict]:
    """``METRICS`` with each function's first docstring line, checked to exist in this module."""
    g = globals()
    out = []
    for m in METRICS:
        fns = [f.strip() for f in m["function"].split("/")]
        missing = [f for f in fns if f not in g]
        if missing:
            raise KeyError(f"registry names missing functions {missing}")
        out.append({**m, "doc": " | ".join((g[f].__doc__ or "").strip().splitlines()[0] for f in fns)})
    return out


# ---------------------------------------------------------------------------------------------
# figures (matplotlib imported lazily)
# ---------------------------------------------------------------------------------------------


def plot_km(ax, eps: list[Episode], tau: float, label: str, color: str, n_boot: int = 500, seed: int = 0) -> None:
    """KM CDF on [0, τ] with its pointwise episode-clustered 95% band (PhAIL Fig. 2 left)."""
    grid = np.linspace(0, tau, 400)
    F, lo, hi = F_band(eps, grid, n_boot, np.random.default_rng(seed))
    ax.step(grid, F, where="post", color=color, label=label)
    ax.fill_between(grid, lo, hi, step="post", color=color, alpha=0.2, linewidth=0)
    ax.set_xlabel("time-to-success T (s)")
    ax.set_ylabel("F(t) = P(T <= t)")
    ax.set_ylim(0, 1)


def plot_pp(ax, model_eps: list[Episode], ref_eps: list[Episode], tau: float, color: str, label: str) -> float:
    """P-P plot (F_ref, F_model) with the diagonal; returns the censored AUC P(T_model < T_ref) (Fig. 2 right)."""
    x, y = pp_curve(model_eps, ref_eps, tau)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, 0, y, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("F_ref(t)")
    ax.set_ylabel("F_model(t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return auc(model_eps, ref_eps)


__all__ = ["Episode", "episode_from_ops", "libero_episode", "collect_TE", "km", "F_curve", "F_at", "median_T", "rmst",
           "completion", "uph", "uph_conditional", "mtbfa", "ghost_event_rate", "ghost_fraction", "censored_rate",
           "macro", "pooled", "hrt_cells", "hrt", "auc_censored", "auc", "macro_auc", "pp_curve", "macro_pp_curve", "ks",
           "macro_ks", "logrank_stratified", "cluster_bootstrap", "hrt_ci", "F_band", "uph_mtbf_sweep", "METRICS",
           "metric_registry", "plot_km", "plot_pp"]
