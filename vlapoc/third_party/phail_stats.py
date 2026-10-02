"""Survival-analysis primitives for the time-to-success framework (FINDINGS D3-D8).

Lifted from `build/markup/_legacy/cdf_analysis.py`, decoupled from its
EpisodeData class. Any object exposing `place_ts: list[float]`,
`duration_s: float`, `total_items: int` works as input to `episode_to_T_pairs`.

Functions:
    episode_to_T_pairs(ep)  -> list[(T, event)]   # D3 inter-success extraction
    kaplan_meier(times, events)  -> (t_grid, S, var_logS)
    km_step(t_grid, S, t_eval)   -> S at arbitrary points (right-continuous)
    km_median(t_grid, S)         -> first t with S(t) <= 0.5, or None
    rmst(t_grid, S, tau)         -> restricted mean survival time over [0, tau]
    ks_distance(t1, S1, t2, S2)  -> sup |F1 - F2|
    episode_cluster_bootstrap_ks(eps_a, eps_b, n_boot, rng) -> (D, p)
        single-pool KS: pool all episodes per arm, one KM, one sup.
    per_obj_macro_bootstrap_ks(eps_a_by_obj, eps_b_by_obj, n_boot, rng) -> (D_macro, p)
        per-object KS, macro-mean across objects; the test used by
        explore_efficiency.py and the framework section of the paper.
    pct_of_human(rmst_model, rmst_human) -> 100 * rmst_human / rmst_model
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _TSource(Protocol):
    place_ts: list[float]
    duration_s: float
    total_items: int
    n_inf_events: int  # drops_outside + (1 if safety-stopped). Default 0 if absent.


def episode_to_T_pairs(ep: _TSource, multi_grip_threshold_s: float = 0.3) -> list[tuple[float, int]]:
    """Return (T, event) pairs per D3. event=1 uncensored, event=0 right-censored,
    and event=1 with T=+inf for hard failures (drops_outside + safety_stops).

    A right-censored tail (last place -> episode_end) is emitted only when
    items remained to be placed (len(place_ts) < total_items). Episodes that
    placed every item ('Success') had no further attempt awaited, so the
    residual recording time is not a censored 'time to next success'.

    **Hard failures as T=inf** (per FINDINGS D2/D4 update):
    - drops_outside_workspace: each item physically lost (gone from the scene)
      contributes one (inf, 1) ghost event — the operation can never succeed.
    - safety_stop (eval.outcome == 'Safety'): episode terminated by operator
      due to dangerous behavior. Contributes one (inf, 1) ghost event for the
      operation that triggered the stop.
    - **Stalls and timeouts are NOT T=inf**: the model was simply slow / idle.
      They live in the censored tail (model could in principle have succeeded
      with more time). This is the user-confirmed framing.

    The caller supplies the count via `ep.n_inf_events`. Callers that don't
    populate it (legacy callers) get 0 by default.

    Multi-grip handling: when the annotator marks one candidate with
    `n_successful=K`, `derive_place_timestamps_from_annotations` emits K
    identical timestamps (one drop placed K items at once). Each duplicate
    is a real success but takes ~0 additional time, so we emit `T=0.0` for
    the K-1 extras and DO NOT advance the inter-arrival anchor — the next
    distinct release's T is measured from the cluster's first ts. Two places
    closer than `multi_grip_threshold_s` are treated as one such cluster,
    which also covers near-duplicate timestamps that round to slightly
    different values.
    """
    out: list[tuple[float, int]] = []
    n_inf = int(getattr(ep, 'n_inf_events', 0) or 0)
    if not ep.place_ts:
        if ep.total_items > 0:
            out.append((ep.duration_s, 0))
    else:
        prev_anchor = 0.0
        last_place: float | None = None
        for p in sorted(ep.place_ts):
            if last_place is not None and (p - last_place) < multi_grip_threshold_s:
                # Multi-grip extra: same physical drop, ~0 extra time.
                out.append((0.0, 1))
            else:
                out.append((p - prev_anchor, 1))
                prev_anchor = p
            last_place = p
        fully_completed = len(ep.place_ts) >= ep.total_items > 0
        if not fully_completed and last_place is not None:
            tail = ep.duration_s - last_place
            if tail > 0:
                out.append((tail, 0))
    for _ in range(n_inf):
        out.append((float('inf'), 1))
    return out


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_grid, S_hat, var_logS) at distinct event times. Greenwood variance."""
    order = np.argsort(times, kind='stable')
    t_sorted = times[order]
    e_sorted = events[order]
    n = len(t_sorted)

    distinct: list[float] = []
    d_list: list[int] = []
    n_at_risk: list[int] = []
    at_risk = n
    i = 0
    while i < n:
        t_j = t_sorted[i]
        d_events = 0
        d_total = 0
        while i < n and t_sorted[i] == t_j:
            d_events += int(e_sorted[i])
            d_total += 1
            i += 1
        if d_events > 0:
            distinct.append(t_j)
            d_list.append(d_events)
            n_at_risk.append(at_risk)
        at_risk -= d_total

    if not distinct:
        return np.array([0.0]), np.array([1.0]), np.array([0.0])

    t_grid = np.array(distinct)
    d = np.array(d_list, dtype=float)
    nrisk = np.array(n_at_risk, dtype=float)
    S = np.cumprod(1.0 - d / nrisk)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(nrisk * (nrisk - d) > 0, d / (nrisk * (nrisk - d)), 0.0)
    var_logS = np.cumsum(terms)
    return t_grid, S, var_logS


def km_step(t_grid: np.ndarray, S: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Step-function evaluation of S at arbitrary t_eval (right-continuous)."""
    out = np.ones_like(t_eval, dtype=float)
    if len(t_grid) == 0:
        return out
    idx = np.searchsorted(t_grid, t_eval, side='right') - 1
    out[idx >= 0] = S[idx[idx >= 0]]
    return out


def km_median(t_grid: np.ndarray, S: np.ndarray) -> float | None:
    """Smallest t with S(t) <= 0.5; None if KM never crosses 0.5."""
    below = np.where(S <= 0.5)[0]
    if len(below) == 0:
        return None
    return float(t_grid[below[0]])


def rmst(t_grid: np.ndarray, S: np.ndarray, tau: float) -> float:
    """Restricted mean survival time = integral_0^tau S(t) dt over a step KM."""
    if tau <= 0:
        return 0.0
    pts = np.concatenate(([0.0], t_grid[t_grid < tau], [tau]))
    pts = np.unique(pts)
    pts.sort()
    seg_lens = np.diff(pts)
    seg_S = km_step(t_grid, S, pts[:-1])
    return float(np.sum(seg_S * seg_lens))


def ks_distance(t1: np.ndarray, S1: np.ndarray, t2: np.ndarray, S2: np.ndarray) -> float:
    """Sup |F1 - F2| evaluated on the union of event-time grids."""
    grid = np.unique(np.concatenate(([0.0], t1, t2)))
    F1 = 1.0 - km_step(t1, S1, grid)
    F2 = 1.0 - km_step(t2, S2, grid)
    return float(np.max(np.abs(F1 - F2)))


def _collect_T_E(eps: list[_TSource]) -> tuple[np.ndarray, np.ndarray]:
    T: list[float] = []
    E: list[int] = []
    for ep in eps:
        for t, ev in episode_to_T_pairs(ep):
            T.append(t)
            E.append(ev)
    return np.array(T), np.array(E)


def episode_cluster_bootstrap_ks(
    eps_a: list[_TSource],
    eps_b: list[_TSource],
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Episode-clustered bootstrap p-value for two-sample KS (D8).

    Returns (observed_ks, p_value). Resamples WITH replacement at the
    episode level, never per-operation, so within-episode correlation is
    preserved.
    """
    rng = rng or np.random.default_rng(0)
    T_a, E_a = _collect_T_E(eps_a)
    T_b, E_b = _collect_T_E(eps_b)
    t_a, S_a, _ = kaplan_meier(T_a, E_a)
    t_b, S_b, _ = kaplan_meier(T_b, E_b)
    obs = ks_distance(t_a, S_a, t_b, S_b)

    pooled = eps_a + eps_b
    n_a = len(eps_a)
    n_b = len(eps_b)
    count = 0
    for _ in range(n_boot):
        a_idx = rng.integers(0, len(pooled), size=n_a)
        b_idx = rng.integers(0, len(pooled), size=n_b)
        a_eps = [pooled[i] for i in a_idx]
        b_eps = [pooled[i] for i in b_idx]
        Ta, Ea = _collect_T_E(a_eps)
        Tb, Eb = _collect_T_E(b_eps)
        ta, Sa, _ = kaplan_meier(Ta, Ea)
        tb, Sb, _ = kaplan_meier(Tb, Eb)
        if ks_distance(ta, Sa, tb, Sb) >= obs:
            count += 1
    p = (count + 1) / (n_boot + 1)
    return obs, p


def per_obj_macro_bootstrap_ks(
    eps_a_by_obj: dict[str, list[_TSource]],
    eps_b_by_obj: dict[str, list[_TSource]],
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Per-object KS, macro-averaged across objects, with episode-clustered
    pooled-resample bootstrap p-value.

    For each object o present in both arms:
      - compute KM curves S_a^o, S_b^o on the object's operations,
      - compute D_o = sup_t |F_a^o(t) - F_b^o(t)|.
    The reported statistic is the equal-weight macro-mean D = (1/J) sum_o D_o.

    Under H_0: F_a^o = F_b^o for all o, the bootstrap pools each object's
    episodes from both arms and resamples with replacement at the episode
    level into fresh arm-A and arm-B episode lists of the original sizes,
    then recomputes the macro-mean. Returns (observed_macro_D, p_value).
    """
    rng = rng or np.random.default_rng(0)
    objs_both = sorted(set(eps_a_by_obj) & set(eps_b_by_obj))
    objs_both = [o for o in objs_both if eps_a_by_obj[o] and eps_b_by_obj[o]]
    if not objs_both:
        return 0.0, 1.0

    def _macro(a_by_o, b_by_o):
        vals = []
        for o in objs_both:
            T_a, E_a = _collect_T_E(a_by_o[o])
            T_b, E_b = _collect_T_E(b_by_o[o])
            if len(T_a) == 0 or len(T_b) == 0:
                continue
            t_a, S_a, _ = kaplan_meier(T_a, E_a)
            t_b, S_b, _ = kaplan_meier(T_b, E_b)
            vals.append(ks_distance(t_a, S_a, t_b, S_b))
        return float(np.mean(vals)) if vals else 0.0

    obs = _macro(eps_a_by_obj, eps_b_by_obj)

    pooled_per_obj = {o: eps_a_by_obj[o] + eps_b_by_obj[o] for o in objs_both}
    n_a_per_obj = {o: len(eps_a_by_obj[o]) for o in objs_both}
    n_b_per_obj = {o: len(eps_b_by_obj[o]) for o in objs_both}

    count = 0
    for _ in range(n_boot):
        a_resampled = {}
        b_resampled = {}
        for o in objs_both:
            pool = pooled_per_obj[o]
            a_idx = rng.integers(0, len(pool), size=n_a_per_obj[o])
            b_idx = rng.integers(0, len(pool), size=n_b_per_obj[o])
            a_resampled[o] = [pool[i] for i in a_idx]
            b_resampled[o] = [pool[i] for i in b_idx]
        if _macro(a_resampled, b_resampled) >= obs:
            count += 1
    p = (count + 1) / (n_boot + 1)
    return obs, p


def pct_of_human(rmst_model: float, rmst_human: float) -> float:
    """D5 headline scalar: % of human throughput.

    UPH ∝ 1 / E[T], so a smaller RMST means faster, which means higher %.
    Returns 100 * rmst_human / rmst_model. Falls back to 0 when rmst_model<=0.
    """
    if rmst_model <= 0:
        return 0.0
    return 100.0 * rmst_human / rmst_model
