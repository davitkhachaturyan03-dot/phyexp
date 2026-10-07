"""The single metric registry: every PoC metric, its declared functional and family, its per-rollout return (or why
none), the verdict metric that checks it, and its transfer status to LIBERO.

Grounding
---------
- PoC §3.4 (p. 7), "Declared targets. Every attribution result states which functional it explains"; §4.4 (p. 10)
  "Label every claim with its metric and time cap τ"; §4.3 (p. 10) the per-metric returns; §3.2 (p. 6) the
  primitives; PhAIL Table 1 (p. 5) the scalars as functionals of F.
- Brief owner decision 4 ("cover all metrics the PoC specifies ... where a metric doesn't transfer to simulation, say
  so"), owner decision 6 (no LIBERO ghost events).
- Evaluation-layer plan r2, "Declared functionals (r2)" (answer to analysis-r1 F-P2/F-P3/E2): **family W** (whole-task
  T; censoring only at τ_episode = 60 s, so min(T, 60) is observed for every rollout) is the declared attribution
  family for both suites, used by S6's per-rollout returns and S7's verdict; **family O** (per-operation T_k, PhAIL
  §3.1 pairs) is declared separately, with the attribution functional ``rmst60_op_cap`` (an unfinished attempt counts
  as reaching the cap) and the PhAIL-semantics operation KM functionals as *descriptive*; **family PhAIL** is PhAIL's
  own semantics (operation pairs with ghosts, τ = 240 s, AUC ties dropped), used only to reproduce PhAIL (S2, S7).

Where it sits (docs/dataflow.md): the shared vocabulary of Step 2 (Evaluation: which scalars describe a policy),
Step 3 (Attribution: which per-rollout return each metric gets) and Step 4 (Verification: which metrics the verdict
tests, in which family). Read it first: every column name in ``returns.parquet`` and every metric in a verdict
comes from an entry here.

This module is data only (no imports of the metric code), so ``vlapoc.evaluation`` (S1), ``vlapoc.returns`` (S6)
and ``vlapoc.verdict`` (S7) can all import it; ``metrics_registry.json`` (written with every returns run) is
``to_json()``. Fields per entry: ``key``, ``metric`` (display name), ``family`` (W / O / O-descriptive / PhAIL /
scoring-only / primitive / procedure / protocol / not-transferable), ``T`` (which time variable), ``tau``,
``censoring``, ``functional``, ``functions`` (``vlapoc.evaluation`` names), ``return_columns``, ``verdict`` (how S7
computes it, or None), ``identity`` (the return ↔ verdict relation), ``transfer_status``, ``reason``, ``source``.
"""

from __future__ import annotations

# The three time variables an entry's functional can be taken of (the ``T`` field); one per family.
T_W = "whole-task time to success T (s, 20 Hz sim time; LIBERO _check_success); censored only at τ_episode = 60 s"
T_O = "inter-completion time T_k of operation k (S4 rule; PhAIL §3.1 pairs; the unfinished attempt is the tail τ_end − t_last)"
T_P = "PhAIL operation pairs (T_i since the previous placement; ghosts (inf, 1); censored tail), wall clock"

ENTRIES: list[dict] = [
    # ---- family W: the declared attribution functionals (S6 returns, S7 verdict) -------------------------------
    {"key": "f30", "metric": "F(30 s) / Success@τ", "family": "W", "T": T_W, "tau": "30 s",
     "censoring": "at τ_episode = 60 s only (> 30 s)", "functional": "F(30) = P(T <= 30 s)", "functions": ["F_at"],
     "return_columns": ["ret_f30"], "verdict": {"compute": "F_at", "args": {"t": 30.0}, "aggregate": "pooled", "higher_is_better": True},
     "identity": "per cell: mean(ret_f30) = F(30)", "transfer_status": "built",
     "reason": "transfers literally: τ_episode = 60 s keeps 30 s an interior point of the CDF", "source": "PoC §4.1 p.9, §4.3 p.10; PhAIL Table 1 p.5"},
    {"key": "rmst60", "metric": "RMST(τ)", "family": "W", "T": T_W, "tau": "60 s = τ_episode",
     "censoring": "at τ_episode only, so min(T, 60) is observed for every rollout", "functional": "RMST(60) = ∫_0^60 (1 - F(t)) dt",
     "functions": ["rmst"], "return_columns": ["ret_rmst60"],
     "verdict": {"compute": "rmst", "args": {"tau": 60.0}, "aggregate": "pooled", "higher_is_better": False},
     "identity": "per cell: mean(ret_rmst60) = -RMST(60)", "transfer_status": "built", "reason": "negative capped time (§4.3)",
     "source": "PoC §4.3 p.10; PhAIL Table 1 p.5"},
    {"key": "auc_h", "metric": "AUC vs reference", "family": "W", "T": T_W + "; human: whole-task T of all 1000 v0 originals",
     "tau": "none (humans all < 60 s)", "censoring": "a censored rollout loses to every human time",
     "functional": "AUC_h = P(T < T_h) + ½ P(T = T_h) (Mann-Whitney; ties ½ because model and human share the 20 Hz grid)",
     "functions": ["auc_ties_half", "macro_auc_ties_half"], "return_columns": ["ret_auc_expected", "ret_auc_sampled"],
     "verdict": {"compute": "auc_ties_half", "args": {}, "aggregate": "macro", "higher_is_better": True},
     "identity": "per cell: mean(ret_auc_expected) = AUC_h; mean(ret_auc_sampled) = AUC_h up to sampling noise",
     "transfer_status": "built", "reason": "§9.1 Step 3 'whether the rollout beat a sampled human time' (sampled) and its expectation",
     "source": "PoC §4.3 p.10, §9.1 Step 3 p.19; PhAIL §3.2 p.5"},
    {"key": "hrt60", "metric": "HRT", "family": "W", "T": T_W + "; human: whole-task T of all 1000 v0 originals (fixed anchor)",
     "tau": "60 s", "censoring": "as rmst60", "functional": "HRT(60) = (1/J) Σ_o RMST_h,o(60) / RMST_m,o(60)",
     "functions": ["hrt"], "return_columns": ["ret_hrt60_lin"],
     "verdict": {"compute": "hrt", "args": {"tau": 60.0}, "aggregate": "macro", "higher_is_better": True},
     "identity": "Σ_cells mean(ret_hrt60_lin) = -HRT(60) at the evaluated policy (first-order weights w_o = RMST_h,o / (J RMST_m,o²))",
     "transfer_status": "built (linearized)", "reason": "a ratio: first-order (delta-method) per-rollout return (§4.3)",
     "source": "PoC §3.2 p.6, §4.3 p.10; PhAIL §3.2 p.5"},
    {"key": "censored60", "metric": "censored rate", "family": "W", "T": T_W, "tau": "60 s = τ_episode",
     "censoring": "the censored tail itself", "functional": "1 - F(60) (no ghosts on LIBERO)", "functions": ["censored_rate", "F_at"],
     "return_columns": ["ret_censored60"],
     "verdict": {"compute": "one_minus_F", "args": {"t": 60.0}, "aggregate": "pooled", "higher_is_better": False},
     "identity": "per cell: mean(ret_censored60) = 1 - F(60)", "transfer_status": "built",
     "reason": "timeouts are censored, not failed (§3.2); 'the censored tail' as a target (§3.4)", "source": "PoC §3.2 p.6, §3.4 p.7"},
    # ---- family O: per sub-task (§3.5 A) -----------------------------------------------------------------------
    {"key": "rmst60_op_cap", "metric": "operation RMST, capped", "family": "O", "T": T_O, "tau": "60 s",
     "censoring": "the unfinished attempt counts as reaching the cap (60 s): pessimistic by declaration",
     "functional": "mean over operation samples of min(T_k, 60), unfinished attempt = 60", "functions": ["rmst_capped"],
     "return_columns": ["ret_rmst60_op_cap"],
     "verdict": {"compute": "rmst_capped", "args": {"tau": 60.0}, "aggregate": "pooled", "higher_is_better": False},
     "identity": "per cell over operation samples: mean(ret_rmst60_op_cap) = -rmst60_op_cap",
     "transfer_status": "built", "reason": "§3.4 'operation-level (T, event) pairs give several outcome samples per rollout'; "
     "an unfinished attempt can never outrank a completed operation (analysis-r1 F-P3)", "source": "PoC §3.4 p.7, §3.5 A p.7"},
    {"key": "rmst60_op", "metric": "operation RMST (descriptive)", "family": "O-descriptive", "T": T_O, "tau": "60 s",
     "censoring": "PhAIL: the unfinished attempt is a right-censored pair", "functional": "KM RMST(60) over operation pairs",
     "functions": ["rmst"], "return_columns": [], "verdict": None, "identity": "none (no return: descriptive CDF, S4)",
     "transfer_status": "built (descriptive)", "reason": "PhAIL-semantics per-sub-task CDF (S4); not an attribution target",
     "source": "PoC §3.5 A p.7; PhAIL §3.1 p.4"},
    {"key": "hrt60_op", "metric": "operation HRT (descriptive)", "family": "O-descriptive", "T": T_O, "tau": "60 s",
     "censoring": "as rmst60_op", "functional": "(1/J) Σ_o RMST_h,o / RMST_m,o over operation pairs", "functions": ["hrt"],
     "return_columns": [], "verdict": None, "identity": "none (descriptive)", "transfer_status": "built (descriptive)",
     "reason": "per-sub-task view of HRT; not an attribution target", "source": "PoC §3.5 A p.7"},
    # ---- family PhAIL: PhAIL's own semantics, for reproduction only ----------------------------------------------
    {"key": "phail_f30", "metric": "F(30 s), PhAIL semantics", "family": "PhAIL", "T": T_P, "tau": "30 s", "censoring": "PhAIL §3.1",
     "functional": "F(30) of pooled operation pairs", "functions": ["F_at"], "return_columns": [],
     "verdict": {"compute": "F_at", "args": {"t": 30.0}, "aggregate": "pooled", "higher_is_better": True},
     "identity": "none (validation only)", "transfer_status": "PhAIL validation", "reason": "PoC §4.1 yardstick on PhAIL's data",
     "source": "PoC §4.1 p.9"},
    {"key": "phail_hrt240", "metric": "HRT(240 s), PhAIL semantics", "family": "PhAIL", "T": T_P, "tau": "240 s",
     "censoring": "PhAIL §3.1, ghosts -> τ", "functional": "(1/J) Σ_o RMST_teleop,o(240) / RMST_m,o(240)", "functions": ["hrt"],
     "return_columns": [], "verdict": {"compute": "hrt", "args": {"tau": 240.0}, "aggregate": "macro", "higher_is_better": True},
     "identity": "none (validation only)", "transfer_status": "PhAIL validation", "reason": "PhAIL headline scalar", "source": "PhAIL §3.2 p.5"},
    {"key": "phail_rmst240", "metric": "global RMST(240 s), PhAIL semantics", "family": "PhAIL", "T": T_P, "tau": "240 s",
     "censoring": "PhAIL §3.1, ghosts -> τ", "functional": "RMST(240) of pooled operation pairs", "functions": ["rmst"],
     "return_columns": [], "verdict": {"compute": "rmst", "args": {"tau": 240.0}, "aggregate": "pooled", "higher_is_better": False},
     "identity": "none (validation only)", "transfer_status": "PhAIL validation", "reason": "PoC §4.1 'global RMST'", "source": "PoC §4.1 p.9"},
    {"key": "phail_auc", "metric": "AUC vs human, PhAIL semantics", "family": "PhAIL", "T": T_P, "tau": "none",
     "censoring": "verified pairs only; ties dropped from the denominator (explore_pp_plot._auc_censored)",
     "functional": "macro over objects of P(T_model < T_teleop)", "functions": ["auc", "macro_auc"], "return_columns": [],
     "verdict": {"compute": "auc_ties_dropped", "args": {}, "aggregate": "macro", "higher_is_better": True},
     "identity": "none (validation only)", "transfer_status": "PhAIL validation",
     "reason": "PhAIL's tie rule; not comparable with the LIBERO ties-½ AUC without saying so (plan R15)", "source": "PhAIL §3.2 p.5, §5.3 p.9"},
    # ---- scoring-only scalars (PhAIL Table 1) --------------------------------------------------------------------
    {"key": "median", "metric": "median", "family": "scoring-only", "T": "per family", "tau": "none", "censoring": "KM",
     "functional": "inf{t : F(t) >= 0.5}", "functions": ["median_T"], "return_columns": [], "verdict": None,
     "identity": "none", "transfer_status": "scoring only", "reason": "no per-rollout return in the PoC", "source": "PhAIL Table 1 p.5"},
    {"key": "uph", "metric": "UPH", "family": "scoring-only", "T": "per family", "tau": "τ_episode", "censoring": "KM",
     "functional": "3600 / RMST(τ_episode) (adopted, as evaluation.uph and PhAIL build/fig_qq_tradeoff.py). Notes: Table 1's "
     "literal 3600 / E[T | T < τ_episode] (evaluation.uph_conditional) and common.py's per-episode 3600·S/T differ by up to 3× (S2 AC2.8)",
     "functions": ["uph", "uph_conditional"], "return_columns": [], "verdict": None, "identity": "none",
     "transfer_status": "projection only", "reason": "production UPH also needs reset and recovery time (PhAIL p.5); sim time excludes inference latency",
     "source": "PhAIL Table 1 p.5; S2 AC2.8"},
    {"key": "completion", "metric": "completion", "family": "scoring-only", "T": "per family", "tau": "τ_episode", "censoring": "KM",
     "functional": "F(τ_episode)", "functions": ["completion"], "return_columns": [], "verdict": None,
     "identity": "on family W equals 1 - mean(ret_censored60)", "transfer_status": "scoring only",
     "reason": "no separate return: 1 - censored rate when there are no ghosts", "source": "PhAIL Table 1 p.5"},
    {"key": "mtbfa", "metric": "MTBF/A", "family": "scoring-only", "T": "per family", "tau": "τ_episode", "censoring": "KM",
     "functional": "RMST(τ_episode) / (1 - F(τ_episode))", "functions": ["mtbfa"], "return_columns": [], "verdict": None,
     "identity": "none", "transfer_status": "adapted, scoring only", "reason": "'failures/assists' become non-completions; no operator in simulation",
     "source": "PhAIL Table 1 p.5"},
    # ---- primitives, procedures, protocol ------------------------------------------------------------------------
    {"key": "T", "metric": "time-to-success T", "family": "primitive", "T": "W: " + T_W + " | O: " + T_O, "tau": "τ_episode = 60 s",
     "censoring": "right-censored at τ_episode", "functional": "the random variable every functional is taken of", "functions": ["libero_episode", "episode_from_ops"],
     "return_columns": [], "verdict": None, "identity": "none", "transfer_status": "built (adapted)",
     "reason": "simulated seconds at 20 Hz, not wall clock (see wall-clock T)", "source": "PoC §3.2 p.6, §3.5 A p.7"},
    {"key": "km", "metric": "Kaplan–Meier CDF F(t)", "family": "primitive", "T": "per family", "tau": "up to τ", "censoring": "KM with Greenwood variance",
     "functional": "F(t) = P(T <= t), estimated with censoring", "functions": ["km", "F_curve"], "return_columns": [], "verdict": None,
     "identity": "each family-W return is the per-rollout term of one functional of F", "transfer_status": "built",
     "reason": "vendored PhAIL kaplan_meier (T = inf allowed)", "source": "PoC §3.2 p.6"},
    {"key": "ghost_rate", "metric": "ghost-event rate", "family": "not-transferable", "T": T_P, "tau": "none", "censoring": "KM mass at T = inf",
     "functional": "1 - F(t -> inf)", "functions": ["ghost_event_rate"], "return_columns": [], "verdict": None, "identity": "none",
     "transfer_status": "not available on LIBERO (built and validated on PhAIL's data)",
     "reason": "LIBERO defines no unrecoverable-state rule (plan P13; owner decision 6): no ghost events exist", "source": "PoC §3.2 p.6, §3.4 p.7"},
    {"key": "ks", "metric": "KS / macro-KS", "family": "procedure", "T": "per verdict family", "tau": "none", "censoring": "KM CDFs",
     "functional": "sup_t |F_a - F_b| per cell; mean over cells; pooled-resample episode-clustered bootstrap p", "functions": ["ks", "macro_ks"],
     "return_columns": [], "verdict": None, "identity": "none", "transfer_status": "significance gate (S7)",
     "reason": "a two-sample test, never a ranker (PhAIL App. I)", "source": "PoC §3.2 p.6, §3.4 p.7; PhAIL §3.3 p.6"},
    {"key": "pp", "metric": "P-P plot", "family": "procedure", "T": "per family", "tau": "plot range", "censoring": "KM CDFs",
     "functional": "(F_ref(t), F_model(t)), t in [0, τ]; area = P(T_model < T_ref)", "functions": ["pp_curve", "macro_pp_curve"],
     "return_columns": [], "verdict": None, "identity": "none", "transfer_status": "visualization", "reason": "not a scalar", "source": "PoC §3.2 p.6"},
    {"key": "bootstrap", "metric": "clustered bootstrap CI", "family": "procedure", "T": "as the scalar", "tau": "as the scalar",
     "censoring": "n/a", "functional": "percentile CI, whole episodes resampled within cells", "functions": ["cluster_bootstrap", "hrt_ci", "F_band"],
     "return_columns": [], "verdict": None, "identity": "none", "transfer_status": "built", "reason": "a CI procedure, not a metric",
     "source": "PoC §3.2 p.6; PhAIL §3.2 p.5"},
    {"key": "logrank", "metric": "stratified logrank (sanity)", "family": "procedure", "T": "per verdict family", "tau": "none",
     "censoring": "KM risk sets", "functional": "Mantel-Haenszel over cells", "functions": ["logrank_stratified"], "return_columns": [],
     "verdict": None, "identity": "none", "transfer_status": "built (sanity line)", "reason": "PhAIL reports it as a sanity check",
     "source": "PhAIL §3.3 p.6"},
    {"key": "intervention", "metric": "intervention / safety-stop rate", "family": "not-transferable", "T": "n/a", "tau": "none", "censoring": "n/a",
     "functional": "share of rollouts with an operator stop", "functions": [], "return_columns": [], "verdict": None, "identity": "none",
     "transfer_status": "not transferable", "reason": "no operator in simulation", "source": "PoC §3.3 p.7"},
    {"key": "wallclock", "metric": "wall-clock T", "family": "protocol", "T": "n/a", "tau": "none", "censoring": "n/a",
     "functional": "time per operation", "functions": [], "return_columns": [], "verdict": None, "identity": "none",
     "transfer_status": "adapted", "reason": "simulated time at 20 Hz; excludes policy inference latency (PhAIL's wall clock includes it)",
     "source": "PoC §3.2 p.6"},
    {"key": "scheduling", "metric": "blind randomized scheduling", "family": "protocol", "T": "n/a", "tau": "none", "censoring": "n/a",
     "functional": "protocol", "functions": [], "return_columns": [], "verdict": None, "identity": "none", "transfer_status": "replaced",
     "reason": "fixed init states + seeds + a fixed batch plan give scene-level pairing PhAIL cannot have ('initial physical configurations are not replayed')",
     "source": "PoC §3.4 p.7; PhAIL §3.3 p.6"},
    {"key": "failure_types_not_built", "metric": "failure types: missed grasp, wrong object, ignored instruction, idle", "family": "not-transferable",
     "T": "n/a", "tau": "none", "censoring": "n/a", "functional": "rules over simulator state", "functions": [], "return_columns": [], "verdict": None,
     "identity": "none", "transfer_status": "not built",
     "reason": "LIBERO defines no such rule (P13); the catalog's exact-zero idle rule does not transfer to policy actions; raw signals are recorded so rules can be added without new rollouts",
     "source": "PoC §2.1 item 3 p.3, §9.1 Step 2 p.18"},
    {"key": "failure_type", "metric": "failure types: success / timeout_no_progress / timeout_partial_k_of_n", "family": "W",
     "T": T_W + "; operations from S4", "tau": "τ_episode", "censoring": "n/a", "functional": "operation rule (S4)", "functions": [],
     "return_columns": ["failure_type"], "verdict": None, "identity": "a subset key, not a return", "transfer_status": "built",
     "reason": "from goal-conjunct traces", "source": "PoC §9.1 Step 2-3 pp.18-19"},
]

VERDICT_FAMILIES = {"W": ["f30", "auc_h", "rmst60", "hrt60", "censored60"], "O": ["rmst60_op_cap"],
                    "PhAIL": ["phail_f30", "phail_hrt240", "phail_rmst240", "phail_auc"]}
FAMILY_TAU_S = {"W": 60.0, "O": 60.0, "PhAIL": 240.0}  # the verdict family's RMST/HRT cap τ (s), recorded with each verdict


def entry(key: str) -> dict:
    """The registry entry with this ``key`` (e.g. ``"rmst60"``); ``KeyError`` if there is none."""
    for e in ENTRIES:
        if e["key"] == key:
            return e
    raise KeyError(key)


def return_columns(family: str = "W") -> list[str]:
    """The per-rollout return columns of a family, in registry order (``failure_type`` excluded)."""
    return [c for e in ENTRIES if e["family"] == family and e["verdict"] is not None for c in e["return_columns"]]


def verdict_metrics(family: str) -> list[dict]:
    """The entries ``vlapoc.verdict.compare_policies`` tests in ``family`` ("W", "O" or "PhAIL"), in
    ``VERDICT_FAMILIES`` order. That order fixes the bootstrap matrix's columns and the Holm family."""
    return [entry(k) for k in VERDICT_FAMILIES[family]]


def to_json() -> list[dict]:
    """Shallow copies of every entry: the content of ``metrics_registry.json`` written with every returns run."""
    return [dict(e) for e in ENTRIES]


__all__ = ["ENTRIES", "VERDICT_FAMILIES", "FAMILY_TAU_S", "entry", "return_columns", "verdict_metrics", "to_json"]
