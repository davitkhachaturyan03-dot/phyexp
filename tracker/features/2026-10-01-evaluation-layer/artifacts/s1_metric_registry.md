| metric | function | functional of F | τ | source |
|---|---|---|---|---|
| F(30 s) / Success@τ | `F_at` | F(30) = P(T <= 30 s) | 30 s | PoC §4.1 p.9, §4.3 p.10; PhAIL Table 1 p.5 |
| RMST(τ) | `rmst` | RMST(60) = ∫_0^60 (1 - F(t)) dt | 60 s = τ_episode | PoC §4.3 p.10; PhAIL Table 1 p.5 |
| AUC vs reference | `auc_ties_half / macro_auc_ties_half` | AUC_h = P(T < T_h) + ½ P(T = T_h) (Mann-Whitney; ties ½ because model and human share the 20 Hz grid) | none (humans all < 60 s) | PoC §4.3 p.10, §9.1 Step 3 p.19; PhAIL §3.2 p.5 |
| HRT | `hrt` | HRT(60) = (1/J) Σ_o RMST_h,o(60) / RMST_m,o(60) | 60 s | PoC §3.2 p.6, §4.3 p.10; PhAIL §3.2 p.5 |
| censored rate | `censored_rate / F_at` | 1 - F(60) (no ghosts on LIBERO) | 60 s = τ_episode | PoC §3.2 p.6, §3.4 p.7 |
| operation RMST, capped | `rmst_capped` | mean over operation samples of min(T_k, 60), unfinished attempt = 60 | 60 s | PoC §3.4 p.7, §3.5 A p.7 |
| operation RMST (descriptive) | `rmst` | KM RMST(60) over operation pairs | 60 s | PoC §3.5 A p.7; PhAIL §3.1 p.4 |
| operation HRT (descriptive) | `hrt` | (1/J) Σ_o RMST_h,o / RMST_m,o over operation pairs | 60 s | PoC §3.5 A p.7 |
| F(30 s), PhAIL semantics | `F_at` | F(30) of pooled operation pairs | 30 s | PoC §4.1 p.9 |
| HRT(240 s), PhAIL semantics | `hrt` | (1/J) Σ_o RMST_teleop,o(240) / RMST_m,o(240) | 240 s | PhAIL §3.2 p.5 |
| global RMST(240 s), PhAIL semantics | `rmst` | RMST(240) of pooled operation pairs | 240 s | PoC §4.1 p.9 |
| AUC vs human, PhAIL semantics | `auc / macro_auc` | macro over objects of P(T_model < T_teleop) | none | PhAIL §3.2 p.5, §5.3 p.9 |
| median | `median_T` | inf{t : F(t) >= 0.5} | none | PhAIL Table 1 p.5 |
| UPH | `uph / uph_conditional` | 3600 / RMST(τ_episode) (adopted, as evaluation.uph and PhAIL build/fig_qq_tradeoff.py). Notes: Table 1's literal 3600 / E[T | T < τ_episode] (evaluation.uph_conditional) and common.py's per-episode 3600·S/T differ by up to 3× (S2 AC2.8) | τ_episode | PhAIL Table 1 p.5; S2 AC2.8 |
| completion | `completion` | F(τ_episode) | τ_episode | PhAIL Table 1 p.5 |
| MTBF/A | `mtbfa` | RMST(τ_episode) / (1 - F(τ_episode)) | τ_episode | PhAIL Table 1 p.5 |
| time-to-success T | `libero_episode / episode_from_ops` | the random variable every functional is taken of | τ_episode = 60 s | PoC §3.2 p.6, §3.5 A p.7 |
| Kaplan–Meier CDF F(t) | `km / F_curve` | F(t) = P(T <= t), estimated with censoring | up to τ | PoC §3.2 p.6 |
| ghost-event rate | `ghost_event_rate` | 1 - F(t -> inf) | none | PoC §3.2 p.6, §3.4 p.7 |
| KS / macro-KS | `ks / macro_ks` | sup_t |F_a - F_b| per cell; mean over cells; pooled-resample episode-clustered bootstrap p | none | PoC §3.2 p.6, §3.4 p.7; PhAIL §3.3 p.6 |
| P-P plot | `pp_curve / macro_pp_curve` | (F_ref(t), F_model(t)), t in [0, τ]; area = P(T_model < T_ref) | plot range | PoC §3.2 p.6 |
| clustered bootstrap CI | `cluster_bootstrap / hrt_ci / F_band` | percentile CI, whole episodes resampled within cells | as the scalar | PoC §3.2 p.6; PhAIL §3.2 p.5 |
| stratified logrank (sanity) | `logrank_stratified` | Mantel-Haenszel over cells | none | PhAIL §3.3 p.6 |
