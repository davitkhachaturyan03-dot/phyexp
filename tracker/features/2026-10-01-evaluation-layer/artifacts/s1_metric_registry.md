| metric | function | functional of F | τ | source |
|---|---|---|---|---|
| F(τ) / Success@τ | `F_at` | F(τ) = P(T <= τ) | any τ (PoC §4.1: 30 s) | PhAIL Table 1 p.5; PoC §3.2 p.6, §4.1 p.9 |
| median | `median_T` | inf{t : F(t) >= 0.5} | none | PhAIL Table 1 p.5 |
| RMST(τ) | `rmst` | ∫_0^τ (1 - F(t)) dt, ghosts -> τ | integration cap τ | PhAIL Table 1 p.5, §3.1 p.4 |
| UPH | `uph` | 3600 / RMST(τ_episode)  (Table 1: 3600/E[T|T<τ_episode] ∝ 1/RMST) | τ_episode | PhAIL Table 1 p.5; build/fig_qq_tradeoff.py |
| completion | `completion` | F(τ_episode) | τ_episode | PhAIL Table 1 p.5 |
| MTBF/A | `mtbfa` | RMST(τ_episode) / (1 - F(τ_episode)) | τ_episode | PhAIL Table 1 p.5 |
| HRT | `hrt` | mean_o RMST_human,o(τ) / RMST_model,o(τ) | integration cap τ | PoC §3.2 p.6; PhAIL §3.2 p.5 |
| AUC vs reference | `auc / macro_auc` | P(T_model < T_ref) = area under P-P | none (censoring-aware pairs) | PoC §3.2 p.6, §4.1 p.9; PhAIL §3.2 p.5 |
| KS / macro-KS | `ks / macro_ks` | sup_t |F_a - F_b|; mean over cells; pooled-resample p | none | PoC §3.2 p.6; PhAIL §3.3 p.6 |
| P-P plot | `pp_curve / macro_pp_curve` | (F_ref(t), F_model(t)), t in [0, τ] | plot range τ | PoC §3.2 p.6; PhAIL §3.2 p.5 |
| ghost-event rate | `ghost_event_rate` | KM mass at T = inf = 1 - F(t -> inf) | none | PoC §3.2 p.6; PhAIL §3.1 p.4 |
| censored rate | `censored_rate` | share of episodes ending in a censored pair | τ_episode (protocol) | PoC §3.2 p.6, §3.4 p.7; PhAIL §3.1 p.4 |
| clustered bootstrap CI | `cluster_bootstrap / hrt_ci / F_band` | percentile CI, episodes resampled within cells | as the scalar | PoC §3.2 p.6; PhAIL §3.2 p.5 |
| stratified logrank (sanity) | `logrank_stratified` | Mantel-Haenszel over cells | none | PhAIL §3.3 p.6 |
