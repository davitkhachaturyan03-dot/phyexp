"""Third-party code vendored verbatim (unmodified), with its license.

``phail_stats.py``
    ``build/stats.py`` of https://github.com/Positronic-Robotics/phail-paper at commit
    ``18ce72d5703dcbbbb10a980336aa5a1622601fb4`` (2026-05-27), **unmodified, byte-identical**
    (sha256 ``802387d343463cd46a56e8d1dc4b4917b90c34b51fc72acce5b6e6cb915b46f2``).
    Copyright 2025-2026 Positronic Robotics. Licensed under the Apache License, Version 2.0; the full
    text is in ``LICENSE-phail-paper`` next to this file (Apache-2.0 §4(a)). The file is not modified,
    so §4(b) (notices on modified files) does not apply; the repo has no NOTICE file (§4(d)).

    It is the defining implementation of the PhAIL time-to-success primitives (PhAIL §3.1-3.3, pp.
    4-6; PoC §3.2 p. 6): T-pair extraction with censored tails and T=inf ghost events, Kaplan-Meier with
    Greenwood variance, KM median, RMST, KS distance, and the episode-clustered pooled-resample
    bootstrap for (macro-)KS. ``vlapoc.evaluation`` builds on it. Do not edit it: changes go into
    ``vlapoc.evaluation``, so this copy can always be re-checked against upstream with sha256.

``cupid_quality.py``
    ``sum_of_sum_influence`` and ``compute_demo_quality_scores`` from https://github.com/agiachris/cupid at commit
    ``2941eba9427a3398d3cbe26b10c01e3ece56bd18`` (``diffusion_policy/common/error_util.py``), function bodies verbatim.
    Copyright (c) 2025 Christopher Agia, MIT License, full text in ``LICENSE-cupid``. It is only the parity reference
    for ``vlapoc.attribution.performance_influence`` (training-attribution S2); no pipeline code imports it.
"""
