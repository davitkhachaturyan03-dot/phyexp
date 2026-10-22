"""CUPID's pairwise-score aggregation and demo-quality score, vendored from the released code (MIT).

Source: https://github.com/agiachris/cupid at commit ``2941eba9427a3398d3cbe26b10c01e3ece56bd18``,
``diffusion_policy/common/error_util.py``: ``sum_of_sum_influence`` (lines 17-19) and ``compute_demo_quality_scores``
(lines 299-335), function bodies **verbatim** (only this header and the two imports are added).
Copyright (c) 2025 Christopher Agia, MIT License (``LICENSE-cupid`` next to this file).

Used only as the reference in ``scripts/verify_attribution.py`` check 7a (training-attribution S2, AC2.2):
``vlapoc.attribution.performance_influence(M, 2*r01 - 1, center=False) * m`` must equal
``compute_demo_quality_scores(<sum_of_sum trajectory scores>, success_mask, "net")``. Not imported by any pipeline code.
"""

from typing import Optional

import numpy as np


def sum_of_sum_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.sum(axis=1).sum()
    return s


def compute_demo_quality_scores(
    traj_scores: np.ndarray, 
    success_mask: np.ndarray, 
    metric: str = "net",
) -> Optional[np.ndarray]:
    """Compute demonstration quality scores. Higher predicted value means higher quality."""
    assert traj_scores.shape[0] == len(success_mask)

    has_succ = np.any(success_mask == True)
    has_fail = np.any(success_mask == False)
    
    demo_quality_scores = None
    if metric == "net":
        succ_scores = compute_demo_quality_scores(
            traj_scores=traj_scores, 
            success_mask=success_mask, 
            metric="succ"
        )
        fail_scores = compute_demo_quality_scores(
            traj_scores=traj_scores, 
            success_mask=success_mask, 
            metric="fail"
        )
        succ_scores = succ_scores if succ_scores is not None else 0.0
        fail_scores = fail_scores if fail_scores is not None else 0.0
        demo_quality_scores = succ_scores + fail_scores

    elif metric == "succ":
        demo_quality_scores = traj_scores[success_mask].sum(axis=0) if has_succ else None

    elif metric == "fail":    
        demo_quality_scores = -traj_scores[~success_mask].sum(axis=0) if has_fail else None
        
    else:
        raise ValueError(f"Quality metric {metric} is not supported.")
    
    return demo_quality_scores
