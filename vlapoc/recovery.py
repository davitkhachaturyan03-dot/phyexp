"""Recovery of planted corruptions by an attribution table: type x harm-column AUROC / recall next to the best heuristic.

Ground-truth side (imports ``vlapoc.groundtruth``): this module reads the answer key and must never be imported by
``vlapoc.influence`` / ``vlapoc.attribution`` or any method code (owner decision 3; boundary scan in
``scripts/verify_training_attribution.py --recovery``).

Grounding
---------
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check whether your attribution
  method flags exactly those. This turns 'the method says X' into 'the method recovered 90% of the injected bad episodes.'"
- PoC §4.5 (p. 11), quoted: "Metric-targeted injections. Extend the injection test with corruptions designed to hit
  specific metrics — for example, idle or hesitant segments aimed at time-based metrics, and noisy gripper release near
  placement aimed at drops. Check whether attribution assigns each corrupted group to the expected metric. These mappings
  are hypotheses the experiment should confirm."
- Ground-truth feature, limitation D (owner): a method must beat each corruption type's best simple heuristic
  (``tracker/features/2026-10-15-ground-truth/artifacts/s5_v3_r3_bands.csv``).

Hypotheses declared BEFORE any attribution score exists (training-attribution S5, committed ahead of the scoring code)
-----------------------------------------------------------------------------------------------------------------------
``EXPECTED_METRICS`` maps a corruption type to the family-W metrics whose ``harm_`` column should rank that type's
episodes highest. Only the two mappings the PoC names are declared: ``idle_padding`` -> time-based metrics (``rmst60``,
``hrt60_lin``); ``gripper_release`` -> drops (``f30``, ``censored60``: a drop ends in no success within the cap). Every
other type has no PoC-stated mapping, so none is invented here: it is ``None`` and reported descriptively. The mapping is
a hypothesis; the output reports whether each type's best column matches it, never that it was "confirmed" by a
synthetic table.
"""

from __future__ import annotations

EXPECTED_METRICS: dict[str, tuple[str, ...] | None] = {
    "idle_padding": ("rmst60", "hrt60_lin"),  # PoC §4.5: "idle or hesitant segments aimed at time-based metrics"
    "gripper_release": ("f30", "censored60"),  # PoC §4.5: "noisy gripper release near placement aimed at drops"
    "action_noise": None,
    "detour": None,
    "dropped_frames": None,
    "instruction_swap": None,
    "time_shift": None,
}
