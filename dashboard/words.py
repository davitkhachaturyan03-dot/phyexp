"""Plain-language names for the dashboard (round 2, AC5.9): one place that turns metric keys, descriptors, groups and quadrants into words a
non-specialist can read. The declared functional of every metric stays reachable (a caption or tooltip next to the element; PoC §3.4 p. 7)."""

from __future__ import annotations

import re

METRICS = {"f30": "Success within 30 s", "rmst60": "Mean time to success (RMST)", "auc_expected": "Beats human (AUC)", "hrt60_lin": "Speed vs human (HRT)",
           "censored60": "Timed out (censored)"}
QUADRANTS = {"bad_and_harmful": "looks bad, hurts", "bad_but_not_harmful": "looks bad, no detectable harm", "clean_but_harmful": "looks clean, hurts",
             "clean_and_not_harmful": "looks clean, no detectable harm"}
GROUPS = {"consistent_harmer": "harms several metrics", "consistent_helper": "helps several metrics", "trade_off": "trade-off (helps some, harms others)",
          "metric_specific": "moves one metric", "neutral": "neutral"}
DESCRIPTORS = {"length_vs_task": "length vs the task's norm", "post_release_share_vs_task": "time after last release vs the task's norm", "duplicate_frame_pairs": "repeated frames",
               "instruction_mismatch": "instruction differs from the task's usual", "length_steps": "length (steps)", "duration_s": "duration (s)", "idle_fraction": "idle share",
               "idle_steps": "idle steps", "gripper_toggles": "gripper toggles", "smoothness": "roughness of actions", "task_name": "task", "libero_task_id": "task id",
               "language_instruction": "instruction", "collection_date": "collection date", "source_file": "source file"}
PLANTED = {"action_noise": "action noise", "detour": "detour", "dropped_frames": "dropped frames", "gripper_release": "gripper release", "idle_padding": "idle padding",
           "instruction_swap": "instruction swap", "time_shift": "time shift", "clean": "clean"}
DIRECTION = {"harm": "hurts", "boost": "helps", "mixed": "helps some, harms others", "none": "no shift"}
LENS = {"slice": "Rule-like slices", "label": "Label groups", "metric_group": "Four metric groups"}


def metric_name(col: str) -> str:
    """Short name of a score column (``harm_hrt60_lin``, ``z__f30``, ``hrt60``): ``Speed vs human (HRT)``."""
    base = re.sub(r"^(harm_|infl_|z__|harm__|quadrant__)", "", col).split("__")[0]
    base = {"hrt60": "hrt60_lin", "auc_h": "auc_expected"}.get(base, base)
    return METRICS.get(base, base.replace("_", " "))


_TOKENS = {**{k: v for k, v in METRICS.items()}, **GROUPS, **DESCRIPTORS, **PLANTED, **QUADRANTS}


def plain(text: str) -> str:
    """Display text with known identifiers replaced by words, then remaining underscores by spaces (task names, scene names). Only for display: ids and run names are
    shown in the lineage panel unchanged."""
    text = str(text)
    for k in sorted(_TOKENS, key=len, reverse=True):
        if "_" in k or k in ("f30", "clean", "detour", "neutral"):
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(k)}(?![A-Za-z0-9])", _TOKENS[k], text)
    return text.replace("_", " ")
