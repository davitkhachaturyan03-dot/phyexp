"""Round B for the planted data (PoC §9.2): the ground-truth-side removal arms and the verdict wording (dashboard S10). Ground-truth side:
reads the answer key; method code never imports it, and the arms it builds are refused by every method entry point.

Where this sits: after Round A (``scripts/server_pipeline.py --subset v3/candidate``) has produced the ensemble scores and the base policies,
``scripts/server_pipeline.py --round-b-planted`` retrains without each planted type, without the most harmful *clean* episodes, without clean
random episodes (the control) and, as a positive control, without every planted episode (the oracle, the ground-truth feature's GT-R5 arm).
``stage.py select-planted`` calls ``planted_select_run`` to build each child subset; ``stage.py train --groundtruth-arm`` trains on it;
``stage.py verify`` compares; ``stage.py discrepancy-verdict`` calls ``discrepancy_verdict_run`` to turn the verdicts into the sentences the
dashboard shows. Reading order: ``build_arm``, ``planted_select_run``, ``statuses``, ``discrepancy_verdict_run``.

Grounding
---------
- PoC §9.2 (p. 19): "episodes that look bad but do not hurt the policy, and episodes that look clean but do." The estimated half is
  ``vlapoc.discrepancy``; this is the causal half: only retraining can support "this bad data did not hurt this policy on this metric" or "this
  clean data did" (§9.1 Step 4 p. 19: "This step turns them from claims into verified results"; §2.6 p. 5; §2.1 item 5 p. 3).
- PoC §2.1 item 4 (p. 3) and owner decision 3: the answer key is read here, to build the arms, on the ground-truth path only. The subsets carry
  ``groundtruth: True`` in their derivation, so ``method_data.open_for_method`` refuses them, and the select and train runs are tagged
  ``vlapoc.groundtruth_arm=true``, which ``attribute_run``, ``features``, ``select --scores``, ``clusters`` and ``analyze`` refuse.
- PoC §4.4 (pp. 10-11): the two-step verdict (macro-KS gate, then per-metric CIs, Holm) is the evidence; §2.3 (p. 4): 2-3 seeds.
- CUPID Fig. 4 caption (p. 8): quality measures "degrade performance when higher-quality demonstrations induce brittle strategies".
- Ground-truth runbook GT-R5 (``tracker/features/2026-10-15-ground-truth/artifacts/server-runbook.md``): the ``oracle_drop`` arm trained directly.

The arms (all children of ``v3/candidate``; a type is exactly one episode in each of 13 tasks, measured in the dashboard plan)
-------------------------------------------------------------------------------------------------------------------
``type-<T>``: minus the 13 episodes of planted type T. ``cleanharm-<m>``: minus the 13 most harmful **clean** episodes by ``harm_<m>__same_task``,
at most one per task. ``cleanrandom<r>``: minus 13 clean episodes, one from each of 13 seeded-random tasks (the matched control of a type arm).
``oracle``: minus all 91 planted episodes. ``cleanrandom91-<r>``: minus clean episodes with the oracle's per-task counts (the oracle's control).

What the evidence may say (the wording rule, stated before any result; inferred design, the owner may override)
--------------------------------------------------------------------------------------------------------------
Per (arm, metric), over the ``curated_vs_random`` verdicts (family W, every control and seed), with the S4 confirmation rule (``vlapoc.confirm``):
- ``harmful (confirmed)``: removing the arm beat removing the same number of clean episodes: expectation met in >= ceil(2n/3) records, none contradicting.
- ``helpful (confirmed)``: the same, in the opposite direction (planted but helpful).
- ``no detectable harm, bounded``: no record significant in the harmful direction **and** the round has a positive control (some arm confirmed
  on some metric). The sentence gives the union of the CIs and says "This is not proof of no harm."
- ``inconclusive``: everything else, and all arms when no arm in the round shows an effect ("no arm showed an effect; with 13 episodes (1.4% of
  the data) per arm the test may lack power").
"Clean but harmful" is claimed only for ``cleanharm-<m>`` arms that are ``harmful (confirmed)``. The word that would claim the opposite of
"harmful" never appears in any output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.confirm import status_of
from vlapoc.recovery import EXPECTED_METRICS

ARM_KINDS = ("type", "cleanharm", "cleanrandom", "oracle", "cleanrandom_oracle")
EFFECT_KINDS = ("type", "cleanharm", "oracle")  # arms whose removal is expected to help (harmful data removed)
TASK = ["suite", "libero_task_id"]
STATUSES = ("harmful (confirmed)", "helpful (confirmed)", "no detectable harm, bounded", "inconclusive")


def _clean_and_tasks(parent_eps: pd.DataFrame, labelled: set[str]) -> pd.DataFrame:
    df = parent_eps[["episode_id", *TASK]].copy()
    df["task"] = df[TASK].astype(str).agg("/".join, axis=1)
    df["planted"] = df["episode_id"].isin(labelled)
    return df.sort_values("episode_id").reset_index(drop=True)


def build_arm(parent, kind: str, labels: pd.DataFrame, *, corruption_type: str | None = None, scores: pd.DataFrame | None = None,
              metric: str | None = None, seed: int = 0, n: int = 13, scope: str = "same_task", lineage_root: Path | None = None):
    """The child subset of one planted arm and its parameters ``(child, params)`` (module docstring). ``labels`` is the answer key
    (``groundtruth.load_answer_key``). ``cleanharm`` needs ``scores`` (an ensemble or seed score table keying ``parent``) and ``metric``.
    The derivation carries ``groundtruth: True``, so ``open_for_method`` refuses the child."""
    from vlapoc.lineage import check_score_keys, drop_ids

    lab = set(labels["episode_id"]) & set(parent.episode_ids)
    eps = _clean_and_tasks(parent.episodes(lineage_root), lab)
    clean = eps[~eps["planted"]]
    base = {"groundtruth": True, "arm_kind": kind}
    if kind == "type":
        ids = sorted(set(labels.loc[labels["corruption_type"] == corruption_type, "episode_id"]) & set(parent.episode_ids))
        if not ids:
            raise ValueError(f"no episodes of type {corruption_type!r} in the parent subset")
        return drop_ids(parent, ids, "groundtruth_type_drop", {**base, "corruption_type": corruption_type}, lineage_root), {"corruption_type": corruption_type, "n": len(ids)}
    if kind == "cleanharm":
        if scores is None or not metric:
            raise ValueError("cleanharm needs a score table and a metric")
        col = f"harm_{metric}__same_task" if scope == "same_task" else f"harm_{metric}"  # "all": the all-rollout column (a demo whose rollouts cover few tasks)
        check_score_keys(parent, scores, col, lineage_root)
        t = scores[["episode_id", col]].merge(clean[["episode_id", "task"]], on="episode_id")
        best = t.sort_values([col, "episode_id"], ascending=[False, True]).drop_duplicates("task")  # the most harmful clean episode of each task
        pick = best.head(n)
        if len(pick) < n:
            raise ValueError(f"only {len(pick)} tasks have clean episodes, {n} needed")
        return (drop_ids(parent, list(pick["episode_id"]), "groundtruth_cleanharm_drop", {**base, "metric": col, "n_tasks": n}, lineage_root),
                {"metric": col, "n": n, "tasks": sorted(pick["task"])})
    if kind == "cleanrandom":
        rng = np.random.default_rng(seed)
        tasks = sorted(clean["task"].unique())
        chosen = [tasks[i] for i in sorted(rng.choice(len(tasks), size=n, replace=False))]
        ids = [rng.choice(sorted(clean.loc[clean["task"] == t, "episode_id"])) for t in chosen]
        return (drop_ids(parent, [str(i) for i in ids], "groundtruth_cleanrandom_drop", {**base, "seed": seed, "n_tasks": n}, lineage_root),
                {"seed": seed, "n": n, "tasks": chosen})
    if kind == "oracle":
        ids = sorted(lab)
        return drop_ids(parent, ids, "oracle_drop", {**base, "dropped_note": "every labelled episode (GT-R5)"}, lineage_root), {"n": len(ids)}
    if kind == "cleanrandom_oracle":
        rng = np.random.default_rng(seed)
        per_task = eps[eps["planted"]].groupby("task").size()
        ids = []
        for t, k in sorted(per_task.items()):
            pool = sorted(clean.loc[clean["task"] == t, "episode_id"])
            ids += [pool[i] for i in rng.choice(len(pool), size=int(k), replace=False)]
        return (drop_ids(parent, ids, "groundtruth_cleanrandom_match_drop", {**base, "seed": seed, "per_task_counts": {t: int(k) for t, k in per_task.items()}}, lineage_root),
                {"seed": seed, "n": len(ids), "per_task_counts": {t: int(k) for t, k in per_task.items()}})
    raise ValueError(f"kind must be one of {ARM_KINDS}")


def expected_for_type(corruption_type: str | None) -> dict | None:
    """The expectation stated before any result (AC10.6): removing harmful data should leave the treated policy better. If the answer key's
    hypothesis names metrics (``recovery.EXPECTED_METRICS``, declared before any score existed: idle_padding -> time-based, gripper_release ->
    drops) the expectation applies to those verdict metrics only (``{metric: "treated_better"}``); otherwise ``None`` = every metric
    ``treated_better`` (the select run's ``direction = highest``)."""
    from vlapoc.metrics_registry import verdict_key_of

    named = EXPECTED_METRICS.get(corruption_type) if corruption_type else None
    return {verdict_key_of(f"harm_{m}"): "treated_better" for m in named} if named else None


def planted_select_run(parent: str, name: str, kind: str, *, corruption_type: str | None = None, scores_run_id: str | None = None,
                       metric: str | None = None, seed: int = 0, n: int = 13, scope: str = "same_task", lineage_root: Path | None = None, chain: str | None = None,
                       allow_dirty: bool = False) -> tuple[str, object]:
    """Stage ``select`` for a planted arm (ground-truth side): ``build_arm`` on ``parent`` (a subset of a version with an answer key), recorded as
    one MLflow run tagged ``vlapoc.groundtruth_arm=true`` whose output is the child subset. Params: ``op``, ``arm_kind``, ``groundtruth_arm``,
    ``corruption_type``, ``n_dropped``, ``child_n_episodes``, ``direction`` (``highest`` for arms expected to help), ``expected_by_metric``
    (where the hypothesis names metrics), ``hypothesis``. Refuses (before any run) a parent whose version has no answer key. Returns ``(run_id, child)``."""
    from vlapoc import results
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import Subset, load_names
    from vlapoc.runs import stage_run

    ps = Subset.load(parent, lineage_root)
    vname = next(k for k, v in load_names(lineage_root)["versions"].items() if v == ps.dataset_version_id)
    labels = load_answer_key(vname, lineage_root)
    scores, src = None, None
    if kind == "cleanharm":
        scores = results.score_table(scores_run_id, lineage_root)
        src = results._get_run(scores_run_id, lineage_root).data.tags["output.attribution_scores.sha256"]
    child, p = build_arm(ps, kind, labels, corruption_type=corruption_type, scores=scores, metric=metric, seed=seed, n=n, scope=scope, lineage_root=lineage_root)
    hyp = labels.groupby("corruption_type")["hypothesis"].first().to_dict().get(corruption_type) if "hypothesis" in labels and corruption_type else None
    expected = expected_for_type(corruption_type) if kind == "type" else None
    with stage_run("select", name, lineage_root, chain=chain, allow_dirty=allow_dirty, extra_tags={"vlapoc.groundtruth_arm": "true"}) as r:
        r.input_subset(ps, "parent", parent if "/" in parent else None)
        if src:
            r.input_entity("score_table", src, "score_run_table")
        r.params({"op": child.derivation["op"], "arm_kind": kind, "groundtruth_arm": True, "corruption_type": corruption_type or "", "metric": p.get("metric", metric or ""),
                  "seed": seed, "n_dropped": len(child.derivation["dropped"]), "child_n_episodes": len(child.episode_ids), "parent_subset_id": ps.subset_id,
                  "direction": "highest" if kind in EFFECT_KINDS else "", "hypothesis": hyp or "",
                  "curation": "groundtruth", "arm_detail": json.dumps(p, sort_keys=True, default=str)[:5000],
                  **({"expected_by_metric": expected} if expected else {})})
        r.tags({"select.dropped": json.dumps(child.derivation["dropped"])[:7900]})
        r.produced("subset", child.subset_id, "subset")
        return r.run_id, child


# ---------------------------------------------------------------------------------------------
# the verdict wording
# ---------------------------------------------------------------------------------------------


def treated_advantage(row: dict) -> tuple[float, float, float]:
    """``(advantage, lo, hi)`` of the treated arm B over the control A in "better" units, from one verdict metric row: ``diff`` and ``ci95`` are
    A minus B in raw units, ``diff_better`` is sign-corrected (> 0: A better). So B's advantage is ``-diff_better`` and its interval is
    ``-sgn * ci95`` with ``sgn = +1`` for a higher-is-better metric (``higher_is_better``; else inferred from ``diff_better / diff``)."""
    sgn = 1.0 if row.get("higher_is_better", True) else -1.0
    if "higher_is_better" not in row and row.get("diff"):
        sgn = math.copysign(1.0, row["diff_better"] / row["diff"]) if row["diff_better"] else 1.0
    lo, hi = sorted(-sgn * c for c in row["ci95"])
    return -float(row["diff_better"]), float(lo), float(hi)


def statuses(records: list[dict], arms: dict[str, dict], n_total: int | None = None, synthetic: bool = False) -> pd.DataFrame:
    """One row per (arm, metric) over the family-W ``curated_vs_random`` verdicts of the planted effect arms (``type``, ``cleanharm``, ``oracle``):
    ``status`` (``STATUSES``), the wording ``sentence``, the by-record calls and the CI summary. ``records`` / ``arms`` are ``step4.collect``'s.
    The positive control and the all-null rule are in the module docstring."""
    from vlapoc.step4 import ALPHA, call

    mine = {a for a, d in arms.items() if d.get("groundtruth_arm") and d.get("arm_kind") in EFFECT_KINDS}
    by: dict = {}
    for r in records:
        if r["treated"] in mine and r["kind"] == "curated_vs_random" and r["verdict"]["family"] == "W":
            by.setdefault(r["treated"], []).append(r)
    rows = []
    for arm, recs in sorted(by.items()):
        recs = sorted(recs, key=lambda r: (r["control"], int(r["seed"])))
        for m in recs[0]["verdict"]["metrics"]:
            calls = [call(r["verdict"]["metrics"][m], r["verdict"].get("alpha", ALPHA), bool(r["verdict"]["gate"]["fires"])) for r in recs]
            adv = [treated_advantage(r["verdict"]["metrics"][m]) for r in recs]
            n_confirm_h = status_of(calls, "T")[0] == "confirmed"
            n_confirm_p = status_of(calls, "C")[0] == "confirmed"
            rows.append({"arm": arm, "arm_kind": arms[arm]["arm_kind"], "corruption_type": arms[arm].get("corruption_type") or "", "metric": m,
                         "confirmed_harmful": n_confirm_h, "confirmed_helpful": n_confirm_p, "any_T": calls.count("T") > 0, "calls": ",".join(calls), "n_records": len(recs),
                         "n_T": calls.count("T"), "n_C": calls.count("C"), "n_gate_fired": sum(bool(r["verdict"]["gate"]["fires"]) for r in recs),
                         "advantage_mean": float(np.mean([a[0] for a in adv])), "ci_lo_union": float(min(a[1] for a in adv)), "ci_hi_union": float(max(a[2] for a in adv)),
                         "n_dropped": int(arms[arm].get("n_dropped") or 0), "controls": ",".join(sorted({r["control"] for r in recs})),
                         "verify_run_ids": json.dumps([r["run_id"] for r in recs])})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    positive = bool((t["confirmed_harmful"] | t["confirmed_helpful"]).any())
    pos_rows = t[t["confirmed_harmful"] | t["confirmed_helpful"]]
    pos_row = (pos_rows[pos_rows["arm_kind"] == "oracle"] if (pos_rows["arm_kind"] == "oracle").any() else pos_rows).iloc[0] if positive else None
    big = positive and pos_row["arm_kind"] == "oracle"
    pos_text = ((f"The positive control in this round ({pos_row['arm']}) removed {int(pos_row['n_dropped'])} episodes, far more than a single type's, so its effect does not show that a "
                 "13-episode arm has the power to show one. " if big else f"The positive control in this round ({pos_row['arm']}) removed {int(pos_row['n_dropped'])} episodes. ")
                if positive else "")
    shared = "The clean control is one fixed draw shared by every type and seed, so the runs are not independent replicates of it. "
    out_status, out_sentence = [], []
    for r in t.itertuples():
        k = f"{r.n_T} of {r.n_records}"
        pct = f"{r.n_dropped} episodes ({100 * r.n_dropped / n_total:.1f}% of the data)" if n_total and r.n_dropped else "this many episodes"
        what = {"type": f"the {r.n_dropped} {r.corruption_type} episodes", "cleanharm": f"the {r.n_dropped} most harmful clean episodes", "oracle": f"all {r.n_dropped} planted episodes"}[r.arm_kind]
        if r.confirmed_harmful:
            st = STATUSES[0]
            sent = (f"Removing {what} improved {r.metric} by {r.advantage_mean:.3g} [{r.ci_lo_union:.3g}, {r.ci_hi_union:.3g}] relative to removing the same number of clean "
                    f"episodes ({r.n_T} of {r.n_records} runs).")
        elif r.confirmed_helpful:
            st = STATUSES[1]
            sent = (f"Removing {what} made {r.metric} worse by {-r.advantage_mean:.3g} [{-r.ci_hi_union:.3g}, {-r.ci_lo_union:.3g}] relative to removing the same number of clean "
                    f"episodes ({r.n_C} of {r.n_records} runs): these episodes helped on this metric.")
        elif positive and not r.any_T:
            st = STATUSES[2]
            sent = (f"No detectable harm on {r.metric}: removing {what} did not improve it relative to removing the same number of clean episodes in any of "
                    f"{r.n_records} runs; the 95% CI on the difference spanned [{r.ci_lo_union:.3g}, {r.ci_hi_union:.3g}] (union over runs), so a benefit from removal "
                    f"larger than {r.ci_hi_union:.3g} on this metric is not supported by this experiment. {pos_text}{shared}This is not proof of no harm.")
        else:
            st = STATUSES[3]
            sent = (f"Inconclusive on {r.metric} for {what}: " + (f"{k} runs were significant in the harmful direction without meeting the rule. " if r.n_T else "")
                    + ("" if positive else f"no arm showed an effect; with {pct} per arm the test may lack power."))
        out_status.append(st)
        out_sentence.append(sent.strip())
    t["status_code"], t["positive_control_in_round"] = out_status, positive
    pre = "SYNTHETIC: " if synthetic else ""  # recorded synthetic verdicts say so in the very text the app shows verbatim
    t["status"] = [pre + x for x in out_status]
    t["sentence"] = [pre + x for x in out_sentence]
    t["claim_clean_but_harmful"] = (t["arm_kind"] == "cleanharm") & (t["status_code"] == STATUSES[0])
    return t


def discrepancy_verdict_run(tag: str, name: str, chain: str | None = None, out_dir: Path | None = None, lineage_root: Path | None = None,
                            allow_dirty: bool = False, run_chain: str | None = None, synthetic: bool = False, n_total: int | None = None) -> dict:
    """Stage ``discrepancy-verdict`` (recorded as stage ``summarize``; PoC §9.2 p. 19, §9.1 Step 4 p. 19): ``statuses`` of the planted arms of chain
    ``tag``, written as ``discrepancy_verdict.parquet`` / ``.json`` (status, sentence and verify run ids per arm and metric) and recorded as one
    run consuming every verdict. ``n_total`` (the base subset's size, default read from the first arm's select run) only words the all-null
    sentence. Refuses a chain without planted-arm verdicts. Returns ``{"run_id", "table", "rows", "positive_control"}``."""
    from vlapoc import step4
    from vlapoc.lineage import rel
    from vlapoc.runs import stage_run

    records, arms, chain_id = step4.collect(tag, chain, lineage_root)
    if n_total is None:
        a0 = next((d for d in arms.values() if d.get("groundtruth_arm") and d.get("child_n_episodes")), None)
        n_total = int(a0["child_n_episodes"]) + int(a0["n_dropped"]) if a0 else None
    table = statuses(records, arms, n_total, synthetic)
    if table.empty:
        raise ValueError(f"chain {chain_id} has no family-W curated_vs_random verdicts of planted arms")
    table["synthetic"] = bool(synthetic)
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path, jpath = out_dir / "discrepancy_verdict.parquet", out_dir / "discrepancy_verdict.json"
    table.to_parquet(path, index=False)
    jpath.write_text(json.dumps({"chain": chain_id, "synthetic": bool(synthetic), "positive_control_in_round": bool(table["positive_control_in_round"].iloc[0]),
                                 "rule": __doc__.split("What the evidence may say")[1][:2400], "rows": table.to_dict("records")}, indent=1, default=str, ensure_ascii=False))
    used = [r for r in records if r["treated"] in set(table["arm"]) and r["kind"] == "curated_vs_random"]
    with stage_run("summarize", name, lineage_root, chain=run_chain or chain_id, allow_dirty=allow_dirty,
                   extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        for rec in used:
            r.input_entity("verdict", rec["verdict_sha256"], f"verify.{rec['name']}"[:200])
        r.params({"kind": "discrepancy-verdict (Round B planted, PoC §9.2)", "tag": tag, "source_chain": chain_id, "n_arms": int(table["arm"].nunique()),
                  "positive_control_in_round": bool(table["positive_control_in_round"].iloc[0]), "n_verify_runs": len(used), "synthetic": bool(synthetic),
                  "statuses": {s: int((table["status_code"] == s).sum()) for s in STATUSES}})
        r.output_file("discrepancy_verdict", path, kind="table")
        r.output_file("discrepancy_verdict_json", jpath, kind="file")
        return {"run_id": r.run_id, "table": rel(path), "rows": len(table), "positive_control": bool(table["positive_control_in_round"].iloc[0])}
