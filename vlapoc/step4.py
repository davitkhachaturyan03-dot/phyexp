"""Step 4 summary: one table over the verify runs of a chain, with a seed-consistency count per comparison.

Where this sits (docs/dataflow.md, Step 4 "Verification"): last stage of the chain. It reads the ``verify`` runs that
``scripts/stage.py verify`` wrote for each retrained arm and summarises them in one ``summarize`` run; it trains and evaluates nothing.
Reading order: ``collect`` (find the verdicts), ``call`` (T/C/0/g per metric and seed), ``summarize`` (one row per arm and comparison),
``summary_run`` (record it).

Grounding
---------
- PoC §9.1 Step 4 (p. 19), quoted: "1 Remove the top-k most harmful episodes and retrain. As a control, remove k random
  episodes and retrain. 2 Evaluate both policies as in Step 2. 3 Apply the macro-KS test ... 4 If it did, check each
  metric's confidence interval for the direction of the change. 5 Repeat for the top-k most helpful episodes (removing
  them should make things worse)." §2.6 (p. 5): "Remove the top-k 'harmful' episodes and retrain: the policy should
  improve more than after removing k random episodes. Remove the top-k 'helpful' episodes and retrain: the policy
  should degrade more than after random removal."
- PoC §4.5 (p. 11), "Same dataset, opposite curations": curate twice, once for HRT and once for reliability (F(30 s)),
  "retrain on each subset, and show that each curated policy wins on its own metric". The ``head_to_head`` comparison
  is that test.
- PoC §2.3 (p. 4): "2–3 seeds per data configuration". Each (arm, comparison) is verified once per training seed;
  this module counts, per metric, in how many seeds the Holm-corrected verdict went each way.
- PoC §3.4 (p. 7) and §4.4 (p. 10-11): every result carries the functional it explains. Each row carries the declared
  functional (column, metric, τ, rollout subset) of the select run that built the treated arm.

What it reads
-------------
``stage.py verify`` tags every Step 4 verify run ``step4.kind`` (``vs_base`` | ``curated_vs_random`` |
``head_to_head``), ``step4.treated`` (arm B, bare arm name such as ``harm-f30-k5``), ``step4.control`` (arm A; ``base`` for the original
policy) and ``step4.seed``, plus the chain tag ``vlapoc.chain``. The run's ``verdict.json`` (``vlapoc.verdict.compare_policies``)
gives, per family-W metric, ``p_holm`` and ``diff_better`` (> 0: arm A, the control, is better). The treated arm's select
run (``select-<tag>-<arm>``) gives its operation, direction, curation mode and declared functional.

Per metric and seed the call is ``T`` (gate fired, Holm p < α and the treated arm is better), ``C`` (gate fired, Holm p < α
and the control is better), ``0`` (gate fired, not significant) or ``g`` (the macro-KS gate did not fire: no directional
support, PoC §4.4 p. 10, §9.1 Step 4.4 p. 19; training-attribution round 2). The two-step verdict (macro-KS gate, then per-metric CIs) is PhAIL §3.3 as
``vlapoc.verdict`` implements it; this module aggregates, it does not re-test. It refuses verdicts from several chains
under one tag unless a chain id is given, and a verify run whose chain differs from the requested one.

Expected direction (stated before any result exists): removing the most harmful episodes should leave the treated policy
better than its control (``treated_better``); removing the most helpful should leave it worse (``treated_worse``);
for the HRT-vs-reliability head-to-head the HRT-curated arm should win on ``hrt60`` and the F(30)-curated arm on ``f30``.
Where no expectation applies the cell reads ``-``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

KINDS = ("vs_base", "curated_vs_random", "head_to_head")
ALPHA = 0.05  # the verdict's own alpha (compare_policies default); recorded per verdict as v["alpha"]


def expected_direction(kind: str, arm_direction: str | None, metric: str, treated: str, control: str,
                       expected_by_metric: dict | None = None) -> str | None:
    """The declared expectation for one metric (module docstring): ``treated_better``, ``treated_worse`` or ``None``.

    ``expected_by_metric`` (dashboard S4/S10): when the treated arm's select run recorded one (a ``drop_cluster`` arm: the estimated sign per
    metric whose interval excludes zero; a planted-type arm whose label names metrics), the expectation is exactly that mapping and every
    metric it does not name reads ``None``, for ``vs_base`` and ``curated_vs_random`` alike."""
    if expected_by_metric is not None and kind != "head_to_head":
        return expected_by_metric.get(metric)
    if kind == "head_to_head":
        if metric == "hrt60" and "hrt60" in treated:
            return "treated_better"
        if metric == "f30" and "f30" in control:
            return "treated_worse"  # the control (F(30)-curated) arm should win its own metric
        return None
    if arm_direction == "highest":
        return "treated_better"  # harmful removed
    if arm_direction == "lowest":
        return "treated_worse"  # helpful removed
    return None


def call(metric_row: dict, alpha: float = ALPHA, gate_fires: bool = True) -> str:
    """``T`` / ``C`` / ``0`` / ``g`` for one verdict metric row (``diff_better`` > 0 means arm A, the control, is better).

    The two-step verdict is gated: PoC §4.4 (p. 10) and §9.1 Step 4.3-4.4 (p. 19), "Apply the macro-KS test ... If it did
    [fire], check each metric's confidence interval for the direction of the change". A seed whose macro-KS gate did not fire
    (``gate_fires`` False) gets ``g`` whatever its per-metric Holm p-values say: it contributes no directional support in
    either direction (consistent with ``vlapoc.verdict.outcome``, which reports outcome "iii" and no direction then)."""
    if not gate_fires:
        return "g"
    if metric_row["p_holm"] >= alpha or metric_row["diff_better"] == 0:
        return "0"
    return "C" if metric_row["diff_better"] > 0 else "T"


def summarize(records: list[dict], arms: dict[str, dict]) -> pd.DataFrame:
    """The Step 4 table: one row per (treated arm, comparison kind, control, family).

    ``records``: one dict per verify run: ``kind``, ``treated``, ``control``, ``seed``, ``verdict`` (a ``verdict.json``).
    ``arms``: ``{arm name: {"op", "direction", "metric", "declaration"}}`` from the select runs (missing arms get blanks).
    Columns: identifiers; ``n_seeds``, ``seeds``, ``gate_fires`` (seeds whose macro-KS gate fired), ``outcome_by_seed``;
    per metric ``<m>__by_seed`` (T/C/0 in seed order), ``<m>__T``, ``<m>__C``, ``<m>__expected``, ``<m>__supports``
    (seeds whose significant call matches the expectation), ``<m>__consistency`` (text); and the declared functional.
    """
    df = pd.DataFrame([{"kind": r["kind"], "treated": r["treated"], "control": r["control"], "seed": int(r["seed"]),
                        "family": r["verdict"]["family"], "verdict": r["verdict"]} for r in records])
    rows = []
    for (treated, kind, control, family), g in df.groupby(["treated", "kind", "control", "family"], sort=True):
        g = g.sort_values("seed")
        arm = arms.get(treated, {})
        decl = arm.get("declaration") or {}
        row = {"arm": treated, "comparison": kind, "control": control, "family": family,
               "op": arm.get("op", ""), "direction": arm.get("direction", ""), "metric_column": arm.get("metric", ""),
               "curation": arm.get("curation", ""), "n_seeds": len(g), "seeds": ",".join(str(s) for s in g["seed"]),
               "gate_fires": f"{sum(v['gate']['fires'] for v in g['verdict'])}/{len(g)}",
               "outcome_by_seed": ",".join(v["outcome"] for v in g["verdict"]),
               "functional": decl.get("functional", ""), "tau": decl.get("tau", ""),
               "rollout_subset": decl.get("rollout_subset", ""), "good_sign": decl.get("good_sign", "")}
        keys = list(g["verdict"].iloc[0]["metrics"])
        for m in keys:
            calls = [call(v["metrics"][m], v.get("alpha", ALPHA), bool(v["gate"]["fires"])) for v in g["verdict"]]
            exp = expected_direction(kind, arm.get("direction"), m, treated, control, arm.get("expected_by_metric"))
            want = {"treated_better": "T", "treated_worse": "C"}.get(exp)
            n_t, n_c = calls.count("T"), calls.count("C")
            supports = sum(c == want for c in calls) if want else None
            row[f"{m}__by_seed"] = ",".join(calls)
            row[f"{m}__T"], row[f"{m}__C"] = n_t, n_c
            row[f"{m}__expected"] = exp or "-"
            row[f"{m}__supports"] = supports if supports is not None else "-"
            row[f"{m}__consistency"] = (f"treated better in {n_t}/{len(g)} seeds, control better in {n_c}/{len(g)}, gate not fired in {calls.count('g')}/{len(g)}"
                                        + (f"; expectation ({exp}) met in {supports}/{len(g)}" if want else ""))
        rows.append(row)
    return pd.DataFrame(rows)


def collect(tag: str, chain: str | None = None, lineage_root: Path | None = None) -> tuple[list[dict], dict[str, dict], str]:
    """Read the Step 4 verify runs of one chain: ``(records, arms, chain id)``.

    Refuses (``ValueError``) when the tag matches several chains and no ``chain`` is given, when nothing matches, and
    when a run's chain is not the requested one.
    """
    from vlapoc.attribution import _artifact_json
    from vlapoc.runs import client, experiment_id

    c = client(lineage_root)
    runs = [r for r in c.search_runs([experiment_id(lineage_root)], "tags.stage = 'verify' and attributes.status = 'FINISHED'",
                                     max_results=10000)
            if "step4.kind" in r.data.tags and r.data.tags.get("vlapoc.chain", "").startswith(f"{tag}-")]
    chains = sorted({r.data.tags["vlapoc.chain"] for r in runs})
    if chain is None:
        if len(chains) > 1:
            raise ValueError(f"tag {tag!r} matches verify runs of several chains {chains}; pass the chain id to summarize one")
        chain = chains[0] if chains else None
    if chain is None:
        raise ValueError(f"no FINISHED step4 verify runs under tag {tag!r}")
    runs = [r for r in runs if r.data.tags["vlapoc.chain"] == chain]  # only the requested chain's verdicts are summarized
    if not runs:
        raise ValueError(f"chain {chain!r} has no FINISHED step4 verify runs under tag {tag!r}")
    records = [{"kind": r.data.tags["step4.kind"], "treated": r.data.tags["step4.treated"], "control": r.data.tags["step4.control"],
                "seed": r.data.tags["step4.seed"], "run_id": r.info.run_id, "name": r.info.run_name,
                "verdict_sha256": r.data.tags["output.verdict.sha256"],
                "verdict": _artifact_json(r, "verdict", lineage_root)} for r in runs]
    arms = {}
    for treated in sorted({x["treated"] for x in records}):
        sel = c.search_runs([experiment_id(lineage_root)], f"tags.mlflow.runName = 'select-{tag}-{treated}' "
                            "and tags.stage = 'select' and attributes.status = 'FINISHED'", max_results=5)
        sel = [s for s in sel if s.data.tags.get("vlapoc.chain") == chain]
        if sel:
            p = sel[0].data.params
            decl = p.get("metric_declaration")
            ebm = p.get("expected_by_metric")
            arms[treated] = {"expected_by_metric": json.loads(ebm) if ebm and ebm.startswith("{") else None, "groundtruth_arm": p.get("groundtruth_arm") == "True", "cluster_run_id": p.get("cluster_run_id"), "cluster_id": p.get("cluster_id"),
                             "definition": p.get("definition"), "arm_kind": p.get("arm_kind"), "corruption_type": p.get("corruption_type"),
                             "n_dropped": p.get("n_dropped"), "child_n_episodes": p.get("child_n_episodes"), "op": p.get("op", ""), "direction": p.get("direction", ""), "metric": p.get("metric", ""),
                             "curation": p.get("curation", ""),
                             "declaration": json.loads(decl) if decl and decl.startswith("{") else None}
    return records, arms, chain


def summary_run(tag: str, name: str, chain: str | None = None, out_dir: Path | None = None, lineage_root: Path | None = None,
                allow_dirty: bool = False, run_chain: str | None = None) -> dict:
    """Stage ``summarize``: write ``step4_summary.csv`` / ``.json`` and record them as one MLflow run (PoC §9.1 Step 4
    p. 19). Returns ``{"run_id", "table", "rows", "n_verify_runs", "chain"}``."""
    from vlapoc.lineage import rel
    from vlapoc.runs import stage_run

    records, arms, chain_id = collect(tag, chain, lineage_root)
    table = summarize(records, arms)
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, js = out_dir / "step4_summary.csv", out_dir / "step4_summary.json"
    table.to_csv(csv, index=False)
    js.write_text(json.dumps({"tag": tag, "chain": chain_id, "alpha": ALPHA, "n_verify_runs": len(records),
                              "verify_run_ids": sorted(r["run_id"] for r in records), "rows": table.to_dict("records")}, indent=1, default=str))
    with stage_run("summarize", name, lineage_root, chain=run_chain or chain_id, allow_dirty=allow_dirty) as r:
        for rec in records:
            r.input_entity("verdict", rec["verdict_sha256"], f"verify.{rec['name']}"[:200])
        r.params({"tag": tag, "source_chain": chain_id, "n_verify_runs": len(records), "alpha": ALPHA, "rows": len(table),
                  "kinds": sorted({x["kind"] for x in records})})
        r.output_file("step4_summary_csv", csv, kind="table")
        r.output_file("step4_summary_json", js, kind="table")
        return {"run_id": r.run_id, "table": rel(csv), "rows": len(table), "n_verify_runs": len(records), "chain": chain_id}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m vlapoc.step4 --tag <t> --name <run name> [--chain <id>]``."""
    p = argparse.ArgumentParser(prog="python -m vlapoc.step4")
    p.add_argument("--tag", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--chain", default=None, help="full chain id (<tag>-<sha8>) when the tag matches several")
    p.add_argument("--lineage-root", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(summary_run(a.tag, a.name, a.chain, lineage_root=a.lineage_root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
