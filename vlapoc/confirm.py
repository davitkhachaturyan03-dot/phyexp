"""The ``confirm`` stage: turn Round B retraining verdicts into a status per discovered cluster and metric (dashboard S4).

Where this sits: after ``scripts/server_pipeline.py --round-b`` has retrained without each picked cluster (``drop_cluster`` select arms) and
``stage.py verify`` has compared every arm with the base policy (and with its matched random arm), ``stage.py confirm`` reads those verdicts
(``vlapoc.step4.collect``) and writes ``cluster_status.parquet``: ``confirmed`` / ``refuted`` / ``not confirmed`` per (cluster, comparison,
metric). The dashboard shows a cluster as *estimated* until a ``confirm`` run says otherwise. Method side: reads verdicts and cluster outputs
only, no answer key.

Grounding
---------
- PoC §9.1 Step 4 (p. 19), quoted: "Influence estimates can be wrong. This step turns them from claims into verified results." and Step 4.3-4.4:
  "Apply the macro-KS test: did the time-to-success distribution change at all? If it did, check each metric's confidence interval for the
  direction of the change." A seed whose macro-KS gate did not fire counts for nothing (``vlapoc.step4.call`` returns ``g``).
- PoC §4.4 (pp. 10-11): two-step verdict per intervention, Holm across metrics (the verdict's ``p_holm``); §2.6 (p. 5): a close comparison between
  two policies needs the significance gate; §2.1 item 5 (p. 3): leave-group-out ablation, "coarse but unambiguous".
- CUPID p. 11: summed individual influence only approximates the effect of removing a group, which is why a cluster stays *estimated* until
  this stage says otherwise.

The rule (inferred, stated before any result; recorded as a run param; the owner may override)
------------------------------------------------------------------------------------------
Per metric with an expectation (the select run's ``expected_by_metric``: ``treated_better`` where the cluster's estimated reading was harmful,
``treated_worse`` where helpful), over the training seeds (3): the call of each seed is ``T`` / ``C`` / ``0`` / ``g`` (``step4.call``).
**confirmed**: the expectation is met in at least ``ceil(2n/3)`` seeds (2 of 3) and contradicted in none. **refuted**: the opposite significant call
appears in at least ``ceil(2n/3)`` seeds. Otherwise **not confirmed**. A metric the select run states no expectation for is ``no expectation``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RULE = ("confirmed: expectation met (significant call in the expected direction, macro-KS gate fired) in >= ceil(2n/3) of n seeds and contradicted in none; "
        "refuted: the opposite significant call in >= ceil(2n/3) seeds; otherwise not confirmed (inferred; stated before any result)")


def status_of(calls: list[str], want: str) -> tuple[str, int, int]:
    """``(status, met, contradicted)`` of one metric's per-seed calls against the wanted call (``T`` or ``C``)."""
    other = "C" if want == "T" else "T"
    met, contra, n = calls.count(want), calls.count(other), len(calls)
    need = math.ceil(2 * n / 3) if n else 1
    if n >= 2 and met >= need and contra == 0:
        return "confirmed", met, contra
    if n >= 2 and contra >= need:
        return "refuted", met, contra
    return "not confirmed", met, contra


def cluster_status(records: list[dict], arms: dict[str, dict], cluster_run_id: str) -> pd.DataFrame:
    """One row per (cluster, comparison, metric) over the family-W verdicts of the ``drop_cluster`` arms of ``cluster_run_id``.
    ``comparison`` is ``vs_base`` (the arm against the original policy) or ``vs_matched_random`` (against the arm that removed the same per-task
    counts at random, kind ``curated_vs_random``). Columns: ``cluster_id``, ``definition``, ``arm``, ``control``, ``metric``, ``expected``,
    ``status``, ``n_seeds``, ``met``, ``contradicted``, ``gate_fired``, ``calls_by_seed``, ``seeds``, ``verify_run_ids`` (JSON, by seed)."""
    from vlapoc.step4 import ALPHA, call

    rows = []
    mine = {a for a, d in arms.items() if d.get("op") == "drop_cluster" and d.get("cluster_run_id") == cluster_run_id}
    keyed: dict = {}
    for r in records:
        if r["treated"] in mine and r["verdict"]["family"] == "W" and r["kind"] in ("vs_base", "curated_vs_random"):
            keyed.setdefault((r["treated"], r["kind"], r["control"]), []).append(r)
    for (arm, kind, control), recs in sorted(keyed.items()):
        recs = sorted(recs, key=lambda r: int(r["seed"]))
        d = arms[arm]
        expected = d.get("expected_by_metric") or {}
        for m in recs[0]["verdict"]["metrics"]:
            calls = [call(r["verdict"]["metrics"][m], r["verdict"].get("alpha", ALPHA), bool(r["verdict"]["gate"]["fires"])) for r in recs]
            exp = expected.get(m)
            want = {"treated_better": "T", "treated_worse": "C"}.get(exp)
            status, met, contra = status_of(calls, want) if want else ("no expectation", 0, 0)
            rows.append({"cluster_id": d["cluster_id"], "definition": d.get("definition"), "arm": arm, "comparison": "vs_base" if kind == "vs_base" else "vs_matched_random",
                         "control": control, "metric": m, "expected": exp or "-", "status": status, "n_seeds": len(recs), "met": met, "contradicted": contra,
                         "gate_fired": sum(bool(r["verdict"]["gate"]["fires"]) for r in recs), "calls_by_seed": ",".join(calls),
                         "seeds": ",".join(str(r["seed"]) for r in recs), "verify_run_ids": json.dumps({str(r["seed"]): r["run_id"] for r in recs})})
    return pd.DataFrame(rows)


def overall(table: pd.DataFrame) -> dict[str, str]:
    """Per cluster, on the ``vs_base`` rows: ``confirmed`` (some metric confirmed, none refuted), ``refuted`` (the reverse), ``mixed`` (both),
    ``not confirmed`` (neither). A cluster with no expectation on any metric is ``no expectation``."""
    out = {}
    for cid, g in table[table["comparison"] == "vs_base"].groupby("cluster_id"):
        st = set(g["status"])
        out[cid] = ("mixed" if {"confirmed", "refuted"} <= st else "confirmed" if "confirmed" in st else "refuted" if "refuted" in st
                    else "not confirmed" if "not confirmed" in st else "no expectation")
    return out


def confirm_run(cluster_run_id: str, tag: str, name: str, chain: str | None = None, out_dir: Path | None = None, lineage_root: Path | None = None,
                allow_dirty: bool = False, run_chain: str | None = None) -> dict:
    """Stage ``confirm`` (recorded as stage ``summarize``): ``cluster_status.parquet`` and ``cluster_status.json`` for the arms of a cluster run,
    from the Step 4 verify runs of chain ``tag``. Consumes the cluster run's ``clusters`` entity and every verdict. Refuses a cluster run that is not
    a FINISHED cluster run, and a chain with no ``drop_cluster`` verdicts of that cluster run. Returns ``{"run_id", "table", "rows", "overall"}``."""
    from vlapoc import step4
    from vlapoc.lineage import rel
    from vlapoc.runs import client, stage_run

    crun = client(lineage_root).get_run(cluster_run_id)
    if crun.info.status != "FINISHED" or crun.data.tags.get("stage") != "cluster":
        raise ValueError(f"run {cluster_run_id[:12]} is not a FINISHED cluster run")
    records, arms, chain_id = step4.collect(tag, chain, lineage_root)
    table = cluster_status(records, arms, cluster_run_id)
    if table.empty:
        raise ValueError(f"chain {chain_id} has no family-W verdicts of drop_cluster arms of cluster run {cluster_run_id[:12]}")
    synthetic = crun.data.params.get("synthetic") == "True" or crun.data.tags.get("vlapoc.synthetic") == "true"
    table["synthetic"] = synthetic
    table["cluster_run_id"] = cluster_run_id
    table["status_display"] = [("SYNTHETIC: " if synthetic else "") + x for x in table["status"]]  # the app shows this column; "status" stays the canonical word
    ov = overall(table)
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path, jpath = out_dir / "cluster_status.parquet", out_dir / "cluster_status.json"
    table.to_parquet(path, index=False)
    jpath.write_text(json.dumps({"cluster_run_id": cluster_run_id, "chain": chain_id, "rule": RULE, "synthetic": synthetic, "overall_vs_base": ov,
                                 "note": "a cluster is confirmed only by this stage; before it, every cluster is estimated"}, indent=1))
    used = [r for r in records if r["treated"] in {a for a, d in arms.items() if d.get("op") == "drop_cluster" and d.get("cluster_run_id") == cluster_run_id}]
    with stage_run("summarize", name, lineage_root, chain=run_chain or chain_id, allow_dirty=allow_dirty,
                   extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        r.input_entity("clusters", crun.data.tags["output.clusters.sha256"], "clusters")
        for rec in used:
            r.input_entity("verdict", rec["verdict_sha256"], f"verify.{rec['name']}"[:200])
        r.params({"kind": "confirm (Round B cluster status)", "cluster_run_id": cluster_run_id, "tag": tag, "source_chain": chain_id, "rule": RULE,
                  "n_verify_runs": len(used), "overall_vs_base": ov, "overall_display": {k: ("SYNTHETIC: " if synthetic else "") + v for k, v in ov.items()},
                  "synthetic": synthetic})
        r.output_file("cluster_status", path, kind="table")
        r.output_file("cluster_status_json", jpath, kind="file")
        return {"run_id": r.run_id, "table": rel(path), "rows": len(table), "overall": ov}
