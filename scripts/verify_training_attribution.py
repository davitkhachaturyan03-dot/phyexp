"""Verify the training-and-attribution feature (``tracker/features/2026-10-20-training-attribution/``), story by story.

Grounding: PoC §2.1 item 4 p.3, §3.4 p.7, §4.2-§4.5 pp.9-11, §9.1 Steps 3-4 p.19, §9.2 p.19; owner decision 3
(ground-truth feature); CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) App. A.1-A.2.

Run from the repo root; each mode exits 0 when every check passes (S2 lives in ``scripts/verify_attribution.py``: checks 6-8
and ``--cpu-smoke``, CUPID parity is its check 7a)::

    .venv/bin/python scripts/verify_training_attribution.py --blind        # S1: refusal table (~40 s)
    .venv/bin/python scripts/verify_training_attribution.py --functionals  # S3: functionals.json, ensemble, provenance
    .venv/bin/python scripts/verify_training_attribution.py --pipeline     # S4: parse a dry run (optional path to write it to)
    .venv/bin/python scripts/verify_training_attribution.py --step4        # S4: step4-summary on synthetic verdicts
    .venv/bin/python scripts/verify_training_attribution.py --recovery     # S5: recovery matrix on synthetic planted scores
    .venv/bin/python scripts/verify_training_attribution.py --names        # S6: site config, chain-scoped names
    .venv/bin/python scripts/verify_training_attribution.py --groups       # S7: four groups, aggregation, figures
    .venv/bin/python scripts/verify_training_attribution.py --integration  # S8(a): whole chain via stage.py, synthetic data
    .venv/bin/python scripts/verify_training_attribution.py --local-smoke --device mps   # S8(b): real checkpoint, ~8 min

Every MLflow-writing check runs on a scratch lineage store under ``data/scratch/`` (``runs.scratch_lineage_copy``);
the real ``data/lineage/mlflow.db`` sha256 is recorded before and after and must not change (plan rule R7). Scratch
runs use ``allow_dirty=True`` because this script itself is under ``scripts/``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-20-training-attribution"
ARTIFACTS = FEATURE_DIR / "artifacts"
SCRATCH = REPO_ROOT / "data" / "scratch" / "training_attribution"
REAL_DB = REPO_ROOT / "data" / "lineage" / "mlflow.db"


def db_sha() -> str:
    """sha256 of the real lineage store's MLflow database (must be unchanged by every check here, rule R7)."""
    return hashlib.sha256(REAL_DB.read_bytes()).hexdigest()


def n_runs(lr: Path) -> int:
    """Number of MLflow runs in the store at ``lr`` (all states)."""
    from vlapoc.runs import client, experiment_id

    return len(client(lr).search_runs([experiment_id(lr)], max_results=10000))


def fake_train_run(lr: Path, name: str, subset, ckpt_sha: str = "0" * 64, chain: str | None = None) -> str:
    """A FINISHED ``train`` run in the scratch store whose training input is ``subset`` and which produces the checkpoint entity
    ``ckpt_sha`` (no model involved)."""
    import mlflow

    from vlapoc.runs import stage_run

    with stage_run("train", name, lr, allow_dirty=True, chain=chain) as r:
        r.input_subset(subset, "training", None)
        mlflow.set_tag("checkpoint.weights.sha256", ckpt_sha)
        r.produced("checkpoint", ckpt_sha, "checkpoint")
        return r.run_id


# ---------------------------------------------------------------------------------------------
# a synthetic attribution fixture on a scratch lineage store (S3; reused by S4-S8)
# ---------------------------------------------------------------------------------------------

FAKE_CKPT_SHA = "ab" * 32
FEATURE_DIM = 16


def write_features(prefix: Path, rows: "np.ndarray", episode_index: "np.ndarray", base: dict, n_shards: int) -> None:
    """Write a feature prefix in ``vlapoc.influence``'s on-disk format (``.npy`` float16, ``.meta.npz``, ``.manifest.json``)."""
    import numpy as np

    prefix.parent.mkdir(parents=True, exist_ok=True)
    parts = np.array_split(np.arange(len(rows)), n_shards)
    for i, idx in enumerate(parts):
        stem = f"{prefix}.shard{i}" if n_shards > 1 else str(prefix)
        np.save(f"{stem}.npy", rows[idx].astype(np.float16))
        np.savez(f"{stem}.meta.npz", episode_index=episode_index[idx], frame_index=np.arange(len(idx)),
                 loss=np.ones(len(idx)))
        Path(f"{stem}.manifest.json").write_text(json.dumps({**base, "shard": [i, n_shards], "n_samples": int(len(idx))}))


def pick_covering(lr: Path, parent, min_per_task: int = 3):
    """Episodes of ``parent`` (a v3 subset) forming a small, stratified training subset: greedily choose tasks until every
    corruption type is present, then take each chosen task's labelled episodes plus clean ones up to ``min_per_task``.
    Ground-truth side (reads the answer key): only the SYNTHETIC test fixture does this, never a method."""
    from vlapoc.groundtruth import load_answer_key

    cat = parent.episodes(lr).sort_values(["suite", "libero_task_id", "episode_index"])
    labels = dict(zip(*(load_answer_key("v3", lr)[c] for c in ("episode_id", "corruption_type"))))
    cat = cat.assign(_t=cat["episode_id"].map(labels).fillna(""))
    tasks = {k: g for k, g in cat.groupby(["suite", "libero_task_id"], sort=True)}
    need, chosen = set(labels.values()), []
    while need:
        k = max(tasks, key=lambda k: (len(need & set(tasks[k]["_t"])), -len(tasks[k])))
        if not need & set(tasks[k]["_t"]):
            break
        chosen.append(k)
        need -= set(tasks[k]["_t"])
    rows = []
    for k in chosen:
        g = tasks[k]
        lab, clean = g[g["_t"] != ""], g[g["_t"] == ""]
        rows.append(pd.concat([lab, clean.head(max(min_per_task - len(lab), 2))]))
    return pd.concat(rows)


def make_train(fx: dict, name: str, subset, sha: str, seed: int = 0, n_shards: int = 2) -> dict:
    """A fake train run on ``subset`` with checkpoint ``sha``, plus synthetic train features (``feats/<name>/train``)."""
    import numpy as np

    from vlapoc.lineage import resolve_version

    lr = fx["lr"]
    eps = subset.episodes(lr).sort_values("episode_index").reset_index(drop=True)
    tid = fake_train_run(lr, name, subset, sha, fx.get("chain"))
    ep_idx = np.repeat(eps["episode_index"].to_numpy(), 3)
    man = {"scheme": "vlapoc.features.v2", "checkpoint": "data/scratch/fake", "checkpoint_sha256": sha, "checkpoint_weights_sha256_recheck": True,
           "dataset_root": resolve_version(subset.dataset_version_id, lr)["lerobot_root"], "dataset_fps": 20, "stride": 2, "rank": 4, "k_draws": 4, "seed": 0,
           "dim": FEATURE_DIM, "n_layers": 1, "layers": ["l0"], "rename": None, "chunk_size": 50, "output_fn": "square", "episodes": [int(e) for e in subset.resolve(lr)]}
    prefix = fx["base"] / "feats" / name / "train"
    write_features(prefix, np.random.default_rng(seed + 100).normal(size=(len(ep_idx), FEATURE_DIM)), ep_idx, man, n_shards)
    return {"train_id": tid, "name": name, "sha": sha, "prefix": prefix, "manifest": man, "subset": subset}


def make_fixture(base: Path, tasks: int = 5, per_task: int = 3, seed: int = 0, parent: str = "v0/candidate", chain: str | None = None,
                 train_name: str = "fake-train-fixture", cover_types: bool = False) -> dict:
    """A scratch copy of the real lineage store with a small training subset of ``parent`` (``tasks`` x ``per_task`` episodes, or, with
    ``cover_types`` on a v3 parent, a type-covering stratified subset), a FINISHED fake ``train`` run on it and synthetic train features
    (2 shards). SYNTHETIC: no model involved. The subset is bound as ``fixture/train`` (and ``v3/fixture20`` for a v3 parent)."""
    import numpy as np

    from vlapoc.lineage import Subset, _bind_name
    from vlapoc.runs import scratch_lineage_copy

    rng = np.random.default_rng(seed)
    lr = scratch_lineage_copy(base / "lineage")
    par = Subset.load(parent, lr)
    if cover_types:
        pick = pick_covering(lr, par)
    else:
        cat = par.episodes(lr).sort_values(["suite", "libero_task_id", "episode_index"])
        pick = pd.concat([g.head(per_task) for _, g in cat.groupby(["suite", "libero_task_id"], sort=True)][:tasks])
    sub = Subset(par.dataset_version_id, tuple(pick["episode_id"]), {"op": "synthetic_fixture", "parent_subset_id": par.subset_id})
    sub.save(lr)
    _bind_name("subsets", "fixture/train", sub.subset_id, lr)
    if parent.startswith("v3"):
        _bind_name("subsets", "v3/fixture20", sub.subset_id, lr)
    fx = {"lr": lr, "subset": sub, "episodes": sub.episodes(lr).sort_values("episode_index").reset_index(drop=True), "base": base,
          "rng": rng, "chain": chain}
    tr = make_train(fx, train_name, sub, FAKE_CKPT_SHA, seed)
    fx |= {"train": tr, "train_id": tr["train_id"], "train_prefix": tr["prefix"], "manifest": tr["manifest"]}
    return fx


def add_eval(fx: dict, tag: str, per_task: int = 6, drop_traj: int | None = None, extra_failure_type: bool = False,
             seed: int = 1, train: dict | None = None, eval_name: str | None = None, returns_name: str | None = None) -> dict:
    """An eval run (rollouts parquet with trajectory indices), its returns run, and rollout features, all synthetic.

    ``train``: the fake train run it evaluates (default the fixture's primary). ``drop_traj``: that rollout (row number) has no
    recorded trajectory and no features. ``extra_failure_type``: rollouts get one more failure type with enough members for its
    own column. Rollout features go to ``feats/<train name>/rollouts-<eval name>`` (where ``stage.py`` looks).
    """
    import mlflow
    import numpy as np

    from vlapoc.catalog import sha256_file
    from vlapoc.metrics_registry import return_columns
    from vlapoc.runs import stage_run

    tr = train or fx["train"]
    lr, rng = fx["lr"], np.random.default_rng(seed)
    eps = fx["episodes"]
    tasks = eps[["suite", "libero_task_id"]].drop_duplicates().to_records(index=False).tolist()
    ro = pd.DataFrame([{"suite": suite, "libero_task_id": int(tid)} for suite, tid in tasks for _ in range(per_task)])
    n = len(ro)
    ro["rollout_id"] = [f"{tag}/{i}" for i in range(n)]
    ro["traj_episode_index"] = np.arange(5000, 5000 + n, dtype=float)
    if drop_traj is not None:
        ro.loc[drop_traj, "traj_episode_index"] = np.nan
    ret = ro[["rollout_id", "suite", "libero_task_id"]].copy()
    for c in return_columns("W"):
        ret[c] = rng.normal(size=n)
    for c in ("ret_f30", "ret_censored60"):
        ret[c] = (rng.random(n) < 0.4).astype(float)
    ft = ["success"] * (n // 2) + ["timeout"] * (n // 4)
    ft += ["drop"] * (n // 5) if extra_failure_type else []
    ft += ["timeout"] * (n - len(ft))
    ret["failure_type"] = ft
    out = fx["base"] / "evals" / tag
    out.mkdir(parents=True, exist_ok=True)
    ro.to_parquet(out / "rollouts.parquet", index=False)
    ret.to_parquet(out / "returns.parquet", index=False)
    ename = eval_name or f"fake-eval-{tag}"
    from vlapoc.lineage import file_digests, rel, sha256_json

    traj_dir = out / "trajectories"  # a stand-in trajectories copy: the eval run records its digest, the rollout features' manifest names its root
    traj_dir.mkdir(exist_ok=True)
    (traj_dir / "synthetic.txt").write_text(f"synthetic trajectories of {ename}")
    with stage_run("eval", ename, lr, allow_dirty=True, chain=fx.get("chain")) as r:
        r.params({"train_run_id": tr["train_id"], "checkpoint_sha256": tr["sha"]})
        r.output_file("rollouts", out / "rollouts.parquet", kind="rollouts")
        mlflow.set_tag("output.trajectories", f"trajectories:{sha256_json(file_digests(traj_dir))}")
        eval_id = r.run_id
    sha = sha256_file(out / "rollouts.parquet")
    with stage_run("returns", returns_name or f"fake-returns-{tag}", lr, allow_dirty=True, chain=fx.get("chain")) as r:
        r.input_entity("rollouts", sha, "rollouts")
        r.output_file("returns", out / "returns.parquet", kind="returns")
        returns_id = r.run_id
    has = ro["traj_episode_index"].notna().to_numpy()
    traj = np.repeat(ro.loc[has, "traj_episode_index"].to_numpy().astype(int), 2)
    man = {**tr["manifest"], "episodes": None, "stride": 4, "dataset_root": rel(traj_dir)}
    prefix = fx["base"] / "feats" / tr["name"] / f"rollouts-{ename}"
    write_features(prefix, rng.normal(size=(len(traj), FEATURE_DIM)), traj, man, 1)
    return {"eval_id": eval_id, "returns_id": returns_id, "rollout_prefix": prefix, "n_rollouts": n, "train": tr, "name": ename}


def run_attribute(fx: dict, ev: dict, name: str, **kw) -> dict:
    """``attribute_run`` on the fixture (CPU, scratch store, allow_dirty)."""
    from vlapoc.attribution import attribute_run

    tr = ev.get("train") or fx["train"]
    return attribute_run(tr["train_id"], ev["eval_id"], ev["returns_id"], tr["prefix"], ev["rollout_prefix"], name,
                         out_dir=fx["base"] / "out" / name, device="cpu", lineage_root=fx["lr"], allow_dirty=True,
                         chain=fx.get("chain"), **kw)


# ---------------------------------------------------------------------------------------------
# S1: blindness
# ---------------------------------------------------------------------------------------------


def blind() -> int:
    """S1, AC1.1-AC1.6: refusal table over four entry points (attribute, features, train, select) x four inputs."""
    import stage as stage_mod  # scripts/stage.py

    from verify_derive_gt_s2 import derive_small
    from verify_method_boundary_s3 import method_sources, scan_boundary
    from vlapoc import method_data
    from vlapoc.attribution import attribute_run
    from vlapoc.lineage import LineageError, Subset, VersionIntegrityError, _bind_name, drop_random_k, version_episodes
    from vlapoc.runs import scratch_lineage_copy

    t0 = time.time()
    sha0 = db_sha()
    shutil.rmtree(SCRATCH / "blind", ignore_errors=True)
    (SCRATCH / "blind").mkdir(parents=True)
    rows: list[dict] = []
    ok_all = True
    dummy = "0" * 32

    def refuse(entry: str, case: str, fn, expect: type, lr: Path, contains: str):
        """Run ``fn``; it must raise ``expect`` mentioning ``contains``, and (for attribute) leave the run count alone."""
        method_data._reset_session()
        before = n_runs(lr)
        try:
            fn()
            got, msg = None, "opened"
        except Exception as e:  # noqa: BLE001
            got, msg = type(e), str(e)
        after = n_runs(lr)
        ok = got is expect and contains in msg and before == after
        rows.append({"entry_point": entry, "input": case, "result": "REFUSED " + got.__name__ if got else "OPENED",
                     "names_reason": contains in msg, "runs_before": before, "runs_after": after, "as_expected": ok,
                     "message": msg[:140]})
        return ok

    try:
        # ---- store A: a copy of the real lineage store (v0..v3, names, fences) -----------------------------
        lr = scratch_lineage_copy(SCRATCH / "blind" / "lineage_a")
        names = json.loads((lr / "names.json").read_text())["subsets"]
        v1c, v2c = Subset.load("v1/candidate", lr), Subset.load("v2/candidate", lr)
        v0c = Subset.load("v0/candidate", lr)
        # a 20-episode ground-truth-derived subset of the clean v0 (built the way verify_method_boundary_s3 builds one)
        oracle = Subset(v0c.dataset_version_id, tuple(v0c.episode_ids[:20]),
                        {"op": "oracle_drop", "parent_subset_id": v0c.subset_id, "groundtruth": True,
                         "dropped": "20 episodes (synthetic oracle subset for the S1 test)"})
        oracle.save(lr)
        _bind_name("subsets", "v0/oracle20", oracle.subset_id, lr)
        child = drop_random_k(oracle, 2, 0, lr)  # a descendant whose chain still contains oracle_drop
        _bind_name("subsets", "v0/oracle20_then_random", child.subset_id, lr)
        train = {"fenced v1": (fake_train_run(lr, "fake-train-v1", v1c), v1c, "fenced=True"),
                 "fenced v2 (superseded)": (fake_train_run(lr, "fake-train-v2", v2c), v2c, "fenced"),
                 "ground-truth-derived (oracle_drop)": (fake_train_run(lr, "fake-train-oracle", oracle), oracle,
                                                        "ground-truth-derived"),
                 "descendant of oracle_drop": (fake_train_run(lr, "fake-train-oracle-child", child), child,
                                               "ground-truth-derived")}
        for case, (rid, sub, contains) in train.items():
            ok_all &= refuse("attribute", case, lambda rid=rid: attribute_run(
                rid, dummy, dummy, Path("x"), Path("y"), "attr-test", lineage_root=lr, allow_dirty=True),
                LineageError, lr, contains)
            ok_all &= refuse("features --which train", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr),
                             LineageError, lr, contains)
            ok_all &= refuse("train", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr), LineageError, lr, contains)
            ok_all &= refuse("select (parent)", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr),
                             LineageError, lr, contains)

        # positive control: the blind v3 opens (so the refusals above are not blanket), and its timing (premise P3)
        method_data._reset_session()
        t1 = time.time()
        view = method_data.open_for_method("v3/candidate", lr)
        t_open = time.time() - t1
        rows.append({"entry_point": "open_for_method (control)", "input": "v3/candidate (blind)", "result": "OPENED",
                     "names_reason": True, "runs_before": 0, "runs_after": 0, "as_expected": len(view.episode_indices) == 960,
                     "message": f"{len(view.episode_indices)} episodes, {t_open:.1f}s incl. integrity hash"})
        ok_all &= rows[-1]["as_expected"]
        method_data._reset_session()

        # ---- store B: a small derived version, one byte of a data file flipped (AC1.4) ----------------------
        lrb, pvid, res, _ = derive_small(SCRATCH / "blind" / "small")
        cvid = res["dataset_version_id"]
        ce = version_episodes(cvid, lrb)
        small = Subset(cvid, tuple(ce["episode_id"]), {"op": "all"})
        small.save(lrb)
        _bind_name("subsets", "small20_gt/all", small.subset_id, lrb)
        rid_b = fake_train_run(lrb, "fake-train-small", small)
        f = next((REPO_ROOT / res["lerobot_root"] / "data").rglob("*.parquet"))
        orig = f.read_bytes()
        try:
            b2 = bytearray(orig)
            b2[len(b2) // 2] ^= 0x01
            f.write_bytes(bytes(b2))
            ok_all &= refuse("attribute", "tampered version (1 byte flipped)", lambda: attribute_run(
                rid_b, dummy, dummy, Path("x"), Path("y"), "attr-test", lineage_root=lrb, allow_dirty=True),
                VersionIntegrityError, lrb, "")
            ok_all &= refuse("features --which train", "tampered version (1 byte flipped)",
                             lambda: stage_mod.method_view(small.subset_id, lrb), VersionIntegrityError, lrb, "")
            ok_all &= refuse("train", "tampered version (1 byte flipped)", lambda: stage_mod.method_view(small.subset_id, lrb),
                             VersionIntegrityError, lrb, "")
            ok_all &= refuse("select (parent)", "tampered version (1 byte flipped)",
                             lambda: stage_mod.method_view(small.subset_id, lrb), VersionIntegrityError, lrb, "")
        finally:
            f.write_bytes(orig)
        method_data._reset_session()
        view_b = method_data.open_for_method(small.subset_id, lrb)
        rows.append({"entry_point": "open_for_method (control)", "input": "same small version, byte restored",
                     "result": "OPENED", "names_reason": True, "runs_before": 0, "runs_after": 0,
                     "as_expected": len(view_b.episode_indices) == 20, "message": f"{len(view_b.episode_indices)} episodes"})
        ok_all &= rows[-1]["as_expected"]

        # ---- AC1.3 static: the stage.py commands actually call method_view ------------------------------------
        tree = ast.parse((REPO_ROOT / "scripts" / "stage.py").read_text())
        calls = {}
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in ("cmd_train", "cmd_features", "cmd_select"):
                calls[fn.name] = any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "method_view" for n in ast.walk(fn))
        attr_src = (REPO_ROOT / "vlapoc" / "attribution.py").read_text()
        calls["attribute_run uses open_for_method, not assert_blind"] = "open_for_method(" in attr_src and "assert_blind(" not in attr_src.split("def attribute_run")[1]
        print(f"AC1.3 stage commands call method_view / attribute_run uses open_for_method: {calls}")
        ok_all &= all(calls.values()) and len(calls) == 4

        # ---- AC1.5 boundary scan covers influence.py and attribution.py ---------------------------------------
        srcs = method_sources()
        names_scanned = [p.name for p in srcs]
        viol = scan_boundary(srcs)
        print(f"AC1.5 boundary scan over {names_scanned}: violations {viol}")
        ok_all &= {"influence.py", "attribution.py"} <= set(names_scanned) and viol == []
    finally:
        shutil.rmtree(SCRATCH / "blind", ignore_errors=True)

    tbl = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(tbl.drop(columns=["message"]).to_string(index=False))
    sha1 = db_sha()
    print(f"real mlflow.db sha256 before {sha0[:16]}... after {sha1[:16]}... unchanged: {sha0 == sha1}")
    ok_all &= bool(tbl["as_expected"].all()) and sha0 == sha1
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(ARTIFACTS / "s1_refusal_table.csv", index=False)
    (ARTIFACTS / "s1_blind.json").write_text(json.dumps(
        {"pass": bool(ok_all), "rows": rows, "static": calls, "boundary_scanned": names_scanned,
         "mlflow_db_sha256_before": sha0, "mlflow_db_sha256_after": sha1, "seconds": round(time.time() - t0)}, indent=1))
    print(f"S1 pass: {ok_all} ({time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------
# S3: declared functionals, provenance of the attribute run
# ---------------------------------------------------------------------------------------------


def functionals(manifest: Path | None) -> int:
    """S3, AC3.1-AC3.6 (and the params half of AC2.5), on synthetic data (SYNTHETIC: labelled as such)."""
    import mlflow
    import numpy as np

    from vlapoc import attribution as at
    from vlapoc.attribution import column_declarations, declaration_of, ensemble_run
    from vlapoc.lineage import Subset
    from vlapoc.metrics_registry import ENTRIES, FAMILY_TAU_S
    from vlapoc.runs import client, select_run

    t0 = time.time()
    sha0 = db_sha()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    def raises(fn, exc, contains: str = "") -> tuple[bool, str]:
        try:
            fn()
        except exc as e:
            return contains in str(e), str(e)[:160]
        return False, "did not raise"

    # ---- AC3.1: one declaration row per column, τ from the registry ------------------------------------------
    rng = np.random.default_rng(3)
    n_ep, n_ro = 8, 30
    eps = pd.DataFrame({"episode_id": [f"e{i}" for i in range(n_ep)], "content_hash": [f"h{i}" for i in range(n_ep)],
                        "episode_index": range(n_ep), "suite": ["s"] * n_ep, "libero_task_id": [0, 0, 1, 1, 2, 2, 3, 3],
                        "task_name": ["t"] * n_ep})
    ret = pd.DataFrame({"rollout_id": range(n_ro), "suite": "s", "libero_task_id": [i % 4 for i in range(n_ro)]})
    for c in at.RETURN_COLUMNS:
        ret[c] = rng.normal(size=n_ro)
    ret["ret_f30"] = (rng.random(n_ro) < 0.4).astype(float)
    ret["ret_censored60"] = (rng.random(n_ro) < 0.3).astype(float)
    ret["failure_type"] = ["success"] * 12 + ["timeout"] * 10 + ["drop"] * 8
    m = rng.normal(size=(n_ro, n_ep))
    table = at.score_table(m, eps, ret, center=True, return_encoding="pm1")
    decls = column_declarations(table, ret, True, "pm1", ("square", "square"))
    cols = at.score_columns(table)
    by_col = {r["column"]: r for r in decls}
    reg_tau = {c: e["tau"] for e in ENTRIES if e["family"] == "W" and e["verdict"] is not None for c in e["return_columns"]}
    ok_rows = len(decls) == len(cols) == len(by_col) and set(by_col) == set(cols)
    ok_tau = all(by_col[c]["tau"] == (reg_tau[by_col[c]["return_column"]] if not by_col[c]["return_column"].startswith("ft_")
                                      else next(e["tau"] for e in ENTRIES if e["key"] == "failure_type")) for c in cols)
    ok_ft = all(by_col[c]["tau_s"] == FAMILY_TAU_S["W"] for c in cols if by_col[c]["return_column"].startswith("ft_"))
    check("3.1 every infl_/harm_ column has exactly one functionals row, and its tau is the metrics_registry tau", ok_rows and ok_tau and ok_ft,
          f"{len(cols)} columns, {len(decls)} rows; tau_s by return: " +
          str({r["return_column"]: r["tau_s"] for r in decls if r["column"].startswith("infl_") and "__" not in r["column"]}))
    enc = {r["return_column"]: (r["return_encoding"], r["centred"]) for r in decls if r["column"].startswith("infl_") and "__" not in r["column"]}
    check("3.1b the declared encoding/centring is what column_plan applied (pm1 on binary columns, centred on the rest)",
          enc["ret_f30"] == ("pm1", False) and enc["ret_rmst60"] == ("raw", True) and enc["ft_timeout"] == ("pm1", False), str(enc))
    check("3.1c harm columns state their sign (good_sign) and subset", by_col["harm_f30__same_task"]["rollout_subset"] == "same_task"
          and by_col["harm_censored60"]["good_sign"] == -1.0 and by_col["harm_f30"]["kind"] == "harm")

    # ---- the integration fixture ------------------------------------------------------------------------------
    shutil.rmtree(SCRATCH / "functionals", ignore_errors=True)
    base = SCRATCH / "functionals"
    base.mkdir(parents=True)
    try:
        fx = make_fixture(base)
        lr = fx["lr"]
        ev_a = add_eval(fx, "a", seed=1)
        ev_b = add_eval(fx, "b", seed=2, extra_failure_type=True)
        n0 = n_runs(lr)
        ra = run_attribute(fx, ev_a, "attribute-fixture-a")
        c = client(lr)
        run_a = c.get_run(ra["run_id"])
        # ---- AC2.5 params half ------------------------------------------------------------------------------------
        prm = run_a.data.params
        need = ("train_output_fn", "rollout_output_fn", "return_encoding", "center_returns", "divergences_from_cupid",
                "min_failure_type_rollouts_inferred")
        check("2.5 attribute run params carry output functions, encoding, centring and DIVERGENCES",
              all(k in prm for k in need) and prm["train_output_fn"] == "square" and set(json.loads(prm["divergences_from_cupid"])) == set(at.DIVERGENCES),
              f"{ {k: prm[k] for k in need if k != 'divergences_from_cupid'} }, divergences keys {sorted(json.loads(prm['divergences_from_cupid']))}")
        # ---- AC3.3 feature digests ---------------------------------------------------------------------------------
        from vlapoc.catalog import sha256_file

        feat_tags = {k: v for k, v in run_a.data.tags.items() if k.startswith("input.features.")}
        files = sorted(list(fx["train_prefix"].parent.glob("train*")) + list(ev_a["rollout_prefix"].parent.glob(ev_a["rollout_prefix"].name + "*")))
        files = [f for f in files if f.name.endswith((".npy", ".meta.npz"))]
        digests_ok = len(feat_tags) == len(files) == 6 and sorted(v.split(":", 1)[1] for v in feat_tags.values()) == sorted(sha256_file(f) for f in files)
        check("3.3 the attribute run lists one `features` input per .npy / .meta.npz file with its sha256", digests_ok,
              f"{len(feat_tags)} features inputs for {len(files)} files: {sorted(k.removeprefix('input.') for k in feat_tags)}")
        target = fx["train_prefix"].parent / "train.shard1.npy"
        orig = target.read_bytes()
        b = bytearray(orig)
        b[len(b) // 2] ^= 0x01
        target.write_bytes(bytes(b))
        try:
            rb = run_attribute(fx, ev_a, "attribute-fixture-a-flipped")
        finally:
            target.write_bytes(orig)
        tags_b = {k: v for k, v in c.get_run(rb["run_id"]).data.tags.items() if k.startswith("input.features.")}
        changed = sorted(k for k in feat_tags if feat_tags[k] != tags_b[k])
        check("3.3b flipping one byte of one feature file changes exactly that file's recorded digest", changed == ["input.features.train.shard1.npy"], str(changed))
        # ---- AC3.4 partial rollouts ---------------------------------------------------------------------------------
        ev_p = add_eval(fx, "partial", seed=4, drop_traj=3)
        ok, msg = raises(lambda: run_attribute(fx, ev_p, "attribute-fixture-partial"), ValueError, "no recorded trajectory")
        check("3.4 attribute raises when one rollout lacks a trajectory", ok, msg)
        rp = run_attribute(fx, ev_p, "attribute-fixture-partial", allow_partial=True)
        pp = c.get_run(rp["run_id"]).data.params
        check("3.4b --allow-partial passes and records n_rollouts_without_trajectory", pp["n_rollouts_without_trajectory"] == "1" and pp["allow_partial"] == "True",
              f"n_rollouts {rp['n_rollouts']} of {ev_p['n_rollouts']}, recorded {pp['n_rollouts_without_trajectory']}")
        # ---- functionals.json of the run (AC3.1 on a real attribute run) ---------------------------------------------
        decl_a = at._artifact_json(run_a, "functionals", lr)
        tab_a = at._artifact_table(run_a, "attribution_scores", lr)
        check("3.1d functionals.json of the attribute run has one row per score column", sorted(r["column"] for r in decl_a) == sorted(at.score_columns(tab_a)),
              f"{len(decl_a)} rows")
        # ---- AC3.5 ensemble ------------------------------------------------------------------------------------------
        rb2 = run_attribute(fx, ev_b, "attribute-fixture-b")
        rc = run_attribute(fx, ev_a, "attribute-fixture-c-nocenter", center=False)
        ok, msg = raises(lambda: ensemble_run([ra["run_id"], ra["run_id"]], "ens-dup", out_dir=base / "out" / "ens-dup", lineage_root=lr, allow_dirty=True), ValueError, "duplicate")
        check("3.5 ensemble_run raises on [id, id] duplicates", ok, msg)
        ok, msg = raises(lambda: ensemble_run([ra["run_id"], rc["run_id"]], "ens-mix", out_dir=base / "out" / "ens-mix", lineage_root=lr, allow_dirty=True), ValueError, "center_returns")
        check("3.5b ensemble_run raises on mixed center_returns", ok, msg)
        re_ = ensemble_run([ra["run_id"], rb2["run_id"]], "ens-ab", out_dir=base / "out" / "ens-ab", lineage_root=lr, allow_dirty=True)
        run_e = c.get_run(re_["run_id"])
        nm = json.loads(run_e.data.params["n_models_per_column"])
        decl_e = at._artifact_json(run_e, "functionals", lr)
        extra = [col for col, n in nm.items() if n == 1]
        check("3.5c ensemble records n_models per column (a failure-type column exists in only one seed)",
              len(extra) > 0 and all(col.startswith(("infl_ft_drop", "harm_ft_drop")) for col in extra) and
              all(r["n_models"] == nm[r["column"]] for r in decl_e) and nm["infl_f30"] == 2,
              f"columns with n_models=1: {extra}; others 2")
        # ---- AC3.6 select records the declaration -------------------------------------------------------------------
        parent = fx["subset"]
        score_path = base / "out" / "ens-ab" / "attribution_scores.parquet"
        metric = "harm_f30__same_task"
        _, child = select_run("fixture/train", "select-fixture", "drop_top_k_per_task", k=1, score_path=score_path, metric=metric,
                              score_run_id=re_["run_id"], lineage_root=lr, allow_dirty=True, require_declaration=True)
        sel = c.search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], "tags.mlflow.runName = 'select-fixture'")[0]
        got = json.loads(sel.data.params["metric_declaration"])
        want = declaration_of(re_["run_id"], metric, lr)
        check("3.6 the select run's params carry the ranked column's declaration", got == want and got["column"] == metric and got["tau"],
              f"{got['column']}: {got['functional']} | tau {got['tau']} | subset {got['rollout_subset']} | good_sign {got['good_sign']}")
        ok, msg = raises(lambda: select_run("fixture/train", "select-undeclared", "drop_top_k_per_task", k=1, score_path=score_path, metric=metric,
                                            score_run_id=fx["train_id"], lineage_root=lr, allow_dirty=True, require_declaration=True),
                         ValueError, "undeclared")
        check("3.6b select refuses (require_declaration) a score run that recorded no functionals.json", ok, msg)
        ok, msg = raises(lambda: select_run("fixture/train", "select-badcol", "drop_top_k_per_task", k=1, score_path=score_path, metric="harm_nonexistent",
                                            score_run_id=re_["run_id"], lineage_root=lr, allow_dirty=True), Exception, "harm_nonexistent")
        check("3.6c select refuses a column the score run neither contains nor declares (before any run is created)", ok, msg)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s3_functionals_example.json").write_text(json.dumps(
            {"SYNTHETIC": "fixture data, not an attribution of any real policy", "source": "attribute run of verify_training_attribution.py --functionals",
             "rows": decl_a}, indent=1, ensure_ascii=False))
        (ARTIFACTS / "s3_ensemble_functionals_example.json").write_text(json.dumps(
            {"SYNTHETIC": "fixture data", "n_models_per_column": nm, "rows": decl_e}, indent=1, ensure_ascii=False))
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # ---- AC3.2 manifests from the featurize smoke are repo-relative ------------------------------------------------
    if manifest is not None:
        man = json.loads(manifest.read_text())
        paths = {k: man[k] for k in ("checkpoint", "dataset_root")}
        ok = all(not v.startswith("/") and (REPO_ROOT / v).exists() for v in paths.values())
        check("3.2 featurize manifests hold repo-relative checkpoint and dataset_root (rel() round-trips to an existing path)", ok, str(paths))
        (ARTIFACTS / "s3_manifest_example.json").write_text(json.dumps({k: man[k] for k in ("scheme", "checkpoint", "dataset_root", "output_fn", "dim", "n_layers", "stride", "k_draws", "rank")}, indent=1))
    else:
        # no manifest given: a real one needs a featurize call on the checkpoint (minutes on MPS), so the documented bare invocation
        # records a SKIP, which does not fail the run; the evidence for 3.2 is in the log and artifacts/s3_manifest_example.json
        results.append(("3.2 featurize manifests are repo-relative [SKIPPED: pass --manifest <a manifest written by influence.featurize>]", True, "SKIP"))
        print("[SKIP] 3.2 featurize manifests are repo-relative (no --manifest given; see artifacts/s3_manifest_example.json)")

    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    ok_all = all(r[1] for r in results)
    (ARTIFACTS / "s3_functionals.json").write_text(json.dumps({"pass": ok_all, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results],
                                                               "seconds": round(time.time() - t0)}, indent=1))
    print(f"S3 pass: {ok_all} ({sum(r[1] for r in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------
# S4: Step 4 completeness (arms, comparisons, curation modes, seed summary)
# ---------------------------------------------------------------------------------------------


def parse_dry_run(text: str) -> list[dict]:
    """Parse ``server_pipeline.py --dry-run`` output into jobs: ``id`` (J<n>), ``name``, ``deps`` (job ids), ``dep_names``, ``cmd``."""
    import re

    jobs = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"\[dry (\d+)\] (\S+)\s+deps=(\S*)", ln)
        if m:
            deps = [] if m.group(3) == "-" else m.group(3).split(",")
            jobs.append({"id": f"J{m.group(1)}", "name": m.group(2), "deps": deps, "cmd": lines[i + 1].strip()})
    names = {j["id"]: j["name"] for j in jobs}
    for j in jobs:
        j["dep_names"] = [names[d] for d in j["deps"]]
    return jobs


def dry_run(*args: str) -> tuple[list[dict], str]:
    """Run ``scripts/server_pipeline.py --tag t --dry-run <args>``; return (parsed jobs, last summary line)."""
    import subprocess

    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "server_pipeline.py"), "--tag", "t", "--dry-run", *args],
                         check=True, capture_output=True, text=True, cwd=REPO_ROOT).stdout
    return parse_dry_run(out), out.strip().splitlines()[-1]


def pipeline(dry_path: Path | None) -> int:
    """S4, AC4.1-AC4.3 and AC4.6: parse the dry-run of the Step 4 chain (not eyeballed)."""
    import re
    import shlex

    t0 = time.time()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    jobs, summary = dry_run()
    if dry_path is not None:
        dry_path.parent.mkdir(parents=True, exist_ok=True)
        dry_path.write_text("\n".join(f"{j['id']} {j['name']} deps={','.join(j['dep_names'])} :: {j['cmd']}" for j in jobs))
    by_name = {j["name"]: j for j in jobs}
    arm_names = re.search(r"arms: (\[.*\])", summary).group(1)
    arm_list = eval(arm_names)  # a printed list of str
    seeds = (0, 1, 2)
    harm_help = [a for a in arm_list if re.match(r"g?(harm|help)-", a)]
    ctrl_of = {a: [b for b in arm_list if re.match(rf"{'g' if a.startswith('g') else ''}random\d+-", b)] for a in harm_help}
    bad, n_checked = [], 0
    for a in harm_help:
        for c in ctrl_of[a]:
            for s in seeds:
                j = by_name.get(f"verify-t-{a}-vs-{c}-s{s}")
                n_checked += 1
                want = {f"eval-t-{a}-s{s}", f"eval-t-{c}-s{s}"}
                if j is None or set(j["dep_names"]) != want or "--comparison curated_vs_random" not in j["cmd"] or "--family W" not in j["cmd"]:
                    bad.append((a, c, s))
    helps = [a for a in arm_list if "help-" in a]
    check("4.1 every harm and help arm has a verify job vs each random control of its family, per seed, depending on exactly "
          "both arms' eval jobs", not bad and n_checked > 0 and len(helps) > 0, f"{n_checked} jobs checked, {len(helps)} help arms, mismatches {bad[:3]}")
    two_arms = [a for a in ("harm-f30-k5", "help-f30-k5") if a in arm_list]
    check("4.1b both a harm-vs-random and a help-vs-random job exist for f30 (per-task) in every seed",
          len(two_arms) == 2 and all(f"verify-t-{a}-vs-random0-k5-s{s}" in by_name for a in two_arms for s in seeds))
    hh = [(s, by_name.get(f"verify-t-harm-hrt60_lin-k5-vs-harm-f30-k5-s{s}")) for s in seeds]
    check("4.2 a harm-hrt60_lin arm and per seed a head-to-head verify (hrt60_lin arm B vs f30 arm A), depending on both evals",
          "harm-hrt60_lin-k5" in arm_list and all(j and set(j["dep_names"]) == {f"eval-t-harm-hrt60_lin-k5-s{s}", "eval-t-harm-f30-k5-s" + str(s)}
                                                  and "--comparison head_to_head" in j["cmd"] for s, j in hh))
    # AC4.3 curation modes: parse the select commands
    def selects(*args: str) -> list[dict]:
        js, _ = dry_run(*args)
        cmd = next(j for j in js if j["name"] == "select-t")["cmd"]
        out = []
        for part in cmd.split(" && "):
            toks = shlex.split(part)
            d = {toks[i].lstrip("-"): toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--") and not toks[i + 1].startswith("--")}
            d["headline"] = "--headline" in toks
            out.append(d)
        return out

    g = selects("--curation", "global")
    pt = selects("--curation", "per_task")
    both = selects("--curation", "both")
    default = selects()
    g_top = [d for d in g if d["op"] == "drop_top_k"]
    g_rand = [d for d in g if d["op"] == "drop_random_k" and d["k"] != "20"]  # k = 20 is AC7.5's own gate arm
    ok_g = (all("__same_task" not in d["metric"] and d["metric"].startswith("harm_") for d in g_top)
            and {d["k"] for d in g_rand} == {d["k"] for d in g_top} == {"100"} and len(g_rand) == 2
            and not any("per_task" in d["op"] for d in g))
    check("4.3 --curation global emits drop_top_k on harm_<m> (no __same_task) and drop_random_k with the same k (N = 5 x 20 = 100)", ok_g,
          f"{len(g_top)} drop_top_k arms (k {sorted({d['k'] for d in g_top})}), {len(g_rand)} random controls (k {sorted({d['k'] for d in g_rand})})")
    ok_pt = all("__same_task" in d["metric"] for d in pt if d["op"] == "drop_top_k_per_task") and not any(d["op"] in ("drop_top_k", "drop_random_k") and d.get("k") != "20" for d in pt)
    check("4.3b --curation per_task emits only the per-task family (plus the k=20 AC7.5 gate)", ok_pt)
    ops_both = {d["op"] for d in both}
    check("4.3c --curation both emits both families, and is the default (owner answer Q1); each select records --pipeline-curation both, "
          "per-task arms --headline",
          {"drop_top_k_per_task", "drop_top_k", "drop_random_k_per_task", "drop_random_k"} <= ops_both and both == default
          and all(d.get("pipeline-curation") == "both" for d in both if d["op"] != "drop_random_k" or d["k"] != "20")
          and all(d["headline"] for d in both if d["op"].endswith("_per_task")) and not any(d["headline"] for d in both if not d["op"].endswith("_per_task")),
          f"{len(both)} select runs: " + str({o: sum(d['op'] == o for d in both) for o in sorted(ops_both)}))
    # step4-summary job last, depending on every verify job
    verifies = [j["name"] for j in jobs if j["name"].startswith("verify-")]
    last = jobs[-1]
    check("4.3d a step4-summary job runs last and depends on every verify job", last["name"] == "step4-summary-t" and sorted(last["dep_names"]) == sorted(verifies),
          f"{len(verifies)} verify jobs")
    n_by = {}
    for j in jobs:
        key = re.sub(r"-(s\d|t-.*)$", "", j["name"].split("-")[0])
        n_by[key] = n_by.get(key, 0) + 1
    check("4.6 job count and arm list of the default dry run recorded", True, f"{len(jobs)} jobs ({n_by}); arms ({len(arm_list)}): {arm_list}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "s4_dry_run_default.txt").write_text("\n".join(f"{j['id']} {j['name']} deps={','.join(j['dep_names'])} :: {j['cmd']}" for j in jobs) + "\n" + summary + "\n")
    ok_all = all(r[1] for r in results)
    (ARTIFACTS / "s4_pipeline.json").write_text(json.dumps({"pass": ok_all, "n_jobs": len(jobs), "arms": arm_list, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results]}, indent=1))
    print(f"S4 pipeline pass: {ok_all} ({sum(r[1] for r in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


def synthetic_verdict(treated_better: bool | None, gate: bool = True, metric_keys=("f30", "auc_h", "rmst60", "hrt60", "censored60"),
                      only: dict | None = None) -> dict:
    """A SYNTHETIC ``verdict.json`` (only the fields ``vlapoc.step4`` reads). ``treated_better``: True/False/None = not significant,
    for every metric except where ``only`` overrides per metric key."""
    def row(v):
        if v is None:
            return {"p_holm": 0.4, "diff_better": 0.01}
        return {"p_holm": 0.004, "diff_better": -0.1 if v else 0.1}  # diff_better > 0: arm A (the control) better
    m = {k: row((only or {}).get(k, treated_better)) for k in metric_keys}
    return {"SYNTHETIC": True, "family": "W", "alpha": 0.05, "gate": {"fires": gate, "p": 0.01 if gate else 0.6}, "metrics": m,
            "outcome": "i" if gate else "iii"}


def step4_synthetic() -> int:
    """S4, AC4.4 and AC4.5: ``step4-summary`` on synthetic verdicts, and the per-task counts of every select arm on the real v0/candidate."""
    import shlex

    import mlflow
    import numpy as np

    from vlapoc import step4
    from vlapoc.lineage import Subset
    from vlapoc.runs import client, scratch_lineage_copy, select_run, stage_run

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import server_pipeline as sp
    import stage as stage_mod

    t0 = time.time()
    sha0 = db_sha()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    shutil.rmtree(SCRATCH / "step4", ignore_errors=True)
    base = SCRATCH / "step4"
    base.mkdir(parents=True)
    try:
        # ---- AC4.4: synthetic verdicts in a scratch store -----------------------------------------------------------
        lr = scratch_lineage_copy(base / "lineage_summary", mlflow_db=False)
        chain, other = "t-aaaa1111", "t-bbbb2222"

        def verify(name, kind, treated, control, seed, verdict, ch):
            p = base / "verdicts" / f"{name}.json"
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(verdict))
            with stage_run("verify", name, lr, chain=ch, allow_dirty=True,
                           extra_tags={"step4.kind": kind, "step4.treated": treated, "step4.control": control, "step4.seed": seed}) as r:
                r.output_file("verdict", p, kind="verdict")

        decl = {"column": "harm_f30__same_task", "functional": "F(30) = P(T <= 30 s)", "tau": "30 s", "rollout_subset": "same_task", "good_sign": 1.0}
        for arm, direction in (("harm-f30-k5", "highest"), ("help-f30-k5", "lowest"), ("harm-hrt60_lin-k5", "highest")):
            with stage_run("select", f"select-t-{arm}", lr, chain=chain, allow_dirty=True) as r:
                r.params({"op": "drop_top_k_per_task", "direction": direction, "metric": decl["column"], "curation": "per_task",
                          "metric_declaration": json.dumps(decl)})
        planted = {0: True, 1: True, 2: None}  # harm-f30 beats random0 on every metric in seeds 0 and 1, not significant in seed 2
        for sd in range(3):
            verify(f"v-harm-s{sd}", "curated_vs_random", "harm-f30-k5", "random0-k5", sd, synthetic_verdict(planted[sd]), chain)
            verify(f"v-help-s{sd}", "curated_vs_random", "help-f30-k5", "random0-k5", sd, synthetic_verdict(False if sd < 2 else None), chain)
            verify(f"v-h2h-s{sd}", "head_to_head", "harm-hrt60_lin-k5", "harm-f30-k5", sd,
                   synthetic_verdict(None, only={"hrt60": True, "f30": False}), chain)
            verify(f"v-base-s{sd}", "vs_base", "harm-f30-k5", "base", sd, synthetic_verdict(True), chain)
        verify("v-other-s0", "curated_vs_random", "harm-f30-k5", "random0-k5", 0, synthetic_verdict(False), other)
        try:
            step4.collect("t", None, lr)
            check("4.4 refuses a tag whose verify runs span two chains unless one is named", False)
        except ValueError as e:
            check("4.4 refuses a tag whose verify runs span two chains unless one is named", "several chains" in str(e), str(e)[:120])
        res = step4.summary_run("t", "step4-summary-t", chain=chain, out_dir=base / "out", lineage_root=lr, allow_dirty=True)
        tbl = pd.read_csv(base / "out" / "step4_summary.csv")
        harm = tbl[(tbl["arm"] == "harm-f30-k5") & (tbl["comparison"] == "curated_vs_random")].iloc[0]
        help_ = tbl[(tbl["arm"] == "help-f30-k5") & (tbl["comparison"] == "curated_vs_random")].iloc[0]
        h2h = tbl[tbl["comparison"] == "head_to_head"].iloc[0]
        check("4.4b planted case: curated beats random on f30 in 2/3 seeds, with the by-seed calls and the expectation counted",
              harm["f30__by_seed"] == "T,T,0" and harm["f30__T"] == 2 and harm["f30__supports"] == 2 and "2/3" in harm["f30__consistency"]
              and harm["gate_fires"] == "3/3" and harm["n_seeds"] == 3, f"{harm['f30__by_seed']} | {harm['f30__consistency']}")
        check("4.4c helpful removal: control better in 2/3 seeds meets the 'treated worse' expectation twice",
              help_["f30__by_seed"] == "C,C,0" and help_["f30__expected"] == "treated_worse" and help_["f30__supports"] == 2)
        check("4.4d head-to-head: HRT arm wins hrt60 and the F30 arm wins f30 in 3/3 seeds, as the PoC §4.5 expects",
              h2h["hrt60__by_seed"] == "T,T,T" and h2h["f30__by_seed"] == "C,C,C" and h2h["hrt60__supports"] == 3 and h2h["f30__supports"] == 3)
        check("4.4e each row carries the declared functional of its treated arm",
              harm["functional"] == decl["functional"] and harm["tau"] == "30 s" and harm["rollout_subset"] == "same_task" and harm["direction"] == "highest")
        # round 2: the macro-KS gate decides whether a seed counts (PoC §4.4 p.10, §9.1 Step 4.4 p.19)
        for sd in range(3):
            verify(f"v-gate-s{sd}", "curated_vs_random", "harm-hrt60_lin-k5", "random0-k5", sd, synthetic_verdict(True, gate=False), chain)
        verify("v-gatemix-s0", "curated_vs_random", "help-f30-k5", "random1-k5", 0, synthetic_verdict(False, gate=True), chain)
        verify("v-gatemix-s1", "curated_vs_random", "help-f30-k5", "random1-k5", 1, synthetic_verdict(False, gate=False), chain)
        verify("v-gatemix-s2", "curated_vs_random", "help-f30-k5", "random1-k5", 2, synthetic_verdict(False, gate=True), chain)
        step4.summary_run("t", "step4-summary-t2", chain=chain, out_dir=base / "out2", lineage_root=lr, allow_dirty=True)
        tbl2 = pd.read_csv(base / "out2" / "step4_summary.csv")
        gz = tbl2[(tbl2["arm"] == "harm-hrt60_lin-k5") & (tbl2["comparison"] == "curated_vs_random")].iloc[0]
        gm = tbl2[(tbl2["arm"] == "help-f30-k5") & (tbl2["control"] == "random1-k5")].iloc[0]
        check("4.4h seeds whose macro-KS gate did not fire add no directional support even when every metric is Holm-significant (planted)",
              gz["f30__by_seed"] == "g,g,g" and gz["f30__T"] == 0 and gz["f30__C"] == 0 and gz["f30__supports"] == 0 and gz["gate_fires"] == "0/3",
              f"by seed {gz['f30__by_seed']}, supports {gz['f30__supports']}, gate_fires {gz['gate_fires']}")
        check("4.4i a mixed case: gate fired in 2 of 3 seeds, so at most 2 count (control better in the 2 fired seeds, none from the third)",
              gm["f30__by_seed"] == "C,g,C" and gm["f30__C"] == 2 and gm["f30__supports"] == 2 and gm["gate_fires"] == "2/3", f"by seed {gm['f30__by_seed']}")
        check("4.4f the other chain's verdict is not in the table (a chain is summarized alone)", int(harm["n_seeds"]) == 3)
        run = client(lr).get_run(res["run_id"])
        check("4.4g the summary is a tracked run (stage summarize) with the table as an output",
              run.data.tags["stage"] == "summarize" and "output.step4_summary_csv.sha256" in run.data.tags, f"{res['rows']} rows from {res['n_verify_runs']} verify runs")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(ARTIFACTS / "s4_step4_summary_SYNTHETIC.csv", index=False)

        # ---- AC4.5: every select arm of the chain on the real v0/candidate manifests ---------------------------------
        lr2 = scratch_lineage_copy(base / "lineage_select")
        parent = Subset.load("v0/candidate", lr2)
        cat = parent.episodes(lr2)
        per_task_parent = cat.groupby(["suite", "libero_task_id"]).size()
        check("4.5 premise: v0/candidate has 20 tasks x 48 episodes", len(per_task_parent) == 20 and set(per_task_parent) == {48}, f"{len(per_task_parent)} tasks, counts {sorted(set(per_task_parent))}")
        rng = np.random.default_rng(5)
        scores = parent.keys(lr2).reset_index(drop=True)
        for m in ("f30", "rmst60", "auc_expected", "hrt60_lin"):
            scores[f"harm_{m}"] = rng.normal(size=len(scores))
            scores[f"harm_{m}__same_task"] = rng.normal(size=len(scores))
        spath = base / "scores.parquet"
        scores.to_parquet(spath, index=False)
        ns = argparse.Namespace(tag="t", subset="v0/candidate", metrics=["f30", "rmst60", "auc_expected", "hrt60_lin"], k_per_task=[5],
                                curation="both", n_tasks=20, random_draws=2, ac75=True)
        plan = sp.arms(ns)
        ok_counts, kept, rows = True, {}, []
        for x in plan:
            toks = shlex.split(x["args"])
            d = {toks[i].lstrip("-"): toks[i + 1] for i in range(len(toks) - 1) if toks[i].startswith("--") and not toks[i + 1].startswith("--")}
            extra = stage_mod.select_params(d["op"], "--headline" in toks, d.get("pipeline-curation"))
            kw = {"metric": d.get("metric"), "direction": d.get("direction", "highest"), "seed": int(d["seed"]) if "seed" in d else None,
                  "score_path": spath if d["op"].startswith("drop_top_k") else None}
            _, child = select_run("v0/candidate", f"select-t-{x['arm']}", d["op"], k=int(d["k"]), lineage_root=lr2, allow_dirty=True,
                                  extra_params=extra, **kw)
            counts = cat[cat["episode_id"].isin(child.episode_ids)].groupby(["suite", "libero_task_id"]).size()
            kept[x["arm"]] = (len(child.episode_ids), sorted(set(counts)), extra)
            if x["mode"] == "per_task":
                ok_counts &= sorted(set(counts)) == [43] and len(counts) == 20 and len(child.episode_ids) == 860
            elif x["mode"] == "global":
                ok_counts &= len(child.episode_ids) == 860  # 960 - 100
            rows.append({"arm": x["arm"], "mode": x["mode"], "episodes_kept": len(child.episode_ids), "per_task_counts": sorted(set(counts)), "curation_param": extra["curation"], "headline": extra["headline"]})
        check("4.5 every per-task arm keeps exactly 43 of 48 episodes in each of the 20 tasks (860 total); global arms keep 860 too (960 - 100)",
              ok_counts, f"{len(plan)} arms: {sorted({(r['mode'], r['episodes_kept']) for r in rows})}")
        sel = client(lr2).search_runs([client(lr2).get_experiment_by_name("vlapoc-lineage").experiment_id], "tags.mlflow.runName = 'select-t-harm-f30-k5'")[0].data.params
        check("4.5b the select run records the curation mode, headline flag and pipeline setting (owner Q1)",
              sel["curation"] == "per_task" and sel["headline"] == "True" and sel["pipeline_curation"] == "both", str({k: sel[k] for k in ("curation", "headline", "pipeline_curation", "op", "k", "direction")}))
        pd.DataFrame(rows).to_csv(ARTIFACTS / "s4_select_counts_SYNTHETIC_SCORES.csv", index=False)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    ok_all = all(r[1] for r in results)
    (ARTIFACTS / "s4_summary_and_selects.json").write_text(json.dumps({"pass": ok_all, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results]}, indent=1))
    print(f"S4 summary/selects pass: {ok_all} ({sum(r[1] for r in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------
# S5: recovery of planted corruptions (SYNTHETIC score tables)
# ---------------------------------------------------------------------------------------------


def recovery() -> int:
    """S5, AC5.1-AC5.6, on SYNTHETIC score tables over the real v3 answer key (no attribution has been run on v3)."""
    import subprocess

    import numpy as np

    from vlapoc import recovery as rec
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import LineageError, Subset
    from vlapoc.runs import client, scratch_lineage_copy, stage_run
    from verify_method_boundary_s3 import method_sources, scan_boundary

    t0 = time.time()
    sha0 = db_sha()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    shutil.rmtree(SCRATCH / "recovery", ignore_errors=True)
    base = SCRATCH / "recovery"
    base.mkdir(parents=True)
    try:
        lr = scratch_lineage_copy(base / "lineage")
        sub = Subset.load("v3/candidate", lr)
        keys = sub.keys(lr).reset_index(drop=True)
        labels = load_answer_key("v3", lr)
        typ = keys["episode_id"].map(dict(zip(labels["episode_id"], labels["corruption_type"]))).fillna("").to_numpy()
        rng = np.random.default_rng(11)
        tab = keys.copy()
        for m in ("f30", "rmst60", "auc_expected", "hrt60_lin", "censored60"):
            tab[f"harm_{m}"] = rng.normal(size=len(tab))
        tab["harm_rmst60__same_task"] = rng.normal(size=len(tab))
        tab["harm_rmst60"] = np.where(typ == "idle_padding", 10.0, 0.0) + rng.normal(size=len(tab)) * 0.1  # PLANTED
        tab["harm_f30"] = np.where(typ == "gripper_release", 10.0, 0.0) + rng.normal(size=len(tab)) * 0.1  # PLANTED
        pq_path = base / "attribution_scores.parquet"
        tab.to_parquet(pq_path, index=False)
        with stage_run("attribute", "fake-attribute-v3", lr, allow_dirty=True) as r:
            r.output_file("attribution_scores", pq_path, kind="score_table")
            rid = r.run_id
        n_before = n_runs(lr)
        res = rec.recovery_run(rid, "v3", "score-synthetic-v3", out_dir=base / "out", lineage_root=lr, allow_dirty=True, synthetic=True)
        m = pd.read_csv(base / "out" / "recovery_matrix.csv")
        sm = pd.read_csv(base / "out" / "recovery_summary.csv").set_index("type")
        au = lambda ty, c: float(m[(m["type"] == ty) & (m["column"] == c)]["auroc"].iloc[0])  # noqa: E731
        pooled_rand = au("pooled", "harm_hrt60_lin")
        check("5.1 planted columns: idle_padding AUROC 1.0 in harm_rmst60 and gripper_release 1.0 in harm_f30", au("idle_padding", "harm_rmst60") == 1.0
              and au("gripper_release", "harm_f30") == 1.0, f"{au('idle_padding', 'harm_rmst60')}, {au('gripper_release', 'harm_f30')}")
        check("5.1b a random column's pooled AUROC is in [0.4, 0.6] (inferred band: 91 positives of 960)", 0.4 <= pooled_rand <= 0.6, f"{pooled_rand:.3f}")
        cols = sorted(c for c in tab.columns if c.startswith("harm_"))
        check("5.2 recovery_matrix.csv has one row per (type or pooled, harm column)",
              len(m) == 8 * len(cols) and not m.duplicated(["type", "column"]).any(), f"{len(m)} rows = 8 x {len(cols)} columns")
        b = pd.read_csv(rec.BANDS_CSV).set_index("type")
        eq = [ty for ty in b.index if np.allclose(m[m["type"] == ty]["best_heuristic_auroc"].unique(), b.loc[ty, "best_auroc_as_scored"])]
        check("5.2b best_heuristic_auroc equals the ground-truth bands CSV for every type", len(eq) == 7, f"{len(eq)}/7")
        check("5.3 matches_hypothesis is True for the two planted types and None where the PoC states no mapping",
              bool(sm.loc["idle_padding", "matches_hypothesis"]) and bool(sm.loc["gripper_release", "matches_hypothesis"])
              and sm.drop(["idle_padding", "gripper_release"])["matches_hypothesis"].isna().all(),
              f"idle_padding best {sm.loc['idle_padding', 'best_column']}; gripper_release best {sm.loc['gripper_release', 'best_column']} (SYNTHETIC, planted)")
        first = subprocess.run(["git", "log", "--format=%h", "--", "vlapoc/recovery.py"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()[-1]
        old = subprocess.run(["git", "show", f"{first}:vlapoc/recovery.py"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        check("5.3b the hypothesis map was committed before any scoring code existed", "EXPECTED_METRICS" in old and "def recovery_matrix" not in old
              and "def recovery_run" not in old, f"first commit of vlapoc/recovery.py: {first}")
        check("5.3c the run records synthetic=True and a tracked score run exists", client(lr).get_run(res["run_id"]).data.params["synthetic"] == "True"
              and n_runs(lr) == n_before + 1)
        # 5.5: a table keyed to another version is refused
        v0keys = Subset.load("v0/candidate", lr).keys(lr).reset_index(drop=True)
        t0_ = v0keys.copy()
        t0_["harm_f30"] = rng.normal(size=len(t0_))
        p0 = base / "v0scores.parquet"
        t0_.to_parquet(p0, index=False)
        with stage_run("attribute", "fake-attribute-v0", lr, allow_dirty=True) as r:
            r.output_file("attribution_scores", p0, kind="score_table")
            rid0 = r.run_id
        try:
            rec.recovery_run(rid0, "v3", "score-wrong", out_dir=base / "out0", lineage_root=lr, allow_dirty=True)
            check("5.5 score refuses a score table keyed to another version", False)
        except LineageError as e:
            check("5.5 score refuses a score table keyed to another version", "keys" in str(e), str(e)[:120])
        # 5.4 dry run
        v3 = subprocess.run([sys.executable, "scripts/server_pipeline.py", "--tag", "t", "--dry-run", "--subset", "v3/candidate"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        v0 = subprocess.run([sys.executable, "scripts/server_pipeline.py", "--tag", "t", "--dry-run"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        j3, j0 = parse_dry_run(v3), parse_dry_run(v0)
        s3 = [j for j in j3 if j["name"].startswith("score-")]
        check("5.4 dry run with --subset v3/candidate has one score job depending on the ensemble job; with v0/candidate none",
              len(s3) == 1 and s3[0]["dep_names"] == ["ensemble-t"] and not [j for j in j0 if j["name"].startswith("score-")], f"{s3[0]['cmd'] if s3 else None}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    # 5.6 boundary
    srcs = [REPO_ROOT / "vlapoc" / "influence.py", REPO_ROOT / "vlapoc" / "attribution.py"]
    mention = [p.name for p in srcs if "recovery" in p.read_text().replace("recovery_", "").replace("# ", "") and "vlapoc.recovery" in p.read_text()]
    tree = ast.parse((REPO_ROOT / "scripts" / "stage.py").read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and "recovery" in ast.dump(n)]
    check("5.6 the scoring code is imported only inside stage.py's score command, never by influence.py / attribution.py (boundary scan clean)",
          scan_boundary(srcs) == [] and not mention and not top and "vlapoc.recovery" not in " ".join(str(p) for p in method_sources()), f"mentions {mention}, top-level imports {len(top)}")
    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    ok_all = all(r[1] for r in results)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "s5_recovery.json").write_text(json.dumps({"SYNTHETIC": "planted score tables over the real v3 answer key", "pass": ok_all, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results]}, indent=1))
    print(f"S5 pass: {ok_all} ({sum(r[1] for r in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------
# S6: testable and portable chain
# ---------------------------------------------------------------------------------------------


def names() -> int:
    """S6, AC6.1-AC6.6 (AC6.4's second half, the whole chain on a scratch store, is S8's ``--integration``)."""
    import glob
    import subprocess

    from vlapoc.runs import scratch_lineage_copy, stage_run
    import stage as stage_mod

    t0 = time.time()
    sha0 = db_sha()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    g = subprocess.run(["git", "grep", "-n", "-E", "/ra" "id|/auto/" "home", "--", "slurm", "scripts", "vlapoc"], cwd=REPO_ROOT, capture_output=True, text=True)
    check("6.1 no machine-specific cluster path (the two cluster roots) is left in slurm, scripts or vlapoc", g.stdout.strip() == "", g.stdout.strip()[:200] or "no matches")
    files = sorted(glob.glob(str(REPO_ROOT / "slurm" / "*.sh")) + glob.glob(str(REPO_ROOT / "slurm" / "*.sbatch")))
    bad = [f for f in files if subprocess.run(["bash", "-n", f]).returncode != 0]
    check("6.2 bash -n passes on every slurm/*.sh and *.sbatch", not bad and len(files) == 4, f"{len(files)} files")
    r = subprocess.run(["bash", "-c", f"cd {REPO_ROOT} && SITE_ENV={SCRATCH}/does_not_exist.env && export SITE_ENV && source slurm/env.sh"], capture_output=True, text=True)
    check("6.2b env.sh without a site.env exits non-zero with a message naming site.env.example", r.returncode != 0 and "site.env.example" in r.stderr, f"rc={r.returncode}: {r.stderr.strip()[:140]}")
    # 6.3 chain-scoped names on a scratch store
    shutil.rmtree(SCRATCH / "names", ignore_errors=True)
    base = SCRATCH / "names"
    base.mkdir(parents=True)
    try:
        lr = scratch_lineage_copy(base / "lineage", mlflow_db=False)
        ids = {}
        for chain in ("t-aaaa1111", "u-bbbb2222", "t-cccc3333"):
            with stage_run("train", "train-t-s0", lr, chain=chain, allow_dirty=True) as r_:
                ids[chain] = r_.run_id
        got = stage_mod.run_id("train-t-s0", "train", "u-", False, lr)
        check("6.3 run_id with a chain prefix returns the run of that chain only", got == ids["u-bbbb2222"], f"u- scope -> {got[:8]} (u-bbbb2222 is {ids['u-bbbb2222'][:8]})")
        try:
            stage_mod.run_id("train-t-s0", "train", "t-", False, lr)
            check("6.3b two runs of one name in scope (t-aaaa1111, t-cccc3333) raise without --latest", False)
        except ValueError as e:
            check("6.3b two runs of one name in scope (t-aaaa1111, t-cccc3333) raise without --latest", "--latest" in str(e), str(e)[:130])
        got = stage_mod.run_id("train-t-s0", "train", "t-", True, lr)
        check("6.3c ... and --latest takes the most recent of them", got == ids["t-cccc3333"])
        check("6.3d without a chain (hand use) the most recent run of the name wins, as before",
              stage_mod.run_id("train-t-s0", "train", None, False, lr) == ids["t-cccc3333"])
        check("6.3e chain prefixes derive from chain ids", stage_mod.chain_prefix_of("v0cand-1a2b3c4d") == "v0cand-" and stage_mod.chain_prefix_of(None) is None)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    # 6.4 every stage call is threaded with the lineage root
    src = (REPO_ROOT / "scripts" / "stage.py").read_text()
    tree = ast.parse(src)
    bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "client" and not n.args]
    bare_lr = [n.lineno for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name.startswith("cmd_") for n in ast.walk(fn)
               if isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("train_run", "eval_run", "returns_run", "attribute_run", "ensemble_run", "select_run", "verify_run", "recovery_run", "summary_run")
               and not any(k.arg == "lineage_root" for k in n.keywords)]
    p = subprocess.run([sys.executable, "scripts/stage.py", "--help"], cwd=REPO_ROOT, capture_output=True, text=True)
    check("6.4 stage.py has a global --lineage-root; no client() call without it; every tracked-stage call passes lineage_root=LR",
          "--lineage-root" in p.stdout and not bare and not bare_lr, f"bare client(): {bare}; calls missing lineage_root: {bare_lr}. The whole chain on a scratch store is S8 (--integration)")
    # 6.5 dry run flags
    out = subprocess.run([sys.executable, "scripts/server_pipeline.py", "--tag", "t", "--dry-run", "--curation", "per_task", "--seeds", "0"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    ex = {k: v for k, v in (ln.split("=", 1) for ln in (REPO_ROOT / "slurm" / "site.env.example").read_text().splitlines() if "=" in ln and not ln.startswith("#"))}
    argv = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("sbatch ")]
    want_log = ex["LOG_DIR"].replace("${SCRATCH_ROOT}", ex["SCRATCH_ROOT"])
    arr = [a_ for a_ in argv if "--array" in a_]
    check("6.5 the dry run prints the full sbatch argv with --partition=<PARTITION> and --output=<LOG_DIR>/... from site.env.example",
          len(argv) > 10 and all(f"--partition={ex['PARTITION']}" in a_ and f"--output={want_log}/" in a_ for a_ in argv) and all("%A_%a" in a_ for a_ in arr) and arr,
          f"{len(argv)} sbatch lines; e.g. {argv[0][:200]}")
    # 6.6 verify_attribution leaves nothing outside data/scratch
    tmp = REPO_ROOT / "data" / "scratch" / "tmpdir_probe"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    env = {**os.environ, "TMPDIR": str(tmp)}
    rr = subprocess.run([sys.executable, "scripts/verify_attribution.py"], cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    left = list(tmp.iterdir())
    pat = "tempfile.TemporaryDirectory()" in (REPO_ROOT / "scripts" / "verify_attribution.py").read_text()
    check("6.6 verify_attribution.py with TMPDIR set to an empty dir under data/scratch leaves it empty, and uses explicit dir=",
          rr.returncode == 0 and not left and not pat, f"rc {rr.returncode}, entries left in TMPDIR: {[p_.name for p_ in left]}; "
          f"{[ln for ln in rr.stdout.splitlines() if ln.startswith('pass:')]}")
    shutil.rmtree(tmp, ignore_errors=True)
    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    # round 2: LOG_DIR is created before the first real sbatch (sbatch does not create the --output directory; a missing one
    # fails the job silently, so the whole chain would have no logs). Stubbed subprocess.run, nothing is submitted.
    import subprocess as sp_

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import server_pipeline as sp

    logd = SCRATCH / "logdir_probe" / "a" / "logs"
    shutil.rmtree(SCRATCH / "logdir_probe", ignore_errors=True)
    seen = {}

    class _R:
        stdout = "4242"

    def _stub(argv, **kw):
        seen["argv"], seen["exists_at_submit"] = argv, logd.is_dir()
        return _R()

    real = sp.subprocess.run
    sp.subprocess.run = _stub
    try:
        sp.Chain(False, [], "deadbeef", "t-deadbeef", "gpu", str(logd)).submit("cpu.sbatch", "python scripts/stage.py returns", "probe")
    finally:
        sp.subprocess.run = real
    check("6.7 a real (non-dry) submission creates LOG_DIR first (mkdir -p) so sbatch can open --output",
          seen.get("exists_at_submit") is True and f"--output={logd}/" in " ".join(seen["argv"]), f"dir existed at the sbatch call: {seen.get('exists_at_submit')}")
    shutil.rmtree(SCRATCH / "logdir_probe", ignore_errors=True)
    ok_all = all(r_[1] for r_ in results)
    (ARTIFACTS / "s6_names.json").write_text(json.dumps({"pass": ok_all, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results]}, indent=1))
    print(f"S6 pass: {ok_all} ({sum(r_[1] for r_ in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------------------------
# S7: metric-resolved outputs (SYNTHETIC inputs)
# ---------------------------------------------------------------------------------------------


def groups() -> int:
    """S7, AC7.1-AC7.4 on SYNTHETIC tables (planted groups; a fixture attribute run for the stage)."""
    import numpy as np

    from vlapoc import analysis as an
    from vlapoc.runs import client

    t0 = time.time()
    sha0 = db_sha()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    rng = np.random.default_rng(21)
    n = 240
    cols = [f"harm_{m}" for m in an.DEFAULT_METRICS]
    tab = pd.DataFrame({"episode_id": [f"e{i}" for i in range(n)], "content_hash": [f"h{i}" for i in range(n)],
                        "episode_index": range(n), "suite": "s", "libero_task_id": [i % 6 for i in range(n)],
                        "task_name": [f"task{i % 6}" for i in range(n)]})
    for c in cols:
        tab[c] = rng.normal(scale=0.3, size=n)
    plant = {"consistent_helper": {c: -5 for c in cols}, "consistent_harmer": {c: 5 for c in cols},
             "trade_off": {"harm_f30": 5, "harm_rmst60": -5}, "metric_specific": {"harm_hrt60_lin": 5}}
    expect = pd.Series("neutral", index=tab.index, dtype=object)
    for k, (g, spec) in enumerate(plant.items()):
        for j in range(5):
            i = 10 * k + j
            for c, v in spec.items():
                tab.loc[i, c] = v
            expect[i] = g
    lab = an.metric_groups(tab, cols)
    planted = expect != "neutral"
    ok = bool((lab[planted] == expect[planted]).all())
    check("7.1 every planted helper, harmer, trade-off and metric-specific episode gets its label (4/4 groups, 20/20 episodes)", ok,
          f"{int((lab[planted] == expect[planted]).sum())}/{int(planted.sum())} planted correct; counts {lab.value_counts().to_dict()}")
    check("7.1b the unplanted noise episodes are neutral", bool((lab[~planted] == "neutral").all()), f"{int((lab[~planted] == 'neutral').sum())}/{int((~planted).sum())}")
    agg = an.aggregate_by(tab, "task_name", cols, n_boot=200)
    check("7.2 aggregate_by(task_name): one row per task, n sums to the table's rows, mean inside its CI",
          len(agg) == 6 and agg["n"].sum() == n and bool(((agg["harm_f30__lo"] <= agg["harm_f30__mean"]) & (agg["harm_f30__mean"] <= agg["harm_f30__hi"])).all()),
          f"{len(agg)} rows, n sum {agg['n'].sum()}")
    errs = []
    for bad in ("not_a_column", "episode_id"):
        try:
            an.aggregate_by(tab, bad, cols)
            errs.append(False)
        except ValueError as e:
            errs.append(True)
    check("7.2b aggregate_by refuses a non-catalog column and a per-episode identifier", all(errs))

    shutil.rmtree(SCRATCH / "groups", ignore_errors=True)
    base = SCRATCH / "groups"
    base.mkdir(parents=True)
    try:
        fx = make_fixture(base)
        ev = add_eval(fx, "a")
        ra = run_attribute(fx, ev, "attribute-fixture-groups")
        res = an.analyze_run(ra["run_id"], "analyze-fixture", out_dir=base / "analysis", lineage_root=fx["lr"], allow_dirty=True, synthetic=True)
        out = base / "analysis"
        pngs = {p_.name: p_.stat().st_size for p_ in out.glob("*.png")}
        check("7.3 the heatmap and scatter PNGs are written", set(pngs) == {"heatmap.png", "tradeoff.png"} and all(v > 5000 for v in pngs.values()), str(pngs))
        lab_json = json.loads((out / "figure_labels.json").read_text())
        decl = {r["column"]: r for r in json.loads((out / "figure_labels.json").read_text()) and at_decls(fx, ra, base)}
        xl, yl = lab_json["tradeoff"]["xlabel"], lab_json["tradeoff"]["ylabel"]
        check("7.3b each axis label names the column's functional and τ from functionals.json, and the titles say SYNTHETIC",
              decl["harm_rmst60"]["functional"] in xl and f"τ {decl['harm_rmst60']['tau']}" in xl and decl["harm_f30"]["functional"] in yl
              and f"τ {decl['harm_f30']['tau']}" in yl and "SYNTHETIC" in lab_json["heatmap"]["title"] and "SYNTHETIC" in lab_json["tradeoff"]["title"]
              and all("τ " in t for t in lab_json["heatmap"]["xticklabels"]), f"x: {xl.splitlines()[0]} | y: {yl.splitlines()[0]}")
        prm = client(fx["lr"]).get_run(res["run_id"]).data.params
        check("7.4 the analyze run's params carry the group threshold and the descriptive-unverified status, and link the score run",
              prm["group_z_threshold"] == "1.0" and prm["group_status"].startswith("descriptive, unverified") and prm["score_run_id"] == ra["run_id"]
              and prm["synthetic"] == "True", f"z={prm['group_z_threshold']}, counts {prm['group_counts']}")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for f in ("heatmap.png", "tradeoff.png"):
            shutil.copy(out / f, ARTIFACTS / f"s7_{f.removesuffix('.png')}_SYNTHETIC.png")
        shutil.copy(out / "figure_labels.json", ARTIFACTS / "s7_figure_labels_SYNTHETIC.json")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    # the planted table too, as figures (clearer than the 15-episode fixture)
    decls_p = {c: {"column": c, "functional": f"({c} functional)", "tau": "60 s", "rollout_subset": "all", "good_sign": 1.0} for c in cols}
    an.plot_heatmap(tab, cols, lab, decls_p, ARTIFACTS / "s7_heatmap_planted_SYNTHETIC.png", True)
    an.plot_tradeoff(tab, lab, decls_p, ARTIFACTS / "s7_tradeoff_planted_SYNTHETIC.png", "harm_rmst60", "harm_f30", True)
    from verify_method_boundary_s3 import scan_boundary

    viol = scan_boundary([REPO_ROOT / "vlapoc" / "analysis.py"])
    check("7.5 the group module passes the ground-truth boundary scan (no groundtruth import, no label files)", viol == [], f"violations {viol}")
    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    ok_all = all(r_[1] for r_ in results)
    (ARTIFACTS / "s7_groups.json").write_text(json.dumps({"SYNTHETIC": True, "pass": ok_all, "checks": [{"name": n_, "pass": p_, "detail": d} for n_, p_, d in results]}, indent=1))
    print(f"S7 pass: {ok_all} ({sum(r_[1] for r_ in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


def at_decls(fx: dict, ra: dict, base: Path) -> list[dict]:
    """The ``functionals.json`` rows of a fixture attribute run."""
    from vlapoc import attribution as at
    from vlapoc.runs import client

    return at._artifact_json(client(fx["lr"]).get_run(ra["run_id"]), "functionals", fx["lr"])


# ---------------------------------------------------------------------------------------------
# S8: the chain end to end on this Mac
# ---------------------------------------------------------------------------------------------


def stage_cli(args: list[str], lr: Path, feats: Path, chain: str) -> dict:
    """``scripts/stage.py`` as a subprocess on a scratch store; returns the JSON it prints (raises with stderr on failure)."""
    import subprocess

    p = subprocess.run([sys.executable, "scripts/stage.py", "--lineage-root", str(lr), "--feats-root", str(feats), "--chain", chain, *args],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"stage.py {' '.join(args)} failed rc={p.returncode}:\n{p.stderr[-1500:]}")
    out = p.stdout
    return json.loads(out[out.index("{"):])


def integration() -> int:
    """S8(a), AC8.1, AC8.2, AC8.4 and the second half of AC6.4: attribute x2 -> ensemble -> select (per-task harm / help / random, global
    harm / random) -> score -> analyze, all through ``scripts/stage.py --lineage-root`` on a scratch copy of the store, with SYNTHETIC
    features, rollouts and returns over a real v3 training subset. Needs a clean ``vlapoc/``+``scripts/`` tree (the CLI has no allow_dirty)."""
    import subprocess

    import numpy as np

    from vlapoc import attribution as at
    from vlapoc.lineage import LineageError, Subset
    from vlapoc.runs import client, select_run, trace

    t0 = time.time()
    sha0 = db_sha()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print(f"commit first: the stage CLI refuses a dirty tree:\n{dirty}")
        return 2
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    shutil.rmtree(SCRATCH / "integration", ignore_errors=True)
    base = SCRATCH / "integration"
    base.mkdir(parents=True)
    CH, T = "ti-fixture0", "ti"
    made_dirs = [REPO_ROOT / "data" / "attribution" / n for n in (f"attribute-{T}-s0", f"attribute-{T}-s1", f"attribute-{T}-ens", f"score-{T}", f"analyze-{T}")]
    try:
        fx = make_fixture(base, parent="v3/candidate", cover_types=True, chain=CH, train_name=f"train-{T}-s0")
        lr, feats = fx["lr"], base / "feats"
        sub = fx["subset"]
        tr0 = fx["train"]
        tr1 = make_train(fx, f"train-{T}-s1", sub, "cd" * 32, seed=1)
        ev0 = add_eval(fx, "s0", train=tr0, eval_name=f"eval-{T}-s0", returns_name=f"returns-{T}-s0", seed=1)
        ev1 = add_eval(fx, "s1", train=tr1, eval_name=f"eval-{T}-s1", returns_name=f"returns-{T}-s1", seed=2)
        eps = fx["episodes"]
        n_types = len(set(eps["episode_id"]) & set(pd.read_parquet(lr / "versions" / sub.dataset_version_id / "labels.parquet")["episode_id"]))
        print(f"fixture (SYNTHETIC features/rollouts/returns): {len(eps)} v3 episodes in {eps[['suite', 'libero_task_id']].drop_duplicates().shape[0]} tasks, {n_types} injected, "
              f"2 seeds; scratch store {lr.relative_to(REPO_ROOT)}")
        C = lambda *a: stage_cli(list(a), lr, feats, CH)  # noqa: E731
        outs = {}
        for s_ in (0, 1):
            outs[f"attribute-s{s_}"] = C("attribute", "--train", f"train-{T}-s{s_}", "--eval", f"eval-{T}-s{s_}", "--returns", f"returns-{T}-s{s_}", "--name", f"attribute-{T}-s{s_}", "--device", "cpu")
        outs["ensemble"] = C("ensemble", "--runs", f"attribute-{T}-s0", f"attribute-{T}-s1", "--name", f"attribute-{T}-ens")
        sels = {"harm": ["--op", "drop_top_k_per_task", "--k", "1", "--metric", "harm_f30__same_task", "--direction", "highest", "--headline"],
                "help": ["--op", "drop_top_k_per_task", "--k", "1", "--metric", "harm_f30__same_task", "--direction", "lowest", "--headline"],
                "random": ["--op", "drop_random_k_per_task", "--k", "1", "--seed", "0", "--headline"],
                "gharm": ["--op", "drop_top_k", "--k", "2", "--metric", "harm_f30", "--direction", "highest"],
                "grandom": ["--op", "drop_random_k", "--k", "2", "--seed", "0"]}
        for k, a_ in sels.items():
            outs[f"select-{k}"] = C("select", "--parent", "fixture/train", "--name", f"select-{T}-{k}", "--scores", f"attribute-{T}-ens", "--pipeline-curation", "both", *a_)
        outs["score"] = C("score", "--scores", f"attribute-{T}-ens", "--version", "v3", "--subset", "fixture20", "--name", f"score-{T}", "--synthetic")
        outs["analyze"] = C("analyze", "--scores", f"attribute-{T}-ens", "--name", f"analyze-{T}", "--synthetic")
        print("stage runs through the CLI: " + ", ".join(f"{k}={v['run_id'][:8]}" for k, v in outs.items()))
        c = client(lr)
        n_eps = len(eps)
        per_task = eps.groupby(["suite", "libero_task_id"]).size()
        check("8.1 attribute x2, ensemble, five selects, score and analyze all ran through the stage CLI on a scratch store (--lineage-root; AC6.4)",
              len(outs) == 10 and outs["attribute-s0"]["n_episodes"] == n_eps and outs["ensemble"]["n_models"] == 2,
              f"{len(outs)} stage runs; attribute table {outs['attribute-s0']['n_episodes']} episodes x {outs['attribute-s0']['n_rollouts']} rollouts")
        ch = c.get_run(outs["select-harm"]["run_id"])
        kept = {k: outs[f"select-{k}"]["n_episodes"] for k in sels}
        check("8.1b selects: per-task arms keep n - (tasks x 1), global arms n - 2, per-task counts of curated and random children are identical",
              kept["harm"] == kept["help"] == kept["random"] == n_eps - len(per_task) and kept["gharm"] == kept["grandom"] == n_eps - 2
              and ch.data.params["curation"] == "per_task" and ch.data.params["headline"] == "True" and "metric_declaration" in ch.data.params,
              f"kept {kept} of {n_eps}; harm select declares {json.loads(ch.data.params['metric_declaration'])['functional']}")
        # AC8.1: trace from the select run
        tr_ = trace(outs["select-harm"]["run_id"], lr)
        stages = sorted({r["stage"] for r in tr_["runs"].values()})
        n_feat = sum(1 for e in tr_["edges"] if e["consumes"].startswith("features:"))
        run_names = {r["name"] for r in tr_["runs"].values()}
        need_runs = {f"attribute-{T}-s0", f"attribute-{T}-s1", f"attribute-{T}-ens", f"returns-{T}-s0", f"returns-{T}-s1", f"eval-{T}-s0", f"eval-{T}-s1", f"train-{T}-s0", f"train-{T}-s1"}
        check("8.2 trace from a select run reaches both attribute runs and the ensemble, both seeds' returns, eval and train runs, every feature digest and the training subset",
              need_runs <= run_names and n_feat == 12 and sub.subset_id in tr_["entities"],
              f"stages {stages}; {len(tr_['runs'])} runs; {n_feat} features inputs (2 seeds x 6 files); training subset {sub.subset_id[:12]} in entities: {sub.subset_id in tr_['entities']}; "
              f"dead ends (no producer): {sorted({e['consumes'].split(':')[0] for e in tr_['edges'] if not e['produced_by']})}")
        # recovery output of the CLI score stage (SYNTHETIC random scores: only the shape is checked)
        rm = pd.read_csv(REPO_ROOT / "data" / "attribution" / f"score-{T}" / "recovery_matrix.csv")
        sc = c.get_run(outs["score"]["run_id"]).data.params
        check("8.3 the score stage ran on the v3 subset (recovery matrix has 7 types + pooled per harm column, flagged synthetic)",
              set(rm["type"]) >= {"idle_padding", "gripper_release", "pooled"} and rm["type"].nunique() == 8 and sc["synthetic"] == "True", f"{len(rm)} rows, {rm['column'].nunique()} harm columns")
        # AC8.2 negatives: nothing written
        n_before = n_runs(lr)
        sha_b = "ef" * 32
        sub_b = Subset(sub.dataset_version_id, tuple(sub.episode_ids[:-1]), {"op": "synthetic_fixture_b", "parent_subset_id": sub.derivation["parent_subset_id"]})
        sub_b.save(lr)
        tr_b = make_train(fx, f"train-{T}-b", sub_b, sha_b, seed=7)
        ev_b = add_eval(fx, "b", train=tr_b, eval_name=f"eval-{T}-b", returns_name=f"returns-{T}-b", seed=8)
        ra_b = run_attribute(fx, ev_b, f"attribute-{T}-b")  # a valid run on another subset (expected to be written)
        n_neg0 = n_runs(lr)
        neg = []

        def refuse(label, fn, exc, contains):
            try:
                fn()
                neg.append((label, False, "did not raise"))
            except exc as e:
                neg.append((label, contains in str(e) and n_runs(lr) == n_neg0, str(e)[:110]))

        alt = feats / "alt"
        for suffix, mutate in (("ckpt", lambda m: m.update(checkpoint_sha256="99" * 32)), ("eps", lambda m: m.update(episodes=m["episodes"][:-1]))):
            for f in (feats / tr0["name"]).glob("train.shard*"):
                shutil.copy(f, alt / suffix / f.name) if (alt / suffix).mkdir(parents=True, exist_ok=True) is None else None
            for mf in (alt / suffix).glob("*.manifest.json"):
                m = json.loads(mf.read_text())
                mutate(m)
                mf.write_text(json.dumps(m))
        refuse("features from another checkpoint", lambda: run_attribute(fx, ev0, "attribute-bad-ckpt") if False else
               at.attribute_run(tr0["train_id"], ev0["eval_id"], ev0["returns_id"], alt / "ckpt" / "train", ev0["rollout_prefix"], "attribute-bad-ckpt",
                                out_dir=base / "out" / "bad1", device="cpu", lineage_root=lr, allow_dirty=True), ValueError, "another checkpoint")
        refuse("train features on a different episode list", lambda: at.attribute_run(tr0["train_id"], ev0["eval_id"], ev0["returns_id"], alt / "eps" / "train", ev0["rollout_prefix"],
                                                                                    "attribute-bad-eps", out_dir=base / "out" / "bad2", device="cpu", lineage_root=lr, allow_dirty=True),
               ValueError, "exactly the training subset")
        # round 2: bind features to the data they claim (PoC §2.1 item 1 p.3; §9.2 p.19)
        for sub_dir, files, upd in (("root", list((feats / tr0["name"]).glob("train.shard*")), {"dataset_root": "data/lerobot/libero_spatial_10"}),
                                    ("eval", list((feats / tr0["name"]).glob("rollouts-" + ev0["name"] + "*")),
                                     {"dataset_root": json.loads((ev1["rollout_prefix"].parent / (ev1["rollout_prefix"].name + ".manifest.json")).read_text())["dataset_root"]})):
            (alt / sub_dir).mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, alt / sub_dir / f.name)
            for mf in (alt / sub_dir).glob("*.manifest.json"):
                m = json.loads(mf.read_text())
                m.update(upd)
                mf.write_text(json.dumps(m))
        refuse("train features from another version's root (v0 root for a v3 subset)", lambda: at.attribute_run(
            tr0["train_id"], ev0["eval_id"], ev0["returns_id"], alt / "root" / "train", ev0["rollout_prefix"], "attribute-bad-root", out_dir=base / "out" / "bad5",
            device="cpu", lineage_root=lr, allow_dirty=True), ValueError, "not on the training subset's version copy")
        refuse("rollout features computed on another eval's trajectories", lambda: at.attribute_run(
            tr0["train_id"], ev0["eval_id"], ev0["returns_id"], tr0["prefix"], alt / "eval" / ("rollouts-" + ev0["name"]), "attribute-bad-eval", out_dir=base / "out" / "bad6",
            device="cpu", lineage_root=lr, allow_dirty=True), ValueError, "not the trajectories copy")
        tab_b = at._artifact_table(c.get_run(ra_b["run_id"]), "attribution_scores", lr)
        pb = base / "tab_b.parquet"
        tab_b.to_parquet(pb, index=False)
        refuse("a score table from another subset at select", lambda: select_run("fixture/train", "select-bad", "drop_top_k_per_task", k=1, score_path=pb, metric="harm_f30__same_task",
                                                                                score_run_id=ra_b["run_id"], lineage_root=lr, allow_dirty=True), LineageError, "keys")
        refuse("an ensemble of two runs on different subsets", lambda: at.ensemble_run([outs["attribute-s0"]["run_id"], ra_b["run_id"]], "ens-bad", out_dir=base / "out" / "bad4",
                                                                                      lineage_root=lr, allow_dirty=True), ValueError, "different subsets")
        for label, ok, msg in neg:
            check(f"8.4 refused with nothing written: {label}", ok, msg)
        check("8.4b the integration run left the real mlflow.db sha256 unchanged", db_sha() == sha0, f"{sha0[:16]}...")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s8_trace_select_SYNTHETIC.json").write_text(json.dumps({"SYNTHETIC": "fixture runs on a scratch store", "stages": stages, "n_runs": len(tr_["runs"]),
                                                                          "runs": sorted(r["name"] for r in tr_["runs"].values()), "n_feature_inputs": n_feat,
                                                                          "edges": [{k: e[k] for k in ("consumes", "role")} | {"produced_by": len(e["produced_by"])} for e in tr_["edges"]]}, indent=1))
    finally:
        shutil.rmtree(base, ignore_errors=True)
        for d in made_dirs:
            shutil.rmtree(d, ignore_errors=True)
    check("R7 real mlflow.db sha256 unchanged", db_sha() == sha0, f"{sha0[:16]}...")
    ok_all = all(r_[1] for r_ in results)
    (ARTIFACTS / "s8_integration.json").write_text(json.dumps({"SYNTHETIC": True, "pass": ok_all, "checks": [{"name": n_, "pass": p_, "detail": d} for n_, p_, d in results], "seconds": round(time.time() - t0)}, indent=1))
    print(f"S8(a) pass: {ok_all} ({sum(r_[1] for r_ in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


def local_smoke(device: str, train_ref: str = "0a33bc75bb71433ca2def5220014dde0") -> int:
    """S8(b), AC8.3: the chain on real objects at toy scale, on this Mac: the tracked SmolVLA train run ``train_ref`` (2 steps on
    v0/reference, checkpoint on disk), a smoke eval (1 task x 2 init states, trajectories recorded), returns, features (one sample per
    training episode, a few per rollout), attribute and select, all through ``stage.py --lineage-root`` on a scratch copy of the store.
    NOT attribution-grade: a 2-step checkpoint, 2 rollouts, 1 sample per episode. The first failing stage is recorded verbatim."""
    import subprocess

    from vlapoc.runs import scratch_lineage_copy

    t0 = time.time()
    sha0 = db_sha()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print(f"commit first: the stage CLI refuses a dirty tree:\n{dirty}")
        return 2
    shutil.rmtree(SCRATCH / "smoke", ignore_errors=True)
    base = SCRATCH / "smoke"
    base.mkdir(parents=True)
    CH, T = "ts-smoke0", "ts-smoke"
    lr = scratch_lineage_copy(base / "lineage")
    feats = base / "feats"
    log: list[dict] = []
    outs: dict = {}
    failed = None
    steps = [
        ("eval", ["eval", "--train", train_ref, "--name", f"eval-{T}", "--smoke", "--tasks", "libero_spatial:0", "--n", "2", "--batch-size", "2", "--device", device]),
        ("returns", ["returns", "--eval", f"eval-{T}", "--name", f"returns-{T}"]),
        ("features-train", ["features", "--train", train_ref, "--which", "train", "--train-stride", "100000", "--batch-size", "8", "--workers", "0", "--device", device]),
        ("features-rollouts", ["features", "--train", train_ref, "--which", "rollouts", "--eval", f"eval-{T}", "--rollout-stride", "150", "--batch-size", "8", "--workers", "0", "--device", device]),
        ("attribute", ["attribute", "--train", train_ref, "--eval", f"eval-{T}", "--returns", f"returns-{T}", "--name", f"attribute-{T}", "--device", "cpu"]),
        ("select", ["select", "--parent", "v0/reference", "--name", f"select-{T}", "--op", "drop_top_k", "--k", "2", "--metric", "harm_f30", "--scores", f"attribute-{T}", "--pipeline-curation", "global"]),
    ]
    try:
        for name, args in steps:
            t1 = time.time()
            try:
                outs[name] = stage_cli(args, lr, feats, CH)
                log.append({"stage": name, "ok": True, "seconds": round(time.time() - t1, 1), "output": {k: v for k, v in outs[name].items() if k != "agreement"}})
                print(f"[ok] {name} {time.time() - t1:.0f}s {json.dumps(log[-1]['output'])[:300]}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed = {"stage": name, "error": str(e)[-1800:]}
                log.append({"stage": name, "ok": False, "seconds": round(time.time() - t1, 1), "error": str(e)[-1800:]})
                print(f"[FAILED] {name}: {str(e)[-1200:]}", flush=True)
                break
        report: dict = {"label": "smoke (not attribution-grade): 2-step SmolVLA checkpoint, 2 rollouts, 1 train sample per episode", "device": device,
                        "train_run": train_ref, "stages": log, "failed": failed}
        if failed is None:
            import numpy as np

            from vlapoc import attribution as at
            from vlapoc.runs import client

            c = client(lr)
            att = c.get_run(outs["attribute"]["run_id"])
            tab = at._artifact_table(att, "attribution_scores", lr)
            agree = json.loads((REPO_ROOT / "data" / "attribution" / f"attribute-{T}" / "agreement.json").read_text())
            fin = tab[[c_ for c_ in tab.columns if c_.startswith("harm_") and "__" not in c_]]
            sel = c.get_run(outs["select"]["run_id"])
            prm = att.data.params
            report |= {"n_rollouts": outs["attribute"]["n_rollouts"], "n_episodes": outs["attribute"]["n_episodes"], "score_table_shape": list(tab.shape),
                       "tau_agreement": {k: v for k, v in agree.items()}, "all_finite_harm_columns": bool(np.isfinite(fin.to_numpy()).all()),
                       "train_output_fn": prm["train_output_fn"], "rollout_output_fn": prm["rollout_output_fn"],
                       "feature_files_recorded": int(prm["n_feature_files"]), "select_declared_functional": json.loads(sel.data.params["metric_declaration"])["functional"],
                       "select_child_episodes": outs["select"]["n_episodes"]}
            print(f"smoke (not attribution-grade): n_rollouts {report['n_rollouts']}, n_episodes {report['n_episodes']}, score table {report['score_table_shape']}, "
                  f"finite harm columns {report['all_finite_harm_columns']}, tau agreement pairs {len(agree)} (values: {[round(v['tau'], 3) if v['tau'] == v['tau'] else None for v in agree.values()][:4]} ... n per pair {[v['n'] for v in agree.values()][:4]})")
        report["mlflow_db_sha256_before"], report["mlflow_db_sha256_after"] = sha0, db_sha()
        report["seconds"] = round(time.time() - t0)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s8_local_smoke.json").write_text(json.dumps(report, indent=1, default=str))
    finally:
        shutil.rmtree(base, ignore_errors=True)
        for d in (REPO_ROOT / "data" / "eval" / f"eval-{T}", REPO_ROOT / "data" / "attribution" / f"attribute-{T}", REPO_ROOT / "data" / "eval" / f"returns-{T}"):
            shutil.rmtree(d, ignore_errors=True)
    unchanged = db_sha() == sha0
    print(f"real mlflow.db unchanged: {unchanged}; failed: {failed['stage'] if failed else None} ({time.time() - t0:.0f}s)")
    return 0 if (failed is None and unchanged) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blind", action="store_true", help="S1")
    ap.add_argument("--functionals", action="store_true", help="S3")
    ap.add_argument("--pipeline", type=Path, nargs="?", const=Path("data/scratch/dry.txt"), default=None,
                    help="S4 AC4.1-4.3, 4.6: parse server_pipeline.py --dry-run (writes the parsed jobs to this path)")
    ap.add_argument("--recovery", action="store_true", help="S5 AC5.1-5.6: recovery matrix on SYNTHETIC planted scores")
    ap.add_argument("--names", action="store_true", help="S6 AC6.1-6.6: site config, chain-scoped names, no /tmp")
    ap.add_argument("--groups", action="store_true", help="S7 AC7.1-7.4: four groups, aggregation, figures (SYNTHETIC)")
    ap.add_argument("--integration", action="store_true", help="S8(a): the chain through stage.py on a scratch store (SYNTHETIC data; clean tree)")
    ap.add_argument("--local-smoke", action="store_true", help="S8(b): real SmolVLA checkpoint, toy-scale chain on this Mac (needs --device)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--step4", action="store_true", help="S4 AC4.4-4.5: step4-summary on synthetic verdicts, select arm counts")
    ap.add_argument("--manifest", type=Path, default=None, help="S3 AC3.2: a manifest written by influence.featurize")
    a = ap.parse_args()
    if a.blind:
        return blind()
    if a.functionals:
        return functionals(a.manifest)
    if a.pipeline is not None:
        return pipeline(a.pipeline)
    if a.step4:
        return step4_synthetic()
    if a.recovery:
        return recovery()
    if a.names:
        return names()
    if a.groups:
        return groups()
    if a.integration:
        return integration()
    if a.local_smoke:
        return local_smoke(a.device)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
