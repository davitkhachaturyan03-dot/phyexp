"""Verify the training-and-attribution feature (``tracker/features/2026-10-20-training-attribution/``), story by story.

Grounding: PoC §2.1 item 4 p.3, §3.4 p.7, §4.2-§4.5 pp.9-11, §9.1 Steps 3-4 p.19, §9.2 p.19; owner decision 3
(ground-truth feature); CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) App. A.1-A.2.

    .venv/bin/python scripts/verify_training_attribution.py --blind        # S1: refusal table (~2-3 min)
    .venv/bin/python scripts/verify_training_attribution.py --estimator    # S2: toy-torch, CUPID parity, centring algebra
    .venv/bin/python scripts/verify_training_attribution.py --functionals  # S3: functionals.json, ensemble, provenance
    .venv/bin/python scripts/verify_training_attribution.py --pipeline data/scratch/dry.txt   # S4: parse a dry run

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


def fake_train_run(lr: Path, name: str, subset, ckpt_sha: str = "0" * 64) -> str:
    """A FINISHED ``train`` run in the scratch store whose training input is ``subset`` (no model involved)."""
    import mlflow

    from vlapoc.runs import stage_run

    with stage_run("train", name, lr, allow_dirty=True) as r:
        r.input_subset(subset, "training", None)
        mlflow.set_tag("checkpoint.weights.sha256", ckpt_sha)
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


def make_fixture(base: Path, tasks: int = 5, per_task: int = 3, seed: int = 0) -> dict:
    """A scratch copy of the real lineage store with: a training subset of ``tasks`` x ``per_task`` clean v0/candidate
    episodes, a FINISHED fake ``train`` run on it, and synthetic train features (2 shards). SYNTHETIC: no model involved."""
    import numpy as np

    from vlapoc.lineage import Subset, _bind_name
    from vlapoc.runs import scratch_lineage_copy

    rng = np.random.default_rng(seed)
    lr = scratch_lineage_copy(base / "lineage")
    v0c = Subset.load("v0/candidate", lr)
    cat = v0c.episodes(lr).sort_values(["suite", "libero_task_id", "episode_index"])
    pick = pd.concat([g.head(per_task) for _, g in cat.groupby(["suite", "libero_task_id"], sort=True)][:tasks])
    sub = Subset(v0c.dataset_version_id, tuple(pick["episode_id"]), {"op": "synthetic_fixture", "parent_subset_id": v0c.subset_id})
    sub.save(lr)
    _bind_name("subsets", "fixture/train", sub.subset_id, lr)
    episodes = sub.episodes(lr).sort_values("episode_index").reset_index(drop=True)
    train_id = fake_train_run(lr, "fake-train-fixture", sub, FAKE_CKPT_SHA)
    n_per = 3
    ep_idx = np.repeat(episodes["episode_index"].to_numpy(), n_per)
    manifest = {"scheme": "vlapoc.features.v2", "checkpoint": "data/scratch/fake", "checkpoint_sha256": FAKE_CKPT_SHA,
                "checkpoint_weights_sha256_recheck": True, "dataset_root": "data/lerobot/libero_spatial_10",
                "dataset_fps": 20, "stride": 2, "rank": 4, "k_draws": 4, "seed": 0, "dim": FEATURE_DIM, "n_layers": 1,
                "layers": ["l0"], "rename": None, "chunk_size": 50, "output_fn": "square",
                "episodes": [int(e) for e in sub.resolve(lr)]}
    train_prefix = base / "feats" / "train"
    write_features(train_prefix, rng.normal(size=(len(ep_idx), FEATURE_DIM)), ep_idx, manifest, 2)
    return {"lr": lr, "subset": sub, "episodes": episodes, "train_id": train_id, "train_prefix": train_prefix,
            "manifest": manifest, "base": base, "rng": rng}


def add_eval(fx: dict, tag: str, per_task: int = 6, drop_traj: int | None = None, extra_failure_type: bool = False,
             seed: int = 1) -> dict:
    """An eval run (rollouts parquet with trajectory indices), its returns run, and rollout features, all synthetic.

    ``drop_traj``: that rollout (row number) has no recorded trajectory and no features. ``extra_failure_type``: rollouts
    get one more failure type with enough members for its own column (so a seed can have columns another lacks).
    """
    import mlflow
    import numpy as np

    from vlapoc.catalog import sha256_file
    from vlapoc.metrics_registry import return_columns
    from vlapoc.runs import stage_run

    lr, rng = fx["lr"], np.random.default_rng(seed)
    eps = fx["episodes"]
    tasks = eps[["suite", "libero_task_id"]].drop_duplicates().to_records(index=False).tolist()
    rows = []
    for suite, tid in tasks:
        for j in range(per_task):
            rows.append({"suite": suite, "libero_task_id": int(tid)})
    ro = pd.DataFrame(rows)
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
    with stage_run("eval", f"fake-eval-{tag}", lr, allow_dirty=True) as r:
        r.params({"train_run_id": fx["train_id"], "checkpoint_sha256": FAKE_CKPT_SHA})
        r.output_file("rollouts", out / "rollouts.parquet", kind="rollouts")
        mlflow.set_tag("output.trajectories", "trajectories:synthetic")
        eval_id = r.run_id
    sha = sha256_file(out / "rollouts.parquet")
    with stage_run("returns", f"fake-returns-{tag}", lr, allow_dirty=True) as r:
        r.input_entity("rollouts", sha, "rollouts")
        r.output_file("returns", out / "returns.parquet", kind="returns")
        returns_id = r.run_id
    has = ro["traj_episode_index"].notna().to_numpy()
    traj = np.repeat(ro.loc[has, "traj_episode_index"].to_numpy().astype(int), 2)
    man = {**fx["manifest"], "episodes": None, "stride": 4}
    prefix = fx["base"] / "feats" / f"rollouts-{tag}"
    write_features(prefix, rng.normal(size=(len(traj), FEATURE_DIM)), traj, man, 1)
    return {"eval_id": eval_id, "returns_id": returns_id, "rollout_prefix": prefix, "n_rollouts": n}


def run_attribute(fx: dict, ev: dict, name: str, **kw) -> dict:
    """``attribute_run`` on the fixture (CPU, scratch store, allow_dirty)."""
    from vlapoc.attribution import attribute_run

    return attribute_run(fx["train_id"], ev["eval_id"], ev["returns_id"], fx["train_prefix"], ev["rollout_prefix"], name,
                         out_dir=fx["base"] / "out" / name, device="cpu", lineage_root=fx["lr"], allow_dirty=True, **kw)

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
                                            score_run_id=re_["run_id"], lineage_root=lr, allow_dirty=True), ValueError, "declares 0")
        check("3.6c select refuses a column the score run does not declare", ok, msg)
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
        check("3.2 featurize manifests are repo-relative", False, "pass --manifest <a manifest written by influence.featurize>")

    sha1 = db_sha()
    check("R7 real mlflow.db sha256 unchanged", sha0 == sha1, f"{sha0[:16]}... == {sha1[:16]}...")
    ok_all = all(r[1] for r in results)
    (ARTIFACTS / "s3_functionals.json").write_text(json.dumps({"pass": ok_all, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in results],
                                                               "seconds": round(time.time() - t0)}, indent=1))
    print(f"S3 pass: {ok_all} ({sum(r[1] for r in results)}/{len(results)}, {time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blind", action="store_true", help="S1")
    ap.add_argument("--functionals", action="store_true", help="S3")
    ap.add_argument("--manifest", type=Path, default=None, help="S3 AC3.2: a manifest written by influence.featurize")
    a = ap.parse_args()
    if a.blind:
        return blind()
    if a.functionals:
        return functionals(a.manifest)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
