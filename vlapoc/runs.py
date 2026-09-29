"""Every PoC stage as an MLflow run; lineage queries ``trace``, ``consumers``, ``diff``; ``check_scores``.

Grounding
---------
- PoC §2.1 item 1 (p. 3): "Every training run should be tied to exact dataset versions (content
  hashes), episode IDs, mixture weights, augmentation settings, the config, the git SHA, and the
  resulting checkpoints. Every eval rollout should be tied back to that checkpoint. W&B Artifacts,
  MLflow, or DVC handle this well. ... it lets you answer "what changed between the good run and the
  bad run?"". ``train_run`` records each listed field; ``diff`` answers the last question.
- PoC §2.3 (p. 4), Controls: "Fixed config and steps; 2–3 seeds per data configuration". ``diff``
  separates data, seed and config differences.
- PoC §9.1 (pp. 18–19): Step 1 screening verdicts "are stored rather than acted on, so they can later
  be compared with what actually happened"; Step 3 input "The training dataset, the trained policy, and
  the rollouts from Step 2"; Step 4 "Remove the top-k most harmful episodes and retrain. As a control,
  remove k random episodes and retrain." PoC §9.2 (p. 19): "The most compelling output is where
  screening and outcome attribution disagree", so every score table is keyed by
  (``episode_id``, ``content_hash``) and ``check_scores`` rejects keys the run never trained on.

Backend (owner answer BLOCKING Q1): MLflow 3.16.1, local SQLite at ``<lineage_root>/mlflow.db``,
artifacts in ``<lineage_root>/mlartifacts``; offline, no account.

How lineage is encoded in MLflow (all verified against mlflow 3.16.1 source/behaviour)
-------------------------------------------------------------------------------------
- Consumed episode sets are logged with ``mlflow.log_input(mlflow.data.from_pandas(<episode_id,
  content_hash table>, source=<repo-relative manifest path>, name=..., digest=...), context=<role>)``.
  MLflow caps a dataset digest at 36 characters (``MAX_DATASET_DIGEST_SIZE = 36``, enforced with an
  ``MlflowException``), so ``digest`` is the first 32 hex chars (128 bits) of our sha256 id; the full id
  is an input tag (``vlapoc.id``) and every query confirms it.
  ``search_runs(filter_string="datasets.digest IN (...)")`` finds consumers natively.
- Other inputs (checkpoints, score tables, rollout plans) are run tags ``input.<role> = <kind>:<id>``.
- Every entity a run creates is tagged ``produced.<id[:32]> = <kind>:<id>``, so the producer of any id
  is one tag search away; ``trace`` walks these edges upstream.
- ``mlflow.source.git.commit`` is set to ``git rev-parse HEAD``. A run refuses to start while tracked or
  untracked files under ``vlapoc/`` or ``scripts/`` differ from that commit, so the recorded SHA is the
  code that ran (``allow_dirty=True`` overrides and is recorded as ``vlapoc.git_dirty=true``).
- Only FINISHED runs count as producers or consumers; a run that fails part-way is kept (status
  FAILED) but never becomes a lineage edge.
- Stand-in stages (no real screening, rollouts or attribution yet) carry ``vlapoc.stand_in=true``
  and write only lineage keys plus clearly named placeholder payloads; no outcome numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted in multiple ways")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from vlapoc.catalog import sha256_file  # noqa: E402
from vlapoc.libero_convert import REPO_ROOT  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LineageError,
    Subset,
    _root,
    drop_random_k,
    drop_top_k,
    rel,
    resolve_version,
    score_table_digest,
    sha256_json,
    verify_version,
    version_episodes,
)

STAGES = ("build", "split", "derive", "screen", "select", "train", "eval", "attribute")
EXPERIMENT = "vlapoc-lineage"
SOURCE_DIRS = ("vlapoc", "scripts")
DIGEST_LEN = 32  # MLflow MAX_DATASET_DIGEST_SIZE = 36
# LeRobot config fields that identify a run rather than define it (excluded from config diffs).
BOOKKEEPING_CONFIG_KEYS = {"output_dir", "job_name", "checkpoint_path", "dataset.episodes", "wandb.run_id"}
BOOKKEEPING_PARAMS = {"output_dir", "run_name", "subset_id", "n_episodes", "n_frames"}


# ---------------------------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------------------------


def tracking_uri(lineage_root: Path | None = None) -> str:
    return f"sqlite:///{(_root(lineage_root) / 'mlflow.db').resolve()}"


def client(lineage_root: Path | None = None) -> MlflowClient:
    _root(lineage_root).mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=tracking_uri(lineage_root))


def experiment_id(lineage_root: Path | None = None) -> str:
    c = client(lineage_root)
    e = c.get_experiment_by_name(EXPERIMENT)
    if e is not None:
        return e.experiment_id
    art = (_root(lineage_root) / "mlartifacts").resolve()
    art.mkdir(parents=True, exist_ok=True)
    return c.create_experiment(EXPERIMENT, artifact_location=art.as_uri())


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCE_DIRS], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    return {"sha": sha, "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:10]}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------------------------
# stage runs
# ---------------------------------------------------------------------------------------------


class StageRun:
    """Handle for the active MLflow run of one stage execution."""

    def __init__(self, run_id: str, lineage_root: Path | None):
        self.run_id = run_id
        self.lineage_root = lineage_root
        self.outputs: dict[str, str] = {}

    # inputs -------------------------------------------------------------------------------
    def input_subset(self, subset: Subset, role: str, label: str | None = None) -> None:
        """Log the exact episode versions this stage consumes (one MLflow dataset input)."""
        path = _root(self.lineage_root) / "subsets" / f"{subset.subset_id}.json"
        if not path.is_file():
            subset.save(self.lineage_root)
        ds = mlflow.data.from_pandas(
            subset.keys(self.lineage_root), source=_local_source(rel(path)),
            name=label or f"subset:{subset.subset_id[:12]}", digest=subset.subset_id[:DIGEST_LEN],
        )
        mlflow.log_input(ds, context=role, tags={"vlapoc.kind": "subset", "vlapoc.id": subset.subset_id,
                                                 "vlapoc.dataset_version_id": subset.dataset_version_id})
        mlflow.set_tag(f"input.{role}", f"subset:{subset.subset_id}")

    def input_version(self, version: str, role: str) -> None:
        v = resolve_version(version, self.lineage_root)
        vid = v["dataset_version_id"]
        keys = version_episodes(vid, self.lineage_root)[["episode_id", "content_hash"]]
        vpath = _root(self.lineage_root) / "versions" / vid / "version.json"
        ds = mlflow.data.from_pandas(keys, source=_local_source(rel(vpath)), name=f"version:{v['name']}",
                                     digest=vid[:DIGEST_LEN])
        mlflow.log_input(ds, context=role, tags={"vlapoc.kind": "version", "vlapoc.id": vid,
                                                 "vlapoc.dataset_version_id": vid})
        mlflow.set_tag(f"input.{role}", f"version:{vid}")

    def input_entity(self, kind: str, entity_id: str, role: str) -> None:
        mlflow.set_tag(f"input.{role}", f"{kind}:{entity_id}")

    # outputs ------------------------------------------------------------------------------
    def produced(self, kind: str, entity_id: str, name: str) -> None:
        mlflow.set_tag(f"produced.{entity_id[:DIGEST_LEN]}", f"{kind}:{entity_id}")
        mlflow.set_tag(f"output.{name}", f"{kind}:{entity_id}")
        self.outputs[name] = f"{kind}:{entity_id}"

    def output_file(self, name: str, path: Path, kind: str = "file", entity_id: str | None = None) -> str:
        """Log a small output file as an artifact with its sha256; returns the sha256."""
        sha = sha256_file(Path(path))
        mlflow.log_artifact(str(path))
        mlflow.set_tag(f"output.{name}.sha256", sha)
        mlflow.set_tag(f"output.{name}.artifact", Path(path).name)
        self.produced(kind, entity_id or sha, name)
        return sha

    def params(self, d: dict) -> None:
        mlflow.log_params({k: (json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v) for k, v in d.items()})

    def tags(self, d: dict) -> None:
        mlflow.set_tags({k: str(v) for k, v in d.items()})


def _local_source(uri: str) -> str:
    """Repo-relative manifest path; MLflow resolves it to a LocalArtifactDatasetSource with the relative
    URI kept (measured). That class is generated dynamically in mlflow.data.artifact_dataset_sources and
    cannot be imported by name, so the string is passed; the resolver's ambiguity warning is silenced."""
    return uri


@contextmanager
def stage_run(stage: str, name: str, lineage_root: Path | None = None, stand_in: bool = False,
              allow_dirty: bool = False, chain: str | None = None, extra_tags: dict | None = None):
    """One MLflow run for one execution of a PoC stage (see module docstring for the encoding)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    g = git_state()
    if g["dirty"] and not allow_dirty:
        raise LineageError(f"source under {SOURCE_DIRS} differs from HEAD {g['sha'][:12]} ({g['dirty_files']}); "
                           "commit first so the recorded git SHA is the code that runs")
    mlflow.set_tracking_uri(tracking_uri(lineage_root))
    exp = experiment_id(lineage_root)
    tags = {"stage": stage, "mlflow.source.git.commit": g["sha"], "vlapoc.git_dirty": str(g["dirty"]).lower(),
            "vlapoc.stand_in": str(stand_in).lower(), "vlapoc.command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])[:4000]}
    if chain:
        tags["vlapoc.chain"] = chain
    tags.update({k: str(v) for k, v in (extra_tags or {}).items()})
    with mlflow.start_run(experiment_id=exp, run_name=name, tags=tags) as run:
        yield StageRun(run.info.run_id, lineage_root)


# ---------------------------------------------------------------------------------------------
# stages with real work
# ---------------------------------------------------------------------------------------------


def build_run(version: str = "v0", lineage_root: Path | None = None, chain: str | None = None,
              verify_raw: bool = True) -> str:
    """Stage ``build`` for an already-converted version: re-verify it against raw and record it.

    Mode ``verify_existing``: the conversion itself ran before tracking existed (S1). This run recomputes
    the version id from the stored bytes, re-hashes every episode from raw HDF5 (1000/1000 must match)
    and re-hashes the raw files, then records the version as this run's product.
    """
    from vlapoc.catalog import build_catalog, raw_file_hashes
    from vlapoc.lineage import recompute_version_id

    v = resolve_version(version, lineage_root)
    vid = v["dataset_version_id"]
    vdir = _root(lineage_root) / "versions" / vid
    with stage_run("build", f"build-{v['name']}", lineage_root, chain=chain) as r:
        t0 = time.time()
        recomputed = recompute_version_id(vid, lineage_root=lineage_root)
        if recomputed != vid:
            raise LineageError(f"recomputed version id {recomputed[:12]} != registered {vid[:12]}")
        verify_version(vid, lineage_root=lineage_root)
        checks = {}
        if verify_raw:
            _, checks = build_catalog(REPO_ROOT / v["lerobot_root"])
            if checks["hash_match_raw"] != checks["n_episodes"]:
                raise LineageError(f"only {checks['hash_match_raw']}/{checks['n_episodes']} episodes match raw")
            raw_now = raw_file_hashes()
            raw_rec = pd.read_parquet(vdir / "raw_files.parquet")
            if not raw_now.sort_values("source_file")["sha256"].tolist() == raw_rec.sort_values("source_file")["sha256"].tolist():
                raise LineageError("raw file sha256s changed since registration")
        raw = pd.read_parquet(vdir / "raw_files.parquet")
        raw_digest = sha256_json(sorted([p, h] for p, h in zip(raw["source_file"], raw["sha256"])))
        r.params({"mode": "verify_existing", "version_name": v["name"], "dataset_version_id": vid,
                  "lerobot_root": v["lerobot_root"], "n_episodes": v["n_episodes"], "n_frames": v["n_frames"],
                  "converter": "vlapoc.libero_convert", "raw_files": len(raw)})
        r.tags({"raw_files.digest": raw_digest, "raw_files.count": len(raw), "build.recomputed_id": recomputed,
                "build.hash_match_raw": f"{checks.get('hash_match_raw', 'skipped')}/{checks.get('n_episodes', v['n_episodes'])}",
                "build.seconds": round(time.time() - t0, 1)})
        mlflow.log_artifact(str(vdir / "raw_files.parquet"))
        mlflow.log_artifact(str(vdir / "version.json"))
        r.produced("version", vid, f"version.{v['name']}")
        all_s = Subset(vid, tuple(version_episodes(vid, lineage_root)["episode_id"]), {"op": "all"})
        r.produced("subset", all_s.subset_id, f"subset.{v['name']}/all")
        return r.run_id


def split_run(version: str = "v0", reference_per_task: int = 2, seed: int = 0, lineage_root: Path | None = None,
              chain: str | None = None) -> str:
    """Stage ``split``: held-out reference / candidate split (deterministic; same ids on re-run)."""
    from vlapoc.lineage import make_splits

    v = resolve_version(version, lineage_root)
    all_s = Subset(v["dataset_version_id"], tuple(version_episodes(v["dataset_version_id"], lineage_root)["episode_id"]), {"op": "all"})
    with stage_run("split", f"split-{v['name']}", lineage_root, chain=chain) as r:
        r.input_subset(all_s, "population", f"{v['name']}/all")
        out = make_splits(v["dataset_version_id"], reference_per_task, seed, lineage_root)
        rec = out["record"]
        r.params({"reference_per_task": reference_per_task, "seed": seed, "rule": rec["rule"],
                  "dataset_version_id": v["dataset_version_id"]})
        split_path = _root(lineage_root) / "splits" / f"{rec['split_id']}.json"
        r.output_file("split_record", split_path, kind="split", entity_id=rec["split_id"])
        r.produced("subset", out["reference"].subset_id, f"subset.{v['name']}/reference")
        r.produced("subset", out["candidate"].subset_id, f"subset.{v['name']}/candidate")
        return r.run_id


def derive_run(parent: str = "v0", name: str = "v1", per_task: int = 5, seed: int = 0,
               lineage_root: Path | None = None, chain: str | None = None) -> tuple[str, dict]:
    """Stage ``derive``: the S4 demo derivation (RoboDrop action noise) recorded as a run."""
    from vlapoc.derive import derive_demo_action_noise

    pv = resolve_version(parent, lineage_root)
    cand = Subset.load(f"{pv['name']}/candidate", lineage_root)
    with stage_run("derive", f"derive-{name}", lineage_root, chain=chain) as r:
        r.input_version(pv["dataset_version_id"], "parent")
        r.input_subset(cand, "selection_pool", f"{pv['name']}/candidate")
        out = derive_demo_action_noise(pv["dataset_version_id"], name, per_task, seed, lineage_root=lineage_root)
        vid = out["dataset_version_id"]
        vdir = _root(lineage_root) / "versions" / vid
        r.params({"parent_version_id": pv["dataset_version_id"], "corruption_type": "action_gaussian_noise",
                  "per_task": per_task, "seed": seed, "n_edits": out["n_edits"], "std_scale": 0.25,
                  "mechanism": "lerobot delete_episodes + writer + merge_datasets"})
        r.tags({f"derive.timing.{k}": v for k, v in out["timing_s"].items()})
        r.produced("version", vid, f"version.{name}")
        r.output_file("labels", vdir / "labels.parquet", kind="labels")
        mlflow.log_artifact(str(vdir / "derivation.json"))
        mlflow.log_artifact(str(vdir / "version.json"))
        names = json.loads((_root(lineage_root) / "names.json").read_text())["subsets"]
        for role in ("all", "reference", "candidate"):
            r.produced("subset", names[f"{name}/{role}"], f"subset.{name}/{role}")
        return r.run_id, out


def _ckpt_files(pretrained: Path) -> dict[str, Path]:
    files = {"weights": pretrained / "model.safetensors", "train_config": pretrained / "train_config.json",
             "policy_config": pretrained / "config.json"}
    norm = sorted(pretrained.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    unnorm = sorted(pretrained.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))
    if len(norm) != 1 or len(unnorm) != 1:
        raise LineageError(f"expected one normalizer and one unnormalizer in {pretrained}")
    files["normalizer_stats"], files["unnormalizer_stats"] = norm[0], unnorm[0]
    return files


def train_run(subset_ref: str, name: str, seed: int = 0, steps: int = 5, batch_size: int = 4,
              policy_type: str = "act", policy_overrides: dict | None = None, device: str = "mps",
              dataset_root: Path | None = None, runs_root: Path | None = None, lineage_root: Path | None = None,
              chain: str | None = None, allow_dirty: bool = False) -> dict:
    """Verify the dataset copy, then run LeRobot training on exactly ``subset`` and record every §2.1 field.

    Refuses to launch (raises ``VersionIntegrityError``, no MLflow run is created) if any file of the
    dataset copy differs from its registered digest.
    """
    subset = Subset.load(subset_ref, lineage_root)
    v = resolve_version(subset.dataset_version_id, lineage_root)
    vid = v["dataset_version_id"]
    root = Path(dataset_root) if dataset_root is not None else REPO_ROOT / v["lerobot_root"]
    verify_version(vid, root=root, lineage_root=lineage_root)  # before any run exists
    episodes = subset.resolve(lineage_root)
    runs_root = Path(runs_root) if runs_root is not None else REPO_ROOT / "data" / "runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = runs_root / f"{name}-{stamp}"
    overrides = {"device": device, "push_to_hub": False, **({"pretrained_backbone_weights": None} if policy_type == "act" else {}),
                 **(policy_overrides or {})}
    spec = {"dataset_repo_id": f"local/{Path(v['lerobot_root']).name}", "dataset_root": v["lerobot_root"],
            "episodes": episodes, "policy_type": policy_type, "policy_overrides": overrides, "steps": steps,
            "batch_size": batch_size, "num_workers": 0, "seed": seed, "output_dir": rel(out_dir / "lerobot"), "job_name": name,
            "image_transforms_enable": False, "use_imagenet_stats": True}
    stats_path = root / "meta" / "stats.json"
    with stage_run("train", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_subset(subset, "training", subset_ref if "/" in subset_ref else None)
        r.params({"dataset_version_id": vid, "subset_id": subset.subset_id, "n_episodes": len(episodes),
                  "n_frames": int(subset.episodes(lineage_root)["length_steps"].sum()), "seed": seed, "steps": steps,
                  "batch_size": batch_size, "policy_type": policy_type, "device": device,
                  "policy_overrides": overrides, "output_dir": rel(out_dir),
                  "mixture_weights": "uniform over the subset's frames (LeRobot default sampler; sample_weighting=None)",
                  "augmentation": "none (dataset.image_transforms.enable=false)",
                  "normalization_stats": "full dataset meta/stats.json of the version (LeRobot default), "
                                         "camera stats replaced by ImageNet stats (use_imagenet_stats=true)"})
        r.tags({"dataset_version_id": vid, "subset_id": subset.subset_id, "seed": seed,
                "dataset.stats_json.sha256": sha256_file(stats_path)})
        out_dir.mkdir(parents=True, exist_ok=True)  # LeRobot writes into out_dir/lerobot, which must not exist yet
        spec_path = out_dir / "vlapoc_train_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        mlflow.log_artifact(str(spec_path))
        mlflow.log_artifact(str(_root(lineage_root) / "subsets" / f"{subset.subset_id}.json"), artifact_path="subset")
        log = out_dir / "train_stdout.log"
        t0 = time.time()
        with open(log, "w") as fh:
            proc = subprocess.run([sys.executable, "-m", "vlapoc.lerobot_train_entry", str(spec_path)], cwd=REPO_ROOT,
                                  stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})
        mlflow.log_artifact(str(log))
        if proc.returncode != 0:
            raise LineageError(f"LeRobot training failed (exit {proc.returncode}); see {rel(log)}")
        pretrained = (out_dir / "lerobot" / "checkpoints" / "last").resolve() / "pretrained_model"
        files = _ckpt_files(pretrained)
        hashes = {k: sha256_file(p) for k, p in files.items()}
        cfg = json.loads(files["train_config"].read_text())
        eps_back = sorted(cfg["dataset"]["episodes"])
        if eps_back != episodes:
            raise LineageError("checkpoint train_config.json episodes differ from the subset")
        mlflow.log_artifact(str(files["train_config"]), artifact_path="checkpoint")
        mlflow.log_artifact(str(files["policy_config"]), artifact_path="checkpoint")
        r.tags({"checkpoint.path": rel(pretrained), "checkpoint.step": pretrained.parent.name,
                **{f"checkpoint.{k}.sha256": h for k, h in hashes.items()},
                "checkpoint.train_config.episodes_match_subset": "true", "train.seconds": round(time.time() - t0, 1)})
        r.produced("checkpoint", hashes["weights"], "checkpoint")
        return {"run_id": r.run_id, "checkpoint": rel(pretrained), "weights_sha256": hashes["weights"],
                "normalizer_sha256": hashes["normalizer_stats"], "episodes_match": True, "out_dir": rel(out_dir),
                "n_episodes": len(episodes), "seconds": round(time.time() - t0, 1)}


def load_checkpoint(pretrained: Path):
    """Load a LeRobot policy checkpoint without ``from_pretrained``.

    ``PreTrainedConfig.from_pretrained`` builds a draccus argparse parser and fails on Python 3.14
    (``TypeError: typing.Dict[str, ...PolicyFeature] | None is not callable``, measured).
    ``draccus.load`` decodes without argparse; weights load with ``safetensors`` in strict mode.
    """
    import draccus
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from safetensors.torch import load_model

    with open(Path(pretrained) / "config.json") as f:
        cfg = draccus.load(PreTrainedConfig, f)
    policy = get_policy_class(cfg.type)(cfg)
    load_model(policy, str(Path(pretrained) / "model.safetensors"), strict=True)
    return policy


# ---------------------------------------------------------------------------------------------
# stand-in stages (lineage keys only; no invented numbers)
# ---------------------------------------------------------------------------------------------


def _placeholder(episode_id: str, content_hash: str, salt: str) -> float:
    """Deterministic uniform [0,1) placeholder from the key; NOT a score of anything."""
    return int(hashlib.sha256(f"{salt}|{episode_id}|{content_hash}".encode()).hexdigest()[:13], 16) / 16 ** 13


def screen_standin(candidate: str, reference: str, name: str, out_dir: Path, lineage_root: Path | None = None,
                   chain: str | None = None) -> tuple[str, Path]:
    """Stage ``screen`` stand-in (PoC §9.1 Step 1): one row per candidate episode version, keyed by
    (episode_id, content_hash); ``suspicion_score`` is NaN (not computed; DemInf/RoboDrop come later)."""
    cand = Subset.load(candidate, lineage_root)
    ref = Subset.load(reference, lineage_root)
    table = cand.keys(lineage_root).copy()
    table["suspicion_score"] = np.nan
    table["stand_in"] = True
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "screening_scores.STAND_IN.parquet"
    table.to_parquet(path, index=False)
    with stage_run("screen", name, lineage_root, stand_in=True, chain=chain) as r:
        r.input_subset(cand, "candidate", candidate)
        r.input_subset(ref, "reference", reference)
        r.params({"method": "stand-in (no screening computed)", "payload": "suspicion_score = NaN for every row",
                  "n_rows": len(table)})
        digest = score_table_digest(table, "suspicion_score")  # NaN payload hashes as 'nan'
        r.output_file("screening_scores", path, kind="score_table", entity_id=digest)
        return r.run_id, path


def eval_standin(train_run_id: str, name: str, out_dir: Path, rollouts_per_task: int = 50, seed_base: int = 0,
                 lineage_root: Path | None = None, chain: str | None = None) -> tuple[str, Path]:
    """Stage ``eval`` stand-in (PoC §9.1 Step 2): load the checkpoint, check its sha256 against the train
    run, and record the rollout PLAN (task, init-state index, seed). Outcomes are ``not_run``: the LIBERO
    simulator needs Linux (LeRobot docs), so no rollout happens and no outcome is recorded."""
    c = client(lineage_root)
    tr = c.get_run(train_run_id)
    pretrained = REPO_ROOT / tr.data.tags["checkpoint.path"]
    sha = sha256_file(pretrained / "model.safetensors")
    if sha != tr.data.tags["checkpoint.weights.sha256"]:
        raise LineageError("checkpoint on disk differs from the one the train run recorded")
    policy = load_checkpoint(pretrained)
    n_params = int(sum(p.numel() for p in policy.parameters()))
    subset = Subset.load(tr.data.tags["subset_id"], lineage_root)
    tasks = subset.episodes(lineage_root)[["suite", "libero_task_id", "task_name"]].drop_duplicates()
    rows = [{"suite": t.suite, "libero_task_id": int(t.libero_task_id), "task_name": t.task_name,
             "init_state_index": i, "seed": seed_base + i, "checkpoint_sha256": sha, "outcome": "not_run",
             "stand_in": True}
            for t in tasks.sort_values(["suite", "libero_task_id"]).itertuples() for i in range(rollouts_per_task)]
    plan = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rollout_plan.STAND_IN.parquet"
    plan.to_parquet(path, index=False)
    with stage_run("eval", name, lineage_root, stand_in=True, chain=chain) as r:
        r.input_entity("checkpoint", sha, "checkpoint")
        r.params({"train_run_id": train_run_id, "checkpoint_sha256": sha, "suites": sorted(tasks["suite"].unique().tolist()),
                  "libero_task_ids": {s: sorted(g["libero_task_id"].astype(int).tolist()) for s, g in tasks.groupby("suite")},
                  "rollouts_per_task": rollouts_per_task, "init_state_indices": f"0..{rollouts_per_task - 1} (LIBERO fixed init states)",
                  "seeds": f"seed_base + init_state_index, seed_base={seed_base}", "outcomes": "not_run (stand-in)"})
        r.tags({"eval.train_run_id": train_run_id, "eval.checkpoint.sha256": sha, "eval.checkpoint.loaded": "true",
                "eval.checkpoint.n_params": n_params})
        r.output_file("rollouts", path, kind="rollouts")
        return r.run_id, path


def attribute_standin(train_run_id: str, eval_run_id: str, name: str, out_dir: Path,
                      lineage_root: Path | None = None, chain: str | None = None) -> tuple[str, Path, str]:
    """Stage ``attribute`` stand-in (PoC §9.1 Step 3): an episode x metric table keyed by the training
    subset's (episode_id, content_hash). ``placeholder_score`` is a deterministic hash of the key, used
    only to exercise drop-top-k downstream; it is not an influence estimate."""
    c = client(lineage_root)
    tr, ev = c.get_run(train_run_id), c.get_run(eval_run_id)
    subset = Subset.load(tr.data.tags["subset_id"], lineage_root)
    table = subset.keys(lineage_root).copy()
    table["placeholder_score"] = [_placeholder(e, h, "attribution-stand-in") for e, h in zip(table["episode_id"], table["content_hash"])]
    table["stand_in"] = True
    check_scores(table, train_run_id, lineage_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attribution_scores.STAND_IN.parquet"
    table.to_parquet(path, index=False)
    digest = score_table_digest(table, "placeholder_score")
    with stage_run("attribute", name, lineage_root, stand_in=True, chain=chain) as r:
        r.input_subset(subset, "training", None)
        r.input_entity("checkpoint", tr.data.tags["checkpoint.weights.sha256"], "checkpoint")
        r.input_entity("rollouts", ev.data.tags["output.rollouts.sha256"], "rollouts")
        r.params({"method": "stand-in (no attribution computed)", "metric": "placeholder_score",
                  "payload": "placeholder_score = sha256(key) mapped to [0,1); not an influence estimate",
                  "train_run_id": train_run_id, "eval_run_id": eval_run_id, "n_rows": len(table)})
        r.output_file("attribution_scores", path, kind="score_table", entity_id=digest)
        return r.run_id, path, digest


def select_run(parent: str, name: str, op: str, k: int, seed: int | None = None, score_path: Path | None = None,
               metric: str | None = None, score_run_id: str | None = None, lineage_root: Path | None = None,
               chain: str | None = None) -> tuple[str, Subset]:
    """Stage ``select``: drop-top-k by a score table, or drop-random-k (PoC §2.6 p.5, §9.1 Step 4)."""
    ps = Subset.load(parent, lineage_root)
    with stage_run("select", name, lineage_root, chain=chain) as r:
        r.input_subset(ps, "parent", parent if "/" in parent else None)
        if op == "drop_top_k":
            table = pd.read_parquet(score_path)
            child = drop_top_k(ps, table, metric, k, "highest", score_source=f"mlflow run {score_run_id}",
                               lineage_root=lineage_root)
            r.input_entity("score_table", child.derivation["score_table_digest"], "scores")
            r.params({"op": op, "k": k, "metric": metric, "direction": "highest", "score_run_id": score_run_id})
        elif op == "drop_random_k":
            child = drop_random_k(ps, k, seed, lineage_root=lineage_root)
            r.params({"op": op, "k": k, "seed": seed})
        else:
            raise ValueError(op)
        r.params({"parent_subset_id": ps.subset_id, "child_n_episodes": len(child.episode_ids)})
        r.tags({"select.dropped": json.dumps(child.derivation["dropped"])[:7900]})
        r.produced("subset", child.subset_id, "subset")
        return r.run_id, child


# ---------------------------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------------------------


def _inputs(run) -> list[dict]:
    """(kind, id, role) of everything a run consumed."""
    out = []
    for di in run.inputs.dataset_inputs:
        t = {x.key: x.value for x in di.tags}
        if "vlapoc.id" in t:
            out.append({"kind": t["vlapoc.kind"], "id": t["vlapoc.id"], "role": t.get("mlflow.data.context"),
                        "digest": di.dataset.digest, "name": di.dataset.name})
    seen = {(i["kind"], i["id"]) for i in out}
    for k, v in run.data.tags.items():
        if k.startswith("input."):
            kind, _, eid = v.partition(":")
            if (kind, eid) not in seen:
                out.append({"kind": kind, "id": eid, "role": k[len("input."):]})
    return out


def producers(entity_id: str, lineage_root: Path | None = None) -> list:
    c = client(lineage_root)
    runs = c.search_runs([experiment_id(lineage_root)], filter_string=f"tags.`produced.{entity_id[:DIGEST_LEN]}` LIKE '%'")
    return [r for r in runs if r.info.status == "FINISHED"
            and r.data.tags[f"produced.{entity_id[:DIGEST_LEN]}"].split(":", 1)[1] == entity_id]


def _short(run) -> dict:
    t = run.data.tags
    return {"run_id": run.info.run_id, "name": run.info.run_name, "stage": t.get("stage"),
            "stand_in": t.get("vlapoc.stand_in") == "true", "git": t.get("mlflow.source.git.commit", "")[:12],
            "status": run.info.status}


def trace(run_id: str, lineage_root: Path | None = None) -> dict:
    """Walk upstream from a run through producer links to the build run and its raw-file sha256s."""
    c = client(lineage_root)
    nodes, edges, entities = {}, [], {}
    raw_files = None
    todo, seen = [run_id], set()
    while todo:
        rid = todo.pop(0)
        if rid in seen:
            continue
        seen.add(rid)
        run = c.get_run(rid)
        nodes[rid] = _short(run)
        for k, v in run.data.tags.items():
            if k.startswith("output."):
                kind, _, eid = v.partition(":")
                if eid and kind in ("version", "subset", "checkpoint", "score_table", "rollouts", "labels", "split"):
                    entities.setdefault(eid, {"kind": kind, "name": k[len("output."):]})
        if run.data.tags.get("stage") == "build":
            path = mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="raw_files.parquet",
                                                       tracking_uri=tracking_uri(lineage_root))
            raw_files = pd.read_parquet(path)[["source_file", "sha256"]].to_dict("records")
        for inp in _inputs(run):
            entities.setdefault(inp["id"], {"kind": inp["kind"], "name": inp.get("name")})
            prods = producers(inp["id"], lineage_root)
            edges.append({"run": rid, "consumes": f"{inp['kind']}:{inp['id']}", "role": inp["role"],
                          "produced_by": [p.info.run_id for p in prods]})
            todo += [p.info.run_id for p in prods]
    return {"root": run_id, "runs": nodes, "edges": edges, "entities": entities, "raw_files": raw_files}


def consumers(episode_id: str, content_hash: str, lineage_root: Path | None = None) -> list[dict]:
    """Runs whose consumed episode sets contain this exact episode version."""
    lr = _root(lineage_root)
    ids = []
    for p in sorted((lr / "subsets").glob("*.json")):
        rec = json.loads(p.read_text())
        if episode_id in rec["episode_ids"]:
            eps = version_episodes(rec["dataset_version_id"], lineage_root)
            if eps.loc[eps["episode_id"] == episode_id, "content_hash"].item() == content_hash:
                ids.append(rec["subset_id"])
    for vdir in sorted((lr / "versions").iterdir()):
        eps = pd.read_parquet(vdir / "episodes.parquet", columns=["episode_id", "content_hash"])
        if ((eps["episode_id"] == episode_id) & (eps["content_hash"] == content_hash)).any():
            ids.append(vdir.name)
    if not ids:
        return []
    c = client(lineage_root)
    quoted = ", ".join(f"'{i[:DIGEST_LEN]}'" for i in ids)
    runs = c.search_runs([experiment_id(lineage_root)], filter_string=f"datasets.digest IN ({quoted})")
    out = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        via = [i for i in _inputs(run) if i["id"] in ids]
        if via:
            out.append({**_short(run), "via": [f"{i['kind']}:{i['id'][:12]} ({i['role']})" for i in via]})
    return sorted(out, key=lambda d: d["name"])


def _training_subset(run, lineage_root) -> Subset:
    for i in _inputs(run):
        if i["kind"] == "subset" and i["role"] == "training":
            return Subset.load(i["id"], lineage_root)
    raise LineageError(f"run {run.info.run_id} has no training subset input")


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(run_a: str, run_b: str, lineage_root: Path | None = None) -> dict:
    """What changed between two train runs: episodes added / removed / changed version, seed and other
    params, LeRobot config, git SHA (PoC §2.1 p.3 "what changed between the good run and the bad run?")."""
    c = client(lineage_root)
    ra, rb = c.get_run(run_a), c.get_run(run_b)
    sa, sb = _training_subset(ra, lineage_root), _training_subset(rb, lineage_root)
    ka = dict(sa.keys(lineage_root).itertuples(index=False, name=None))
    kb = dict(sb.keys(lineage_root).itertuples(index=False, name=None))
    added = sorted(set(kb) - set(ka))
    removed = sorted(set(ka) - set(kb))
    changed = sorted(e for e in set(ka) & set(kb) if ka[e] != kb[e])
    pa, pb = ra.data.params, rb.data.params
    params = {k: (pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb))
              if pa.get(k) != pb.get(k) and k not in BOOKKEEPING_PARAMS}

    def cfg(run):
        p = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="checkpoint/train_config.json",
                                                tracking_uri=tracking_uri(lineage_root))
        return _flatten(json.loads(Path(p).read_text()))

    ca, cb = cfg(ra), cfg(rb)
    config = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb))
              if ca.get(k) != cb.get(k) and k not in BOOKKEEPING_CONFIG_KEYS}
    ga, gb = ra.data.tags.get("mlflow.source.git.commit"), rb.data.tags.get("mlflow.source.git.commit")
    return {
        "runs": [_short(ra), _short(rb)],
        "data": {"subset_a": sa.subset_id, "subset_b": sb.subset_id,
                 "dataset_version_a": sa.dataset_version_id, "dataset_version_b": sb.dataset_version_id,
                 "added": added, "removed": removed, "changed_version": changed,
                 "n_added": len(added), "n_removed": len(removed), "n_changed_version": len(changed)},
        "params": params, "config": config,
        "git": {"a": ga, "b": gb, "same": ga == gb},
        "ignored_bookkeeping": sorted(BOOKKEEPING_CONFIG_KEYS | BOOKKEEPING_PARAMS),
    }


def check_scores(table: pd.DataFrame, train_run_id: str, lineage_root: Path | None = None) -> dict:
    """Reject score rows whose (episode_id, content_hash) is not in the run's training subset (§9.2 p.19)."""
    run = client(lineage_root).get_run(train_run_id)
    subset = _training_subset(run, lineage_root)
    keys = set(subset.keys(lineage_root).itertuples(index=False, name=None))
    rows = list(zip(table["episode_id"], table["content_hash"]))
    bad = [r for r in rows if r not in keys]
    if bad:
        raise LineageError(f"{len(bad)}/{len(rows)} score rows are not episode versions trained on by run "
                           f"{train_run_id[:12]} (e.g. {bad[0][0]}@{bad[0][1][:12]})")
    return {"rows": len(rows), "training_episodes": len(keys), "covered": len(set(rows)), "ok": True}
