"""Run LeRobot's own training loop from a JSON spec (subprocess entry used by ``vlapoc.runs.train_run``).

Why not the ``lerobot-train`` CLI: on this environment (Python 3.14, draccus 0.11.6, the newest release
and the version lerobot 0.6.1 pins, ``draccus<0.12.0,>=0.11.6``) every ``lerobot-train`` invocation,
even ``--help``, fails while building the argument parser: ``TypeError: str | None is not callable``
for the first optional field (``dataset.root``). On 3.14 ``str | None`` evaluates to ``typing.Union``,
which draccus 0.11.6 does not treat as Optional (measured 2026-09-29). LeRobot's ``parser.wrap`` skips
CLI parsing when it is handed a config object (``lerobot/configs/parser.py``: ``if len(args) > 0 and
type(args[0]) is argtype: cfg = args[0]``), so this entry builds ``TrainPipelineConfig`` in Python and
calls ``lerobot.scripts.lerobot_train.train(cfg)``: the same training function ``lerobot-train``
runs, with only argument parsing bypassed. On a Python <= 3.13 machine the saved
``train_config.json`` can be replayed with ``lerobot-train --config_path=...`` (inferred, not tested
here).

Grounding: PoC §2.1 item 2 p.3 ("SmolVLA via LeRobot"); PoC p.1 "model, config, training schedule and
evaluation protocol are held fixed so that data is the only variable": only ``dataset.episodes``
and ``seed`` vary between runs of one experiment.

Per-subset normalization (owner decision 3, evaluation-layer S8; brief: "Every training run normalizes with
mean/std computed only from its own training episodes"): when the spec names ``subset_stats_json`` (written by
``train_run`` from ``vlapoc.subset_stats``), ``install_subset_stats`` wraps LeRobot's
``make_train_eval_datasets`` inside ``lerobot.scripts.lerobot_train`` so that, right after the dataset is
built (and after LeRobot puts ImageNet stats on the camera keys), ``dataset.meta.stats`` holds the subset's
stats for every non-camera feature. ``train()`` reads ``dataset.meta.stats`` for the policy
(``make_policy(ds_meta=...)``), the processors' ``dataset_stats`` and, with a pretrained policy, the
normalizer/unnormalizer overrides (``lerobot_train.py:314, 351, 376-377``), so all three see the subset stats.
LeRobot has no option for externally supplied stats (``DatasetConfig`` has no stats field), hence this one
wrap of a module-level name in the training subprocess; LeRobot's code is not modified.

Pretrained policies (``policy_pretrained_path``, e.g. ``lerobot/smolvla_base``): the policy config is read from
that directory's ``config.json`` (``draccus.load``, as ``vlapoc.runs.load_checkpoint``; ``from_pretrained``'s
argparse path fails on 3.14), its input/output features are cleared so LeRobot derives them from our dataset
(``make_policy``: ``if not cfg.input_features``), ``policy_overrides`` are applied and ``pretrained_path`` is
set: what ``lerobot-train --policy.path=...`` does in ``TrainPipelineConfig.validate``.

Where it sits: the training run between Steps 1 and 2 of docs/dataflow.md. ``vlapoc.runs.train_run``
writes the spec, launches this module in a subprocess inside its MLflow run, and afterwards hashes the
checkpoint LeRobot saved under the spec's ``output_dir``.

Usage (from the repo root; normally only ``train_run`` calls it)::

    .venv/bin/python -m vlapoc.lerobot_train_entry data/runs/<name>-<UTC stamp>/vlapoc_train_spec.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def build_config(spec: dict):
    """LeRobot ``TrainPipelineConfig`` from the spec ``train_run`` writes.

    Required keys: ``policy_type``, ``dataset_repo_id``, ``dataset_root`` (repo-relative),
    ``episodes`` (LeRobot ``episode_index`` list), ``steps``, ``batch_size``, ``seed``, ``output_dir``
    (must not exist; LeRobot refuses to overwrite) and ``job_name``. Optional: ``policy_overrides``,
    ``policy_pretrained_path``, ``subset_stats_json`` (read by ``main``), ``use_imagenet_stats``,
    ``image_transforms_enable``, ``num_workers``, ``log_freq``. One checkpoint is
    saved, at the last step (``save_freq = steps``), and wandb is always off.
    """
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig

    if spec.get("policy_pretrained_path"):
        import draccus

        import lerobot.policies  # noqa: F401  (registers the policy config choice classes draccus decodes into)

        pre = Path(spec["policy_pretrained_path"])
        with open(pre / "config.json") as f:
            policy = draccus.load(PreTrainedConfig, f)
        if policy.type != spec["policy_type"]:
            raise ValueError(f"pretrained config type {policy.type!r} != spec policy_type {spec['policy_type']!r}")
        policy.input_features, policy.output_features = {}, {}
        for k, v in spec.get("policy_overrides", {}).items():
            setattr(policy, k, v)
        policy.pretrained_path = pre
    else:
        policy_cls = PreTrainedConfig.get_choice_class(spec["policy_type"])
        policy = policy_cls(**spec.get("policy_overrides", {}))
    ds = DatasetConfig(repo_id=spec["dataset_repo_id"], root=spec["dataset_root"], episodes=list(spec["episodes"]),
                       use_imagenet_stats=spec.get("use_imagenet_stats", True))
    ds.image_transforms.enable = bool(spec.get("image_transforms_enable", False))
    return TrainPipelineConfig(
        dataset=ds, policy=policy, steps=spec["steps"], batch_size=spec["batch_size"],
        num_workers=spec.get("num_workers", 0), save_freq=spec["steps"], log_freq=spec.get("log_freq", 1),
        seed=spec["seed"], output_dir=Path(spec["output_dir"]), job_name=spec["job_name"],
        wandb=WandBConfig(enable=False),
    )


def install_subset_stats(stats_json: Path) -> dict:
    """Make LeRobot's ``train()`` normalize with ``stats_json`` (a ``vlapoc.subset_stats.to_json`` dict) for every
    non-camera feature it names; camera keys keep LeRobot's handling. Returns the loaded stats."""
    import numpy as np

    import lerobot.scripts.lerobot_train as lt

    stats = {f: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()} for f, d in json.loads(Path(stats_json).read_text()).items()}
    original = lt.make_train_eval_datasets

    def with_subset_stats(cfg):
        """LeRobot's ``make_train_eval_datasets``, then the subset stats copied over every non-camera feature."""
        dataset, eval_dataset = original(cfg)
        for ds in (dataset, eval_dataset):
            if ds is None:
                continue
            for feature, st in stats.items():
                if feature in ds.meta.camera_keys:
                    continue
                ds.meta.stats[feature] = {k: v.copy() for k, v in st.items()}
        return dataset, eval_dataset

    lt.make_train_eval_datasets = with_subset_stats
    return stats


def main(argv: list[str] | None = None) -> int:
    """Read the spec at ``argv[0]`` and run LeRobot's ``train()`` on it (exceptions exit non-zero)."""
    argv = sys.argv[1:] if argv is None else argv
    spec = json.loads(Path(argv[0]).read_text())
    from lerobot.scripts.lerobot_train import train

    if spec.get("subset_stats_json"):
        install_subset_stats(Path(spec["subset_stats_json"]))
    train(build_config(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
