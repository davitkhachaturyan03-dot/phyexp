"""Per-sample gradient features of a SmolVLA checkpoint, for CUPID's influence estimator (``vlapoc.attribution``).

Grounding
---------
- PoC §4.3 (p. 10): "CUPID-style influence on returns ... with one return definition per metric"; §9.1 Step 3
  (p. 19), input "The training dataset, the trained policy, and the rollouts from Step 2". This module turns the
  training subset and the rollouts' trajectories into the gradient features that estimator needs.
- CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) §5.1 p. 5, Prop. 1: the action influence of a
  training pair (s, a) on a rollout pair (s', a') is ∇ℓ(s', a')ᵀ H⁻¹ ∇ℓ(s, a), evaluated, as in CUPID's code, with
  TRAK-style randomly projected gradients. ``vlapoc.rollouts`` records every (s', a') losslessly in the training
  schema (``record_trajectories``), so train demos and rollouts go through the same code here.
- ATHENA (docs/papers/ATHENA_Influence_Functions_arXiv-2606.16208.pdf): gradient projection and parameter subsets to
  scale influence to a VLA. The projection is Kronecker-structured per linear layer (LoGra-style, inferred: the
  factorization below is exact algebra, not a quote): for a layer with inputs x_t and output gradients d_t, the
  per-sample weight gradient is Σ_t d_t x_tᵀ and its projection P_outᵀ (Σ_t d_t x_tᵀ) P_in = Σ_t (P_outᵀ d_t)(P_inᵀ x_t)ᵀ
  needs only the r-dim projected activations, so the 100M-dim per-sample gradient is never materialised.

What is computed
----------------
- A sample is one training sample exactly as LeRobot builds it: an observation and its ``chunk_size``-step action
  chunk (``delta_timestamps`` at the dataset's own fps; out-of-episode steps are padded and masked by SmolVLA's loss).
- Only the parameters SmolVLA trains (``policy.model.set_requires_grad()``: the action expert and projections with
  the VLM frozen, the configuration ``lerobot-train`` fine-tunes) are differentiated, and only their ``nn.Linear``
  weights are projected (rank r per side, r² per layer; 117 layers × 64 = 7,488 dims for a SmolVLA fine-tuned
  from smolvla_base, measured on the cluster's Oct 2026 libero_10 features).
- SmolVLA's flow-matching loss is stochastic in (noise, time). K fixed (noise, time) draws, shared by every train
  and test sample, are averaged, so all features are gradients of the same loss. Time follows SmolVLA's training
  distribution Beta(1.5, 1)·0.999 + 0.001 (``lerobot/policies/smolvla/modeling_smolvla.py`` ``sample_time``).
- Dropout is off (``policy.eval()``): a feature must not depend on a dropout mask.

Outputs (one prefix, possibly sharded): ``<prefix>[.shard<i>].npy`` (float16 memmap, samples × dim),
``<prefix>[.shard<i>].meta.npz`` (``episode_index``, ``frame_index``, ``loss`` per sample) and
``<prefix>[.shard<i>].manifest.json`` (checkpoint sha256, dataset root, episodes, projection seed/rank, K, stride).
``vlapoc.attribution`` refuses to combine features whose manifests disagree on checkpoint, seed, rank or K.

Where it sits (docs/dataflow.md): Step 3 (Attribution), between the eval stage's trajectories and the influence
estimate. Feature files are intermediate, like the HF datasets cache: they are not a lineage entity; the
``attribute`` stage run records their manifests.

Usage (one GPU; ``--shard i/n`` splits the samples into n contiguous parts)::

    .venv/bin/python -m vlapoc.influence --checkpoint <pretrained_model dir> --root data/lerobot/libero_spatial_10 \\
        --episodes data/scratch/subset_episodes.json --stride 2 --out data/attribution/feats/<train run>/train --shard 0/4
    .venv/bin/python -m vlapoc.influence --checkpoint <pretrained_model dir> --root data/eval/<eval>/trajectories \\
        --stride 4 --out data/attribution/feats/<train run>/rollouts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

FEATURE_SCHEME = "vlapoc.features.v1"


class KroneckerProjector:
    """Hooks every trainable ``nn.Linear`` and accumulates an (r·r)-dim projected weight gradient per layer and sample."""

    def __init__(self, model, rank: int, seed: int = 0, device="cuda"):
        import torch
        from torch import nn

        self.rank = rank
        self.layers = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.p_in, self.p_out = {}, {}
        for name, m in self.layers:
            self.p_in[name] = (torch.randn(m.in_features, rank, generator=g) / math.sqrt(rank)).to(device)
            self.p_out[name] = (torch.randn(m.out_features, rank, generator=g) / math.sqrt(rank)).to(device)
        self.feats: dict = {}
        self.handles = [m.register_forward_hook(self._hook(name)) for name, m in self.layers]

    @property
    def dim(self) -> int:
        """Feature dimension: layers × r²."""
        return len(self.layers) * self.rank * self.rank

    def _hook(self, name):
        """Forward hook: keep the projected input; register a backward hook on the output that projects its gradient."""
        import torch

        def fwd(module, inputs, output):
            if not output.requires_grad:
                return
            x = inputs[0].detach()
            x = x.reshape(x.shape[0], -1, x.shape[-1]).float() @ self.p_in[name]  # (B, T, r)

            def bwd(grad):
                d = grad.reshape(grad.shape[0], -1, grad.shape[-1]).float() @ self.p_out[name]  # (B, T, r)
                f = torch.einsum("btr,bts->brs", d, x).flatten(1)
                self.feats[name] = self.feats[name] + f if name in self.feats else f

            output.register_hook(bwd)

        return fwd

    def collect(self):
        """The batch's (B, dim) projected gradients, layers in a fixed order; raises if a hooked layer got no gradient."""
        import torch

        missing = [name for name, _ in self.layers if name not in self.feats]
        if missing:
            raise RuntimeError(f"{len(missing)} hooked layers got no gradient, e.g. {missing[:3]}")
        out = torch.cat([self.feats[name] for name, _ in self.layers], dim=1)
        self.feats = {}
        return out

    def remove(self) -> None:
        """Detach every hook."""
        for h in self.handles:
            h.remove()


def fixed_draws(k: int, chunk: int, action_dim: int, seed: int, device) -> list:
    """K shared (noise, time) draws; time ~ Beta(1.5, 1)·0.999 + 0.001, SmolVLA's training distribution.

    Both come from one seeded generator, so the draws depend only on ``seed`` (not on how much of torch's global RNG
    model construction consumed). Beta(a, 1) has CDF xᵃ, so u^(1/a) with u ~ U(0, 1) is an exact Beta(1.5, 1) draw."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = [torch.randn(chunk, action_dim, generator=g) for _ in range(k)]
    times = torch.rand(k, generator=g, dtype=torch.float64) ** (1 / 1.5) * 0.999 + 0.001
    return [(n.to(device), float(t)) for n, t in zip(noise, times)]


def load_policy(pretrained: Path, device: str = "cuda", rename: dict | None = None):
    """The checkpoint through ``vlapoc.rollouts.checkpoint_policy`` (the same loader and saved processors the eval
    stage uses), with SmolVLA's trainable set enabled and dropout off. ``rename`` overrides the preprocessor's rename
    map, only for checkpoints trained under other camera names (e.g. LeRobot ``camera1``/``camera2``)."""
    from lerobot.policies.factory import make_pre_post_processors

    from vlapoc.rollouts import SMOLVLM_DIR, checkpoint_policy

    spec = checkpoint_policy(Path(pretrained), device=device)
    policy = spec.policy
    if policy.config.type != "smolvla":
        raise ValueError(f"gradient features are implemented for SmolVLA, not {policy.config.type!r}")
    policy.model.set_requires_grad()
    policy.eval()
    pre = spec.preprocessor
    if rename:
        over = {"device_processor": {"device": device}, "tokenizer_processor": {"tokenizer_name": str(SMOLVLM_DIR)},
                "rename_observations_processor": {"rename_map": rename}}
        pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(pretrained), preprocessor_overrides=over)
    return policy, pre, spec.checkpoint_sha256


def sample_indices(ds, episodes: list[int] | None, stride: int) -> np.ndarray:
    """Global frame indices of ``ds`` to featurize: every ``stride``-th frame of each selected episode, from its first."""
    ep = np.asarray(ds.hf_dataset["episode_index"])
    fr = np.asarray(ds.hf_dataset["frame_index"])
    keep = (fr % stride == 0) if episodes is None else np.isin(ep, episodes) & (fr % stride == 0)
    return np.flatnonzero(keep)


def featurize(checkpoint: Path, root: Path, out: Path, episodes: list[int] | None = None, rank: int = 8,
              k_draws: int = 4, stride: int = 1, batch_size: int = 32, workers: int = 8, shard: tuple[int, int] = (0, 1),
              seed: int = 0, device: str = "cuda", rename: dict | None = None, log=print) -> dict:
    """Write the features of ``root``'s frames (restricted to ``episodes``, every ``stride``-th) under prefix ``out``.

    The full dataset is opened without ``episodes=`` and frames are selected by index, so no per-subset HF datasets
    cache is created (``vlapoc.lineage.Subset`` docstring: ~7 GB per distinct episode list). Returns the manifest.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from vlapoc.catalog import sha256_file

    policy, pre, ckpt_sha = load_policy(checkpoint, device, rename)
    cfg = policy.config
    meta = LeRobotDatasetMetadata(f"local/{Path(root).name}", root=root)
    ds = LeRobotDataset(f"local/{Path(root).name}", root=root,
                        delta_timestamps={"action": [i / meta.fps for i in range(cfg.chunk_size)]})
    idx = sample_indices(ds, episodes, stride)
    si, sn = shard
    idx = np.array_split(idx, sn)[si]
    proj = KroneckerProjector(policy, rank, seed=seed, device=device)
    draws = fixed_draws(k_draws, cfg.chunk_size, cfg.max_action_dim, seed, device)
    log(f"[influence] {len(proj.layers)} layers, dim {proj.dim}, {len(idx)} samples (shard {si}/{sn}), K={k_draws}, fps {meta.fps}")
    suffix = f".shard{si}" if sn > 1 else ""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(f"{out}{suffix}.npy", mode="w+", dtype=np.float16, shape=(len(idx), proj.dim))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=batch_size,
                                         num_workers=workers, shuffle=False)
    rec = {"episode_index": [], "frame_index": [], "loss": []}
    row = 0
    for b, raw in enumerate(loader):
        rec["episode_index"].append(raw["episode_index"].numpy())
        rec["frame_index"].append(raw["frame_index"].numpy())
        batch = pre(raw)
        n = batch["action"].shape[0]
        acc, losses = None, 0.0
        for noise, t in draws:
            policy.zero_grad(set_to_none=True)
            loss, _ = policy.forward(dict(batch), noise=noise.expand(n, -1, -1).contiguous(),
                                     time=torch.full((n,), t, device=device), reduction="none")
            loss.sum().backward()
            f = proj.collect()
            acc = f if acc is None else acc + f
            losses = losses + loss.detach()
        feats[row: row + n] = (acc / len(draws)).cpu().numpy().astype(np.float16)
        rec["loss"].append((losses / len(draws)).cpu().numpy())
        row += n
        if b % 50 == 0:
            log(f"[influence] {row}/{len(idx)}")
    feats.flush()
    proj.remove()
    np.savez(f"{out}{suffix}.meta.npz", **{k: np.concatenate(v) for k, v in rec.items()})
    manifest = {"scheme": FEATURE_SCHEME, "checkpoint": str(checkpoint), "checkpoint_sha256": ckpt_sha,
                "checkpoint_weights_sha256_recheck": sha256_file(Path(checkpoint) / "model.safetensors") == ckpt_sha,
                "dataset_root": str(root), "dataset_fps": meta.fps, "episodes": episodes, "stride": stride,
                "rank": rank, "k_draws": k_draws, "seed": seed, "dim": proj.dim, "n_layers": len(proj.layers),
                "layers": [n for n, _ in proj.layers], "shard": [si, sn], "n_samples": int(row), "rename": rename,
                "chunk_size": cfg.chunk_size}
    Path(f"{out}{suffix}.manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"[influence] wrote {row} features to {out}{suffix}.npy")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI (usage in the module docstring)."""
    p = argparse.ArgumentParser(prog="python -m vlapoc.influence")
    p.add_argument("--checkpoint", type=Path, required=True, help="a LeRobot pretrained_model dir")
    p.add_argument("--root", type=Path, required=True, help="a LeRobot v3 dataset (training version or trajectories)")
    p.add_argument("--episodes", type=Path, default=None, help="JSON list of episode_index values (default: all)")
    p.add_argument("--out", type=Path, required=True, help="output prefix")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--k-draws", type=int, default=4)
    p.add_argument("--stride", type=int, default=1, help="every stride-th frame of each episode")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", default="0/1", help="i/n: the i-th of n contiguous parts")
    p.add_argument("--seed", type=int, default=0, help="projection and draw seed; must match across train and rollouts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rename", default=None, help="JSON rename map (only for checkpoints trained under other camera names)")
    a = p.parse_args(argv)
    episodes = json.loads(a.episodes.read_text()) if a.episodes else None
    si, sn = (int(x) for x in a.shard.split("/"))
    featurize(a.checkpoint, a.root, a.out, episodes, a.rank, a.k_draws, a.stride, a.batch_size, a.workers, (si, sn),
              a.seed, a.device, json.loads(a.rename) if a.rename else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
