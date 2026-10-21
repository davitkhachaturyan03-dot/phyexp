#!/usr/bin/env python
"""Checks for the attribution feature (``vlapoc.influence``, ``vlapoc.attribution``, the per-task select ops).

Default (CPU, seconds, no data needed):

1. ``influence_matrix`` equals the brute-force CUPID double sum Σ_{j∈τ} Σ_{i∈ξ} φ_jᵀ H⁻¹ φ_i over sharded features.
2. ``performance_influence``: ``center=False`` is Prop. 1 literally, (1/m) Σ_τ R(τ) M[τ, ξ]; centring subtracts R̄ ×
   the mean rollout row; a constant return scores 0 when centred.
3. ``score_table``: ``__same_task`` columns equal the per-task computation by hand; ``harm_`` = −GOOD_SIGN × ``infl_``;
   per-failure-type columns only for types with ≥ ``MIN_FAILURE_TYPE_ROLLOUTS`` rollouts.
4. ``zscore_ensemble`` is the mean of per-table z-scores and refuses tables keyed differently.
5. ``drop_top_k_per_task`` / ``drop_random_k_per_task`` on a scratch lineage store: k dropped from every task, the
   top-k are the k most harmful (ties by episode_id), the same seed reproduces the same subset, curated and random
   children have identical per-task counts, a stale score table and an emptying k are refused.

6. (training-attribution S2, AC2.1) the ``square`` and ``loss`` output functions on a toy torch module: features from
   ``influence.batch_features`` equal the brute-force per-sample gradient, projected with the same P_in/P_out; the
   ``OutputCapture`` call-count guard (premise P1) raises when the captured module runs twice.
7. (AC2.2, AC2.3) CUPID parity against the vendored released functions (``vlapoc/third_party/cupid_quality.py``,
   commit 2941eba9) and the centring algebra with its counterexample.
8. (AC2.5) ``check_compatible`` refuses shards that mix train-side output functions; ``return_encoding`` guards.

``--cpu-smoke <pretrained_model> <dataset root> [--device mps|cpu]`` (AC2.4): ``featurize`` ≤ 2 episodes of ``v0/reference``
on this Mac with ``output_fn=square`` and ``loss``: dimension, manifest, finite, bit-for-bit repeat, wall time, and the
cosine between the two feature sets.

``--regress <old attribution dir> <old feats dir> <old rollouts dir>``: recompute M from the October
VLA-data-curation features (``outputs/feats/<run>``) with this ``influence_matrix`` and compare with the
``rollout_demo_influence.npy`` that code wrote (same estimator, ported).

``--gpu-smoke <pretrained_model> <dataset root> <old train feature prefix>`` (one GPU): ``featurize`` a few
episodes through ``checkpoint_policy`` and compare with the cluster code's own loader on the same samples and
draws. The October feature *files* are not the reference: that code left flow-matching times unseeded.

    .venv/bin/python scripts/verify_attribution.py
    .venv/bin/python scripts/verify_attribution.py --regress /raid/chem/gor/vla-curation/outputs/attribution/clean_full_s0 \\
        /raid/chem/gor/vla-curation/outputs/feats/clean_full_s0 /raid/chem/gor/vla-curation/outputs/rollouts/clean_full_s0_attrib
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlapoc import attribution as at  # noqa: E402
from vlapoc import lineage as lin  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def synthetic_m() -> None:
    rng = np.random.default_rng(0)
    n_ep, n_ro, dim = 7, 5, 12
    tg = rng.integers(0, n_ep, 200)
    tg[:n_ep] = np.arange(n_ep)
    rg = rng.integers(0, n_ro, 90)
    rg[:n_ro] = np.arange(n_ro)
    tf = rng.normal(size=(len(tg), dim)).astype(np.float16)
    rf = rng.normal(size=(len(rg), dim)).astype(np.float16)
    m = at.influence_matrix([tf[:120], tf[120:]], tg, n_ep, [rf[:40], rf[40:]], rg, n_ro, damping=1e-3, device="cpu")
    phi, psi = tf.astype(np.float64), rf.astype(np.float64)
    h = phi.T @ phi / len(phi)
    h += 1e-3 * np.diag(h).mean() * np.eye(dim)
    hinv = np.linalg.inv(h)
    brute = np.zeros((n_ro, n_ep))
    for j in range(len(psi)):
        for i in range(len(phi)):
            brute[rg[j], tg[i]] += psi[j] @ hinv @ phi[i]
    err = np.abs(m - brute).max() / np.abs(brute).max()
    check("1 influence_matrix == brute-force double sum (2 shards each)", err < 1e-9, f"max rel err {err:.2e}")
    # plan P5: numpy matmul RuntimeWarnings were seen on this Mac while checks pass; make sure nothing is non-finite
    check("1b the brute-force H, its inverse and both matrices are finite (the matmul warnings hide no overflow)",
          bool(np.isfinite(h).all() and np.isfinite(hinv).all() and np.isfinite(m).all() and np.isfinite(brute).all()),
          f"cond(H) = {np.linalg.cond(h):.1f}")

    r = rng.normal(size=n_ro)
    raw = at.performance_influence(m, r, center=False)
    check("2a center=False is Prop. 1: (1/m) Σ R M", np.allclose(raw, (r[:, None] * m).sum(0) / n_ro, atol=1e-12))
    cen = at.performance_influence(m, r)
    check("2b centred = raw − R̄·mean_τ M", np.allclose(cen, raw - r.mean() * m.mean(0), atol=1e-12))
    check("2c constant return scores 0 when centred", np.allclose(at.performance_influence(m, np.full(n_ro, 3.0)), 0))


def synthetic_table() -> None:
    rng = np.random.default_rng(1)
    eps = pd.DataFrame({"episode_id": [f"e{i:02d}" for i in range(8)], "content_hash": [f"h{i}" for i in range(8)],
                        "episode_index": range(8), "suite": ["libero_10"] * 4 + ["libero_spatial"] * 4,
                        "libero_task_id": [0, 0, 1, 1, 0, 0, 1, 1], "task_name": ["t"] * 8})
    n_ro = 24
    ret = pd.DataFrame({"rollout_id": range(n_ro), "suite": (["libero_10"] * 12) + (["libero_spatial"] * 12),
                        "libero_task_id": ([0] * 6 + [1] * 6) * 2})
    for c in at.RETURN_COLUMNS:
        ret[c] = rng.normal(size=n_ro)
    ret["failure_type"] = ["success"] * 10 + ["timeout"] * 6 + ["drop"] * 4 + ["rare"] * 4
    m = rng.normal(size=(n_ro, 8))
    t = at.score_table(m, eps, ret)
    want = np.full(8, np.nan)
    for (s, k), g in eps.groupby(["suite", "libero_task_id"]):
        rows = ((ret["suite"] == s) & (ret["libero_task_id"] == k)).to_numpy()
        want[g.index] = at.performance_influence(m[:, g.index], ret["ret_f30"].to_numpy(), rows)
    check("3a infl_f30__same_task equals the per-task computation", np.allclose(t["infl_f30__same_task"], want))
    check("3b harm = −sign·infl (f30 +1, censored60 −1, ft_* −1)",
          np.allclose(t["harm_f30"], -t["infl_f30"]) and np.allclose(t["harm_censored60"], t["infl_censored60"])
          and np.allclose(t["harm_ft_timeout"], t["infl_ft_timeout"]))
    fts = sorted(c for c in t.columns if c.startswith("infl_ft_") and "__" not in c)
    check("3c failure-type columns: types with >= 5 rollouts only", fts == ["infl_ft_timeout"], str(fts))
    ag = at.agreement(t, ["infl_f30__same_task", "infl_rmst60__same_task"])
    check("3d agreement gives Kendall τ per pair", len(ag) == 1 and -1 <= next(iter(ag.values()))["tau"] <= 1)

    t2 = t.copy()
    t2["infl_f30"] = t2["infl_f30"] * 5 + 2
    ens = at.zscore_ensemble([t, t2])
    z = (t["infl_f30"] - t["infl_f30"].mean()) / t["infl_f30"].std()
    check("4a zscore_ensemble: affine-rescaled seed averages to the same z", np.allclose(ens["infl_f30"], z))
    bad = t2.iloc[::-1].reset_index(drop=True)
    try:
        at.zscore_ensemble([t, bad])
        check("4b zscore_ensemble refuses misaligned keys", False)
    except ValueError:
        check("4b zscore_ensemble refuses misaligned keys", True)


def synthetic_select() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vid = "f" * 64
        n_task = 6
        rows = []
        for s in ("libero_10", "libero_spatial"):
            for k in range(3):
                for d in range(n_task):
                    rows.append({"episode_id": f"{s}/{k}/demo_{d}", "content_hash": f"h-{s}-{k}-{d}", "suite": s,
                                 "libero_task_id": k, "episode_index": len(rows)})
        eps = pd.DataFrame(rows)
        (root / "versions" / vid).mkdir(parents=True)
        eps.to_parquet(root / "versions" / vid / "episodes.parquet")
        orig = lin.resolve_version
        lin.resolve_version = lambda ref, lineage_root=None: {"dataset_version_id": vid}
        try:
            parent = lin.Subset(vid, tuple(eps["episode_id"]), {"op": "all"})
            scores = eps[["episode_id", "content_hash"]].copy()
            scores["harm"] = np.random.default_rng(2).normal(size=len(eps))
            scores.loc[0, "harm"] = scores.loc[1, "harm"] = 99.0  # a tie at the top of task (libero_10, 0)
            cur = lin.drop_top_k_per_task(parent, scores, "harm", 2, lineage_root=root)
            per_task = eps.assign(kept=eps["episode_id"].isin(cur.episode_ids)).groupby(["suite", "libero_task_id"])["kept"].sum()
            check("5a drop_top_k_per_task drops k from every task", (per_task == n_task - 2).all(), per_task.tolist().__str__())
            top = scores.assign(t=eps["suite"] + eps["libero_task_id"].astype(str)).sort_values(
                ["harm", "episode_id"], ascending=[False, True]).groupby("t").head(2)["episode_id"]
            check("5b ... and they are each task's 2 highest (ties by episode_id)", sorted(top) == cur.derivation["dropped"])
            r1 = lin.drop_random_k_per_task(parent, 2, seed=0, lineage_root=root)
            r2 = lin.drop_random_k_per_task(parent, 2, seed=0, lineage_root=root)
            r3 = lin.drop_random_k_per_task(parent, 2, seed=1, lineage_root=root)
            check("5c random control: same seed same subset, other seed differs",
                  r1.subset_id == r2.subset_id and r1.episode_ids != r3.episode_ids)
            cnt = lambda s: eps[eps["episode_id"].isin(s.episode_ids)].groupby(["suite", "libero_task_id"]).size().tolist()  # noqa: E731
            check("5d curated and random children hold identical per-task counts", cnt(cur) == cnt(r1))
            stale = scores.copy()
            stale.loc[3, "content_hash"] = "other"
            for label, fn in (("stale score table", lambda: lin.drop_top_k_per_task(parent, stale, "harm", 2, lineage_root=root)),
                              ("k that empties a task", lambda: lin.drop_random_k_per_task(parent, n_task, 0, lineage_root=root))):
                try:
                    fn()
                    check(f"5e refuses a {label}", False)
                except (lin.LineageError, ValueError):
                    check(f"5e refuses a {label}", True)
        finally:
            lin.resolve_version = orig


def toy_output_functions() -> None:
    """AC2.1: ``batch_features`` on a toy module vs brute-force per-sample gradients (both output functions)."""
    import torch
    from torch import nn

    from vlapoc import influence as inf

    torch.manual_seed(0)
    b, t, din, dh, dout, act_dim, rank, k = 5, 4, 6, 10, 5, 4, 3, 2

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(din, dh)
            self.action_out_proj = nn.Linear(dh, dout)

        def forward(self, x):
            return self.action_out_proj(torch.relu(self.body(x)))

    model = Toy()
    x = torch.randn(b, t, din)
    target = torch.randn(b, t, dout)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[1, 2:] = True
    pad[3, 3] = True
    draws = [(torch.randn(b, t, din), 0.5), (torch.randn(b, t, din), 0.7)][:k]
    proj = inf.KroneckerProjector(model, rank, seed=0, device="cpu")
    capture = inf.OutputCapture(model.action_out_proj)

    def sample_obj(v, target, pad_row, fn):  # one sample, written out from the definition
        valid = ~pad_row
        nv = max(int(valid.sum()) * act_dim, 1)
        e = (v - target) ** 2 if fn == "loss" else v**2
        return (e[:, :act_dim] * valid[:, None]).sum() / nv

    def forward(noise, _t):
        v = model(x + noise)
        return torch.stack([sample_obj(v[i], target[i], pad[i], "loss") for i in range(b)])

    for fn in ("square", "loss"):
        capture.reset()  # the brute-force reference below calls the module outside the measured pass, which also
        proj.feats = {}  # fires the projector's backward hooks, so both are cleared before each measured pass
        got, loss, obj = inf.batch_features(forward, lambda: model.zero_grad(set_to_none=True), proj, capture, draws, fn, b,
                                            pad, act_dim)
        want = torch.zeros_like(got)
        for noise, _ in draws:
            for i in range(b):
                v = model(x[i:i + 1] + noise[i:i + 1])[0]
                o = sample_obj(v, target[i], pad[i], fn)
                gs = torch.autograd.grad(o, [m_.weight for _, m_ in proj.layers])
                f = [(proj.p_out[n].T @ g @ proj.p_in[n]).flatten() for (n, _), g in zip(proj.layers, gs)]
                want[i] += torch.cat(f) / len(draws)
        rel = ((got - want).norm(dim=1) / want.norm(dim=1)).max().item()
        check(f"6a toy module: '{fn}' features == brute-force projected per-sample gradient (masked, K=2, B=5)", rel < 1e-5,
              f"max per-sample rel err {rel:.2e}, dim {got.shape[1]} = {len(proj.layers)} layers x {rank}²")
        if fn == "square":
            ref = torch.stack([sample_obj(model(x[i:i + 1] + draws[0][0][i:i + 1])[0], target[i], pad[i], "square")
                               for i in range(b)])
            direct = inf.square_objective(model(x + draws[0][0]), pad, act_dim)
            check("6b square_objective == the definition (mean of v² over valid steps x action dims)",
                  bool(torch.allclose(ref, direct, atol=1e-6)), f"max abs diff {(ref - direct).abs().max().item():.1e}")
    capture.reset()
    model.action_out_proj(torch.randn(1, 1, dh))
    model.action_out_proj(torch.randn(1, 1, dh))
    try:
        capture.pop()
        check("6c OutputCapture raises when the captured module ran twice (premise P1 guard)", False)
    except RuntimeError as e:
        check("6c OutputCapture raises when the captured module ran twice (premise P1 guard)", "2 times" in str(e))
    proj.remove()
    capture.remove()


def cupid_parity() -> None:
    """AC2.2 and AC2.3: ``performance_influence`` vs CUPID's released ``net`` score; the centring algebra."""
    from scipy.stats import kendalltau

    from vlapoc.third_party.cupid_quality import compute_demo_quality_scores, sum_of_sum_influence

    rng = np.random.default_rng(7)
    n_ro, n_ep, n_rs, n_ts = 11, 9, 60, 80
    rg = np.concatenate([np.arange(n_ro), rng.integers(0, n_ro, n_rs - n_ro)])  # rollout row of each rollout sample
    tg = np.concatenate([np.arange(n_ep), rng.integers(0, n_ep, n_ts - n_ep)])  # episode row of each train sample
    pair = rng.normal(size=(n_rs, n_ts))  # Ψ_a-inf for every (rollout sample, train sample) pair
    m_ours = np.zeros((n_ro, n_ep))
    for i in range(n_rs):
        np.add.at(m_ours[rg[i]], tg, pair[i])  # Σ over steps and training samples = "sum_of_sum"
    succ = np.array([True, False, True, True, False, False, True, False, False, True, False])
    traj = np.array([[sum_of_sum_influence(pair[np.ix_(rg == j, tg == e)], bool(succ[j])) for e in range(n_ep)]
                     for j in range(n_ro)])
    cupid = compute_demo_quality_scores(traj, succ, "net")
    ours = at.performance_influence(m_ours, succ.astype(float), center=False, return_encoding="pm1") * n_ro
    ours2 = at.performance_influence(m_ours, 2 * succ.astype(float) - 1, center=False) * n_ro
    err = max(np.abs(ours - cupid).max(), np.abs(ours2 - cupid).max())
    check("7a CUPID parity: performance_influence(M, ±1, center=False)·m == compute_demo_quality_scores(.., 'net') "
          "(vendored 2941eba9)", err < 1e-9 and np.abs(m_ours - traj).max() < 1e-9, f"max abs err {err:.2e}")

    m = rng.normal(size=(40, 25))
    for p in (0.3, 0.5):
        s = np.zeros(40)
        s[: int(round(p * 40))] = 1
        pp = s.mean()
        ms, mf = m[s == 1].mean(0), m[s == 0].mean(0)
        centred = at.performance_influence(m, s) * 40
        net = at.performance_influence(m, s, center=False, return_encoding="pm1") * 40
        id1 = np.abs(centred - pp * (1 - pp) * 40 * (ms - mf)).max()
        id2 = np.abs(net - 40 * (pp * ms - (1 - pp) * mf)).max()
        tau = kendalltau(centred, net).statistic
        if p == 0.5:
            check("7b centring algebra and p = 1/2: centred = p(1-p)·m·(ms-mf), net = m·[p·ms-(1-p)·mf], same ranking",
                  id1 < 1e-9 and id2 < 1e-9 and abs(tau - 1) < 1e-12, f"p={pp}, identity err {max(id1, id2):.1e}, Kendall tau {tau:.3f}")
        else:
            check("7c ... but at p = 0.3 the two rankings differ (counterexample)", id1 < 1e-9 and id2 < 1e-9 and tau < 0.999,
                  f"p={pp}, identity err {max(id1, id2):.1e}, Kendall tau {tau:.3f}")
    for label, fn in (("pm1 with centring", lambda: at.performance_influence(m, np.ones(40), return_encoding="pm1")),
                      ("pm1 on a non-binary return", lambda: at.performance_influence(m, np.linspace(0, 2, 40), center=False,
                                                                                       return_encoding="pm1"))):
        try:
            fn()
            check(f"7d refuses {label}", False)
        except ValueError:
            check(f"7d refuses {label}", True)


def output_fn_guards() -> None:
    """AC2.5 (part): manifests with mixed train-side output functions are refused; one side may differ from the other."""
    base = {"checkpoint_sha256": "a", "seed": 0, "rank": 8, "k_draws": 4, "dim": 4, "layers": ["l"], "dataset_root": "d"}
    sq, lo = [{**base, "output_fn": "square"}], [{**base, "output_fn": "loss"}]
    ok = at.check_compatible(sq, sq)["output_fns"] == ["square", "square"]
    ok &= at.check_compatible(lo, sq)["output_fns"] == ["loss", "square"]  # ATHENA's mixed form is allowed across sides
    ok &= at.check_compatible([{**base}], sq)["output_fns"] == ["loss", "square"]  # a v1 manifest without the key means loss
    check("8a check_compatible accepts one output function per side (incl. ATHENA loss/square, legacy manifests = loss)", ok)
    try:
        at.check_compatible(sq + lo, sq)
        check("8b check_compatible refuses shards that mix train-side output functions", False)
    except ValueError as e:
        check("8b check_compatible refuses shards that mix train-side output functions", "mix output functions" in str(e), str(e)[:90])
    check("8c DIVERGENCES names every departure listed in the plan",
          {"centred_returns", "draws", "projection", "damping", "strides", "numerics"} <= set(at.DIVERGENCES))


def cpu_smoke(ckpt: Path, root: Path, device: str, scratch: Path) -> None:
    """AC2.4: ``featurize`` on the local SmolVLA checkpoint, ≤ 2 episodes of v0/reference, stride 20, both output functions."""
    import time

    from vlapoc import influence
    from vlapoc.method_data import open_for_method

    view = open_for_method("v0/reference")  # the smoke reads data through the method loader too
    episodes = list(view.episode_indices)[:2]
    scratch.mkdir(parents=True, exist_ok=True)
    feats, secs = {}, {}
    for fn in ("square", "loss", "square"):
        prefix = scratch / f"smoke_{fn}_{len(feats)}"
        t0 = time.time()
        man = influence.featurize(ckpt, root, prefix, episodes=episodes, stride=20, batch_size=4, workers=0, device=device,
                                  output_fn=fn)
        secs[prefix.name] = round(time.time() - t0, 1)
        feats[prefix.name] = (np.load(f"{prefix}.npy").astype(np.float32), man, np.load(f"{prefix}.meta.npz"))
    (sq0, m0, meta0), (lo, m1, meta1), (sq1, m2, _) = feats.values()
    check("S1 featurize(output_fn=square) writes features, dim 7,488 = 117 layers x 8²", sq0.shape[1] == 7488 and m0["dim"] == 7488,
          f"shape {sq0.shape}, n_layers {m0['n_layers']}, device {device}")
    check("S2 the manifest records the output function", m0["output_fn"] == "square" and m1["output_fn"] == "loss")
    check("S3 no NaN/inf in either feature set", bool(np.isfinite(sq0).all() and np.isfinite(lo).all()))
    check("S4 two square calls agree bit-for-bit (seeded draws)", bool(np.array_equal(sq0, sq1)),
          f"max abs diff {np.abs(sq0 - sq1).max():.1e}")
    cos = (sq0 * lo).sum(1) / (np.linalg.norm(sq0, axis=1) * np.linalg.norm(lo, axis=1))
    check("S5 square vs loss features of the same samples (how much the output function changes the gradient)", True,
          f"cosine min {cos.min():.3f} median {np.median(cos):.3f} max {cos.max():.3f} over {len(cos)} samples; "
          f"norm ratio |square|/|loss| median {np.median(np.linalg.norm(sq0, axis=1) / np.linalg.norm(lo, axis=1)):.3g}")
    check("S6 wall time recorded", True, f"{secs} (s, incl. model load) for {len(cos)} samples on {device}")
    print(json.dumps({"episodes": episodes, "n_samples": int(len(cos)), "seconds": secs, "cosine_square_vs_loss": cos.tolist(),
                      "objective_square": meta0["objective"].tolist(), "objective_loss": meta1["objective"].tolist()}, indent=1))


def regress(attr_dir: Path, feats_dir: Path, rollouts_dir: Path) -> None:
    old = np.load(attr_dir / "rollout_demo_influence.npy")
    tf, tmeta = [], []
    for f in sorted(glob.glob(str(feats_dir / "train.shard*.npy")), key=lambda f: int(f.rsplit(".shard", 1)[1].split(".")[0])):
        tf.append(np.load(f, mmap_mode="r"))
        tmeta.append(np.load(f[:-4] + ".meta.npz")["episode_index"])
    rf, rg, off = [], [], 0
    for i in range(len(glob.glob(str(rollouts_dir / "part*")))):
        rf.append(np.load(feats_dir / f"attrib_part{i}.npy", mmap_mode="r"))
        rg.append(np.load(feats_dir / f"attrib_part{i}.meta.npz")["episode_index"] + off)
        off += len(pd.read_parquet(rollouts_dir / f"part{i}" / "rollouts.parquet"))
    m = at.influence_matrix(tf, np.concatenate(tmeta), old.shape[1], rf, np.concatenate(rg), off, damping=1e-3)
    err = np.abs(m - old).max() / np.abs(old).max()
    check("R influence_matrix reproduces the October rollout_demo_influence.npy", m.shape == old.shape and err < 1e-6,
          f"shape {m.shape} vs {old.shape}, max rel err {err:.2e}")


OLD_PY = "/raid/chem/gor/vla-curation/envs/lerobot/bin/python"
OLD_SRC = "/auto/home/gor/projects/VLA-data-curation/src"
OLD_REPLAY = """
import json, sys, torch
from lerobot.policies.factory import make_pre_post_processors
from vlacur.influence import load_policy
policy, cfg = load_policy(sys.argv[1], "cuda")
make_pre_post_processors(policy_cfg=cfg, pretrained_path=sys.argv[1], preprocessor_overrides={"device_processor": {"device": "cuda"}})
beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
print(json.dumps((beta.sample((int(sys.argv[2]),)) * 0.999 + 0.001).tolist()))
"""


def old_times(ckpt: Path, k: int) -> list[float]:
    """The K times the October code would draw in a fresh process: it seeded only the noise and took the times from
    torch's global RNG, whose initial seed is random per process."""
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": OLD_SRC, "HF_HOME": "/raid/chem/gor/vla-curation/hf_cache", "HF_HUB_OFFLINE": "1"}
    out = subprocess.run([OLD_PY, "-c", OLD_REPLAY, str(ckpt), str(k)], env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _old_reference(ckpt: Path, root: Path, ep: np.ndarray, fr: np.ndarray, draws: list, dtypes: dict | None = None) -> np.ndarray:
    """Features of the given (episode, frame) samples through the October code's own loader, processors and projector
    (``vlacur.influence``), with the given draws, in this process. ``dtypes`` (name → dtype) recasts its parameters."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    sys.path.insert(0, OLD_SRC)
    from vlacur.influence import KroneckerProjector, load_policy

    policy, cfg = load_policy(str(ckpt), "cuda")
    for name, p in policy.named_parameters():
        if dtypes and p.dtype != dtypes[name]:
            p.data = p.data.to(dtypes[name])
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(ckpt),
                                      preprocessor_overrides={"device_processor": {"device": "cuda"}})
    ds = LeRobotDataset("lerobot/libero_10", root=root, delta_timestamps={"action": [i / 10 for i in range(cfg.chunk_size)]})
    gi = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ds.hf_dataset["episode_index"], ds.hf_dataset["frame_index"]))}
    idx = [gi[(int(e), int(f))] for e, f in zip(ep, fr)]
    proj = KroneckerProjector(policy, 8, seed=0, device="cuda")
    out = []
    for raw in torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=16, shuffle=False):
        batch = pre(raw)
        n = batch["action"].shape[0]
        acc = None
        for noise, t in draws:
            policy.zero_grad(set_to_none=True)
            loss, _ = policy.forward(dict(batch), noise=noise.expand(n, -1, -1).contiguous(),
                                     time=torch.full((n,), t, device="cuda"), reduction="none")
            loss.sum().backward()
            f = proj.collect()
            acc = f if acc is None else acc + f
        out.append((acc / len(draws)).cpu().numpy().astype(np.float16).astype(np.float32))
    proj.remove()
    return np.concatenate(out)


def gpu_smoke(ckpt: Path, root: Path, old_prefix: Path, episodes: list[int], scratch: Path) -> None:
    """``featurize`` (through the eval stage's ``checkpoint_policy`` loader) on a few episodes vs the October code's own
    path on the same samples with the same draws; seeded draws; and why the October feature files cannot serve as the
    reference (their times were not seeded)."""
    from vlapoc import influence

    out = scratch / "smoke_train"
    man = influence.featurize(ckpt, root, out, episodes=episodes, stride=2, batch_size=16, workers=4)
    new = np.load(f"{out}.npy").astype(np.float32)
    nm = np.load(f"{out}.meta.npz")
    check("G1 featurize loads the checkpoint via checkpoint_policy and writes the manifest",
          man["checkpoint_weights_sha256_recheck"] and man["n_samples"] == len(new), f"{man['n_layers']} layers, dim {man['dim']}")
    check("G2 per-episode stride: frames are every 2nd frame_index of each episode", bool((nm["frame_index"] % 2 == 0).all()))

    import torch

    pol, _, _ = influence.load_policy(ckpt, "cuda")
    dtypes = {n: p.dtype for n, p in pol.named_parameters()}
    del pol
    torch.cuda.empty_cache()
    draws = influence.fixed_draws(4, 50, 32, 0, "cuda")

    def compare(ref):
        cos = (new * ref).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(ref, axis=1))
        rel = np.linalg.norm(new - ref, axis=1) / np.linalg.norm(ref, axis=1)
        return cos, f"{len(new)} samples, min cos {cos.min():.7f}, median rel diff {np.median(rel):.2e}, max {rel.max():.2e}"

    cos, d = compare(_old_reference(ckpt, root, nm["episode_index"], nm["frame_index"], draws, dtypes))
    check("G3 featurize == the October code's loader/processors/projector (same samples, draws, parameter dtypes)",
          cos.min() > 0.999, d)
    n_up = sum(dt == torch.float32 for dt in dtypes.values())
    cos, d = compare(_old_reference(ckpt, root, nm["episode_index"], nm["frame_index"], draws))
    check("G3b (finding) LeRobot from_pretrained keeps the file's bf16 tensors; vlapoc.runs.load_checkpoint upcasts some "
          "to fp32 (the eval stage's numerics). Effect on features, recorded", True, f"{n_up} fp32 params under load_checkpoint; {d}")

    t1, t2 = old_times(ckpt, 4), old_times(ckpt, 4)
    old_rows = np.load(sorted(glob.glob(f"{old_prefix}.shard*.npy"))[0], mmap_mode="r")
    check("G5 (finding) the October code's times differ between processes, so its feature files are not reproducible",
          not np.allclose(t1, t2), f"two fresh processes: {np.round(t1, 4).tolist()} vs {np.round(t2, 4).tolist()}; "
          f"October shard0 has {len(old_rows)} rows, each job drew its own times")

    draws = [t for _, t in influence.fixed_draws(4, 50, 32, 0, "cpu")]
    runs = []
    for i in range(2):
        prefix = scratch / f"smoke_seeded{i}"
        influence.featurize(ckpt, root, prefix, episodes=episodes[:1], stride=4, batch_size=16, workers=0)
        runs.append(np.load(f"{prefix}.npy").astype(np.float32))
    d = np.abs(runs[0] - runs[1]).max() / np.abs(runs[0]).max()
    check("G4 seeded draws ignore torch's global RNG: two featurize calls in one process (the second model build "
          "advances it) agree", d < 1e-2, f"times {np.round(draws, 4).tolist()}, max rel diff {d:.2e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regress", nargs=3, type=Path, metavar=("ATTR_DIR", "FEATS_DIR", "ROLLOUTS_DIR"))
    p.add_argument("--gpu-smoke", nargs=3, type=Path, metavar=("CKPT", "DATASET_ROOT", "OLD_TRAIN_PREFIX"))
    p.add_argument("--cpu-smoke", nargs=2, type=Path, metavar=("CKPT", "DATASET_ROOT"), help="AC2.4 (S2)")
    p.add_argument("--device", default="cpu", help="device for --cpu-smoke (mps or cpu)")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--scratch", type=Path, default=Path("data/scratch/verify_attribution"))
    p.add_argument("--out", type=Path, default=None, help="write the results JSON here")
    a = p.parse_args()
    if a.regress:
        regress(*a.regress)
    elif a.gpu_smoke:
        gpu_smoke(*a.gpu_smoke, a.episodes, a.scratch)
    elif a.cpu_smoke:
        cpu_smoke(*a.cpu_smoke, a.device, a.scratch)
    else:
        synthetic_m()
        synthetic_table()
        synthetic_select()
        toy_output_functions()
        cupid_parity()
        output_fn_guards()
    ok = all(r[1] for r in RESULTS)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"pass": ok, "checks": [{"name": n, "pass": p_, "detail": d} for n, p_, d in RESULTS]}, indent=1))
    print(f"pass: {ok} ({sum(r[1] for r in RESULTS)}/{len(RESULTS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
