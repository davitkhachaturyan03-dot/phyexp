"""Verify ground-truth S4 (the full-scale injected version, now v3) against AC4.1-AC4.8, on the stored bytes.

Grounding: PoC §2.5 p.5 ("Inject 5-10% corrupted episodes"), §8.7 p.15 / §8.3 p.14 (clean references),
§2.1 item 1 p.3 (lineage) and item 4 p.3 (blindness); RoboDrop App. A.1 p.13 ("10% of the episodes for every task").

    .venv/bin/python scripts/verify_injection_s4.py --version v3 --parent v0 --prefix s4_v3 --prior v2 \
        --run-id d3dd9fd4dff84d37b5be21fa563fcc57          # the derive run of v3 (AC4.7 reads it from the real store)

Reads only; everything is recomputed from the stored dataset, never from the derivation's in-memory results.
Writes ``artifacts/s4_verify.json``, ``artifacts/s4_allocation.csv`` and ``artifacts/s4_filmstrip.png``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import corruptions as C  # noqa: E402
from vlapoc.catalog import hash_stored_dataset, episode_content_hash  # noqa: E402
from vlapoc.derive import read_stored_episode  # noqa: E402
from vlapoc.groundtruth import load_answer_key  # noqa: E402
from vlapoc.lineage import Subset, check_blind, resolve_version, verify_version, version_episodes  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"


def check_stored(ctype: str, par: dict, ch: dict, params: dict, parent_tasks: set, suite_of_task: dict) -> list[str]:
    """S1's invariants (AC1.2-AC1.4) on a stored child episode against its stored parent episode."""
    fails = []
    a, b = par["action"], ch["action"]
    T, T2 = len(a), len(b)
    if not ((b >= -1).all() and (b <= 1).all()):
        fails.append("range")
    if not np.isin(b[:, 6], [-1.0, 1.0]).all():
        fails.append("gripper not binary")
    if T2 != (T + 40 if ctype in ("detour", "idle_padding") else T - 40 if ctype == "time_shift" else T):
        fails.append(f"length {T} -> {T2}")
    same = lambda k: len(ch[k]) == len(par[k]) and np.array_equal(ch[k], par[k])  # noqa: E731
    pd_ = {k: set(C.frame_digests(par[k])) for k in ("image", "image2")}
    if ctype == "action_noise":
        if not (same("state") and same("image") and same("image2") and ch["task"] == par["task"] and np.array_equal(a[:, 6], b[:, 6])):
            fails.append("non-action array or gripper changed")
    elif ctype == "time_shift":  # r2: a trim (obs 0..T-41 with actions 40..T-1)
        if not (np.array_equal(ch["state"], par["state"][:T - 40]) and np.array_equal(ch["image"], par["image"][:T - 40])
                and np.array_equal(ch["image2"], par["image2"][:T - 40])) or not np.array_equal(b, a[40:]):
            fails.append("trim: state/images/actions are not the parent's frames 0..T-41 / actions 40..T-1")
        if params.get("length_change") != -40:
            fails.append("length_change != -40")
        if int((b[:, :6] == 0).all(axis=1).sum()) != int((a[40:, :6] == 0).all(axis=1).sum()):
            fails.append("AC1.8 idle_steps")
    elif ctype == "instruction_swap":
        if not (same("action") and same("state") and same("image") and same("image2")):
            fails.append("arrays changed")
        if ch["task"] == par["task"] or ch["task"] not in parent_tasks or suite_of_task[ch["task"]] != suite_of_task[par["task"]]:
            fails.append("task not a different same-suite parent task")
    elif ctype == "dropped_frames":
        if not (same("action") and same("state")):
            fails.append("action/state changed")
        for k in ("image", "image2"):
            if not set(C.frame_digests(ch[k])) <= pd_[k]:
                fails.append(f"{k} non-parent image")
    elif ctype in ("detour", "idle_padding"):
        for k in ("image", "image2"):
            if not set(C.frame_digests(ch[k])) <= pd_[k]:
                fails.append(f"{k} non-parent image")
        if ctype == "detour" and sum(int((ch[k][1:] == ch[k][:-1]).all(axis=(1, 2, 3)).sum()) for k in ("image", "image2")):
            fails.append("consecutive identical frames")
    elif ctype == "gripper_release":
        (s, e), = params["corrupted_frames"]
        diff = np.where((a != b).any(axis=1))[0]
        if not (same("state") and same("image") and same("image2") and np.array_equal(a[:, :6], b[:, :6])
                and set(diff) <= set(range(s, e)) and (b[s:e, 6] == -1).all()):
            fails.append("release edit outside declared range")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--parent", default="v0")
    ap.add_argument("--run-id")
    ap.add_argument("--prefix", default="s4", help="artifact file prefix (s4, or s4_v3 for the round-2 version)")
    ap.add_argument("--prior", help="AC4.8: the superseded version (v2) to compare labels with and to check it is fenced")
    ap.add_argument("--artifacts", type=Path, default=FEATURE_DIR / "artifacts")
    a = ap.parse_args()
    t0 = time.time()
    out: dict = {}
    ok = True
    v, p = resolve_version(a.version), resolve_version(a.parent)
    vroot, proot = REPO_ROOT / v["lerobot_root"], REPO_ROOT / p["lerobot_root"]
    ve = version_episodes(a.version)
    pe = version_episodes(a.parent).set_index("episode_id")
    labels = load_answer_key(a.version)
    cand0, ref0 = Subset.load(f"{p['name']}/candidate"), Subset.load(f"{p['name']}/reference")

    # ---- AC4.1 ---------------------------------------------------------------------------------------
    per_type = labels["corruption_type"].value_counts().to_dict()
    in_cand, in_ref = set(labels["episode_id"]) <= set(cand0.episode_ids), set(labels["episode_id"]) & set(ref0.episode_ids)
    ok1 = len(ve) == 1000 and len(labels) == 91 and set(per_type.values()) == {13} and len(per_type) == 7 and in_cand and not in_ref
    print(f"AC4.1 {a.version} {v['dataset_version_id'][:12]}: {len(ve)} episodes; labels rows {len(labels)}; per type {per_type}; "
          f"all in {p['name']}/candidate: {in_cand}; in reference: {len(in_ref)}")
    out["AC4.1"] = {"episodes": len(ve), "labels": len(labels), "per_type": per_type, "in_candidate": in_cand,
                    "in_reference": len(in_ref), "pass": ok1}
    ok &= ok1

    # ---- AC4.2 ---------------------------------------------------------------------------------------
    lab = labels.merge(ve[["episode_id", "suite", "task_name", "libero_task_id", "episode_index"]], on="episode_id")
    tt = lab.groupby(["suite", "libero_task_id", "corruption_type"]).size().unstack(fill_value=0)
    tt["total"] = tt.sum(axis=1)
    print("\nAC4.2 task x type allocation (rows: suite, libero_task_id)")
    print(tt.to_string())
    tt.to_csv(a.artifacts / f"{a.prefix}_allocation.csv")
    tasks_per_type = lab.groupby("corruption_type")["task_name"].nunique().to_dict()
    suites_per_type = {t: sorted(g.groupby("suite").size().to_list()) for t, g in lab.groupby("corruption_type")}
    per_task = sorted(set(tt["total"]))
    no_repeat = int(tt.drop(columns="total").max().max()) == 1
    n_cand = len(cand0.episode_ids)
    ok2 = set(tasks_per_type.values()) == {13} and all(v_ in ([6, 7], [7, 6]) for v_ in suites_per_type.values()) \
        and set(per_task) <= {4, 5} and no_repeat
    print(f"AC4.2 distinct tasks per type {tasks_per_type}; suite split per type {suites_per_type}; corrupted per task {per_task}; "
          f"type repeated within a task: {not no_repeat}; {len(labels)}/{n_cand} = {len(labels) / n_cand:.2%} of candidates")
    out["AC4.2"] = {"tasks_per_type": tasks_per_type, "suite_split": suites_per_type, "per_task": per_task, "no_repeat": no_repeat,
                    "fraction": len(labels) / n_cand, "pass": ok2}
    ok &= ok2

    # ---- AC4.3 ---------------------------------------------------------------------------------------
    t1 = time.time()
    blind = check_blind(a.version)
    ver = verify_version(a.version)
    cand2, ref2 = Subset.load(f"{v['name']}/candidate"), Subset.load(f"{v['name']}/reference")
    ref_hash_eq = bool((ve.set_index("episode_id").loc[list(ref2.episode_ids), "content_hash"].to_numpy()
                        == pe.loc[list(ref2.episode_ids), "content_hash"].to_numpy()).all())
    ok3 = blind == [] and ver["ok"] and len(ref2.episode_ids) == 40 and len(cand2.episode_ids) == 960 \
        and set(ref2.episode_ids) == set(ref0.episode_ids) and ref_hash_eq
    print(f"\nAC4.3 check_blind({a.version}) = {blind}; verify_version ok={ver['ok']} ({ver['files_checked']} files); "
          f"{a.version}/reference {len(ref2.episode_ids)} == {p['name']}/reference ids: {set(ref2.episode_ids) == set(ref0.episode_ids)}; "
          f"/candidate {len(cand2.episode_ids)}; reference content hashes equal parent's: {ref_hash_eq} ({time.time() - t1:.0f}s)")
    out["AC4.3"] = {"check_blind": blind, "verify_ok": ver["ok"], "ref": len(ref2.episode_ids), "cand": len(cand2.episode_ids),
                    "ref_hashes_equal_parent": ref_hash_eq, "pass": ok3}
    ok &= ok3

    # ---- AC4.4 ---------------------------------------------------------------------------------------
    t1 = time.time()
    stored = hash_stored_dataset(vroot).set_index("episode_index")
    vi = ve.set_index("episode_index")
    table_matches_bytes = bool((stored.loc[vi.index, "content_hash"].to_numpy() == vi["content_hash"].to_numpy()).all())
    unedited = ve[~ve["episode_id"].isin(labels["episode_id"])]
    unedited_equal = bool((unedited["content_hash"].to_numpy() == pe.loc[unedited["episode_id"], "content_hash"].to_numpy()).all())
    stored_unedited_equal = bool((stored.loc[unedited["episode_index"], "content_hash"].to_numpy()
                                  == pe.loc[unedited["episode_id"], "content_hash"].to_numpy()).all())
    parent_tasks = set(pe["language_instruction"])
    suite_of_task = dict(zip(pe["language_instruction"], pe["suite"]))
    fails, hash_ok = {}, 0
    cache_ep = {}
    for r in lab.itertuples():
        ch = read_stored_episode(vroot, int(r.episode_index))
        par = read_stored_episode(proot, int(pe.loc[r.episode_id, "episode_index"]))
        h = episode_content_hash(ch["task"], ch["action"], ch["state"], ch["image"], ch["image2"])
        hash_ok += int(h == r.content_hash)
        params = json.loads(r.params)
        f = check_stored(r.corruption_type, par, ch, params, parent_tasks, suite_of_task)
        if f:
            fails[r.episode_id] = f
        if r.corruption_type not in cache_ep:
            cache_ep[r.corruption_type] = (par, ch, params)
        if r.corruption_type == "action_noise":
            cache_ep.setdefault("_noise", []).append((ch["action"], par["action"], params))
    print(f"\nAC4.4 recomputed all 1000 hashes from the stored bytes: equal to the episode table: {table_matches_bytes}; "
          f"909 unedited: table == {p['name']} {unedited_equal}, stored bytes == {p['name']} {stored_unedited_equal} (n={len(unedited)})")
    print(f"AC4.4 91 labelled episodes read back: {hash_ok}/91 hash to their label's content_hash; S1 invariants (range, gripper "
          f"binary, length, content, image provenance) failing episodes: {len(fails)} {dict(list(fails.items())[:3])} "
          f"({time.time() - t1:.0f}s)")
    ok4 = table_matches_bytes and unedited_equal and stored_unedited_equal and hash_ok == 91 and not fails and len(unedited) == 909
    out["AC4.4"] = {"table_matches_bytes": table_matches_bytes, "unedited_n": len(unedited), "unedited_equal_parent": unedited_equal,
                    "stored_unedited_equal_parent": stored_unedited_equal, "labelled_hash_ok": hash_ok, "invariant_failures": fails,
                    "pass": ok4}
    ok &= ok4

    # ---- AC4.5 ---------------------------------------------------------------------------------------
    nz = cache_ep["_noise"]
    lab_clip = {k: sum(p_[k] for _, _, p_ in nz) for k in ("n_clipped", "n_clipped_dims0_2", "n_beyond_clean_range", "n_values")}
    n3 = sum(3 * len(c) for c, _, _ in nz)
    # recount from the stored bytes: clipping makes a value exactly +-1 on dims 0-2 (clean maximum is 0.9375, so no
    # clean value is +-1), and values in (0.9375, 1] on dims 0-2 are the 'beyond clean range' ones
    rec_clip = sum(int((np.abs(c[:, :3]) == 1.0).sum()) for c, _, _ in nz)
    rec_beyond = sum(int((np.abs(c[:, :3]) > C.CLEAN_TRANSLATION_MAX).sum()) for c, _, _ in nz)
    rec_clean_beyond = sum(int((np.abs(pa[:, :3]) > C.CLEAN_TRANSLATION_MAX).sum()) for _, pa, _ in nz)
    ok5 = lab_clip["n_clipped"] == rec_clip and lab_clip["n_beyond_clean_range"] == rec_beyond and rec_clean_beyond == 0
    print(f"\nAC4.5 13 action_noise labels: n_values {lab_clip['n_values']}, n_clipped {lab_clip['n_clipped']} "
          f"(frac {lab_clip['n_clipped'] / lab_clip['n_values']:.4%} of all values; {lab_clip['n_clipped_dims0_2'] / n3:.3%} of dims-0-2 values), "
          f"n_beyond_clean_range {lab_clip['n_beyond_clean_range']} ({lab_clip['n_beyond_clean_range'] / n3:.3%} of dims-0-2 values)")
    print(f"AC4.5 recount from stored bytes: values exactly +-1 on dims 0-2 = {rec_clip}; values with 0.9375 < |a| <= 1 = {rec_beyond}; "
          f"same count in the parent's episodes = {rec_clean_beyond}; match labels: {ok5}")
    out["AC4.5"] = {"labels": lab_clip, "frac_clipped_dims0_2": lab_clip["n_clipped_dims0_2"] / n3,
                    "frac_beyond": lab_clip["n_beyond_clean_range"] / n3, "recount_clipped": rec_clip, "recount_beyond": rec_beyond,
                    "parent_beyond": rec_clean_beyond, "pass": ok5}
    ok &= ok5

    # ---- AC4.6 / 4.7 ---------------------------------------------------------------------------------
    du = subprocess.run(["du", "-sh", str(vroot)], capture_output=True, text=True).stdout.split()[0]
    deriv = json.loads((REPO_ROOT / "data" / "lineage" / "versions" / v["dataset_version_id"] / "derivation.json").read_text())
    print(f"\nAC4.6 du -sh {v['lerobot_root']}: {du}; labels_sha256 {deriv['labels_sha256'][:16]}; "
          f"release before 0.8T (R10): {sum(json.loads(r.params)['final_release_rel_position'] < 0.8 for r in lab[lab.corruption_type == 'gripper_release'].itertuples())}/13")
    out["AC4.6"] = {"du": du}
    if a.run_id:
        from vlapoc.runs import client, trace

        run = client().get_run(a.run_id)
        tr = trace(a.run_id)
        stages = sorted({r["stage"] for r in tr["runs"].values()}) if tr["runs"] and "stage" in next(iter(tr["runs"].values())) else None
        reached_v0 = p["dataset_version_id"] in tr["entities"]
        has_raw = bool(tr["raw_files"])
        tags = run.data.tags
        ok7 = reached_v0 and has_raw and tags.get("labels_sha256") == deriv["labels_sha256"]
        print(f"AC4.7 trace({a.run_id[:8]}): {len(tr['runs'])} runs {stages}, reaches {p['name']} version entity: {reached_v0}; raw files listed: "
              f"{len(tr['raw_files'] or [])}; tag labels_sha256 == derivation.json: {tags.get('labels_sha256') == deriv['labels_sha256']}; "
              f"timing tags { {k: v_ for k, v_ in tags.items() if k.startswith('derive.timing')} }")
        out["AC4.7"] = {"run_id": a.run_id, "reached_parent": reached_v0, "raw_files": len(tr["raw_files"] or []),
                        "labels_sha256_tag": tags.get("labels_sha256"), "git": tags.get("mlflow.source.git.commit"), "pass": ok7}
        ok &= ok7
    # ---- AC4.8 (round 2): v2 vs v3, retirement, AC1.8 on the stored bytes --------------------------------------
    if a.prior:
        from vlapoc.lineage import fence_record
        from vlapoc.method_data import _reset_session, open_for_method

        old = load_answer_key(a.prior).set_index("episode_id")
        new = labels.set_index("episode_id")
        same_ids = set(old.index) == set(new.index) and (old["corruption_type"] == new.loc[old.index, "corruption_type"]).all()
        nts = new["corruption_type"] != "time_shift"
        eq = (old.loc[new.index[nts], "content_hash"] == new.loc[nts, "content_hash"]).all()
        ts = new.index[~nts]
        differ = (old.loc[ts, "content_hash"] != new.loc[ts, "content_hash"]).all()
        lc = {json.loads(new.loc[i, "params"]).get("length_change") for i in ts}
        old_lc = {json.loads(old.loc[i, "params"]).get("length_change") for i in ts}
        print(f"\nAC4.8 {a.prior} vs {a.version}: same 91 episode_ids and types: {same_ids}; the {int(nts.sum())} non-time_shift rows keep their content_hash: "
              f"{eq}; the {len(ts)} time_shift rows differ: {differ}; length_change in {a.version}: {sorted(lc)} (in {a.prior}: {sorted(map(str, old_lc))})")
        fr = fence_record(a.prior)
        vid_new = v["dataset_version_id"]
        _reset_session()
        try:
            open_for_method(f"{a.prior}/candidate")
            refused = None
        except Exception as e:  # noqa: BLE001
            refused = f"{type(e).__name__}: {e}"
        _reset_session()
        opened = open_for_method(f"{a.version}/candidate")
        _reset_session()
        child_tbl = ve.set_index("episode_id")
        tr = {i: int(child_tbl.loc[i, "length_steps"]) for i in ts}
        idle_ok = fails == {} and not any("AC1.8" in " ".join(f_) for f_ in fails.values())
        print(f"AC4.8 fence_record({a.prior}): reason {fr['reason']!r}, superseded_by {fr.get('superseded_by', '')[:12]} == {a.version} {vid_new[:12]}: "
              f"{fr.get('superseded_by') == vid_new}")
        print(f"AC4.8 open_for_method({a.prior}/candidate) refused: {refused is not None}: {str(refused)[:160]}; open_for_method({a.version}/candidate) opened "
              f"{len(opened.episode_indices)} episodes; AC1.8 (child idle_steps == parent's idle in a[40:T]) on the stored bytes for the {len(ts)} time_shift episodes: {idle_ok}")
        prior_root = REPO_ROOT / resolve_version(a.prior)["lerobot_root"]
        print(f"AC4.8 {a.prior}'s physical copy still exists (owner cleanup, not deleted): {prior_root.is_dir()}")
        ok8 = bool(same_ids and eq and differ and lc == {-40} and fr and fr.get("superseded_by") == vid_new
                   and "superseded by" in fr["reason"] and refused and len(opened.episode_indices) == 960 and idle_ok and prior_root.is_dir())
        out["AC4.8"] = {"same_ids_types": bool(same_ids), "non_time_shift_hashes_equal": bool(eq), "time_shift_differ": bool(differ),
                        "length_change": sorted(lc), "fence_reason": fr["reason"], "v2_refused": refused, "v3_opened": len(opened.episode_indices),
                        "ac1_8_stored": idle_ok, "pass": ok8}
        ok &= ok8
    out["pass"] = bool(ok)
    (a.artifacts / f"{a.prefix}_verify.json").write_text(json.dumps(out, indent=2, default=str))

    # filmstrip: one stored episode per type, before/after
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(7, 6, figsize=(13, 15))
    for r_, t in enumerate(C.TYPES):
        par, ch, params = cache_ep[t]
        s_, e_ = params["corrupted_frames"][0]
        pick = np.linspace(s_, min(e_ - 1, len(ch["image"]) - 1), 6).astype(int)
        for c_, j in enumerate(pick):
            axes[r_, c_].imshow(ch["image"][j] if t != "gripper_release" else ch["image"][j])
            axes[r_, c_].set_title(f"stored child frame {j}" if c_ else f"{t}: child {j}", fontsize=7)
            axes[r_, c_].axis("off")
    fig.suptitle("S4: frames inside the corrupted range of one stored v2 episode per type (images unchanged for noise, shift, swap, release)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.artifacts / f"{a.prefix}_filmstrip.png", dpi=80)
    print(f"\nall S4 checks pass: {ok} ({time.time() - t0:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
