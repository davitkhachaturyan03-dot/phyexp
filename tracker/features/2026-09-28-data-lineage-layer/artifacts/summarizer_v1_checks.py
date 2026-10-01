"""Summarizer check (2026-10-01): what v1's ground truth looks like on disk, read with pyarrow only.

Reads v0/v1 episode tables, v1 labels.parquet and the action column of v1's data files directly (no
LeRobotDataset, so no HF datasets cache is written). Run from the repo root.
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

V0 = "3c4dca68d6f196c10d249b43bd94ae32f51b905735f25c7865e3c943caf553b7"
V1 = "16076b9c5cb2a0e8c5d291fc0766977dc7286c6b9f87df39bd87fe5c17cd85f8"
e0 = pd.read_parquet(f"data/lineage/versions/{V0}/episodes.parquet")
e1 = pd.read_parquet(f"data/lineage/versions/{V1}/episodes.parquet")
lab = pd.read_parquet(f"data/lineage/versions/{V1}/labels.parquet")

print(f"v0 episode table: {e0.shape[0]} rows x {e0.shape[1]} cols; v1 episode table: {e1.shape[0]} rows x {e1.shape[1]} cols")
print("label columns in v1 episode table:", [c for c in ("corruption_type", "params", "seed") if c in e1.columns])
print("labels.parquet:", lab.shape[0], "rows, columns", list(lab.columns))
print("labels join v1 on (episode_id, content_hash):", len(lab.merge(e1, on=["episode_id", "content_hash"])))
m = e1.merge(e0[["episode_id", "content_hash", "gripper_toggles"]], on="episode_id", suffixes=("_v1", "_v0"))
chg = m.content_hash_v1 != m.content_hash_v0
print(f"content_hash v0 -> v1: changed {int(chg.sum())}, identical {int((~chg).sum())}")
tasks = lab.episode_id.str.rsplit("/", n=1).str[0]
print("labelled episodes per task:", sorted(tasks.value_counts().unique().tolist()), "over", tasks.nunique(), "tasks")
idx = e1[e1.episode_id.isin(lab.episode_id)].episode_index
print(f"v1 episode_index of the 100 edited episodes: {idx.min()}..{idx.max()} (appended after the 900 kept ones)")
print(f"gripper_toggles on the 100 corrupted episodes: v1 {m[chg].gripper_toggles_v1.min()}-{m[chg].gripper_toggles_v1.max()}, "
      f"same episodes in v0 {m[chg].gripper_toggles_v0.min()}-{m[chg].gripper_toggles_v0.max()}")

cidx = set(idx)
A, E = [], []
files = sorted(glob.glob("data/lerobot/libero_spatial_10__v1/data/*/*.parquet"))
for f in files:
    t = pq.read_table(f, columns=["action", "episode_index"])
    A.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)))
    E.append(t.column("episode_index").to_numpy())
A, E = np.concatenate(A), np.concatenate(E)
c = np.isin(E, list(cidx))
out = (np.abs(A) > 1).any(1)
print(f"v1 data files read: {len(files)}; frames {len(A)} (corrupted {int(c.sum())}, clean {int((~c).sum())})")
print(f"frames with any action dim outside [-1, 1]: corrupted {out[c].mean():.4f}, clean {out[~c].mean():.4f}")
print("corrupted action min per dim:", [round(float(x), 3) for x in A[c].min(0)])
print("corrupted action max per dim:", [round(float(x), 3) for x in A[c].max(0)])
print("clean action min per dim:    ", [round(float(x), 3) for x in A[~c].min(0)])
print("clean action max per dim:    ", [round(float(x), 3) for x in A[~c].max(0)])
g = A[:, 6]
print(f"gripper action exactly +-1: corrupted {np.isin(g[c], [-1, 1]).mean():.4f}, clean {np.isin(g[~c], [-1, 1]).mean():.4f}")
