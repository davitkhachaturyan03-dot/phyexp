import numpy as np, pandas as pd
from sliceline.slicefinder import Slicefinder
d=pd.read_parquet("data/scratch/dash_exec/draft/desc_v3.parquet")
n=len(d)
def bins(x,nb=5):
    if x.nunique()<=nb: return pd.Series(pd.factorize(x,sort=True)[0],index=x.index)
    c=pd.qcut(x,nb,labels=False,duplicates="drop"); return c
cols=["length_steps","duration_s","smoothness","idle_fraction","idle_steps","gripper_toggles","length_vs_task","post_release_share_vs_task","duplicate_frame_pairs","instruction_mismatch"]
X=pd.DataFrame({c:bins(d[c]) for c in cols})
X["task_name"]=pd.factorize(d.task_name,sort=True)[0]; X["scene"]=pd.factorize(d.scene,sort=True)[0]
print({c:int(X[c].nunique()) for c in X})
top=X["length_vs_task"]==X["length_vs_task"].max()
ct=pd.crosstab(top, d.scene); print(ct.loc[True].to_dict())
rng=np.random.default_rng(0)
sc=ct.loc[True].idxmax(); sidx=pd.factorize(d.scene,sort=True)[1].get_loc(sc)
mask=(top&(X["scene"]==sidx)).to_numpy(); print("planted n",mask.sum(), sc)
for shift in (1.5,2.0):
  for alpha in (0.95,):
    z=rng.normal(size=n); z[mask]+=shift
    sf=Slicefinder(alpha=alpha,k=5,max_l=2,min_sup=max(10,int(.02*n)),verbose=False).fit(X.to_numpy(),np.maximum(z,0))
    print("shift",shift,"alpha",alpha,"found",len(sf.top_slices_))
    for s,st in zip(sf.top_slices_,sf.top_slices_statistics_):
        print("  ",[(X.columns[i],int(v)) for i,v in enumerate(s) if not pd.isna(v)],round(st["slice_score"],2),st["slice_size"])
