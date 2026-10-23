import numpy as np, pandas as pd
from sliceline.slicefinder import Slicefinder
rng=np.random.default_rng(1)
n=960
df=pd.DataFrame({"len":rng.lognormal(0,.3,n),"scene":rng.integers(0,10,n),"tog":rng.integers(0,6,n)})
code=pd.qcut(df["len"],5,labels=False).to_numpy()
for shift in (1.0,1.5,2.0):
  for planted_bins,scenes in (((4,),(3,)),((3,4),(3,))):
    z=rng.normal(size=n); mask=np.isin(code,planted_bins)&np.isin(df["scene"],scenes); z[mask]+=shift
    X=np.c_[code,df["scene"],df["tog"]]
    for alpha in (0.95,0.99):
      for ms in (10,19):
        sf=Slicefinder(alpha=alpha,k=5,max_l=2,min_sup=ms,verbose=False).fit(X,np.maximum(z,0))
        top=[tuple(None if (isinstance(v,float) and np.isnan(v)) or v is None else int(v) for v in s) for s in sf.top_slices_]
        rank=[i for i,s in enumerate(top) if s[0] in planted_bins and s[1] in scenes and s[0] is not None and s[1] is not None]
        print(f"shift {shift} planted n={mask.sum()} alpha {alpha} min_sup {ms}: n_found {len(top)} planted-slice rank {rank[:1]}")
