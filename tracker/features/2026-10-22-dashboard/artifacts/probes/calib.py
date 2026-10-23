import numpy as np
from scipy import stats
rng=np.random.default_rng(0)
n1,n2=13,869
res={"boot300":0,"boot1000":0,"welch":0,"perm":0}
T=1500
def boot(a,b,nb):
    da=a[rng.integers(0,len(a),(nb,len(a)))].mean(1); db=b[rng.integers(0,len(b),(nb,len(b)))].mean(1); d=da-db
    lo,hi=np.quantile(d,[.025,.975]); return lo,hi
for t in range(T):
    a=rng.normal(size=n1); b=rng.normal(size=n2)
    d=a.mean()-b.mean()
    for k,nb in (("boot300",300),("boot1000",1000)):
        lo,hi=boot(a,b,nb); res[k]+= (lo>0) or (hi<0)
    se=np.sqrt(a.var(ddof=1)/n1+b.var(ddof=1)/n2); dfw=se**4/((a.var(ddof=1)/n1)**2/(n1-1)+(b.var(ddof=1)/n2)**2/(n2-1))
    tc=stats.t.ppf(.975,dfw); res["welch"]+= (d-tc*se>0) or (d+tc*se<0)
    # permutation test two-sided
    allv=np.r_[a,b]; cnt=0; P=500
    for _ in range(P):
        rng.shuffle(allv); cnt+= abs(allv[:n1].mean()-allv[n1:].mean())>=abs(d)
    res["perm"]+= (cnt+1)/(P+1)<0.05
print({k:v/T for k,v in res.items()})
