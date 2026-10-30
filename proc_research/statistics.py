"""Paired original-scene inference, never mean-of-image IoUs."""
import numpy as np
from .evaluate import metrics


def paired_bootstrap(candidate,baseline,iterations=2000,seed=20261030):
    a=np.asarray(candidate,dtype=np.int64);b=np.asarray(baseline,dtype=np.int64)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1]!=a.shape[2] or len(a)==0:
        raise ValueError('paired scene confusion matrices required')
    def delta(x,y):
        mx,my=metrics(x)['miou'],metrics(y)['miou']
        if mx is None or my is None:raise ValueError('no evaluated pixels')
        return 100*(mx-my)
    observed=delta(a.sum(0),b.sum(0));rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(iterations):
        idx=rng.integers(0,len(a),len(a));draws.append(delta(a[idx].sum(0),b[idx].sum(0)))
    ci=np.percentile(draws,[2.5,97.5]).tolist()
    return {'delta_pp':observed,'ci95_pp':ci,'scenes':len(a),'iterations':iterations,'seed':seed,
            'status':'INCONCLUSIVE' if len(a)<2 else ('SUPERIOR' if ci[0]>0 else 'NOT_ESTABLISHED')}
