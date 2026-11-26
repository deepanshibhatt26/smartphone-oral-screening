"""Frozen-backbone predictions on EXACTLY the LoRA splits (val/test ids read from LoRA csvs). CPU only.
Usage: python ens_step1_frozen_preds.py <backbone> <exp>   (exp in 1,2,3)"""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.join(HERE, 'mac_kit'); D = os.path.join(HERE, 'mac_results_srilanka')
TR = os.path.join(D, 'train'); OUT = os.path.join(D, 'ensemble', 'frozen_preds'); os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(os.path.join(KIT, 'meta.csv')); sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
meta['id'] = meta.md5; sl['id'] = sl.image; sl['set'] = 'srilanka'
cols = ['id', 'y_bin', 'group', 'set']; df = pd.concat([meta[cols], sl[cols]], ignore_index=True).drop_duplicates('id')
df['country'] = np.where(df.set == 'srilanka', 'SL', 'IN')
bb, exp = sys.argv[1], int(sys.argv[2])
F = {}
for s in ['internal', 'srilanka']:
    z = np.load(os.path.join(D, f'sl_feats_{bb}_{s}.npz'), allow_pickle=True); F.update(dict(zip(z['ids'], z['X'])))
def w_balanced(d):
    w = np.zeros(len(d)); cs = d.country.unique()
    for lab in [1, 0]:
        for c in cs:
            i = ((d.y_bin == lab) & (d.country == c)).values
            if i.sum(): w[i] = .5 / len(cs) / i.sum()
    return w * len(d)
def X(ids): return np.stack([F[i] for i in ids])
pool = {1: df.set.isin(['internal', 'karnataka']), 2: df.set == 'srilanka', 3: df.set.isin(['internal', 'srilanka'])}[exp]
runs = ['run0'] if exp < 3 else [f'fold{k}' for k in range(5)]
for tag in runs:
    base = os.path.join(TR, f'exp{exp}_lora_{tag}')
    va = pd.read_csv(base + '_val.csv'); tests = {f.split('_test_')[1][:-4]: pd.read_csv(os.path.join(TR, f)) for f in os.listdir(TR) if f.startswith(f'exp{exp}_lora_{tag}_test_')}
    excl = set(va.id) | set().union(*[set(t.id) for t in tests.values()])
    trd = df[pool & ~df.id.isin(excl)]
    Xtr = X(trd.id); Xva = X(va.id); best = None
    for C in [0.01, 0.1, 1]:
        sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), trd.y_bin.values, sample_weight=w_balanced(trd))
        ap = average_precision_score(va.y, m.predict_proba(sc.transform(Xva))[:, 1])
        if best is None or ap > best[0]: best = (ap, sc, m, C)
    _, sc, m, C = best
    pd.DataFrame({'id': va.id, 'p': m.predict_proba(sc.transform(Xva))[:, 1]}).to_csv(os.path.join(OUT, f'exp{exp}_{bb}_{tag}_val.csv'), index=False)
    for nm, t in tests.items():
        pd.DataFrame({'id': t.id, 'p': m.predict_proba(sc.transform(X(t.id)))[:, 1]}).to_csv(os.path.join(OUT, f'exp{exp}_{bb}_{tag}_test_{nm}.csv'), index=False)
    print(bb, exp, tag, 'train', len(trd), 'val', len(va), 'C', C, 'valAP', round(best[0], 3), flush=True)
