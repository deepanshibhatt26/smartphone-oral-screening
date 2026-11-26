"""Frozen-feature (linear probe) versions of the three cross-country experiments. CPU only."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.join(HERE, 'mac_kit'); D = os.path.join(HERE, 'mac_results_srilanka')
meta = pd.read_csv(os.path.join(KIT, 'meta.csv')); sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
meta['id'] = meta.md5; sl['id'] = sl.image; sl['set'] = 'srilanka'
cols = ['id', 'y_bin', 'group', 'fold', 'set']; df = pd.concat([meta[cols], sl[cols]], ignore_index=True)
df['country'] = np.where(df.set == 'srilanka', 'SL', 'IN')
SEED = 42; rows = []; preds = {}
def pat(d, p):
    t = d.assign(p=p).groupby('group').agg(y=('y_bin', 'max'), p=('p', 'max')); return t.y.values, t.p.values
def fit(Xtr, ytr, Xva, yva, wtr):
    best = None
    for C in [0.01, 0.1, 1]:
        sc = StandardScaler().fit(Xtr); m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr, sample_weight=wtr)
        ap = average_precision_score(yva, m.predict_proba(sc.transform(Xva))[:, 1])
        if best is None or ap > best[0]: best = (ap, sc, m)
    return best[1], best[2]
def w_balanced(d):
    w = np.zeros(len(d)); cs = d.country.unique()
    for lab, share in [(1, .5), (0, .5)]:
        for c in cs:
            i = ((d.y_bin == lab) & (d.country == c)).values
            if i.sum(): w[i] = share / len(cs) / i.sum()
    return w * len(d)
import sys
for bb in sys.argv[1:]:
    F = {}
    for s in ['internal', 'srilanka']:
        z = np.load(os.path.join(D, f'sl_feats_{bb}_{s}.npz'), allow_pickle=True); F.update(dict(zip(z['ids'], z['X'])))
    X = np.stack([F[i] for i in df.id]); IN = (df.set == 'internal').values; KA = (df.set == 'karnataka').values; SLm = (df.set == 'srilanka').values
    rng = np.random.default_rng(SEED); kav = KA & (rng.random(len(df)) < 0.2)
    def run(tr, va, te, name, extra=None):
        sc, m = fit(X[tr], df.y_bin.values[tr], X[va], df.y_bin.values[va], w_balanced(df[tr]))
        p = m.predict_proba(sc.transform(X[te]))[:, 1]; d = df[te]
        r = {'backbone': bb, 'experiment': name, 'photo AUC': round(roc_auc_score(d.y_bin, p), 3)}
        if name.endswith('Karnataka'): r['patient AUC'] = ''
        else: yy, pp = pat(d, p); r['patient AUC'] = round(roc_auc_score(yy, pp), 3)
        rows.append(r); return p
    run((IN & (df.fold != 4).values) | (KA & ~kav), (IN & (df.fold == 4).values) | kav, SLm, 'exp1 India(+Karnataka) -> Sri Lanka')
    run(SLm & (df.fold != 4).values, SLm & (df.fold == 4).values, IN, 'exp2 Sri Lanka -> India')
    run(SLm & (df.fold != 4).values, SLm & (df.fold == 4).values, KA, 'exp2 Sri Lanka -> Karnataka')
    oof = np.zeros(len(df)); kap = []
    both = IN | SLm
    for k in range(5):
        vk = (k + 1) % 5; tr = both & ~df.fold.isin([k, vk]).values; va = both & (df.fold == vk).values; te = both & (df.fold == k).values
        sc, m = fit(X[tr], df.y_bin.values[tr], X[va], df.y_bin.values[va], w_balanced(df[tr]))
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]; kap.append(m.predict_proba(sc.transform(X[KA]))[:, 1])
    for nm, msk in [('exp3 both, CV India', both & IN), ('exp3 both, CV Sri Lanka', both & SLm)]:
        d = df[msk]; yy, pp = pat(d, oof[msk])
        rows.append({'backbone': bb, 'experiment': nm, 'photo AUC': round(roc_auc_score(d.y_bin, oof[msk]), 3), 'patient AUC': round(roc_auc_score(yy, pp), 3)})
    rows.append({'backbone': bb, 'experiment': 'exp3 both -> Karnataka', 'photo AUC': round(roc_auc_score(df.y_bin[KA], np.mean(kap, 0)), 3), 'patient AUC': ''})
    print(bb, 'done', flush=True)
os.makedirs(os.path.join(D, 'analysis'), exist_ok=True)
for bb in sys.argv[1:]: pd.DataFrame([r for r in rows if r['backbone']==bb]).to_csv(os.path.join(D, 'analysis', f'frozen_{bb}.csv'), index=False)
import glob
out = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'analysis', 'frozen_*-*.csv')))])
out.to_csv(os.path.join(D, 'analysis', 'frozen_cross_country.csv'), index=False)
print(out.pivot_table(index='experiment', columns='backbone', values='photo AUC', aggfunc='first').to_string())
print(out.pivot_table(index='experiment', columns='backbone', values='patient AUC', aggfunc='first').to_string())
