"""Shared helpers for the final-model analyses (A-D): final ensemble patient/photo scores per experiment, classes, habits.
Final model per experiment = configuration chosen on validation in ens_step2_combine.py (ensemble_selection_val.csv)."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
_ns = {'__file__': os.path.join(HERE, 'ens_step2_combine.py')}; exec(open(os.path.join(HERE, 'ens_step2_combine.py')).read().split('EXPS =')[0], _ns)
load, score, lg, FROZEN, KIT = _ns['load'], _ns['score'], _ns['lg'], _ns['FROZEN'], _ns['KIT']
ANA = os.path.join(HERE, 'mac_results_srilanka', 'analysis_final'); os.makedirs(ANA, exist_ok=True)
CFG = {1: (0.75, ('ConvNeXt-B',), 'top2'), 2: (0.6, ('ConvNeXt-B',), 'max'), 3: (0.6, tuple(FROZEN), 'top2')}
TAGS = {1: ['run0'], 2: ['run0'], 3: [f'fold{k}' for k in range(5)]}
TESTS = {1: ['srilanka'], 2: ['india', 'karnataka'], 3: ['cv', 'karnataka']}
RANK = {'Normal': 0, 'Healthy': 0, 'Variation': 1, 'Benign': 1, 'OPMD': 2, 'Malignant': 3, 'Cancer': 3, 'OCA': 3}
CLS = {0: 'healthy', 1: 'benign', 2: 'OPMD', 3: 'cancer'}
m = pd.read_csv(os.path.join(KIT, 'meta.csv')); s = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
m['id'] = m.md5; s['id'] = s.image; s['set'] = 'srilanka'
PH = pd.concat([m[['id', 'cls', 'set', 'dataset']], s[['id', 'cls', 'set', 'dataset']]]).drop_duplicates('id').set_index('id')
PH['r'] = PH.cls.map(RANK); PH['country'] = np.where(PH.set == 'srilanka', 'Sri Lanka', np.where(PH.set == 'karnataka', 'Karnataka', 'India'))
def stats_for(exp, tag):
    v = load(exp, tag, 'val'); return {c: (lg(v[c].values).mean(), lg(v[c].values).std()) for c in ['LoRA'] + FROZEN}
def photos(exp, split):
    """photo-level final-model scores (z-logit scale) for every run of the experiment; adds class, country, run tag"""
    out = []
    for t in TAGS[exp]:
        d = load(exp, t, split); d['s'] = score(d, stats_for(exp, t), CFG[exp]); d['tag'] = t; out.append(d)
    d = pd.concat(out); d['r'] = PH.loc[d.id, 'r'].values; d['country'] = PH.loc[d.id, 'country'].values; d['dataset'] = PH.loc[d.id, 'dataset'].values
    return d
def patients(d, exp):
    """patient-level score with the chosen aggregation; worst finding; one row per (run, group)"""
    how = CFG[exp][2]; g = d.sort_values('s', ascending=False).groupby(['tag', 'group'])
    p = g.s.max() if how == 'max' else g.s.apply(lambda v: v.iloc[:2].mean())
    P = pd.DataFrame({'s': p, 'y': g.y.max(), 'r': g.r.max(), 'country': g.country.first(), 'dataset': g.dataset.first()}).reset_index()
    return P
def boot_auc(y, p, n=500, seed=0):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.choice(len(y), len(y))
        if len(set(y[i])) == 2: a.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), np.percentile(a, 2.5), np.percentile(a, 97.5)
def fmt(t): return f'{t[0]:.3f} ({t[1]:.3f}-{t[2]:.3f})'
