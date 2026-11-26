"""Baseline = RAW LoRA probabilities + max (as reported in paper).
Choose ensemble weights + patient aggregation on VALIDATION ONLY, then evaluate once on test.
Combination = weighted mean of z-scored logits (z stats from that run's validation set)."""
import os, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.join(HERE, 'mac_kit'); D = os.path.join(HERE, 'mac_results_srilanka')
TR = os.path.join(D, 'train'); FP = os.path.join(D, 'ensemble', 'frozen_preds'); OUT = os.path.join(D, 'ensemble')
meta = pd.read_csv(os.path.join(KIT, 'meta.csv')); sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
meta['id'] = meta.md5; sl['id'] = sl.image; sl['set'] = 'srilanka'
info = pd.concat([meta[['id', 'group', 'set']], sl[['id', 'group', 'set']]]).drop_duplicates('id').set_index('id')
FROZEN = ['ConvNeXt-B', 'DINOv2-B14', 'CLIP-B16']
def lg(p): p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))
def load(exp, tag, split):
    d = pd.read_csv(os.path.join(TR, f'exp{exp}_lora_{tag}_{split}.csv')).rename(columns={'p': 'LoRA'})
    for bb in FROZEN: d = d.merge(pd.read_csv(os.path.join(FP, f'exp{exp}_{bb}_{tag}_{split}.csv')).rename(columns={'p': bb}), on='id')
    d['group'] = info.loc[d.id, 'group'].values; d['set'] = info.loc[d.id, 'set'].values; return d
def agg(d, s, how):
    t = d.assign(s=s).sort_values('s', ascending=False).groupby('group')
    y = t.y.max()
    if how == 'max': p = t.s.max()
    elif how == 'top2': p = t.s.apply(lambda v: v.iloc[:2].mean())
    else: p = t.s.mean()
    return y.values, p.loc[y.index].values
# candidate configs: LoRA weight, which frozen members (share the rest equally), patient aggregation
cands = []
for wl in [1.0, .75, .6, .5, .4, .25]:
    for mem in ([()] if wl == 1.0 else [m for r in [1, 2, 3] for m in itertools.combinations(FROZEN, r)]):
        for how in ['max', 'top2', 'mean']: cands.append((wl, mem, how))
def score(d, stats, cfg):
    wl, mem, how = cfg; z = lambda c: (lg(d[c].values) - stats[c][0]) / stats[c][1]
    s = wl * z('LoRA')
    for c in mem: s = s + (1 - wl) / len(mem) * z(c)
    return s
EXPS = {1: (['run0'], ['srilanka']), 2: (['run0'], ['india', 'karnataka']), 3: ([f'fold{k}' for k in range(5)], ['cv', 'karnataka'])}
rng = np.random.default_rng(0); rows = []; sel_rows = []
def boot_pat(y, a, b, n=500):
    """patient-level bootstrap of AUC for score a, and paired delta a-b"""
    A, Dl = [], []; idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx));
        if len(set(y[i])) < 2: continue
        A.append(roc_auc_score(y[i], a[i])); Dl.append(A[-1] - roc_auc_score(y[i], b[i]))
    return np.percentile(A, [2.5, 97.5]), np.percentile(Dl, [2.5, 97.5])
for exp, (tags, tests) in EXPS.items():
    vals = {t: load(exp, t, 'val') for t in tags}
    stats = {t: {c: (lg(v[c].values).mean(), lg(v[c].values).std()) for c in ['LoRA'] + FROZEN} for t, v in vals.items()}
    # validation selection: mean over runs of val patient AUC
    res = []
    for cfg in cands:
        aucs = [roc_auc_score(*agg(vals[t], score(vals[t], stats[t], cfg), cfg[2])) for t in tags]
        res.append((np.mean(aucs), cfg))
    res.sort(key=lambda r: -r[0]); best = res[0][1]; base = (1.0, (), 'max')
    base_val = [r for r in res if r[1] == base][0][0]
    sel_rows.append({'exp': exp, 'chosen LoRA weight': best[0], 'frozen members': '+'.join(best[1]) or '-', 'patient score': best[2],
                     'val patient AUC chosen': round(res[0][0], 3), 'val patient AUC LoRA-max': round(base_val, 3)})
    for tn in tests:
        te = pd.concat([load(exp, t, f'test_{tn}').assign(s_best=lambda d, t=t: score(d, stats[t], best), s_base=lambda d: d.LoRA, tag=t) for t in tags])
        if tn == 'karnataka':   # same photos predicted by each fold model -> average fold scores; photo level only
            te = te.groupby('id').agg(y=('y', 'first'), s_best=('s_best', 'mean'), s_base=('s_base', 'mean'), set=('set', 'first'), group=('group', 'first'), tag=('tag','first')).reset_index()
            subsets = [('Karnataka', te)]
        elif tn == 'cv': subsets = [('pooled', te), ('India part', te[te.set != 'srilanka']), ('Sri Lanka part', te[te.set == 'srilanka'])]
        else: subsets = [(tn, te)]
        for nm, d in subsets:
            r = {'exp': exp, 'test': nm, 'photo AUC LoRA': round(roc_auc_score(d.y, d.s_base), 3), 'photo AUC ens': round(roc_auc_score(d.y, d.s_best), 3)}
            if nm != 'Karnataka':
                yb, pb = agg(d, d.s_base.values, 'max'); ye, pe = agg(d, d.s_best.values, best[2])
                assert (yb == ye).all()
                ci, dci = boot_pat(ye, pe, pb)
                r.update({'patient AUC LoRA': round(roc_auc_score(yb, pb), 3), 'patient AUC ens': round(roc_auc_score(ye, pe), 3),
                          'per-fold mean patient AUC LoRA': round(np.mean([roc_auc_score(*agg(g, g.s_base.values, 'max')) for _, g in d.groupby('tag')]), 3),
                          'per-fold mean patient AUC ens': round(np.mean([roc_auc_score(*agg(g, g.s_best.values, best[2])) for _, g in d.groupby('tag')]), 3),
                          'ens 95% CI': f'{ci[0]:.3f}-{ci[1]:.3f}', 'delta 95% CI': f'{dci[0]:+.3f} to {dci[1]:+.3f}', 'n patients': len(ye)})
            rows.append(r)
    print('exp', exp, 'top-5 val configs:', [(round(a, 3), c) for a, c in res[:5]], flush=True)
sel = pd.DataFrame(sel_rows); out = pd.DataFrame(rows)
sel.to_csv(os.path.join(OUT, 'ensemble_selection_val.csv'), index=False); out.to_csv(os.path.join(OUT, 'ensemble_test_results.csv'), index=False)
pd.set_option('display.width', 250); print(sel.to_string(index=False)); print(out.to_string(index=False))
