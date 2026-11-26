"""Analysis of the Sri Lanka cross-country runs (no GPU needed).
For each experiment: photo- and patient-level AUC/AUPRC (95% CI, bootstrap over patients), calibration (ECE),
and decisions at a threshold set on the validation set for >=90% sensitivity, before and after recalibration:
  raw            : model probabilities as trained (25% positive sampling)
  platt          : Platt scaling fitted on the source validation set (no target information)
  prior-shift    : platt + label-free prevalence correction to the target (EM, Saerens et al. 2002)
  local-50       : platt refitted on 50 labelled target patients (few-label recalibration), evaluated on the rest
  oracle         : threshold chosen with all target labels (upper bound)
Output: mac_results_srilanka/analysis/*.csv"""
import os, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); KIT = os.path.join(HERE, 'mac_kit')
R = os.path.join(HERE, 'mac_results_srilanka', 'train'); A = os.path.join(HERE, 'mac_results_srilanka', 'analysis')
os.makedirs(A, exist_ok=True); rng0 = 0
meta = pd.read_csv(os.path.join(KIT, 'meta.csv')); sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
GROUP = dict(zip(meta.md5, meta.group)); GROUP.update(dict(zip(sl.image, sl.group)))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))

def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())

def thr_sens(y, p, target=0.90):
    pos = np.sort(p[y == 1]); k = int(np.floor((1 - target) * len(pos)))
    return pos[k] if len(pos) else 0.5

def sens_spec(y, p, t): pr = p >= t; return pr[y == 1].mean(), (~pr[y == 0]).mean()

def boot(y, s, g, f, n=500, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(g); gi = {u: np.where(g == u)[0] for u in ug}; v = []
    for _ in range(n):
        ii = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])
        if 0 < y[ii].sum() < len(ii): v.append(f(y[ii], s[ii]))
    return np.percentile(v, [2.5, 97.5])

def fmt(x, ci): return f'{x:.3f} ({ci[0]:.3f}-{ci[1]:.3f})'

def patient(df):
    d = df.assign(g=df.id.map(GROUP)).groupby('g').agg(y=('y', 'max'), p=('p', 'max')).reset_index()
    return d.y.values, d.p.values, d.g.values

def prior_shift(p_src, pi_src, iters=100):
    """EM estimate of target prevalence + adjusted posteriors (labels never used)"""
    pi = pi_src
    for _ in range(iters):
        w1, w0 = pi / pi_src, (1 - pi) / (1 - pi_src)
        q = w1 * p_src / (w1 * p_src + w0 * (1 - p_src)); pi_new = q.mean()
        if abs(pi_new - pi) < 1e-6: break
        pi = pi_new
    return q, pi

def evaluate(name, val, test, level):
    """val/test: DataFrames id,y,p. level: 'photo' or 'patient' (patient = max photo score)"""
    if level == 'patient':
        yv, pv, gv = patient(val); yt, pt, gt = patient(test)
    else:
        yv, pv = val.y.values, val.p.values; yt, pt = test.y.values, test.p.values
        gt = test.id.map(GROUP).fillna(test.id).values
    rows = []
    auc = roc_auc_score(yt, pt); ap = average_precision_score(yt, pt)
    base = {'experiment': name, 'level': level, 'n': len(yt), 'positives': int(yt.sum()), 'prevalence': round(yt.mean(), 3),
            'AUC': fmt(auc, boot(yt, pt, gt, roc_auc_score)), 'AUPRC': fmt(ap, boot(yt, pt, gt, average_precision_score))}
    pl = LogisticRegression(C=1e6).fit(logit(pv)[:, None], yv)
    cal = lambda p: pl.predict_proba(logit(p)[:, None])[:, 1]
    pv_c, pt_c = cal(pv), cal(pt)
    pt_ps, pi_hat = prior_shift(pt_c, yv.mean())
    # few-label: 50 target units (patients or photos) used to refit Platt, evaluated on the remaining units
    rng = np.random.default_rng(0); ug = np.unique(gt); pick = set(rng.choice(ug, min(50, len(ug) // 3), replace=False))
    m50 = np.isin(gt, list(pick))
    l50 = LogisticRegression(C=1e6).fit(logit(pt[m50])[:, None], yt[m50]) if 0 < yt[m50].sum() < m50.sum() else pl
    pt_l = l50.predict_proba(logit(pt)[:, None])[:, 1]
    variants = {'raw': (pt, thr_sens(yv, pv)), 'platt': (pt_c, thr_sens(yv, pv_c)),
                'prior-shift': (pt_ps, None), 'local-50': (pt_l, None)}
    for vname, (pp, t) in variants.items():
        mask = ~m50 if vname == 'local-50' else np.ones(len(yt), bool)
        if vname == 'prior-shift':   # keep source sensitivity target: threshold on adjusted scale from val
            w1, w0 = pi_hat / yv.mean(), (1 - pi_hat) / (1 - yv.mean())
            pv_adj = w1 * pv_c / (w1 * pv_c + w0 * (1 - pv_c)); t = thr_sens(yv, pv_adj)
        if vname == 'local-50':
            t = thr_sens(yt[m50], pp[m50]) if yt[m50].sum() >= 5 else 0.5
        se, sp = sens_spec(yt[mask], pp[mask], t)
        rows.append({**base, 'calibration': vname, 'ECE': round(ece(yt[mask], pp[mask]), 3),
                     'mean predicted risk': round(pp[mask].mean(), 3), 'sens': round(se, 3), 'spec': round(sp, 3),
                     'est. target prevalence': round(pi_hat, 3) if vname == 'prior-shift' else ''})
    t_or = thr_sens(yt, pt); se, sp = sens_spec(yt, pt, t_or)
    rows.append({**base, 'calibration': 'oracle threshold (uses target labels)', 'ECE': '', 'mean predicted risk': '',
                 'sens': round(se, 3), 'spec': round(sp, 3), 'est. target prevalence': ''})
    return rows

def load(exp, model, tag, part): return pd.read_csv(os.path.join(R, f'exp{exp}_{model}_{tag}_{part}.csv'))

rows = []
for model in ['lora', 'convnext']:
    if os.path.exists(os.path.join(R, f'exp1_{model}_run0.json')):
        v, t = load(1, model, 'run0', 'val'), load(1, model, 'run0', 'test_srilanka')
        for lv in ['photo', 'patient']: rows += evaluate(f'exp1 India->Sri Lanka ({model})', v, t, lv)
    if os.path.exists(os.path.join(R, f'exp2_{model}_run0.json')):
        v = load(2, model, 'run0', 'val')
        for lv in ['photo', 'patient']: rows += evaluate(f'exp2 Sri Lanka->India ({model})', v, load(2, model, 'run0', 'test_india'), lv)
        rows += evaluate(f'exp2 Sri Lanka->Karnataka ({model})', v, load(2, model, 'run0', 'test_karnataka'), 'photo')
    if all(os.path.exists(os.path.join(R, f'exp3_{model}_fold{k}.json')) for k in range(5)):
        cv = pd.concat([load(3, model, f'fold{k}', 'test_cv') for k in range(5)])
        va = pd.concat([load(3, model, f'fold{k}', 'val') for k in range(5)])
        ka = load(3, model, 'fold0', 'test_karnataka')[['id', 'y']].assign(
            p=np.mean([load(3, model, f'fold{k}', 'test_karnataka').p.values for k in range(5)], 0))
        cv['country'] = np.where(cv.id.isin(sl.image), 'SL', 'IN')
        for lv in ['photo', 'patient']:
            rows += evaluate(f'exp3 both countries, CV all ({model})', va, cv, lv)
            for c in ['IN', 'SL']: rows += evaluate(f'exp3 both countries, CV {c} only ({model})', va, cv[cv.country == c], lv)
        rows += evaluate(f'exp3 both countries -> Karnataka ({model})', va, ka, 'photo')
out = pd.DataFrame(rows); out.to_csv(os.path.join(A, 'cross_country_results.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(out.to_string(index=False) if len(out) else 'no finished runs yet')
