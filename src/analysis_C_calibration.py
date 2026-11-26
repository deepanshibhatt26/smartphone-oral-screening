"""C: calibration and referral thresholds for the FINAL ensemble (patient level).
Platt scaling fitted on each run's validation patients. Threshold = highest score giving >=90% sensitivity on validation.
exp3: shared threshold (all validation patients) vs per-country thresholds (that country's validation patients).
exp1/exp2 (new country): source-validation calibration/threshold; 'local-50' = refit Platt + threshold on 50 random
target patients (200 repeats, evaluated on the remaining target patients); label-free prior-shift EM (Saerens 2002);
oracle threshold on target labels (reference only)."""
import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from common_final import photos, patients, ANA
def ece(y, p, bins=10):
    o = np.argsort(p); e = 0
    for b in np.array_split(o, bins):
        if len(b): e += len(b) / len(p) * abs(y[b].mean() - p[b].mean())
    return e
def platt(s, y): m = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y); return lambda x: m.predict_proba(np.asarray(x).reshape(-1, 1))[:, 1]
def thr90(s, y):
    pos = np.sort(s[y == 1]); return pos[int(np.floor(0.10 * len(pos)))] if len(pos) else np.inf   # >=90% of positives at or above
def ops(y, s, t):
    ref = s >= t; return {'sensitivity': ref[y == 1].mean(), 'specificity': (~ref[y == 0]).mean(), 'referral rate': ref.mean()}
def em_prior(p, pi_s, it=200):
    pi = pi_s
    for _ in range(it):
        q = (pi / pi_s * p) / (pi / pi_s * p + (1 - pi) / (1 - pi_s) * (1 - p)); pi = q.mean()
    return pi, q
rows = []; rng = np.random.default_rng(0)
# ---------------- exp3: both countries, per fold
V = patients(photos(3, 'val'), 3); T = patients(photos(3, 'test_cv'), 3)
res = {k: [] for k in ['raw', 'platt']}
for t in sorted(T.tag.unique()):
    v, te = V[V.tag == t], T[T.tag == t].copy()
    f = platt(v.s.values, v.y.values); te['p'] = f(te.s.values)
    te['t_shared'] = thr90(v.s.values, v.y.values)
    te['t_country'] = te.country.map({c: thr90(v.s.values[v.country == c], v.y.values[v.country == c]) for c in ['India', 'Sri Lanka']})
    fc = {c: platt(v.s.values[v.country == c], v.y.values[v.country == c]) for c in ['India', 'Sri Lanka']}
    te['p_country'] = [fc[c](x)[0] for c, x in zip(te.country, te.s)]
    res['platt'].append(te)
T3 = pd.concat(res['platt'])
for c in ['India', 'Sri Lanka', 'all']:
    d = T3 if c == 'all' else T3[T3.country == c]; y = d.y.values
    raw_p = 1 / (1 + np.exp(-d.s.values))   # uncalibrated: logistic of the ensemble score
    r = {'experiment': 'exp3 both (5-fold)', 'test patients': c, 'n': len(d), 'prevalence': round(y.mean(), 3),
         'ECE raw': round(ece(y, raw_p), 3), 'ECE Platt (shared)': round(ece(y, d.p.values), 3), 'ECE Platt (per country)': round(ece(y, d.p_country.values), 3) if c != 'all' else ''}
    for nm, col in [('shared thr', 't_shared'), ('per-country thr', 't_country')]:
        o = ops(y, d.s.values, d[col].values); r.update({f'{nm} {k}': round(v, 3) for k, v in o.items()})
    rows.append(r)
# ---------------- exp1 / exp2: new country
for exp, tn, name in [(1, 'test_srilanka', 'exp1 India->Sri Lanka'), (2, 'test_india', 'exp2 Sri Lanka->India')]:
    v = patients(photos(exp, 'val'), exp); te = patients(photos(exp, tn), exp)
    f = platt(v.s.values, v.y.values); p = f(te.s.values); y = te.y.values; s = te.s.values
    pi_hat, q = em_prior(p, v.y.mean())
    r = {'experiment': name, 'test patients': 'target', 'n': len(te), 'prevalence': round(y.mean(), 3), 'source prevalence': round(v.y.mean(), 3),
         'ECE raw': round(ece(y, 1 / (1 + np.exp(-s))), 3), 'ECE Platt (source)': round(ece(y, p), 3),
         'ECE prior-shift EM': round(ece(y, q), 3), 'EM estimated prevalence': round(pi_hat, 3)}
    r.update({f'source thr {k}': round(x, 3) for k, x in ops(y, s, thr90(v.s.values, v.y.values)).items()})
    r.update({f'oracle thr {k}': round(x, 3) for k, x in ops(y, s, thr90(s, y)).items()})
    L = []
    for _ in range(200):
        i = rng.permutation(len(te)); a, b = i[:50], i[50:]
        if len(set(y[a])) < 2 or y[a].sum() < 2: continue
        fl = platt(s[a], y[a]); o = ops(y[b], s[b], thr90(s[a], y[a])); o['ECE'] = ece(y[b], fl(s[b])); L.append(o)
    L = pd.DataFrame(L)
    for k in ['ECE', 'sensitivity', 'specificity', 'referral rate']:
        r[f'local-50 {k}'] = f'{L[k].mean():.3f} ({L[k].quantile(.025):.3f}-{L[k].quantile(.975):.3f})'
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(ANA, 'C_calibration_thresholds.csv'), index=False)
pd.set_option('display.width', 250); print(R.T.to_string())
