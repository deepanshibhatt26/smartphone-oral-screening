"""Step 9b analysis (cloud side): calibrate CLIP-B16-LoRA fold models with their validation-fold predictions.
Inputs: KIT/meta.csv, KIT/step6/CLIP-B16-LoRA_fold{k}.csv (+ _karnataka.csv, .json), RES/step9b_val_fold{k}.csv"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

KIT, RES, OUTD = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTD, exist_ok=True)
N_FOLDS, EPS, N_BOOT = 5, 1e-6, 1000

def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
class Platt:
    def fit(s, p, y): s.m = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p)[:, None], y); return s
    def __call__(s, p): return s.m.predict_proba(logit(p)[:, None])[:, 1]
def ece(y, p, n=10):
    b = np.array_split(np.argsort(p), n); return sum(len(i) / len(p) * abs(y[i].mean() - p[i].mean()) for i in b)
def slope_int(y, p):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p)[:, None], y); return float(m.coef_[0, 0]), float(m.intercept_[0])
def thr_for_sens(y, p, t=0.90):
    fpr, tpr, thr = roc_curve(y, p); return thr[np.argmax(tpr >= t)]
def sens(y, pr): return pr[y == 1].mean()
def spec(y, pr): return 1 - pr[y == 0].mean()
def boot(y, v, g, fn):
    rng = np.random.default_rng(0); idx = pd.Series(np.arange(len(y))).groupby(g).apply(np.array).values; out = []
    for _ in range(N_BOOT):
        ii = np.concatenate(idx[rng.integers(0, len(idx), len(idx))])
        if 0 < y[ii].sum() < len(ii): out.append(fn(y[ii], v[ii]))
    return np.percentile(out, [2.5, 97.5])
def fmt(v, ci): return f'{v:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]'
def pmax(p, g, y):
    s = pd.Series(p).groupby(g).max(); return s.values, pd.Series(y).groupby(g).max().reindex(s.index).values, s.index.values

meta = pd.read_csv(f'{KIT}/meta.csv'); inn = meta[meta.set == 'internal'].set_index('md5'); kar = meta[meta.set == 'karnataka']
yk = kar.y_bin.values
rows, test_ph, test_pt, kraw, kcal, ruleA = [], [], [], [], [], []
for k in range(N_FOLDS):
    va = pd.read_csv(f'{RES}/step9b_val_fold{k}.csv').join(inn[['y_bin', 'group']], on='md5')
    te = pd.read_csv(f'{KIT}/step6/CLIP-B16-LoRA_fold{k}.csv').join(inn[['y_bin', 'group']], on='md5')
    assert va.y_bin.notna().all() and te.y_bin.notna().all(), 'photo missing from meta'
    assert set(va.group) & set(te.group) == set(), 'validation and test folds share a person'
    cph = Platt().fit(va.p.values, va.y_bin.values)
    sv, yv, _ = pmax(va.p.values, va.group.values, va.y_bin.values); cpt = Platt().fit(sv, yv)
    test_ph.append(te.assign(fold=k, cal=cph(te.p.values)))
    st, yt, gt = pmax(te.p.values, te.group.values, te.y_bin.values)
    test_pt.append(pd.DataFrame({'group': gt, 'y': yt, 'raw': st, 'cal': cpt(st), 'fold': k}))
    kp = pd.read_csv(f'{KIT}/step6/CLIP-B16-LoRA_fold{k}_karnataka.csv').set_index('md5').p.reindex(kar.md5).values
    kraw.append(kp); kcal.append(cph(kp))
    ruleA.append(json.load(open(f'{KIT}/step6/CLIP-B16-LoRA_fold{k}.json')))
PH = pd.concat(test_ph, ignore_index=True); PT = pd.concat(test_pt, ignore_index=True)

# ---- calibration quality, out-of-fold --------------------------------------------------------------
for level, y, g, a, b in [('photo', PH.y_bin.values.astype(int), PH.group.values, PH.p.values, PH.cal.values),
                          ('patient', PT.y.values.astype(int), PT.group.values, PT.raw.values, PT.cal.values)]:
    for kind, p in [('raw', a), ('calibrated', b)]:
        sl, ic = slope_int(y, p)
        rows.append({'level': level, 'scores': kind, 'AUC': round(roc_auc_score(y, p), 3), 'mean predicted risk': round(p.mean(), 4),
                     'observed rate': round(y.mean(), 4), 'Brier': round(brier_score_loss(y, p), 4), 'ECE': round(ece(y, p), 4),
                     'cal. slope': round(sl, 2), 'cal. intercept': round(ic, 2)})
cal = pd.DataFrame(rows)

# ---- thresholds: A (validation fold, pre-specified) vs B (pooled other folds) raw vs calibrated -----
trows = []
y, g = PH.y_bin.values.astype(int), PH.group.values; yp, gp = PT.y.values.astype(int), PT.group.values
prA = np.concatenate([(PH.p.values[PH.fold == k] >= ruleA[k]['photo']).astype(int) for k in range(N_FOLDS)])
prA_pt = np.concatenate([(PT.raw.values[PT.fold == k] >= ruleA[k]['patient']).astype(int) for k in range(N_FOLDS)])
order_ph = np.concatenate([np.where(PH.fold == k)[0] for k in range(N_FOLDS)]); order_pt = np.concatenate([np.where(PT.fold == k)[0] for k in range(N_FOLDS)])
trows.append({'threshold rule': 'A: validation fold (pre-specified), raw',
              'photo sens': fmt(sens(y[order_ph], prA), boot(y[order_ph], prA, g[order_ph], sens)),
              'photo spec': fmt(spec(y[order_ph], prA), boot(y[order_ph], prA, g[order_ph], spec)),
              'patient sens': fmt(sens(yp[order_pt], prA_pt), boot(yp[order_pt], prA_pt, gp[order_pt], sens)),
              'patient spec': fmt(spec(yp[order_pt], prA_pt), boot(yp[order_pt], prA_pt, gp[order_pt], spec))})
for kind in ['raw', 'cal']:
    ph = PH.p.values if kind == 'raw' else PH.cal.values; pt = PT[kind].values
    a, b = np.zeros(len(ph), int), np.zeros(len(pt), int)
    for k in range(N_FOLDS):
        t, o = PH.fold.values == k, PH.fold.values != k; a[t] = ph[t] >= thr_for_sens(y[o], ph[o])
        t, o = PT.fold.values == k, PT.fold.values != k; b[t] = pt[t] >= thr_for_sens(yp[o], pt[o])
    trows.append({'threshold rule': f'B: pooled other 4 folds, {"raw" if kind == "raw" else "calibrated"}',
                  'photo sens': fmt(sens(y, a), boot(y, a, g, sens)), 'photo spec': fmt(spec(y, a), boot(y, a, g, spec)),
                  'patient sens': fmt(sens(yp, b), boot(yp, b, gp, sens)), 'patient spec': fmt(spec(yp, b), boot(yp, b, gp, spec))})
thr = pd.DataFrame(trows)

# ---- Karnataka: 5-fold ensemble, raw vs calibrated --------------------------------------------------
krows = []
for kind, p in [('raw', np.mean(kraw, 0)), ('calibrated', np.mean(kcal, 0))]:
    sl, ic = slope_int(yk, p)
    krows.append({'scores': kind, 'AUC': round(roc_auc_score(yk, p), 3), 'mean predicted risk': round(p.mean(), 3),
                  'observed rate': round(yk.mean(), 3), 'Brier': round(brier_score_loss(yk, p), 4), 'ECE': round(ece(yk, p), 4),
                  'cal. slope': round(sl, 2), 'cal. intercept': round(ic, 2)})
kt = pd.DataFrame(krows)

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 12)
print('9b.1  CLIP-B16-LoRA calibration (out-of-fold; calibrators fitted on each model\'s validation fold)'); print(cal.to_string(index=False))
print('\n9b.2  Thresholds (target sensitivity >= 0.90)'); print(thr.to_string(index=False))
print('\n9b.3  Karnataka (5-fold ensemble)'); print(kt.to_string(index=False))
cal.to_csv(f'{OUTD}/step9b_lora_calibration.csv', index=False); thr.to_csv(f'{OUTD}/step9b_lora_thresholds.csv', index=False)
kt.to_csv(f'{OUTD}/step9b_lora_calibration_karnataka.csv', index=False)
print(f'\nSaved to {OUTD}')
