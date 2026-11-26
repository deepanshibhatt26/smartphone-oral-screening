"""Image + habit fusion in ALL THREE experiments, with logistic regression (LR), gradient boosting (GBDT) and TabPFN.
exp1 India->Sri Lanka and exp2 Sri Lanka->India (late fusion):
   habit model fitted on SOURCE-country training patients; combined score = z(image) + w * z(habit logit),
   w chosen on SOURCE-country validation patients (patient AUC); target country evaluated once.
exp3 both countries (5-fold): image score + habits as features, fitted on 4 folds, tested on the 5th (as fusion_sl.py).
Habits (harmonised): smoking, chewing (betel / areca / smokeless tobacco), alcohol, age, sex. CODE and Karnataka have none.
Run:  tabpfn_env/bin/python fusion_all.py      (MOCK=1 -> TabPFN replaced by logistic regression, for testing)"""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'ens_step2_combine.py')).read().split('EXPS =')[0])   # load(), score(), lg(), info, FROZEN, KIT
OUT = os.path.join(HERE, 'mac_results_srilanka', 'fusion'); os.makedirs(OUT, exist_ok=True)
MOCK = os.environ.get('MOCK') == '1'
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
if MOCK: tp = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)); print('MOCK run: TabPFN replaced by LR')
else:
    from tabpfn import TabPFNClassifier; import tabpfn; print('TabPFN', getattr(tabpfn, '__version__', '?'))
    tp = lambda: TabPFNClassifier(random_state=0)
HMODELS = {'LR': lr, 'GBDT': gb, 'TabPFN': tp}
HAB = ['smoking', 'chewing', 'alcohol', 'age', 'male']
# ---------- harmonised habits per patient
h = pd.read_csv(os.path.join(HERE, '..', 'data', 'processed', 'smart_om_metadata_flat.csv')); h.columns = [c.strip() for c in h.columns]
u = lambda c: h[c].astype(str).str.lower().str.contains('user').astype(int)
hi = pd.DataFrame({'group': h.patient_id, 'smoking': u('Smoking|Habit_status'), 'chewing': ((u('Chewing|Habit_status') + u('Arecanut|Habit_status')) > 0).astype(int),
                   'alcohol': u('Alcohol|Habit status'), 'age': pd.to_numeric(h.Age, errors='coerce'), 'male': h.Sex.astype(str).str.strip().str.lower().eq('male').astype(int), 'country': 'India'})
s = pd.read_csv(os.path.join(KIT, 'meta_sl.csv')).drop_duplicates('group')
hs = pd.DataFrame({'group': s.group, 'smoking': s.smoking.eq('Yes').astype(int), 'chewing': s.betel.eq('Yes').astype(int), 'alcohol': s.alcohol.eq('Yes').astype(int),
                   'age': s.age, 'male': s.sex.eq('M').astype(int), 'country': 'Sri Lanka'})
H = pd.concat([hi, hs]).drop_duplicates('group'); H = H[H.age.notna()]
m = pd.read_csv(os.path.join(KIT, 'meta.csv')); lab = pd.concat([m[['group', 'y_bin']], s[['group', 'y_bin']]]).groupby('group').y_bin.max()
H['y'] = H.group.map(lab); H = H[H.y.notna()].set_index('group')
def patient_img(exp, split, cfg, st):
    d = load(exp, 'run0', split); d['s'] = score(d, st, cfg); t = d.sort_values('s', ascending=False).groupby('group')
    p = t.s.max() if cfg[2] == 'max' else t.s.apply(lambda v: v.iloc[:2].mean())
    return pd.DataFrame({'y': t.y.max(), 'img': p})
rng = np.random.default_rng(0)
def evaluate(name, y, preds, rows):
    base = preds['Image only']; B = [b for b in (rng.choice(len(y), len(y)) for _ in range(500)) if len(set(y[b])) == 2]
    for mname, p in preds.items():
        a = [roc_auc_score(y[b], p[b]) for b in B]; dd = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], base[b]) for b in B]
        rows.append({'experiment': name, 'model': mname, 'patients': len(y), 'positive': int(y.sum()), 'patient AUC': round(roc_auc_score(y, p), 3),
                     '95% CI': f'{np.percentile(a, 2.5):.3f}-{np.percentile(a, 97.5):.3f}',
                     'vs image only (95% CI)': '' if mname == 'Image only' else f'{np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f}'})
rows, wsel = [], []
# ---------- exp1 / exp2: late fusion, everything chosen in the source country
CFG = {1: (0.75, ('ConvNeXt-B',), 'top2'), 2: (0.6, ('ConvNeXt-B',), 'max')}   # from ensemble_selection_val.csv
SETUP = {1: ('India', 'srilanka', 'exp1 India -> Sri Lanka'), 2: ('Sri Lanka', 'india', 'exp2 Sri Lanka -> India (SMART-OM)')}
for exp, (src, tname, label) in SETUP.items():
    v = load(exp, 'run0', 'val'); st = {c: (lg(v[c].values).mean(), lg(v[c].values).std()) for c in ['LoRA'] + FROZEN}
    VI = patient_img(exp, 'val', CFG[exp], st).join(H[HAB], how='inner'); TI = patient_img(exp, f'test_{tname}', CFG[exp], st).join(H[HAB], how='inner')
    TRN = H[(H.country == src) & ~H.index.isin(VI.index)]
    zi = (VI.img.mean(), VI.img.std())
    preds = {'Image only': TI.img.values}
    for hn, f in HMODELS.items():
        mdl = f().fit(TRN[HAB].values, TRN.y.values)
        hv, ht = lg(mdl.predict_proba(VI[HAB].values)[:, 1]), lg(mdl.predict_proba(TI[HAB].values)[:, 1]); zh = (hv.mean(), hv.std() + 1e-9)
        preds[f'Habits only, {hn}'] = ht
        cand = [(roc_auc_score(VI.y, (VI.img - zi[0]) / zi[1] + w * (hv - zh[0]) / zh[1]), w) for w in [0, .25, .5, .75, 1, 1.5, 2]]
        best = max(cand, key=lambda c: (round(c[0], 4), -c[1])); w = best[1]
        wsel.append({'experiment': label, 'habit model': hn, 'chosen weight w': w, 'source val patient AUC': round(best[0], 3), 'source val patients': len(VI), 'source train patients': len(TRN)})
        preds[f'Image + habits, {hn}'] = (TI.img.values - zi[0]) / zi[1] + w * (ht - zh[0]) / zh[1]
    evaluate(label, TI.y.values.astype(int), preds, rows); print(label, 'done', flush=True)
# ---------- exp3: features model on 4 folds -> 5th (same as fusion_sl.py), now also TabPFN
D = pd.read_csv(os.path.join(OUT, 'fusion_patient_predictions.csv'))
p3 = {'Image only': D['pred_Image only (score)'].values}
for hn, f in HMODELS.items():
    for kind, cols in [('Habits only', HAB), ('Image + habits', ['img'] + HAB)]:
        p = np.zeros(len(D))
        for k in range(5):
            tr, te = (D.fold != k).values, (D.fold == k).values
            p[te] = f().fit(D.loc[tr, cols].values, D.y.values[tr]).predict_proba(D.loc[te, cols].values)[:, 1]
        p3[f'{kind}, {hn}'] = p
for nm, msk in [('exp3 both, India (SMART-OM)', D.country == 'India'), ('exp3 both, Sri Lanka', D.country == 'Sri Lanka'), ('exp3 both, pooled', D.country.notna())]:
    msk = msk.values; evaluate(nm, D.y.values[msk], {k: v[msk] for k, v in p3.items()}, rows)
print('exp3 done', flush=True)
R = pd.DataFrame(rows); W = pd.DataFrame(wsel); tag = '_MOCK' if MOCK else ''
R.to_csv(os.path.join(OUT, f'fusion_all_results{tag}.csv'), index=False); W.to_csv(os.path.join(OUT, f'fusion_all_weights{tag}.csv'), index=False)
pd.set_option('display.width', 220); print(W.to_string(index=False)); print(R.to_string(index=False)); print('saved in', OUT)
