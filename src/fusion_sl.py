"""Image + habit fusion on India (SMART-OM questionnaire) + Sri Lanka, using the SAME 5 patient folds as exp3.
Image score = exp3 final ensemble patient score (out-of-fold). Fusion models are fitted on the other 4 folds,
evaluated on fold k. No tuning on test: model settings fixed in advance."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'mac_results_srilanka', 'fusion')
exec(open(os.path.join(HERE, 'ens_step2_combine.py')).read().split('EXPS =')[0])   # load(), score(), agg(), info, FROZEN
OUT = os.path.join(HERE, 'mac_results_srilanka', 'fusion')   # re-set: the exec above overwrites OUT
BEST = (0.6, tuple(FROZEN), 'top2')     # exp3 config chosen on validation (ensemble_selection_val.csv)
# ---- 1. out-of-fold image score per patient
rows = []
for k in range(5):
    t = f'fold{k}'; v = load(3, t, 'val'); st = {c: (lg(v[c].values).mean(), lg(v[c].values).std()) for c in ['LoRA'] + FROZEN}
    te = load(3, t, 'test_cv'); y, p = agg(te, score(te, st, BEST), 'top2')
    g = te.assign(s=1).groupby('group').y.max().index
    rows.append(pd.DataFrame({'group': g, 'y': y, 'img': p, 'fold': k}))
P = pd.concat(rows); assert P.group.is_unique
# ---- 2. habits (harmonised: smoking, chewing [betel/areca/smokeless], alcohol, age, sex)
h = pd.read_csv(os.path.join(HERE, '..', 'data', 'processed', 'smart_om_metadata_flat.csv')); h.columns = [c.strip() for c in h.columns]
u = lambda c: h[c].astype(str).str.lower().str.contains('user').astype(int)
hi = pd.DataFrame({'group': h.patient_id, 'smoking': u('Smoking|Habit_status'),
                   'chewing': ((u('Chewing|Habit_status') + u('Arecanut|Habit_status')) > 0).astype(int),
                   'alcohol': u('Alcohol|Habit status'), 'age': pd.to_numeric(h.Age, errors='coerce'),
                   'male': h.Sex.astype(str).str.strip().str.lower().eq('male').astype(int), 'country': 'India'})
s = pd.read_csv(os.path.join(KIT, 'meta_sl.csv')).drop_duplicates('group')
hs = pd.DataFrame({'group': s.group, 'smoking': s.smoking.eq('Yes').astype(int), 'chewing': s.betel.eq('Yes').astype(int),
                   'alcohol': s.alcohol.eq('Yes').astype(int), 'age': s.age, 'male': s.sex.eq('M').astype(int), 'country': 'Sri Lanka'})
H = pd.concat([hi, hs]).drop_duplicates('group')
D = P.merge(H, on='group'); D = D[D.age.notna()].reset_index(drop=True)
HAB = ['smoking', 'chewing', 'alcohol', 'age', 'male']
print('fusion cohort:', D.groupby('country').agg(patients=('y', 'size'), positive=('y', 'sum')).to_string(), flush=True)
# ---- 3. models, fitted on 4 folds, tested on fold k
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
def gb(): return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
MODELS = {'Image only (score)': None, 'Habits only, LR': (lr, HAB), 'Habits only, GBDT': (gb, HAB),
          'Image + habits, LR': (lr, ['img'] + HAB), 'Image + habits, GBDT': (gb, ['img'] + HAB)}
pred = {m: np.zeros(len(D)) for m in MODELS}; imp = []
for k in range(5):
    tr, te = (D.fold != k).values, (D.fold == k).values
    for m, spec in MODELS.items():
        if spec is None: pred[m][te] = D.img.values[te]; continue
        f, cols = spec; mdl = f().fit(D.loc[tr, cols], D.y[tr]); pred[m][te] = mdl.predict_proba(D.loc[te, cols])[:, 1]
        if m == 'Image + habits, GBDT':
            pi = permutation_importance(mdl, D.loc[te, cols], D.y[te], scoring='roc_auc', n_repeats=20, random_state=k)
            imp.append(pd.Series(pi.importances_mean, index=cols))
# ---- 4. metrics
rng = np.random.default_rng(0); res = []
for nm, msk in [('India (SMART-OM)', D.country == 'India'), ('Sri Lanka', D.country == 'Sri Lanka'), ('Both', D.country.notna())]:
    y = D.y.values[msk]; base = pred['Image only (score)'][msk]; idx = np.arange(len(y)); B = [rng.choice(idx, len(idx)) for _ in range(500)]
    B = [b for b in B if len(set(y[b])) == 2]
    for m in MODELS:
        p = pred[m][msk]; a = [roc_auc_score(y[b], p[b]) for b in B]; d = [roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], base[b]) for b in B]
        res.append({'cohort': nm, 'model': m, 'patients': len(y), 'positive': int(y.sum()), 'patient AUC': round(roc_auc_score(y, p), 3),
                    '95% CI': f'{np.percentile(a, 2.5):.3f}-{np.percentile(a, 97.5):.3f}',
                    'vs image-only (95% CI)': '' if m.startswith('Image only') else f'{np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}'})
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'fusion_results.csv'), index=False)
I = pd.concat(imp, axis=1).mean(1).sort_values(ascending=False).round(3); I.to_csv(os.path.join(OUT, 'fusion_permutation_importance.csv'))
D.assign(**{f'pred_{m}': v for m, v in pred.items()}).to_csv(os.path.join(OUT, 'fusion_patient_predictions.csv'), index=False)
pd.set_option('display.width', 220); print(R.to_string(index=False)); print('\nPermutation importance (drop in AUC, GBDT image+habits):\n', I.to_string())
print('\nHabit prevalence by country and label:\n', D.groupby(['country', 'y'])[HAB].mean().round(2).to_string())
