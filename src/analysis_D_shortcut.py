"""D: shortcut check - could the model be recognising the country/camera instead of disease?
1. How easily the frozen image features reveal the country (patient-grouped 5-fold, same folds as exp3).
2. A 'country-only' predictor of disease on the pooled exp3 test set (what pooled AUC a pure shortcut would give).
3. Final-model scores of HEALTHY patients by country (a country bias would push one country's healthy mouths up).
4. Pooled AUC after removing the between-country shift (score centred within country) vs raw pooled AUC."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common_final import photos, patients, boot_auc, fmt, ANA, KIT, HERE
D = os.path.join(HERE, 'mac_results_srilanka'); rows = []
m = pd.read_csv(os.path.join(KIT, 'meta.csv')); s = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
m['id'] = m.md5; s['id'] = s.image; s['set'] = 'srilanka'
df = pd.concat([m[['id', 'fold', 'set', 'y_bin']], s[['id', 'fold', 'set', 'y_bin']]]).drop_duplicates('id'); df = df[df.set.isin(['internal', 'srilanka'])]
df['sl'] = (df.set == 'srilanka').astype(int)
for bb in ['CLIP-B16', 'ConvNeXt-B']:
    F = {}
    for st in ['internal', 'srilanka']:
        z = np.load(os.path.join(D, f'sl_feats_{bb}_{st}.npz'), allow_pickle=True); F.update(dict(zip(z['ids'], z['X'])))
    X = np.stack([F[i] for i in df.id]); p = np.zeros(len(df))
    for k in range(5):
        tr, te = (df.fold != k).values, (df.fold == k).values; sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[tr]), df.sl.values[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    for nm, msk in [('all photos', np.ones(len(df), bool)), ('healthy/benign photos only', (df.y_bin == 0).values)]:
        rows.append({'check': f'1. country decodable from frozen {bb} features ({nm})', 'value': fmt(boot_auc(df.sl.values[msk], p[msk]))})
T = patients(photos(3, 'test_cv'), 3); y = T.y.values; isl = (T.country == 'Sri Lanka').astype(int).values
rows.append({'check': '2. disease AUC of a country-only predictor (pooled exp3 test)', 'value': fmt(boot_auc(y, isl))})
rows.append({'check': '2. disease AUC of the final model (pooled exp3 test)', 'value': fmt(boot_auc(y, T.s.values))})
T['pct'] = T.s.rank(pct=True) * 100
for c in ['India', 'Sri Lanka']:
    h = T[(T.country == c) & (T.r == 0)]; rows.append({'check': f'3. healthy patients, {c}: median score percentile (n={len(h)})', 'value': round(h.pct.median(), 1)})
h = T[T.r == 0]; rows.append({'check': '3. among healthy patients, does the score reveal the country? (AUC)', 'value': fmt(boot_auc((h.country == 'Sri Lanka').astype(int).values, h.s.values))})
T['s_c'] = T.s - T.groupby(['tag', 'country']).s.transform('median')
rows.append({'check': '4. pooled AUC with score centred within country (shift removed)', 'value': fmt(boot_auc(y, T.s_c.values))})
for c in ['India', 'Sri Lanka']:
    d = T[T.country == c]; rows.append({'check': f'4. within-country AUC, {c}', 'value': fmt(boot_auc(d.y.values, d.s.values))})
R = pd.DataFrame(rows); R.to_csv(os.path.join(ANA, 'D_shortcut_check.csv'), index=False)
pd.set_option('display.max_colwidth', 90); print(R.to_string(index=False))
