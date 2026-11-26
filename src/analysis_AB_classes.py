"""A: why cross-country transfer fails -> score per class and AUC on sub-tasks (final ensemble, test sets only).
B: lesion-only AUC (benign vs OPMD/cancer) within each country, compared with Karnataka (benign vs malignant photos)."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common_final import photos, patients, boot_auc, fmt, ANA, CLS
from sklearn.metrics import roc_auc_score
SETS = [('exp1 India->Sri Lanka', 1, 'test_srilanka', None), ('exp2 Sri Lanka->India', 2, 'test_india', None),
        ('exp3 both: India', 3, 'test_cv', 'India'), ('exp3 both: Sri Lanka', 3, 'test_cv', 'Sri Lanka')]
TASKS = [('all: neg vs pos', lambda r: r >= 0, lambda r: r >= 2), ('without benign: healthy vs OPMD/cancer', lambda r: r != 1, lambda r: r >= 2),
         ('lesion-only: benign vs OPMD/cancer', lambda r: r >= 1, lambda r: r >= 2), ('benign vs healthy (is benign flagged?)', lambda r: r <= 1, lambda r: r == 1),
         ('cancer vs healthy/benign', lambda r: r != 2, lambda r: r == 3)]
rows, pct, figdat = [], [], []
for name, exp, split, country in SETS:
    d = photos(exp, split)
    if country: d = d[d.country == country]
    for level, D in [('patient', patients(d, exp)), ('photo', d)]:
        D = D.copy(); D['pct'] = D.s.rank(pct=True) * 100        # percentile of score within this test set
        for tn, keep, pos in TASKS:
            k = D[keep(D.r)]; y = pos(k.r).astype(int).values
            if 0 < y.sum() < len(y): rows.append({'test set': name, 'level': level, 'task': tn, 'n': len(k), 'n pos': int(y.sum()), 'AUC (95% CI)': fmt(boot_auc(y, k.s.values))})
        if level == 'patient':
            for r in range(4):
                v = D.pct[D.r == r]
                if len(v): pct.append({'test set': name, 'class': CLS[r], 'patients': len(v), 'median score percentile': round(v.median(), 1), 'mean percentile': round(v.mean(), 1)})
            figdat.append((name, D))
# Karnataka (photo level only; benign vs malignant) for exp2 and exp3 (exp3: mean over the 5 fold models)
for name, exp in [('exp2 Sri Lanka->Karnataka', 2), ('exp3 both->Karnataka', 3)]:
    d = photos(exp, 'test_karnataka').groupby('id').agg(s=('s', 'mean'), y=('y', 'first'), r=('r', 'first')).reset_index()
    rows.append({'test set': name, 'level': 'photo', 'task': 'lesion-only: benign vs malignant', 'n': len(d), 'n pos': int(d.y.sum()), 'AUC (95% CI)': fmt(boot_auc(d.y.values, d.s.values))})
R = pd.DataFrame(rows); C = pd.DataFrame(pct)
R.to_csv(os.path.join(ANA, 'A_B_subtask_auc.csv'), index=False); C.to_csv(os.path.join(ANA, 'A_class_percentiles.csv'), index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_colwidth', 60)
print(C.pivot(index='test set', columns='class', values='median score percentile')[['healthy', 'benign', 'OPMD', 'cancer']].to_string())
print(R[R.level == 'patient'].to_string(index=False)); print(R[R.level == 'photo'].to_string(index=False))
# figure: score percentile by worst finding, patient level (colour-blind-safe Okabe-Ito colours)
TITLES = {'exp1 India->Sri Lanka': 'Trained: India\nTested: Sri Lanka', 'exp2 Sri Lanka->India': 'Trained: Sri Lanka\nTested: India',
          'exp3 both: India': 'Trained: both\nTested: India', 'exp3 both: Sri Lanka': 'Trained: both\nTested: Sri Lanka'}
COL = ['#009E73', '#56B4E9', '#E69F00', '#D55E00']   # healthy, benign, OPMD, cancer
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 4, figsize=(7.16, 2.15), sharey=True)
for a, (name, D) in zip(ax, figdat):
    data = [D.pct[D.r == r].values for r in range(4)]
    bp = a.boxplot(data, widths=0.6, showfliers=False, patch_artist=True, medianprops=dict(color='black', linewidth=1.2))
    for patch, c in zip(bp['boxes'], COL): patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor('black'); patch.set_linewidth(0.6)
    for w in bp['whiskers'] + bp['caps']: w.set_linewidth(0.6)
    a.set_xticks(range(1, 5)); a.set_xticklabels(['Healthy', 'Benign', 'OPMD', 'Cancer'], rotation=35, fontsize=6.5)
    a.set_title(TITLES[name], fontsize=7); a.tick_params(axis='y', labelsize=6.5); a.grid(axis='y', alpha=0.3, linewidth=0.5)
    for sp in ['top', 'right']: a.spines[sp].set_visible(False)
ax[0].set_ylabel('Patient score (percentile)', fontsize=7); fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(os.path.join(ANA, 'figA_scores_by_class.png'), dpi=300); fig.savefig(os.path.join(ANA, 'figA_scores_by_class.pdf'))
print('saved', ANA)
