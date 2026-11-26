"""Step E analysis (CPU): localisation of Grad-CAM / HiResCAM / attention rollout against expert outlines, and deletion test.
Baselines: random (= lesion area for the pointing game), centre (Gaussian), oral-cavity (uniform over the outlined mouth; Sri Lanka).
CIs and paired tests by resampling PATIENTS (2000 bootstraps).  Usage: python analyze_xai.py [--test]"""
import os, sys, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); XD = os.path.join(HERE, 'mac_results_srilanka', 'xai')
TEST = '--test' in sys.argv; files = sorted(glob.glob(os.path.join(XD, 'part_*_test.npz' if TEST else 'part_f[0-9]_[0-9][0-9][0-9].npz')))
assert files, 'no xai result files found - run run_xai.py first'
METHODS = ['Grad-CAM', 'HiResCAM', 'Attention rollout']; N_BOOT = int(os.environ.get('N_BOOT', 2000))
yy, xx = np.mgrid[0:224, 0:224] / 223 - 0.5; CENTRE = np.exp(-(xx ** 2 + yy ** 2) / (2 * 0.25 ** 2))
def scores(cam, m):
    area = m.mean(); hot = np.unravel_index(np.argmax(cam), cam.shape); cam = cam - cam.min()
    energy = cam[m].sum() / (cam.sum() + 1e-8); top = cam >= np.quantile(cam, 1 - area)
    return {'hit': float(m[hot]), 'energy': energy, 'iou': (top & m).sum() / (top | m).sum()}
rows, curves, keep = [], [], []
CACHE = os.path.join(XD, 'xai_stage1' + ('_TEST' if TEST else '') + '.npz')
STAGE1 = '--stage1' in sys.argv
for f in ([] if (os.path.exists(CACHE) and not STAGE1) else files):
    _z = np.load(f, allow_pickle=True); z = {k: _z[k] for k in _z.files}   # load each array once
    les = np.unpackbits(z['les'], axis=1)[:, :224 * 224].reshape(-1, 224, 224).astype(bool)
    cav = np.unpackbits(z['cav'], axis=1)[:, :224 * 224].reshape(-1, 224, 224).astype(bool)
    for i in range(len(z['pid'])):
        m = les[i]
        if m.sum() == 0: continue
        r = {'pid': z['pid'][i], 'src': z['src'][i], 'group': z['group'][i], 'cls': z['cls'][i], 'p': float(z['p'][i]), 'area': m.mean()}
        for j, nm in enumerate(METHODS):
            for k, v in scores(z['maps'][i][j].astype(np.float32), m).items(): r[f'{nm}|{k}'] = v
        for k, v in scores(CENTRE, m).items(): r[f'Centre|{k}'] = v
        if cav[i].sum() > 0:
            u = cav[i].astype(np.float32) + 1e-3 * CENTRE          # uniform over the mouth (tiny centre term breaks ties)
            for k, v in scores(u, m).items(): r[f'Oral cavity|{k}'] = v
        r['Random|hit'] = r['area']; r['Random|energy'] = r['area']
        d = np.clip(z['dele'][i].astype(np.float64), 1e-6, 1 - 1e-6); lo = np.log(d / (1 - d)); dl = lo - lo[:, :1]   # change in log-odds vs intact photo
        for j, nm in enumerate(METHODS + ['Random order']): r[f'{nm}|deletion AUC'] = float(np.mean(dl[j, 1:]))      # lower = score falls faster = more faithful
        rows.append(r); curves.append(dl)
        if len(keep) < 400: keep.append((z['pid'][i], z['src'][i], z['maps'][i].astype(np.float32), m, cav[i], r))
if rows:
    pd.DataFrame(rows).to_pickle(CACHE + '.rows.pkl'); np.savez_compressed(CACHE, curves=np.array(curves))
    import pickle; pickle.dump(keep[:60], open(CACHE + '.keep.pkl', 'wb'))
    if STAGE1: print('stage 1 saved', len(rows), 'photos'); sys.exit(0)
else:
    import pickle; rows = pd.read_pickle(CACHE + '.rows.pkl').to_dict('records'); curves = list(np.load(CACHE)['curves']); keep = pickle.load(open(CACHE + '.keep.pkl', 'rb'))
R = pd.DataFrame(rows)
def boot(df, cols, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed); g = df.groupby('group').indices; keys = np.array(list(g)); X = df[cols].values; out = []
    for _ in range(n):
        ii = np.concatenate([g[k] for k in keys[rng.integers(0, len(keys), len(keys))]]); out.append(np.nanmean(X[ii], 0))
    return np.array(out)
summ, tests = [], []
for src in ['SMART-OM', 'Sri Lanka']:
    S = R[R.src == src]
    if len(S) < 2: continue
    for metric, names in [('hit', METHODS + ['Centre', 'Oral cavity', 'Random']), ('energy', METHODS + ['Centre', 'Oral cavity', 'Random']),
                          ('iou', METHODS + ['Centre', 'Oral cavity']), ('deletion AUC', METHODS + ['Random order'])]:
        cols = [f'{n}|{metric}' for n in names if f'{n}|{metric}' in S and S[f'{n}|{metric}'].notna().any()]
        B = boot(S, cols)
        for c, b in zip(cols, B.T):
            summ.append({'photos': src, 'metric': metric, 'method': c.split('|')[0], 'n photos': int(S[c].notna().sum()), 'n patients': S.group.nunique(),
                         'mean (95% CI)': f'{np.nanmean(S[c]):.3f} ({np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f})'})
        ref = f'HiResCAM|{metric}'
        for c in cols:
            if c == ref: continue
            i, j = cols.index(ref), cols.index(c); dd = B[:, i] - B[:, j]
            tests.append({'photos': src, 'metric': metric, 'comparison': f'HiResCAM - {c.split("|")[0]}', 'difference': round(float(np.nanmean(S[ref] - S[c])), 3),
                          '95% CI': f'{np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f}', 'p (two-sided)': round(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())), 4)})
    # rollout vs grad-cam too
S_ = pd.DataFrame(summ); T_ = pd.DataFrame(tests)
tag = '_TEST' if TEST else ''
R.to_csv(os.path.join(XD, f'xai_per_photo{tag}.csv'), index=False); S_.to_csv(os.path.join(XD, f'xai_summary{tag}.csv'), index=False); T_.to_csv(os.path.join(XD, f'xai_tests{tag}.csv'), index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
print(S_.pivot_table(index=['photos', 'method'], columns='metric', values='mean (95% CI)', aggfunc='first').to_string())
print(T_.to_string(index=False))
# ---- figure 1: deletion curves; figure 2: examples (Sri Lanka first, best-energy HiResCAM, plus a typical one)
C = np.array(curves); x = np.linspace(0, 100, C.shape[2])
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.4), sharey=True)
for a, src in zip(ax, ['SMART-OM', 'Sri Lanka']):
    idx = ((R.src == src) & (R.p >= 0.5)).values      # deletion is only meaningful where the model says 'disease'
    if idx.sum() == 0: continue
    for j, nm in enumerate(METHODS + ['Random order']): a.plot(x, C[idx, j].mean(0), label=nm, lw=1.5, ls='--' if nm == 'Random order' else '-')
    a.set_title(f'{src}, photos the model flags (n={idx.sum()})', fontsize=8); a.set_xlabel('% of patches removed (most important first)', fontsize=7); a.tick_params(labelsize=7)
ax[0].set_ylabel('Change in log-odds of disease', fontsize=7); ax[1].legend(fontsize=6); fig.tight_layout()
fig.savefig(os.path.join(XD, f'figE_deletion{tag}.png'), dpi=300); fig.savefig(os.path.join(XD, f'figE_deletion{tag}.pdf'))
K = sorted(keep, key=lambda t: (t[1] != 'Sri Lanka', -t[5]['HiResCAM|energy']))[:3]
if len(K):
    fig, axs = plt.subplots(len(K), 4, figsize=(7.16, 1.9 * len(K))); axs = np.atleast_2d(axs)
    for r_, (pid, src, maps, m, cav, rr) in enumerate(K):
        from PIL import Image
        p = os.path.join(HERE, 'mac_kit', 'img512_sl' if src == 'Sri Lanka' else 'img512', pid if src == 'Sri Lanka' else f'{pid}.jpg')
        img = np.asarray(Image.open(p).convert('RGB').resize((224, 224)))
        axs[r_, 0].imshow(img); axs[r_, 0].contour(m, [0.5], colors='lime', linewidths=1); axs[r_, 0].set_title(f'{src}: {rr["cls"]}', fontsize=7)
        for j, nm in enumerate(METHODS):
            axs[r_, j + 1].imshow(img); axs[r_, j + 1].imshow(maps[j], cmap='jet', alpha=0.45); axs[r_, j + 1].contour(m, [0.5], colors='white', linewidths=1)
            axs[r_, j + 1].set_title(f'{nm} (energy {rr[f"{nm}|energy"]:.0%})', fontsize=7)
        for a in axs[r_]: a.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(XD, f'figE_examples{tag}.png'), dpi=300)
print('saved in', XD)
