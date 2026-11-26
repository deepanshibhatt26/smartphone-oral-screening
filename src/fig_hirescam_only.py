"""Paper figure: HiResCAM only. Medium lesions (5-30% of photo) the model flags (p>=0.5):
best SMART-OM, best Sri Lanka (by HiResCAM energy / lesion area) and a TYPICAL Sri Lanka photo (median).
Top row: photo + expert outline (green). Bottom row: HiResCAM heatmap + outline (white). One IEEE column (3.5 in)."""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); XD = os.path.join(HERE, 'mac_results_srilanka', 'xai'); KIT = os.path.join(HERE, 'mac_kit')
R = pd.read_csv(os.path.join(XD, 'xai_per_photo.csv'))
C = R[(R.p >= 0.5) & R.area.between(0.05, 0.30)].copy(); C['ratio'] = C['HiResCAM|energy'] / C.area
sl = C[C.src == 'Sri Lanka'].sort_values('ratio')
pick = [('(a)', C[C.src == 'SMART-OM'].sort_values('ratio').iloc[-1]), ('(b)', sl.iloc[-1]), ('(c)', sl.iloc[len(sl) // 2])]
want = {r.pid for _, r in pick}; found = {}
for f in sorted(glob.glob(os.path.join(XD, 'part_f[0-9]_[0-9][0-9][0-9].npz'))):
    z = np.load(f, allow_pickle=True); pids = list(z['pid'])
    for p in want & set(pids):
        i = pids.index(p); found[p] = (z['maps'][i][1].astype(np.float32), np.unpackbits(z['les'][i])[:224 * 224].reshape(224, 224).astype(bool))
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.55))
for j, (lab, r) in enumerate(pick):
    cam, m = found[r.pid]
    p = os.path.join(KIT, 'img512_sl', r.pid) if r.src == 'Sri Lanka' else os.path.join(KIT, 'img512', f'{r.pid}.jpg')
    img = np.asarray(Image.open(p).convert('RGB').resize((224, 224)))
    axs[0, j].imshow(img); axs[0, j].contour(m, [0.5], colors='lime', linewidths=0.8)
    axs[0, j].set_title(f'{lab} {r.src}' + (' (typical)' if lab == '(c)' else ''), fontsize=6.5)
    axs[1, j].imshow(img); axs[1, j].imshow(cam, cmap='jet', alpha=0.45); axs[1, j].contour(m, [0.5], colors='white', linewidths=0.8)
    axs[1, j].set_title(f'{r["HiResCAM|energy"]:.0%} of heat in lesion', fontsize=6.5)
    for a in axs[:, j]: a.set_xticks([]); a.set_yticks([])
axs[0, 0].set_ylabel('Photo + expert outline', fontsize=6.5); axs[1, 0].set_ylabel('HiResCAM', fontsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.4)
fig.savefig(os.path.join(XD, 'fig_hirescam.png'), dpi=300); fig.savefig(os.path.join(XD, 'fig_hirescam.pdf'))
print([(lab, r.src, r.cls, round(r.area, 3), round(r['HiResCAM|energy'], 3)) for lab, r in pick])
