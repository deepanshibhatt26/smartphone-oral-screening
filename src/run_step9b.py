"""Step 9b (Mac GPU part): validation-fold predictions of the CLIP-B16-LoRA fold models, for calibration.
For fold k the model was trained on 3 folds, early-stopped on validation fold v=(k+1)%5 and tested on fold k.
Calibrators are fitted on fold v's predictions and applied to fold k's saved test predictions (no test labels used).
Resumable: a finished fold is skipped. Output: mac_results/step9b_val_fold{k}.csv, step9b_check.csv"""
import os, time
import numpy as np, pandas as pd
from mac_common import KIT, OUT, DEV, N_FOLDS, load_meta, load_lora, predict, stamp

meta = load_meta(); internal = meta[meta.set == 'internal']
print(f'Device: {DEV} | internal photos: {len(internal)}')
rng = np.random.default_rng(0); checks = []
t0 = time.time()
for k in range(N_FOLDS):
    f = os.path.join(OUT, f'step9b_val_fold{k}.csv')
    v = (k + 1) % N_FOLDS
    if os.path.exists(f):
        print(f'fold {k}: already done, skipped'); continue
    m, tf = load_lora(k)
    # 1) safety check: re-predict 200 test photos (all suspicious + random others) and compare with Colab's saved values
    saved = pd.read_csv(os.path.join(KIT, 'step6', f'CLIP-B16-LoRA_fold{k}.csv'))
    te = internal[internal.fold == k].merge(saved, on='md5')
    pick = pd.concat([te[te.y_bin == 1], te[te.y_bin == 0].sample(min(200, (te.y_bin == 0).sum()), random_state=0)]).head(200)
    pm = predict(m, tf, pick.md5.tolist())
    r = np.corrcoef(pm, pick.p)[0, 1]; mad = np.abs(pm - pick.p).max()
    checks.append({'fold': k, 'n': len(pick), 'corr_with_colab': round(r, 5), 'max_abs_diff': round(mad, 4)})
    print(f'fold {k}: reproduce check on {len(pick)} test photos -> corr {r:.5f}, max |diff| {mad:.4f}  ({stamp(t0)})')
    # 2) validation-fold predictions (what calibration needs)
    va = internal[internal.fold == v]
    pv = predict(m, tf, va.md5.tolist())
    pd.DataFrame({'md5': va.md5.values, 'fold_model': k, 'val_fold': v, 'p': pv}).to_csv(f, index=False)
    print(f'fold {k}: {len(va)} validation photos (fold {v}) predicted  ({stamp(t0)})')
    del m
if checks:
    cf = os.path.join(OUT, 'step9b_check.csv')
    old = pd.read_csv(cf) if os.path.exists(cf) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(checks)]).drop_duplicates('fold', keep='last').to_csv(cf, index=False)
print(f'Step 9b Mac part finished in {stamp(t0)}. Results in {OUT}')
