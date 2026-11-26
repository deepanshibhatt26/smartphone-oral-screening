"""Sri Lanka training (Mac GPU): the three cross-country experiments, same recipe as Step 6.
  exp1  India (SMART-OM + CODE + Karnataka)  ->  test on all Sri Lanka
  exp2  Sri Lanka                            ->  test on all India (SMART-OM + CODE + Karnataka)
  exp3  India (SMART-OM + CODE) + Sri Lanka, 5-fold patient-level CV  ->  Karnataka external
Models: CLIP ViT-B/16 + LoRA ('lora') and ConvNeXt-B full fine-tuning ('convnext').
Resumable: each finished run is skipped. Outputs: mac_results_srilanka/train/
Run:  oral_env/bin/python run_sl_train.py --exp 1 2 3 --models lora
"""
import os, sys, json, time, random, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, average_precision_score
from mac_common import KIT, HERE, DEV, N_FOLDS, stamp
import torch.multiprocessing as _mp
_mp.set_sharing_strategy('file_system')   # avoids 'Too many open files' on macOS

MAX_EPOCHS, PATIENCE, POS_SHARE, SEED = 8, 2, 0.25, 42
BS = int(os.environ.get('BS', 32))
BIC = T.InterpolationMode.BICUBIC
RES = os.path.join(HERE, 'mac_results_srilanka', 'train'); os.makedirs(RES, exist_ok=True)

def load_all():
    m = pd.read_csv(os.path.join(KIT, 'meta.csv'))
    m['id'] = m.md5; m['path'] = [os.path.join(KIT, 'img512', f'{h}.jpg') for h in m.md5]
    m['country'] = 'IN'
    sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
    sl = sl.assign(id=sl.image, path=[os.path.join(KIT, 'img512_sl', f'{n}.jpg') for n in sl.image],
                   country='SL', set='srilanka')
    cols = ['id', 'path', 'y_bin', 'group', 'fold', 'dataset', 'country', 'set']
    return pd.concat([m[cols], sl[cols]], ignore_index=True)

def build(name):
    if name == 'convnext':
        m = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=1, drop_path_rate=0.2)
        groups = [{'params': [p for n, p in m.named_parameters() if not n.startswith('head.fc')], 'lr': 1e-4},
                  {'params': list(m.head.fc.parameters()), 'lr': 1e-3}]
        cfg = m.pretrained_cfg
    else:
        base = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=1)
        lcfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, bias='none', modules_to_save=['head'],
                          target_modules=r'.*blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)')
        m = get_peft_model(base, lcfg)
        groups = [{'params': [p for p in m.parameters() if p.requires_grad], 'lr': 5e-4}]
        cfg = base.pretrained_cfg
    return m, groups, cfg['mean'], cfg['std']

class Photos(Dataset):
    def __init__(s, paths, ys, tf): s.p, s.y, s.tf = list(paths), np.asarray(ys, np.float32), tf
    def __len__(s): return len(s.p)
    def __getitem__(s, i): return s.tf(Image.open(s.p[i]).convert('RGB')), torch.tensor(s.y[i])

def tfs(mean, std):
    tr = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.25), interpolation=BIC),
                    T.RandomHorizontalFlip(), T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.1, 0.02),
                    T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.Resize((224, 224), interpolation=BIC), T.ToTensor(), T.Normalize(mean, std)])
    return tr, te

@torch.no_grad()
def predict(m, paths, tf, nw):
    m.eval(); out = []
    for xb, _ in DataLoader(Photos(paths, np.zeros(len(paths)), tf), batch_size=64, num_workers=nw):
        xb = xb.to(DEV)
        lo = (m(xb).float() + m(torch.flip(xb, [3])).float()) / 2
        out.append(torch.sigmoid(lo).squeeze(1).cpu().numpy())
    return np.concatenate(out)

def weights(tr):
    """25% suspicious per batch; with two countries, each country gives half of each class (no country shortcut)"""
    y, c = tr.y_bin.values, tr.country.values
    share = {1: POS_SHARE, 0: 1 - POS_SHARE}; cs = sorted(set(c))
    w = np.zeros(len(tr))
    for lab in (0, 1):
        for cc in cs:
            idx = (y == lab) & (c == cc)
            if idx.sum(): w[idx] = share[lab] / len(cs) / idx.sum()
    return w

def runs_for(exp, df):
    IN = df[df.set == 'internal']; KA = df[df.set == 'karnataka']; SL = df[df.set == 'srilanka']
    if exp == 1:
        ka_val = KA.sample(frac=0.2, random_state=SEED)
        tr = pd.concat([IN[IN.fold != 4], KA.drop(ka_val.index)]); va = pd.concat([IN[IN.fold == 4], ka_val])
        yield 'run0', tr, va, {'test_srilanka': SL}
    elif exp == 2:
        yield 'run0', SL[SL.fold != 4], SL[SL.fold == 4], {'test_india': IN, 'test_karnataka': KA}
    else:
        for k in range(N_FOLDS):
            vk = (k + 1) % N_FOLDS
            both = pd.concat([IN, SL])
            yield f'fold{k}', both[(both.fold != k) & (both.fold != vk)], both[both.fold == vk], \
                  {'test_cv': both[both.fold == k], 'test_karnataka': KA}

def run(exp, name, tag, tr, va, tests, nw):
    f = os.path.join(RES, f'exp{exp}_{name}_{tag}')
    if os.path.exists(f + '.json'):
        print(f'exp{exp} {name} {tag}: already done'); return
    seed = SEED + sum(map(ord, tag)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m, groups, mean, std = build(name); m = m.to(DEV); ttf, etf = tfs(mean, std)
    sampler = WeightedRandomSampler(torch.tensor(weights(tr), dtype=torch.double), num_samples=len(tr), replacement=True)
    dl = DataLoader(Photos(tr.path, tr.y_bin.values, ttf), batch_size=BS, sampler=sampler, num_workers=nw, drop_last=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=MAX_EPOCHS * len(dl), pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss()
    print(f'exp{exp} {name} {tag}: train {len(tr)} ({int(tr.y_bin.sum())} suspicious, {dict(tr.country.value_counts())}) | val {len(va)}')
    best_ap, best_state, best_ep, t0 = -1, None, -1, time.time()
    for ep in range(MAX_EPOCHS):
        m.train(); tl = 0
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = lossf(m(xb).squeeze(1).float(), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
            opt.step(); sched.step(); tl += loss.item()
        pv = predict(m, va.path, etf, nw)
        ap, auc = average_precision_score(va.y_bin, pv), roc_auc_score(va.y_bin, pv)
        if ap > best_ap: best_ap, best_ep, best_state = ap, ep, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        print(f'   epoch {ep+1}: loss {tl/len(dl):.3f} | val AUC {auc:.3f} AUPRC {ap:.3f}{" *" if ep == best_ep else ""} ({stamp(t0)})')
        if ep - best_ep >= PATIENCE: break
    m.load_state_dict(best_state)
    pd.DataFrame({'id': va.id.values, 'y': va.y_bin.values, 'p': predict(m, va.path, etf, nw)}).to_csv(f + '_val.csv', index=False)
    for tname, t in tests.items():
        pd.DataFrame({'id': t.id.values, 'y': t.y_bin.values, 'p': predict(m, t.path, etf, nw)}).to_csv(f + f'_{tname}.csv', index=False)
    json.dump({'best_epoch': best_ep + 1, 'val_auprc': best_ap, 'minutes': (time.time() - t0) / 60}, open(f + '.json', 'w'))
    print(f'exp{exp} {name} {tag}: saved ({stamp(t0)})')
    del m

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--exp', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--models', nargs='+', default=['lora']); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(); df = load_all()
    print(f'Device: {DEV} | photos: {dict(df.set.value_counts())} | batch {BS}')
    for name in a.models:
        for exp in a.exp:
            for tag, tr, va, tests in runs_for(exp, df):
                run(exp, name, tag, tr, va, tests, a.workers)
    print('All requested runs finished.')
