"""Step E (Mac GPU): compare explanation methods for the CLIP-B16-LoRA models (India-only Step-6 fold models).
Methods: Grad-CAM (old), HiResCAM (faithful variant), attention rollout (transformer-native, class-agnostic).
Photos: (1) SMART-OM suspicious photos with expert masks, explained by the fold model for which they were TEST photos;
        (2) Sri Lanka OPMD/cancer photos with expert lesion polygons (never seen by any of these models), fold = index mod 5.
Per photo: maps + deletion test (remove the most important 10%,20%..100% of patches; lower curve = more faithful).
Output: mac_results_srilanka/xai/part_*.npz (resumable).   Run:  oral_env/bin/python run_xai.py   [--limit N for a quick test]"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw
from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import mac_common as mc
ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--chunk', type=int, default=100); A = ap.parse_args()
KIT, DEV = mc.KIT, mc.DEV
XD = os.path.join(mc.HERE, 'mac_results_srilanka', 'xai'); os.makedirs(XD, exist_ok=True)
STEPS = np.linspace(0, 1, 11)
# ---------------- photo list
meta = pd.read_csv(os.path.join(KIT, 'meta.csv'))
sm = meta[(meta.set == 'internal') & (meta.y_bin == 1)].copy()
sm = sm[sm.md5.apply(lambda h: os.path.exists(os.path.join(KIT, 'masks', f'{h}.png')))]
L1 = pd.DataFrame({'pid': sm.md5, 'src': 'SMART-OM', 'fold': sm.fold.astype(int), 'group': sm.group, 'cls': sm.cls})
sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv')); sl = sl[sl.y_bin == 1].reset_index(drop=True)
sl['fname'] = sl.image.map(lambda i: i if str(i).lower().endswith('.jpg') else f'{i}.jpg')
ann = json.load(open(os.path.join(mc.HERE, '..', 'data', 'raw', 'srilanka_zenodo', 'Annotation.json')))
fid = {im['id']: im['file_name'] for im in ann['images']}; POLY = {}
for a in ann['annotations']:
    POLY.setdefault(fid[a['image_id']], {1: [], 2: []})[a['category_id']].append(a['segmentation'])
sl = sl[sl.fname.map(lambda f: f in POLY and len(POLY[f][1]) > 0)].reset_index(drop=True)
L2 = pd.DataFrame({'pid': sl.fname, 'src': 'Sri Lanka', 'fold': np.arange(len(sl)) % 5, 'group': sl.group, 'cls': sl.cls})
PH = pd.concat([L1, L2], ignore_index=True)
if A.limit: PH = pd.concat([PH[PH.src == 'SMART-OM'].head(A.limit), PH[PH.src == 'Sri Lanka'].head(A.limit)], ignore_index=True)
WH = dict(zip(sl.fname, zip(sl.orig_w, sl.orig_h)))
print(f'{(PH.src=="SMART-OM").sum()} SMART-OM + {(PH.src=="Sri Lanka").sum()} Sri Lanka photos', flush=True)
def poly_mask(fname, cat):
    w, h = WH[fname]; im = Image.new('L', (224, 224), 0); d = ImageDraw.Draw(im)
    for seg in POLY[fname][cat]:
        pts = [(seg[i] * 224 / w, seg[i + 1] * 224 / h) for i in range(0, len(seg) - 1, 2)]
        if len(pts) >= 3: d.polygon(pts, fill=1)
    return np.asarray(im) > 0
def get(row):
    if row.src == 'SMART-OM':
        img = mc.load_img(row.pid); les = np.asarray(Image.open(os.path.join(KIT, 'masks', f'{row.pid}.png'))) > 0; cav = np.zeros((224, 224), bool)
    else:
        img = Image.open(os.path.join(KIT, 'img512_sl', row.pid)).convert('RGB'); les = poly_mask(row.pid, 1); cav = poly_mask(row.pid, 2)
    return img, les, cav
def vit_reshape(t, h=14, w=14):
    t = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)); return t.permute(0, 3, 1, 2)
ATT = []; CAP = [False]          # attention is recorded only while CAP[0] is True (rollout step)
def rollout(m, vit, x):
    ATT.clear(); CAP[0] = True
    with torch.no_grad(): m(x)
    CAP[0] = False
    R = torch.eye(ATT[0].shape[-1], device=ATT[0].device)
    for a in ATT:
        a = a.mean(1)[0]; a = a + torch.eye(a.shape[-1], device=a.device); a = a / a.sum(-1, keepdim=True); R = a @ R
    r = R[0, 1:].reshape(1, 1, 14, 14)
    r = torch.nn.functional.interpolate(r, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].float().cpu().numpy()
    ATT.clear()
    return (r - r.min()) / (r.max() - r.min() + 1e-8)
@torch.no_grad()
def deletion(m, x, maps):
    """for each map: delete patches in order of importance (set to 0 = channel mean after normalisation)"""
    xs = []
    for mp in maps:
        imp = torch.tensor(mp).reshape(14, 16, 14, 16).mean((1, 3)).flatten()
        order = torch.argsort(imp, descending=True)
        for f in STEPS:
            keep = torch.ones(196); keep[order[:int(round(f * 196))]] = 0
            mk = keep.reshape(1, 1, 14, 14).repeat_interleave(16, 2).repeat_interleave(16, 3).to(DEV)
            xs.append(x * mk)
    out = []
    for i in range(0, len(xs), 32): out.append(torch.sigmoid(m(torch.cat(xs[i:i + 32])).float()).squeeze(1).cpu())
    return torch.cat(out).numpy().reshape(len(maps), len(STEPS))
t0 = time.time(); rng = np.random.default_rng(0)
for k in range(5):
    sub = PH[PH.fold == k].reset_index(drop=True)
    chunks = [sub.iloc[i:i + A.chunk] for i in range(0, len(sub), A.chunk)]
    todo = [(j, c) for j, c in enumerate(chunks) if not os.path.exists(os.path.join(XD, f'part_f{k}_{j:03d}{"_test" if A.limit else ""}.npz'))]
    if not todo: print(f'fold {k}: done before, skipped'); continue
    m, tf = mc.load_lora(k); vit = m.base_model.model
    hooks = []
    for blk in vit.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda mod, i, o: ATT.append(o.detach()) if CAP[0] else None))
    cams = {'gradcam': GradCAM(model=m, target_layers=[vit.blocks[-1].norm1], reshape_transform=vit_reshape),
            'hirescam': HiResCAM(model=m, target_layers=[vit.blocks[-1].norm1], reshape_transform=vit_reshape)}
    for j, c in todo:
        rec = {'pid': [], 'src': [], 'group': [], 'cls': [], 'p': [], 'les': [], 'cav': [], 'maps': [], 'dele': []}
        for row in c.itertuples():
            img, les, cav = get(row)
            x = tf(img).unsqueeze(0).to(DEV)
            with torch.no_grad(): p = float(torch.sigmoid(m(x).float()))
            mp = [cams[n](input_tensor=x.clone().requires_grad_(True), targets=[ClassifierOutputTarget(0)])[0] for n in ['gradcam', 'hirescam']]
            mp.append(rollout(m, vit, x)); rnd = rng.random((224, 224)).astype(np.float32)
            dl = deletion(m, x, mp + [rnd])
            for kk, v in zip(['pid', 'src', 'group', 'cls', 'p'], [row.pid, row.src, row.group, row.cls, p]): rec[kk].append(v)
            rec['les'].append(np.packbits(les)); rec['cav'].append(np.packbits(cav))
            rec['maps'].append(np.stack(mp).astype(np.float16)); rec['dele'].append(dl.astype(np.float32))
        if DEV == 'mps': torch.mps.empty_cache()
        np.savez_compressed(os.path.join(XD, f'part_f{k}_{j:03d}{"_test" if A.limit else ""}.npz'), **{kk: np.array(v) for kk, v in rec.items()})
        print(f'fold {k} chunk {j + 1}/{len(chunks)} saved ({mc.stamp(t0)})', flush=True)
    for h in hooks: h.remove()
    del m, cams
print(f'finished in {mc.stamp(t0)}; results in {XD}')
