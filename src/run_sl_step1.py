"""Sri Lanka Step 1 (Mac GPU): 
 Frozen 224-px embeddings for India+Karnataka (8,251) and Sri Lanka (2,988) photos, 4 timm backbones.
Resumable: finished parts are skipped. Outputs go to mac_results_srilanka/.  Run:  oral_env/bin/python run_sl_step1.py"""
import os, time
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from PIL import Image
from mac_common import KIT, HERE, DEV, stamp
OUT = os.path.join(HERE, 'mac_results_srilanka'); os.makedirs(OUT, exist_ok=True)

t0 = time.time()
sl = pd.read_csv(os.path.join(KIT, 'meta_sl.csv'))
inn = pd.read_csv(os.path.join(KIT, 'meta.csv'))
SL_DIR, IN_DIR = os.path.join(KIT, 'img512_sl'), os.path.join(KIT, 'img512')
print(f'Device: {DEV} | Sri Lanka photos: {len(sl)} | India+Karnataka photos: {len(inn)}')

def batches(paths, tf, bs=32):
    for i in range(0, len(paths), bs):
        yield torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).to(DEV)

# ---------- B) frozen embeddings ----------
BACKBONES = {
    'DINOv2-B14': dict(name='vit_base_patch14_dinov2.lvd142m', kw=dict(img_size=224)),
    'CLIP-B16': dict(name='vit_base_patch16_clip_224.openai', kw={}),
    'ConvNeXt-B': dict(name='convnext_base.fb_in22k', kw={}),
    'EfficientNet-B0': dict(name='efficientnet_b0.ra_in1k', kw={}),
}
sets = {'srilanka': [os.path.join(SL_DIR, f'{n}.jpg') for n in sl.image],
        'internal': [os.path.join(IN_DIR, f'{h}.jpg') for h in inn.md5]}
ids = {'srilanka': sl.image.values, 'internal': inn.md5.values}
for bname, spec in BACKBONES.items():
    m = None
    for sname, paths in sets.items():
        f = os.path.join(OUT, f'sl_feats_{bname}_{sname}.npz')
        if os.path.exists(f):
            print(f'B) {bname} {sname}: already done'); continue
        if m is None:
            m = timm.create_model(spec['name'], pretrained=True, num_classes=0, **spec['kw']).to(DEV).eval()
            cfg = m.pretrained_cfg
            tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                            T.Normalize(cfg['mean'], cfg['std'])])
        out = []
        with torch.no_grad():
            for xb in batches(paths, tf):
                out.append(m(xb).float().cpu().numpy())
        np.savez_compressed(f, ids=ids[sname], X=np.concatenate(out))
        print(f'B) {bname} {sname}: {len(paths)} photos ({stamp(t0)})')
    del m
print(f'Sri Lanka Step 1 finished in {stamp(t0)}. Results in {OUT}')
