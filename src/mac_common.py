"""Shared helpers for the Mac (Apple GPU) runs. Rebuilds the CLIP-B16-LoRA fold models exactly as in Step 6."""
import os, time
import numpy as np, pandas as pd, torch, timm
import torchvision.transforms as T
from PIL import Image
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, 'mac_kit')                       # unzipped kit (photos, models, split)
OUT = os.path.join(HERE, 'mac_results'); os.makedirs(OUT, exist_ok=True)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
BIC = T.InterpolationMode.BICUBIC
N_FOLDS = 5


def load_meta():
    m = pd.read_csv(os.path.join(KIT, 'meta.csv'))
    assert {'md5', 'fold', 'y_bin', 'set'} <= set(m.columns), 'meta.csv is incomplete'
    return m


def load_lora(k):
    """Step 6 model: CLIP ViT-B/16 (OpenAI, QuickGELU) + LoRA adapters + trained head, fold k."""
    base = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=1)
    cfg = base.pretrained_cfg
    m = PeftModel.from_pretrained(base, os.path.join(KIT, 'step6', f'CLIP-B16-LoRA_fold{k}_adapter'))
    tf = T.Compose([T.Resize((224, 224), interpolation=BIC), T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])
    return m.to(DEV).eval(), tf


def load_img(md5):
    return Image.open(os.path.join(KIT, 'img512', f'{md5}.jpg')).convert('RGB')


@torch.no_grad()
def predict(m, tf, md5s, bs=32):
    """probability, averaged with a horizontally flipped copy (same TTA as Step 6)"""
    out = []
    for i in range(0, len(md5s), bs):
        xb = torch.stack([tf(load_img(h)) for h in md5s[i:i + bs]]).to(DEV)
        lo = (m(xb).float() + m(torch.flip(xb, [3])).float()) / 2
        out.append(torch.sigmoid(lo).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def stamp(t0):
    return f'{(time.time() - t0) / 60:.1f} min'
