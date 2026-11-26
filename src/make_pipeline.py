"""Study pipeline figure for the IEEE paper (full width, 7.16 in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})

FS, LFS = 6.6, 7.4                 # box / lane font sizes (pt)
W_IN, H_IN = 7.16, 4.25
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

EDGE, TXT = "#404040", "#1a1a1a"
LANE = ["#EEF3F9", "#F4F4F4", "#EEF3F9", "#F4F4F4"]
BAND = "#FBF6EC"

def lane(x, y, w, h, title, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec="#9E9E9E", lw=0.6))
    ax.text(x + 0.8, y + h - 0.8, title, ha="left", va="top", fontsize=LFS, fontweight="bold", color=TXT)

def box(x, y, w, h, text, dashed=False, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                fc="white", ec=EDGE, lw=0.7, ls=(0, (3, 2)) if dashed else "-"))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        # first line bold, rest normal, vertically centred as a block
        n = len(lines); lh = 1.45
        y0 = y + h / 2 + (n - 1) * lh / 2
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y0 - i * lh, ln, ha="center", va="center", fontsize=FS,
                    fontweight="bold" if i == 0 else "normal", color=TXT)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, color=TXT, linespacing=1.25)
    return dict(x=x, y=y, w=w, h=h, l=(x, y + h / 2), r=(x + w, y + h / 2),
                t=(x + w / 2, y + h), b=(x + w / 2, y))

def arrow(p, q, dashed=False, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.75, color=EDGE,
                                 ls=(0, (3, 2)) if dashed else "-", connectionstyle=conn,
                                 shrinkA=0, shrinkB=0))

# ---- layout ---------------------------------------------------------------
M, G = 0.8, 1.6                    # outer margin, gap between lanes
LW = (100 - 2 * M - 3 * G) / 4     # lane width
X = [M + i * (LW + G) for i in range(4)]
TOP_Y, TOP_H = 21.5, 37.7          # upper lanes
BOT_Y, BOT_H = 0.8, 19.0           # bottom bands
PAD = 1.0; BW = LW - 2 * PAD
titles = ["1 · Public data", "2 · Curation and split", "3 · Models", "4 · Decision and assessment"]
for x, t, c in zip(X, titles, LANE):
    lane(x, TOP_Y, LW, TOP_H, t, c)
lane(X[0], BOT_Y, 2 * LW + G, BOT_H, "6 · Habit-from-photo (mouths clear of lesions)", BAND)
lane(X[2], BOT_Y, 2 * LW + G, BOT_H, "5 · Explainability", BAND)
bx = [x + PAD for x in X]

# 1 data
d1 = box(bx[0], 48.5, BW, 7.0, "CODE v5 + SMART-OM\nsmartphone intraoral photos")
d2 = box(bx[0], 38.5, BW, 7.0, "Habit questionnaire\ntobacco · smokeless ·\nareca nut · alcohol")
d4 = box(bx[0], 24.0, BW, 7.5, "Karnataka oral images\n323 photos\nexternal test only", dashed=True)
# 2 curation
c1 = box(bx[1], 49.0, BW, 6.5, "De-duplication (MD5)\nEXIF-upright · 512 px")
c2 = box(bx[1], 40.5, BW, 6.5, "7,928 unique photos · 1,009 people\n158 OPMD/cancer photos (78 people)")
c3 = box(bx[1], 30.5, BW, 8.0, "Frozen patient-level split\n5 folds, stratified by\ndiagnosis × dataset × tobacco")
c4 = box(bx[1], 23.0, BW, 5.5, "Leakage check\nno person in two folds")
# 3 models
m1 = box(bx[2], 47.0, BW, 8.5, "Frozen foundation models\n+ linear probe: DINOv2, CLIP,\nBiomedCLIP, ConvNeXt,\nEfficientNet")
m2 = box(bx[2], 36.5, BW, 8.0, "Fine-tuning\nConvNeXt-B (full)\nCLIP ViT-B/16 + LoRA")
m3 = box(bx[2], 26.5, BW, 7.5, "Patient score\n(all photos of a person)\nmax · mean · attention-MIL")
# 4 decision
e1 = box(bx[3], 48.5, BW, 7.0, "Calibration & threshold\nPlatt scaling\n≥ 90 % sensitivity")
e2 = box(bx[3], 38.5, BW, 7.5, "Image + habit fusion\nlogistic / boosting\nSHAP explanations")
e3 = box(bx[3], 23.0, BW, 12.5, "Evaluation\ninternal 5-fold CV\n(photo & patient level)\nexternal Karnataka test\npatient-bootstrap 95 % CI")
# bottom bands
BBW = 2 * LW + G - 2 * PAD
h1 = box(bx[0], 9.8, BBW, 6.0, "Matched cohort: 65 tobacco users vs 181 non-users\nexact sex · dataset · clinical finding; age ± 5 y")
h2 = box(bx[0], 2.2, BBW, 5.8, "Image-only CV · within-set permutation test\nunmatched contrast · resolution 224 → 518 px")
x1 = box(bx[2], 9.8, BBW, 6.0, "Grad-CAM vs 144 expert lesion outlines\npointing game · energy · IoU vs centre prior")
x2 = box(bx[2], 2.2, BBW, 5.8, "Attention-MIL check\ndoes the top-weighted photo show the lesion?")

def elbow(p, q, dashed=False):
    """horizontal - vertical - horizontal connector ending in an arrow"""
    mx = (p[0] + q[0]) / 2
    ls = (0, (3, 2)) if dashed else "-"
    ax.plot([p[0], mx, mx], [p[1], p[1], q[1]], color=EDGE, lw=0.75, ls=ls, solid_capstyle="butt")
    arrow((mx, q[1]), q, dashed)

# ---- arrows ---------------------------------------------------------------
arrow(d1["r"], (c1["x"], d1["r"][1]))
for a, b in [(c1, c2), (c2, c3), (c3, c4), (m2, m3), (e1, e2), (e2, e3)]:
    arrow(a["b"], b["t"])
elbow(c3["r"], m1["l"])
elbow(c3["r"], m2["l"])
arrow(m1["r"], (e1["x"], m1["r"][1]))
elbow(m3["r"], (e1["x"], e1["y"] + 1.2))
# habits -> fusion: run along the top gap above lane 2/3 boxes would clutter; label instead
ax.text(e2["x"] + 0.4, e2["y"] + e2["h"] + 0.35, "habits from 1", fontsize=5.6, style="italic", color="#555555")
ax.plot([d4["b"][0]]*2, [d4["b"][1], 20.6], color=EDGE, lw=0.75, ls=(0, (3, 2)))
ax.plot([d4["b"][0], e3["x"] + e3["w"] / 2], [20.6, 20.6], color=EDGE, lw=0.75, ls=(0, (3, 2)))
arrow((e3["x"] + e3["w"] / 2, 20.6), e3["b"], dashed=True)
for out in ("pipeline.pdf", "pipeline.png"):
    fig.savefig(out, dpi=300)
print("saved")
