"""Study pipeline figure for the IEEE paper (full width, 7.16 in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "pdf.fonttype": 42, "ps.fonttype": 42})

FS, LFS = 6.6, 7.4                 # box / lane font sizes (pt)
W_IN, H_IN = 7.16, 3.75
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

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


M, G = 0.8, 1.6
LW = (100 - 2 * M - 3 * G) / 4
X = [M + i * (LW + G) for i in range(4)]
TOP_Y, TOP_H = 17.5, 37.7
BOT_Y, BOT_H = 0.8, 15.2
PAD = 1.0; BW = LW - 2 * PAD
titles = ["1 · Public data", "2 · Curation and grouping", "3 · Experiments and modelling", "4 · Decision-making"]
for x, t, c in zip(X, titles, LANE):
    lane(x, TOP_Y, LW, TOP_H, t, c)
lane(X[0], BOT_Y, 2 * LW + G, BOT_H, "5 · Barriers to generalisability", BAND)
lane(X[2], BOT_Y, 2 * LW + G, BOT_H, "6 · Model explainability", BAND)
bx = [x + PAD for x in X]
# 1 data
d1 = box(bx[0], 44.5, BW, 7.0, "India: CODE and SMART-OM\n7,928 images · 1,009 patients")
d2 = box(bx[0], 35.0, BW, 7.0, "Sri Lanka (Zenodo)\n2,988 images · 712 patients\nlesion and mouth markings")
d3 = box(bx[0], 27.0, BW, 5.5, "Behaviour data\nSMART-OM · Sri Lanka")
d4 = box(bx[0], 19.0, BW, 6.0, "Karnataka (India)\n323 lesion images\nexternal test only", dashed=True)
# 2 curation
c1 = box(bx[1], 45.0, BW, 6.5, "De-duplication\nMD5 and difference hashing")
c2 = box(bx[1], 35.5, BW, 7.0, "5 patient-level folds\nstratified by diagnosis ×\norigin × tobacco/betel")
c3 = box(bx[1], 27.5, BW, 5.5, "Leakage check\nno patient in >1 fold")
c4 = box(bx[1], 19.0, BW, 6.5, "Class definitions\npositive = OPMD or cancer\nnegative = healthy or benign")
# 3 experiments + model
m1 = box(bx[2], 42.0, BW, 9.5, "Three experiments\nIndia → Sri Lanka\nSri Lanka → India (+ Karnataka)\nbi-national, 5-fold CV")
m2 = box(bx[2], 30.5, BW, 9.0, "Model (224 px)\nCLIP ViT-B/16 and LoRA\nplus frozen ConvNeXt, DINOv2, CLIP\nweights chosen on validation")
m3 = box(bx[2], 20.5, BW, 7.5, "Patient score\nmean of a patient's\ntwo highest image scores")
# 4 decision
e1 = box(bx[3], 44.0, BW, 7.5, "Calibration\nPlatt scaling · prior shift\n50 local labelled patients")
e2 = box(bx[3], 34.0, BW, 7.5, "Decision threshold\n≥ 90 % sensitivity\nglobal or country-specific")
e3 = box(bx[3], 26.0, BW, 5.5, "Behaviour and image fusion\nlogistic regression · TabPFN")
e4 = box(bx[3], 19.0, BW, 5.0, "Evaluation\npatient and image AUC (95 % CI)")
BBW = 2 * LW + G - 2 * PAD
h1 = box(bx[0], 8.6, BBW, 5.0, "Sub-task AUCs: no benign patients · lesions only · cancer vs non-cancer")
h2 = box(bx[0], 2.0, BBW, 5.0, "Country signature in the images (shortcut check)")
x1 = box(bx[2], 8.6, BBW, 5.0, "HiResCAM heatmaps of the CLIP-LoRA model")
x2 = box(bx[2], 2.0, BBW, 5.0, "Heatmap energy inside expert markings: SMART-OM 144 · Sri Lanka 1,523 images")

def elbow(p, q, dashed=False):
    mx = (p[0] + q[0]) / 2; ls = (0, (3, 2)) if dashed else "-"
    ax.plot([p[0], mx, mx], [p[1], p[1], q[1]], color=EDGE, lw=0.75, ls=ls, solid_capstyle="butt")
    arrow((mx, q[1]), q, dashed)
for a, b in [(c1, c2), (c2, c3), (m1, m2), (m2, m3), (e1, e2), (e2, e3), (e3, e4)]:
    arrow(a["b"], b["t"])
elbow(d1["r"], c1["l"]); elbow(d2["r"], c1["l"])
elbow(c3["r"], m1["l"])
elbow(m3["r"], e1["l"])
ax.text(e3["x"] + 0.3, e3["y"] + e3["h"] + 0.35, "behaviour from 1", fontsize=5.4, style="italic", color="#555555")
for out in ("pipeline_v7.pdf", "pipeline_v7.png"):
    fig.savefig(out, dpi=300)
print("saved")
