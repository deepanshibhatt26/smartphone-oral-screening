# Cross-country smartphone oral cancer screening (India and Sri Lanka)

Code for the paper *Cross-Country Generalisation of Smartphone-Based Oral Cancer Screening: India and Sri Lanka* (Deepanshi Bhatt, Delhi Technological University).

We test whether a model that flags oral cancer and oral potentially malignant disorders (OPMD) in smartphone mouth photographs still works when it moves between India and Sri Lanka, find out why it fails, and test simple fixes (training on both countries, per-country thresholds, local recalibration, habit data).

**No photographs, patient metadata, extracted features or model weights are in this repository.** The datasets have their own licences and one is released on request only. Download each one from its source (see [DATA.md](DATA.md)).

## Main results (patient-level AUC, 95% bootstrap CI)

| Training → test | Patient AUC | Photo AUC |
|---|---|---|
| India → Sri Lanka | 0.707 (0.668–0.744) | 0.654 |
| Sri Lanka → India | 0.700 (0.634–0.756) | 0.783 |
| Sri Lanka → Karnataka (external) | – | 0.703 |
| Both (5-fold CV) → India | 0.959 (0.933–0.979) | 0.977 |
| Both (5-fold CV) → Sri Lanka | 0.834 (0.802–0.861) | 0.807 |
| Both → Karnataka (external) | – | 0.819 |

Karnataka has no patient identifiers, so only photograph-level AUC is reported. Full tables are in `results/tables/`.

## Repository layout

```
src/            all scripts (run order below)
notebooks/      early Colab notebook (India-only stage)
splits/         patient-level 5-fold assignments used in every experiment
results/tables  final metrics as CSV
results/figures figures from the paper that contain no patient photographs
env/            package versions
```

## Method in brief

- CLIP ViT-B/16 fine-tuned with LoRA (r = 16) at 224 × 224 px, ensembled with frozen ConvNeXt-B, DINOv2-B/14 and CLIP + logistic regression; ensemble weights and the patient score are chosen on validation patients only.
- Positive = OPMD or cancer; negative = healthy or benign. Photos split by patient into 5 folds; validation fold = (k + 1) mod 5.
- Experiments: (1) India (+ Karnataka) → Sri Lanka, (2) Sri Lanka → India and Karnataka, (3) both countries, 5-fold CV, Karnataka external.
- Platt calibration, prior-shift (EM) correction, 50-patient local recalibration, thresholds for ≥ 90 % sensitivity (shared vs per country).
- Habit fusion (smoking, chewing, alcohol, age, sex) with logistic regression and TabPFN.
- HiResCAM heatmaps compared with expert lesion outlines.

## How to run

Tested on a MacBook (Apple silicon, PyTorch MPS), Python 3.11; TabPFN in a separate Python 3.13 environment.

```bash
cd src
bash setup_mac.sh                      # creates oral_env (see env/ for exact versions)
```

Prepare the images (de-duplicated, resized to 512 px) and metadata into `src/mac_kit/` following `splits/` and [DATA.md](DATA.md). Then, from `src/`:

| Step | Command | Output |
|---|---|---|
| 1 | `oral_env/bin/python run_sl_step1.py` | frozen backbone features |
| 2 | `oral_env/bin/python frozen_sl.py` | frozen-backbone baselines, exp 1–3 |
| 3 | `oral_env/bin/python run_sl_train.py --exp 1` (then 2, 3) | CLIP-LoRA models and predictions |
| 4 | `oral_env/bin/python analyze_sl.py` | AUC, calibration, thresholds for LoRA alone |
| 5 | `oral_env/bin/python ens_step1_frozen_preds.py <backbone> <exp>` | frozen predictions on the LoRA splits |
| 6 | `oral_env/bin/python ens_step2_combine.py` | final ensemble (Table II) |
| 7 | `tabpfn_env/bin/python fusion_all.py` | habit fusion, LR and TabPFN |
| 8 | `oral_env/bin/python analysis_AB_classes.py` | sub-task AUCs, Fig. 2 |
| 9 | `oral_env/bin/python analysis_C_calibration.py` | calibration and thresholds |
| 10 | `oral_env/bin/python analysis_D_shortcut.py` | country shortcut check |
| 11 | `oral_env/bin/python run_xai.py` → `analyze_xai.py --stage1` → `analyze_xai.py` | HiResCAM localisation |

`run_step9b.py`, `analyze_9b.py` and the notebook belong to the earlier India-only stage, whose LoRA models are used for the heatmaps. `fusion_sl.py` is the first fusion version for the two-country experiment.

## Citation

Paper under review. Please cite the original datasets listed in [DATA.md](DATA.md).

## Licence

Code: MIT (see LICENSE). The datasets keep their own licences.
