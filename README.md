# Low-Light Heart Rate Estimation from Facial Video

> Estimating heart rate non-invasively from facial video - exploring the limits of remote photoplethysmography (rPPG) under varying illumination conditions.

---

## Overview

Contact-based PPG sensors require physical devices, dedicated hardware, and continuous skin contact - making them impractical for telemedicine, driver safety monitoring, and passive clinical surveillance. This project investigates whether heart rate can be reliably estimated from ordinary facial video using rPPG deep learning models, with a specific focus on **low-light conditions**.

We evaluate **PhysNet** - a 3D CNN encoder-decoder that extracts blood volume pulse (BVP) waveforms directly from video - across three experimental phases:

| Phase | Train | Test | Description |
|-------|-------|------|-------------|
| Baseline | UBFC-rPPG | UBFC-rPPG | Normal illumination, single-dataset |
| Pretraining | UBFC-rPPG | MMPD (low-light) | Cross-dataset domain shift |
| Retraining | MMPD (low-light) | MMPD (low-light) | 5-fold cross-validation on low-light data |

---

## Key Findings

- **PhysNet trained on UBFC-rPPG performs well under normal illumination** but degrades significantly when tested on low-light MMPD data - a confirmed domain mismatch (SNR: −5.12 dB).
- **Low-light failure has three root causes:** SNR collapse (pulse signal drowns in sensor noise), difference normalization breaking down (σ becomes noise std, not signal std), and FFT locking onto wrong frequencies due to a flat power spectrum.
- **Retraining on MMPD low-light data (5-fold CV) with PhysNet improves robustness** over the pretrained baseline - confirming that domain-specific fine-tuning is necessary for reliable low-light HR estimation.
- An image enhancement pipeline using **Self-Calibrated Illumination (SCI)** based on Retinex theory was explored as a preprocessing step to recover pulse-bearing skin color dynamics. Implementation challenges around training stability and compute constraints prevented full evaluation within the project scope.

---

## Pipeline

```
Facial Video
     │
     ▼
Face Detection (Haar Cascade / MediaPipe)
     │  Crop + Resize → 72×72
     ▼
Normalization
     │  Difference Normalization + Z-Score Standardization
     ▼
PhysNet (3D CNN Encoder-Decoder)
     │  Input: [N, 6, T, 72, 72]  Output: BVP waveform [T, 1]
     ▼
Post-Processing
     │  Butterworth Filter (0.75–2.5 Hz) → FFT → Peak Frequency
     ▼
Heart Rate (BPM)
```

**Loss Function:** Negative Pearson Correlation - scale-invariant and phase-sensitive, penalizing waveform misalignment rather than absolute amplitude differences.

---

## Repository Structure

```
Low-Light-Heart-Rate-Estimation/
├── rPPG-Toolbox/
│   ├── configs/
│   │   ├── train_configs/
│   │   │   ├── UBFC/                  # PhysNet UBFC training config
│   │   │   └── MMPD_5fold/            # PhysNet 5-fold MMPD configs (folds 1–5)
│   │   └── infer_configs/             # Inference-only configs
│   ├── dataset/
│   │   └── data_loader/               # MMPD and UBFC-rPPG data loaders
│   ├── neural_methods/                # Model definitions (PhysNet, etc.)
│   ├── tools/                         # Signal visualization and analysis utilities
│   └── main.py
├── results/
│   ├── UBFC_training/                 # Plots, Bland-Altman, test outputs
│   ├── UBFC_MMPD_pretraining/         # Bland-Altman plots
│   ├── MMPD_retrained/                # Plots, Bland-Altman, test outputs
│   └── helper_functions/
│       └── print_bpm_results.py       # Script to extract and print BPM from pickles
├── data/                              # Dataset folder (not included — see below)
│   ├── MMPD/
│   └── UBFC-rPPG/
├── docs/                              # Reference papers
└── README.md
```
---

## Setup

### Prerequisites

```bash
pip install torch torchvision numpy scipy opencv-python
```

For the full rPPG-Toolbox dependency list:
```bash
pip install -r rPPG-Toolbox/requirements.txt
```

### Datasets

Datasets are **not included** in this repository due to access restrictions. Obtain them through official channels and place them as described in [`data/README.md`](data/README.md).

| Dataset | Access |
|---------|--------|
| **UBFC-rPPG** | [Request here](https://sites.google.com/view/ybenezeth/ubfcrppg) - registered users only |
| **MMPD** | [Formal written request](https://github.com/McJackTang/MMPD_rPPG_dataset) to the authors - cannot be redistributed |

### Pretrained Weights

PhysNet weights pretrained on UBFC-rPPG are available from the official [rPPG-Toolbox releases](https://github.com/ubicomplab/rPPG-Toolbox). Place them under `pretrained_models/PhysNet/`.

---

## Running Experiments

All experiments use rPPG-Toolbox's config-driven runner. From the `rPPG-Toolbox/` directory:

```bash
# UBFC baseline (train + test)
python main.py --config configs/train_configs/UBFC/UBFC_PHYSNET.yaml

# MMPD 5-fold retraining (repeat for folds 1–5)
python main.py --config configs/train_configs/MMPD_5fold/MMPD_5FOLD_PHYSNET_fold1.yaml

# Inference only (UBFC pretrained → MMPD test)
python main.py --config configs/infer_configs/MMPD_PHYSNET_infer.yaml
```

> All training was done locally on a single GPU (CUDA). No cloud compute was used.

---

## Extracting BPM from Results

After running experiments, pickle outputs are saved under `results/pickled_outputs/`. Use the helper script to print per-subject estimated vs ground-truth BPM:

```bash
python results/helper/print_bpm_results.py
```

---

## Team

Aditi Anil Pulikottil  
Ubika Singh  
Anushka Rawlani  

**Faculty Mentor:** Dr. Ruchira Naskar  
**Research Guide:** Ms. Shreya Bhadra (PhD Scholar)

---

## Acknowledgements

This project builds on the [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) by the UbiComp Lab. The image enhancement approach is based on [SCI (Self-Calibrated Illumination)](https://arxiv.org/abs/2312.15199) and the low-light rPPG methodology from:

> *An image enhancement based method for improving rPPG extraction under low-light illumination* - Biomedical Signal Processing and Control, 100 (2025) 106963.

---

## License

This repository is for academic purposes. Dataset usage is governed by the terms of each dataset's respective license. rPPG-Toolbox is used under its original license.

---

© 2025 Aditi Anil Pulikottil, Anushka Rawlani, Ubika Singh. All rights reserved. This project was developed for academic purposes at the direction of Dr. Ruchira Naskar. Unauthorized reproduction or distribution is prohibited.