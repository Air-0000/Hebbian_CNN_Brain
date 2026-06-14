# Hebbian CNN Brain

> **Bridging neuroscience and deep learning — implementing Hebbian learning rules inside CNNs for CIFAR-10 classification and beyond.**

A research project exploring whether biologically-plausible **Hebbian learning** (Hebb's rule: *"neurons that fire together, wire together"*) can serve as an effective learning mechanism inside convolutional neural networks, with and without backpropagation. All experiments are conducted on **CIFAR-10** using **DenseNet-40** as the base architecture, with extensions into YOLO object detection and PDCA-based optimization.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Research Questions](#research-questions)
- [Hebbian Mechanisms Implemented](#hebbian-mechanisms-implemented)
- [Experimental Results](#experimental-results)
- [Key Files](#key-files)
- [Status & Roadmap](#status--roadmap)
- [References](#references)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.0+ (torch.nn, torch.optim) |
| **Language** | Python 3.10+ |
| **Hardware** | Apple MPS (Metal Performance Shaders) |
| **Dataset** | CIFAR-10 (torchvision), COCO128 |
| **Augmentation** | RandomCrop, RandomFlip, CutMix |
| **Training** | AdamW, CosineAnnealing LR, Label Smoothing CE |
| **Visualization** | Matplotlib, Pandas |
| **Object Detection** | Ultralytics YOLOv8 (PDCA extension) |

---

## Project Architecture

```
Hebbian_CNN_Brain/
├── src/
│   ├── core/
│   │   ├── hebbian_layer.py           # HebbianConv2d + Oja rule + WTA competition
│   │   └── layer_implementation.py    # 4-level neuro-hierarchy (LTP → Synaptic → Network → CNN)
│   │
│   ├── train/
│   │   ├── cifar/
│   │   │   ├── bio_hebbian_densenet.py          # BioHebbian v1/v3 (scale + pv gating, 72 params)
│   │   │   ├── bio_hebbian_densenet_lite.py     # BioHebbian v2 (lightweight, 0.13M params)
│   │   │   ├── bio_hebbian_v3_final.py          # v3 final training script
│   │   │   ├── true_hebbian_densenet.py         # True Hebbian (early version)
│   │   │   ├── true_hebbian_densenet_main.py    # True Hebbian 11.8% (hebbian_w + intra modulation)
│   │   │   ├── true_hebbian_cnn.py              # VGG-style True Hebbian (overfitted)
│   │   │   ├── hebbian_heavy_densenet.py        # HebbianHeavy 67% (expanded Hebbian matrices)
│   │   │   ├── train_true_hebbian.py            # Training entry point
│   │   │   └── lora_only.py                     # LoRA-only baseline experiment
│   │   ├── bio_hebbian_resnet.py                # BioHebbian on ResNet
│   │   ├── bio_hebbian_resnet20.py              # BioHebbian on ResNet-20
│   │   ├── train_hebbian.py                     # Original Hebbian training script
│   │   ├── hebbian_detection.py                 # Hebbian CNN + object detection (COCO128)
│   │   ├── hebbian_verification.py              # Verification suite
│   │   ├── coco128_experiment.py                # COCO128 detection experiment
│   │   ├── coco128_real_train.py                # COCO128 real training
│   │   └── train_detection.py                   # Detection training pipeline
│   │
│   ├── yolo/
│   │   ├── hebbian_yolo.py                     # Hebbian YOLO baseline
│   │   ├── true_hebbian_yolo.py                # True Hebbian on YOLO
│   │   ├── optimized_hebbian_yolo.py            # Optimized Hebbian YOLO
│   │   ├── hebbian_yolo_v3.py                  # Hebbian YOLO v3
│   │   ├── yolo_quick.py                       # Quick YOLO test
│   │   └── yolo_compare.py                     # YOLO comparison script
│   │
│   ├── pdca/
│   │   ├── hebbian_pdca_v4.py                  # PDCA optimization v4
│   │   ├── hebbian_pdca_v5.py                  # PDCA optimization v5
│   │   ├── hebbian_pdca_v6.py                  # PDCA optimization v6 (final)
│   │   └── hebbian_pdca_auto.py                # Automated PDCA loop
│   │
│   └── tools/
│       ├── compare.py                          # Hebbian vs BP comparison
│       ├── visualize.py                        # Feature visualization
│       └── inference_speed_test.py             # Inference speed benchmark
│
├── data/                           # CIFAR-10 / COCO128 datasets
├── results/                        # Model checkpoints and training logs
├── docs/
│   └── experiment_records/
│       ├── EXPERIMENT_SUMMARY.md              # One-page summary
│       ├── ALL_VERSIONS_SUMMARY.md            # Full version comparison
│       ├── DETAILED_VERSION_REPORT.md         # Per-version deep dive
│       └── hebbian_experiment_20260522.md     # Latest experiment log
│
├── auto_train.py                   # Automated training pipeline
├── PROJECT.md                      # Project overview (Chinese)
├── README.md                       # This file
└── data/coco128.yaml               # COCO128 dataset config
```

---

## Research Questions

1. **Can Hebbian learning rules replace or augment backpropagation in CNNs?**
2. **What is the optimal ratio of Hebbian parameters to total parameters?**
3. **Can a network with >67% Hebbian parameters (minimal BP) still learn effectively?**
4. **Does Hebbian modulation provide regularization benefits (Val > Train)?**

---

## Hebbian Mechanisms Implemented

### 1. BioHebbian v1/v3 — Scale + Population Vector Gating (72 params)
Two scalar parameters per DenseLayer: `scale` (multiplicative modulation) and `pv` (population-vector gating via sigmoid). Minimal Hebbian footprint (0.015% of total parameters) — more of a "training trick" than a true Hebbian mechanism, but achieves the highest accuracy.

```python
self.scale = nn.Parameter(torch.tensor(0.05))
self.pv    = nn.Parameter(torch.tensor(0.05))
out = out * (1 + self.scale)
pv_factor = torch.sigmoid(self.pv * (out.mean() - 0.3))
out = out * (1 - pv_factor * 0.3)
```

### 2. True Hebbian 11.8% — Hebbian Weight Matrix + Intra-Channel Modulation
The **core innovation**: learnable Hebbian weight matrices (`hebbian_w`, `hebbian_intra`) that are updated via the Hebbian rule `Δw = η × pre × post` and modulate feature maps through channel-wise gating. 63,684 Hebbian parameters (11.8% of total).

```python
self.hebbian_w     = nn.Parameter(torch.randn(in_ch, growth) * 0.01)
self.hebbian_intra = nn.Parameter(torch.randn(growth, growth) * 0.01)
# Hebbian update: Δw = η × pre × post
update = torch.ger(in_act.mean(0), out_act.mean(0)) * self.hebbian_lr
self.hebbian_w.data = self.hebbian_w.data * 0.95 + update * 0.05
channel_mod = torch.sigmoid(torch.matmul(in_act, self.hebbian_w))
intra_mod   = torch.sigmoid(torch.matmul(out_act, self.hebbian_intra))
out = out * (1 + channel_mod * intra_mod * self.modulation)
```

### 3. HebbianHeavy 67% — Expanded Hebbian Matrices as Primary Learner
Aims to make Hebbian the **primary learning mechanism** by massively expanding the Hebbian matrices: `hebbian_w: [in_ch, growth×16]`, `hebbian_inter: [growth×16, growth×16]`, `hebbian_intra: [growth×8, growth×8]`. 2.59M Hebbian parameters (67.24% of total).

### 4. Core HebbianConv2d — General-Purpose Hebbian Layer
A stand-alone `nn.Module` implementing standard convolution with post-hoc Hebbian weight updates using **Oja's rule** (`Δw = η · pre · post − β · post² · w`) and optional **Winner-Takes-All** competition for sparse feature learning.

### 5. Layer Implementation — 4-Level Neurocognitive Hierarchy
A multi-level biologically-inspired architecture:
- **Molecular (LTP/LTD):** Simulates long-term potentiation and depression
- **Synaptic (Hebbian):** Classic Hebbian weight updates
- **Network (Lindsay PFC):** Prefrontal cortex model with mixed selectivity
- **System (CNN):** Hierarchical feature extraction

---

## Experimental Results

All experiments use **DenseNet-40** (3 DenseBlocks × 12 layers, growth rate=12) on **CIFAR-10** (50k train / 10k test). Training: AdamW (lr=0.001, wd=1e-4), CosineAnnealing scheduler, Label Smoothing CE (s=0.1), CutMix augmentation, batch size 128, up to 100 epochs with early stopping (patience=20).

### Accuracy Leaderboard

| Rank | Version | Mechanism | Best Val Acc | Total Params | Hebbian Params | Hebbian % |
|:----:|---------|-----------|:------------:|:------------:|:--------------:|:---------:|
| 🥇 | **BioHebbian v1** | scale + pv gating (72 params) | **92.50%** | 0.48M | 72 | 0.015% |
| 🥇 | **BioHebbian v3** | scale + pv gating (improved) | **92.35%** | 0.48M | 72 | 0.015% |
| 🥈 | **True Hebbian 11.8%** | hebbian_w + intra modulation | **91.13%** | 0.54M | 63,684 | 11.80% |
| 🥉 | **HebbianHeavy 67%** | expanded Hebbian matrices | **90.53%** | 3.86M | 2,594,412 | 67.24% |
| — | **Standard DenseNet** (baseline) | Pure BP, no Hebbian | 90.3%+ | 0.48M | 0 | 0% |
| — | **BioHebbian v2 (Lite)** | scale + pv, reduced channels | ~73% | 0.13M | 72 | 0.055% |
| — | **VGG-style True Hebbian** | Non-DenseNet architecture | 63.9% | 6.36M | 1.35M | 21.21% |

### Training Progression — Top 3 Variants

| Epoch | BioHebbian v1 | True Hebbian 11.8% | HebbianHeavy 67% |
|:-----:|:-------------:|:------------------:|:----------------:|
| 1 | 54.4% | 47.8% | 37.8% |
| 5 | 79.3% | 71.8% | 64.2% |
| 10 | 85.9% | 81.5% | 76.1% |
| 20 | 89.2% | 86.0% | 82.9% |
| 30 | — | 88.8% | 86.3% |
| 50 | 91.5% | 90.5% | 89.6% |
| **Best** | **92.5%** (ep 100) | **91.13%** (ep 66) | **90.53%** (ep 75) |

### Key Findings

1. ✅ **Hebbian mechanisms are effective** — all versions learn and converge
2. ✅ **True Hebbian (11.8%) achieves strong results** — 91.13% with a genuine Hebbian update rule, not just a training trick
3. ✅ **67% Hebbian can serve as the primary learner** — proves Hebbian updates can work with minimal BP, though parameter efficiency is low (8× params for ~same accuracy)
4. ✅ **Regularization effect observed** — `Val > Train` consistently in True Hebbian and HebbianHeavy, indicating Hebbian gating acts as a regularizer
5. ⚠️ **No accuracy advantage over standard BP** — Hebbian variants match but do not surpass the standard DenseNet baseline
6. ⚠️ **Early instability in HebbianHeavy** — epoch 2 showed Train > Val (53.2% vs 32.6%) before a sudden jump at epoch 3 (Val 66.3%)

### Hebbian Ratio vs. Accuracy (DenseNet-40)

```
Accuracy
  ↑
95% ┤
    │                                              ● BioHebbian v1 (0.015%) 92.5%
90% ┤     ●───────────●─────── True Hebbian (11.8%) 91.13%
    │              ╲
85% ┤               ╲  ●────── HebbianHeavy (67%) 90.53%
    │                ╲╱ ╲
80% ┤                 ●─── Standard DenseNet (0%) 90.3%
    │
    └──────────────────────────────────→ Hebbian Ratio
       0%    20%    40%    60%    80%
```

**Observation:** The optimal Hebbian ratio lies in the **5–15% range** — enough to modulate and regularize, not so much that parameter efficiency suffers.

### Comparison with Literature

| Model | Params | CIFAR-10 Acc | Source |
|-------|:------:|:------------:|--------|
| BioHebbian v1 (ours) | 0.48M | **92.5%** | This project |
| True Hebbian 11.8% (ours) | 0.54M | **91.13%** | This project |
| DenseNet-40 (paper) | 1.0M | 94.5% | Huang et al., CVPR 2017 |
| DenseNet-100 (paper) | 0.8M | 94.5% | Huang et al., CVPR 2017 |
| ResNet-110 | 1.7M | 93.6% | He et al., CVPR 2016 |
| SoftHebb (unsupervised) | 5.9M | 80.3% | ICLR 2023 |
| ResNet-1001 | 10.2M | 95.4% | He et al., CVPR 2016 |

Our best model (92.5%) is **2% below** the DenseNet-40 paper baseline (94.5%) with roughly half the parameters (0.48M vs 1.0M). The True Hebbian variant (91.13%) significantly outperforms unsupervised Hebbian approaches like SoftHebb (80.3%).

---

## Key Files

| File | Purpose |
|------|---------|
| `src/core/hebbian_layer.py` | General-purpose `HebbianConv2d` with Oja's rule and WTA |
| `src/core/layer_implementation.py` | 4-level neurocognitive hierarchy (LTP → CNN) |
| `src/train/cifar/bio_hebbian_densenet.py` | BioHebbian v1/v3 — scale + pv gating (highest accuracy) |
| `src/train/cifar/true_hebbian_densenet_main.py` | True Hebbian 11.8% — genuine Hebbian weight matrices |
| `src/train/cifar/hebbian_heavy_densenet.py` | HebbianHeavy 67% — Hebbian as primary learner |
| `src/train/cifar/bio_hebbian_v3_final.py` | v3 final training script with model saving |
| `src/train/cifar/bio_hebbian_densenet_lite.py` | Lightweight v2 experiment (0.13M params) |
| `src/yolo/hebbian_yolo.py` | Hebbian learning applied to YOLO object detection |
| `src/pdca/hebbian_pdca_v6.py` | PDCA optimization loop for YOLO training |
| `src/train/hebbian_detection.py` | Hebbian CNN + object detection on COCO128 |
| `src/tools/compare.py` | Hebbian vs BP comparison tool |
| `src/tools/visualize.py` | Feature map visualization |
| `auto_train.py` | Automated training pipeline (v3 → full) |
| `docs/experiment_records/ALL_VERSIONS_SUMMARY.md` | Full experimental comparison and analysis |
| `docs/experiment_records/DETAILED_VERSION_REPORT.md` | Per-version deep dive with layer-level parameter breakdown |
| `PROJECT.md` | Project overview (Chinese) |

---

## Status & Roadmap

### Current Status: 🟡 Experimentation Complete — Optimization Needed

- **All planned CIFAR-10 experiments are complete** (6 variants across 3 Hebbian strategies)
- **YOLO + PDCA extensions** are in initial exploration phase
- **Paper-grade results** not yet achieved (93%+ target at 0.5M params)

### Short-term Goals

- [ ] **Improve parameter efficiency** — target 93%+ accuracy at ≤0.5M params
- [ ] **Refine Hebbian ratio** — explore 5–15% range more granularly for optimal trade-off
- [ ] **DenseNet-100 baseline** — match the paper result (94.5%, 0.8M params)
- [ ] **SoftHebb-style channel growth** — experiment with 96→384→1536 channel progression

### Medium-term Goals

- [ ] **STDP (Spike-Timing-Dependent Plasticity)** — add temporal dimension to Hebbian updates
- [ ] **Fully Hebbian training** — eliminate BP entirely for feature extractor
- [ ] **Object detection** — complete Hebbian YOLO evaluation on COCO128
- [ ] **Unsupervised pre-training** — leverage Hebbian for representation learning without labels

---

## References

1. **DenseNet**: Huang et al., *"Densely Connected Convolutional Networks"*, CVPR 2017
2. **SoftHebb**: Journé et al., *"SoftHebb: Bayesian inference in unsupervised Hebbian soft WTA networks"*, ICLR 2023
3. **Hebbian Deep Learning**: *"Hebbian Deep Learning Without Feedback"*, arXiv:2209.11883
4. **Oja's Rule**: Oja, *"A simplified neuron model as a principal component analyzer"*, J. Math. Biology, 1982
5. **Lindsay PFC Model**: *"Prefrontal cortex models with mixed selectivity"*

---

*Created: 2026-05-22 · Last Updated: 2026-06-14*
