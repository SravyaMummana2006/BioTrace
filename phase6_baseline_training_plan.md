# BIOTrace Phase 6 — Fast-Track Baseline Training Specification & Readiness Report

**Author:** Senior Computer Vision Engineer & Deepfake Detection Researcher  
**Status:** Prepared & Ready for Execution  
**Dataset:** HiDF Protocol C Derived Frames ($224 \times 224$ pixels)  
**Task:** Binary Video Deepfake Classification (Real vs. Fake)  

---

## 1. Executive Summary & Recommended Architecture

Following completion of Phase 5 frame extraction, the dataset is primed for immediate baseline deepfake classification. To prioritize implementation velocity, scientific rigor, and zero partition leakage, we recommend an **EfficientNet-B0** (or **ResNet18**) transfer learning baseline trained under PyTorch.

### Recommended Baseline Configuration
| Component | Recommendation | Scientific Rationale |
| :--- | :--- | :--- |
| **Framework** | PyTorch 2.10+ / Torchvision | Native ecosystem, standard benchmarking library, robust dataloading. |
| **Backbone Model** | `efficientnet_b0` (or `resnet18`) | 5.3M parameters (EfficientNet-B0) / 11.7M (ResNet18). High throughput, low memory footprint, excellent feature representations for subtle blending/synthesis artifacts. |
| **Input Resolution** | $224 \times 224 \times 3$ RGB | Matches pre-extracted frame dimensions exactly; eliminates image resizing latency during data loading. |
| **Pretrained Weights** | ImageNet-1K (`DEFAULT`) | Rapid convergence, pre-learned low-level edge/texture filters. |
| **Objective Function** | Binary Cross Entropy with Logits (`BCEWithLogitsLoss`) | Standard numerically stable binary classification objective. |
| **Optimization** | AdamW ($\text{lr}=10^{-4}$, weight decay $10^{-2}$) | Robust generalization, decoupled weight decay prevents overfitting. |
| **Batch Size** | 64 (GPU) or 32 (CPU) | Balanced gradient variance and hardware memory utilization. |
| **Frame-to-Video Aggregation** | Mean logit pooling across the 10 frames of each video clip | Evaluates clip-level authenticity without temporal recurrent overhead. |

---

## 2. Partition & Class Balance Alignment

The model will be trained, validated, and evaluated strictly on the frozen Protocol C partitions:

| Split | Videos (Real / Fake) | Extracted Frames (Real / Fake) | Class Ratio | Identity Leakage |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | $3,053$ real / $3,053$ fake ($6,106$ total) | $30,530$ real / $30,530$ fake ($61,060$ total) | Exactly $1.000$ (50% / 50%) | **0.0%** (Verified) |
| **Validation** | $654$ real / $654$ fake ($1,308$ total) | $6,540$ real / $6,540$ fake ($13,080$ total) | Exactly $1.000$ (50% / 50%) | **0.0%** (Verified) |
| **Test** | $654$ real / $654$ fake ($1,308$ total) | $6,540$ real / $6,540$ fake ($13,080$ total) | Exactly $1.000$ (50% / 50%) | **0.0%** (Verified) |

---

## 3. Data Augmentation & Normalization Pipeline

### 3.1 Training Pipeline (Lightweight, Scientifically Sound)
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
*Note: Random horizontal flipping is safe for face deepfake detection as bilateral facial symmetry is preserved while doubling variation.*

### 3.2 Validation & Test Pipeline (Deterministic)
```python
eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 4. Evaluation Metrics & Scientific Rigor

1. **Primary Metrics:**
   - **Video-level AUC-ROC:** Area under the Receiver Operating Characteristic curve.
   - **Video-level Equal Error Rate (EER):** Threshold where false positive rate equals false negative rate.
   - **Accuracy & F1-Score:** At standard threshold ($0.5$) and calibrated threshold.
2. **Early Stopping:**
   - Monitored on validation video-level AUC-ROC with patience = 3 epochs.
3. **Checkpoint Saving:**
   - Best checkpoint saved to `models/checkpoints/efficientnet_b0_best_val_auc.pt`.
   - Final checkpoint saved to `models/checkpoints/efficientnet_b0_final.pt`.
4. **Reproducibility:**
   - Fixed random seeds: `seed = 42` for PyTorch, NumPy, Python `random`.
   - Deterministic CuDNN flags (`torch.backends.cudnn.deterministic = True`).

---

## 5. Execution Readiness

Phase 6 script template will be placed in `scripts/train_baseline.py`. Training can be initiated immediately upon completion of Phase 5 audit.
