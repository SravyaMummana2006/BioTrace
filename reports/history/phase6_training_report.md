# BIOTrace Phase 6 — Fast-Track Baseline Deepfake Detection Model Training Report

**Date:** 2026-11-04  
**Investigator:** Senior Deep Learning Engineer & Computer Vision Researcher  
**Status:** COMPLETE & EMPIRICALLY VALIDATED  
**Architecture:** EfficientNet-B0 (torchvision official) + ImageNet-1K Pretrained Backbone  
**Split Protocol:** Frozen Protocol C (Bipartite Connected Components Isolation)  
**Dataset Integrity SHA-256:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`  

---

## 1. Executive Summary

Phase 6 of project BIOTrace successfully trained, validated, and evaluated the first reproducible baseline visual deepfake detection model on the HiDF dataset under strict Protocol C bipartite graph partitioning.

Key Highlights:
1. **Zero Data Leakage:** Video identities and connected components are strictly isolated across partitions. Video frames were never mixed or leaked between train, validation, and test sets.
2. **Primary Evaluation Unit:** In adherence to forensic standards, evaluation metrics were computed strictly at the **video level** using mean-logit pooling across all 10 temporal frames per video, eliminating the independence fallacy.
3. **Single-Pass Test Evaluation:** The frozen test partition remained untouched during training, hyperparameter selection, and early stopping, and was evaluated exactly once on the frozen best validation checkpoint.
4. **Generalization & Performance:**
   - **Validation Video-Level ROC-AUC:** **0.7812** (Accuracy: **70.72%**, F1: **0.7305**, EER: **0.3043**)
   - **Test Video-Level ROC-AUC:** **0.7821** (Accuracy: **72.25%**, F1: **0.7394**, EER: **0.2813**)
   - **Video Aggregation Gain:** Mean-logit temporal pooling improved video-level ROC-AUC over individual frame predictions by **+0.0197 AUC** (from 0.7624 to 0.7821), empirically proving the value of temporal aggregation.
5. **Production Checkpoint:** Fully serialized end-to-end EfficientNet-B0 model saved at `ai-engine/models/baseline/best_model.pth` with verified SHA-256 hash `332846ff1479d1f6cdb1313e3247e484ec4a832d5e7c2146fb3a2e0719f29cbc`.

---

## 2. Hardware and Execution Environment

| Parameter | Specification |
|---|---|
| **Operating System** | Microsoft Windows 11 |
| **CPU Architecture** | Intel Core i5 (8 physical cores, 12 logical processors) |
| **System Memory (RAM)** | 15.67 GB Total |
| **GPU / CUDA** | CUDA Unavailable (`torch.cuda.is_available() == False`) |
| **Execution Mode** | Multithreaded CPU Optimization (`torch.set_num_threads(8)`) |
| **Python Version** | 3.11.9 (64-bit) |
| **PyTorch Version** | 2.10.0+cpu |
| **Torchvision Version** | 0.25.0+cpu |
| **Pretrained Weights Source** | Offline cached torchvision official ImageNet-1K (`efficientnet_b0_rwightman-7f5810bc.pth`) |
| **Available Storage** | > 80 GB SSD available on primary volume |

---

## 3. Dataset Composition & Protocol C Split Accounting

All frames were extracted at $224 \times 224$ RGB in Phase 5 from authentic and fake videos without alteration or loss of biometric integrity.

| Partition | Source Videos (Real) | Source Videos (Fake) | Total Videos | Frames per Video | Total Frames | Class Ratio |
|---|---|---|---|---|---|---|
| **Train** | 3,053 | 3,053 | 6,106 | 10 | 61,060 | 50.0% : 50.0% |
| **Validation** | 654 | 654 | 1,308 | 10 | 13,080 | 50.0% : 50.0% |
| **Test** | 654 | 654 | 1,308 | 10 | 13,080 | 50.0% : 50.0% |
| **Total** | **4,361** | **4,361** | **8,722** | **10** | **87,220** | **50.0% : 50.0%** |

- **Partition Disjointness:** Verified via automated assertions:
  $$\text{Train Videos} \cap \text{Val Videos} = \emptyset, \quad \text{Train Videos} \cap \text{Test Videos} = \emptyset, \quad \text{Val Videos} \cap \text{Test Videos} = \emptyset$$
  $$\text{Train Components} \cap \text{Val Components} = \emptyset, \quad \text{Train Components} \cap \text{Test Components} = \emptyset, \quad \text{Val Components} \cap \text{Test Components} = \emptyset$$

---

## 4. Model Architecture & Training Configuration

### 4.1 Architecture Specification
- **Backbone:** Torchvision `models.efficientnet_b0(weights=None)` initialized from official ImageNet-1K pretrained weights.
- **Feature Extraction Dimension:** 1,280 features after adaptive average pooling.
- **Classification Head:**
  ```python
  nn.Sequential(
      nn.Dropout(p=0.3),
      nn.Linear(in_features=1280, out_features=1),  # Single binary logit
  )
  ```
- **Total Parameters:** 4,008,829 parameters (1,281 trainable in linear classification head).
- **Loss Function:** `nn.BCEWithLogitsLoss()` (numerically stable binary cross-entropy with sigmoid integration).

### 4.2 Training Hyperparameters
- **Optimizer:** AdamW (`torch.optim.AdamW`)
- **Learning Rate:** $1.0 \times 10^{-4}$ (0.0001)
- **Weight Decay:** 0.01
- **Batch Size:** 64
- **Maximum Epochs:** 10
- **Early Stopping:** Monitored on Validation Video-Level ROC-AUC (patience = 3 epochs).
- **Random Seed:** 42 (deterministic NumPy, Python, PyTorch seeds).
- **Input Normalization:** Standard ImageNet statistics:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$

### 4.3 Smoke Test Verification
Prior to full training, an automated 5-step smoke test verified:
1. Model forward pass on batched inputs $(4, 3, 224, 224) \rightarrow (4, 1)$.
2. Loss calculation producing finite non-negative scalar.
3. Gradient backpropagation confirming non-zero gradients on trainable parameters.
4. Optimizer parameter update step.
5. Checkpoint serialization and deserialization with `weights_only=True`.
**Smoke Test Result:** `[OK] ALL 5 SMOKE TEST CRITERIA PASSED.`

---

## 5. Training Progression & History

Training ran for all 10 epochs without early stopping triggers because validation video-level ROC-AUC continued improving across epochs.

| Epoch | Train BCE Loss | Val BCE Loss | Val Video ROC-AUC | Val Video Accuracy | Val Video Precision | Val Video Recall | Val Video F1 | Val Video EER | Duration (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.672389 | 0.655226 | 0.7023 | 0.6468 | 0.6437 | 0.6575 | 0.6505 | 0.3517 | 2.86 |
| 2 | 0.645749 | 0.637415 | 0.7283 | 0.6674 | 0.6531 | 0.7141 | 0.6823 | 0.3326 | 2.67 |
| 3 | 0.631503 | 0.625834 | 0.7440 | 0.6766 | 0.6742 | 0.6835 | 0.6788 | 0.3211 | 2.75 |
| 4 | 0.622564 | 0.617402 | 0.7546 | 0.6835 | 0.6852 | 0.6789 | 0.6820 | 0.3150 | 2.73 |
| 5 | 0.615114 | 0.610558 | 0.7629 | 0.6843 | 0.6801 | 0.6957 | 0.6878 | 0.3135 | 2.75 |
| 6 | 0.612120 | 0.605862 | 0.7678 | 0.6812 | 0.6761 | 0.6957 | 0.6858 | 0.3203 | 2.69 |
| 7 | 0.607572 | 0.601935 | 0.7715 | 0.6911 | 0.6844 | 0.7095 | 0.6967 | 0.3165 | 2.78 |
| 8 | 0.604739 | 0.598623 | 0.7760 | 0.6896 | 0.6873 | 0.6957 | 0.6915 | 0.3104 | 2.67 |
| 9 | 0.601710 | 0.595947 | 0.7789 | 0.6934 | 0.6937 | 0.6927 | 0.6932 | 0.3050 | 2.74 |
| **10** | **0.598906** | **0.594179** | **0.7812** | **0.7072** | **0.6767** | **0.7936** | **0.7305** | **0.3043** | **2.70** |

- **Total Feature Extraction Time:** 1,222.6s (20.38 min)
  - Train: 891.2s (68.6 fps across 61,060 frames)
  - Val: 162.1s (80.8 fps across 13,080 frames)
  - Test: 169.3s (77.3 fps across 13,080 frames)
- **Total Training Execution Time:** 27.34s (Average: 2.73s per epoch)
- **Best Validation Epoch:** Epoch 10 (ROC-AUC = 0.7812)

---

## 6. Video-Level Evaluation Results

### 6.1 Validation Partition Results ($N=1,308$ Videos, 13,080 Frames)
- **ROC-AUC:** **0.7812** (0.781191)
- **Average Precision (PR-AUC):** **0.7729** (0.772856)
- **Accuracy (Threshold = 0.50):** **70.72%** (0.707187)
- **Precision:** **0.6767** (0.676662)
- **Recall (Sensitivity):** **0.7936** (0.793578)
- **F1-Score:** **0.7305** (0.730471)
- **Equal Error Rate (EER):** **0.3043** (0.304281)
- **Optimal Operating Threshold:** **0.5343** (Accuracy at optimal threshold: 69.57%)
- **Brier Score:** 0.2001
- **Confusion Matrix ($N=1,308$):**

$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 406 & 248 \\ 135 & 519 \end{pmatrix}$$

### 6.2 Test Partition Results ($N=1,308$ Videos, Single Untouched Pass)
Evaluated strictly **once** on the final frozen best checkpoint without threshold tuning or hyperparameter searching.

- **ROC-AUC:** **0.7821** (0.782059)
- **Average Precision (PR-AUC):** **0.7550** (0.755027)
- **Accuracy (Threshold = 0.50):** **72.25%** (0.722477)
- **Precision:** **0.6969** (0.696888)
- **Recall (Sensitivity):** **0.7875** (0.787462)
- **F1-Score:** **0.7394** (0.739411)
- **Equal Error Rate (EER):** **0.2813** (0.281346)
- **Empirical Test Operating Threshold (Quarantined):** **0.5321** (Accuracy at post-hoc threshold: 71.87%)  
  > [!IMPORTANT]
  > **Threshold Governance & Anti-Leakage Quarantine:** The value 0.5321 is an empirical post-hoc observation derived solely from the test set for retrospective ROC curve characterization. In accordance with zero-leakage forensic standards, **0.5321 is strictly quarantined and forbidden from operational use**. Operational inference thresholds are derived strictly from validation data (0.5000 standard or 0.5343 validation-optimal).
- **Brier Score:** 0.1991
- **Confusion Matrix ($N=1,308$):**

$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 430 & 224 \\ 139 & 515 \end{pmatrix}$$

| Metric | Validation (N=1,308) | Test (N=1,308) | Delta |
|---|:---:|:---:|:---:|
| **ROC-AUC** | 0.7812 | **0.7821** | +0.0009 |
| **Accuracy** | 70.72% | **72.25%** | +1.53% |
| **Precision** | 67.67% | **69.69%** | +2.02% |
| **Recall** | 79.36% | **78.75%** | -0.61% |
| **F1-Score** | 0.7305 | **0.7394** | +0.0089 |
| **EER** | 0.3043 | **0.2813** | -0.0230 |

The consistency between validation and test metrics ($\Delta \text{AUC} = +0.0009$) confirms that the model is **free of overfitting** and generalizes reliably to unseen donor identities under Protocol C.

---

## 7. Forensic Methodological Audit: Video-Level vs Frame-Level Comparison

To demonstrate the scientific necessity of temporal pooling and expose the fallacy of frame-level independence claims:

| Evaluation Level | Sample Unit | Partition | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Frame-Level (Correlated)** | Individual Frame | Test (N=13,080) | 0.7624 | 69.98% | 67.56% | 76.85% | 0.7191 |
| **Video-Level (Aggregated)** | Video (10-frame mean logit) | Test (N=1,308) | **0.7821** | **72.25%** | **69.69%** | **78.75%** | **0.7394** |
| **Gain from Aggregation** | — | — | **+0.0197** | **+2.27%** | **+2.13%** | **+1.90%** | **+0.0203** |

**Forensic Finding:** Mean-logit temporal pooling across 10 video frames mitigates single-frame detector instability, reducing false positives and improving ROC-AUC by approximately 2.0 percentage points.

---

## 8. Artifact Inventory & Cryptographic Provenance

All artifacts are persisted and verifiable via SHA-256 digests:

| Artifact | File Path | SHA-256 Digest | Status |
|---|---|---|:---:|
| **Best Model Checkpoint** | `ai-engine/models/baseline/best_model.pth` | `332846ff1479d1f6cdb1313e3247e484ec4a832d5e7c2146fb3a2e0719f29cbc` | **VERIFIED** |
| **Training History CSV** | `../data/phase6_training_history.csv` | `9d554a9fc5d14df1bcad9d2906e57ba56736dd653194a28fc9a2eb2d3269b2be` | **VERIFIED** |
| **Evaluation Metrics JSON** | `../data/phase6_metrics.json` | `5ec0a5417088b9a2432a2f8c5b045330e6a3f4e24eb3606cf44927cbba03ce4b` | **VERIFIED** |
| **Train Feature Cache** | `data/cache/features/train_features.pt` | `f389808d4b3df5da3141443659cfdfa1a8df063ff39cf5c777e0344d2d48c894` | **VERIFIED** |
| **Val Feature Cache** | `data/cache/features/val_features.pt` | `2924bf63d41f0a1c6a213e4b78996e8e89f81ca9942a490cf2606553835a6431` | **VERIFIED** |
| **Test Feature Cache** | `data/cache/features/test_features.pt` | `1da92817ba7efc8623ba9ec5a0b73c7fc0176868dfd05d54988e404be1f8876c` | **VERIFIED** |

---

## 9. Automated Test Suite Validation

The test suite in `ai-engine/tests/test_phase6_baseline.py` was executed to verify all operational invariants:

| Test Case | Target Invariant | Result |
|---|---|:---:|
| `test_dataset_loading_and_labels` | 87,220 frames loaded, 50:50 real/fake balance confirmed | **PASSED** |
| `test_split_isolation_and_disjointness` | Zero video or component leakage across train, val, and test | **PASSED** |
| `test_model_output_dimensions` | EfficientNet-B0 outputs $(B, 1)$ logits with finite numbers | **PASSED** |
| `test_bce_loss_and_gradients` | `BCEWithLogitsLoss` produces valid positive loss and non-zero gradients | **PASSED** |
| `test_video_level_prediction_aggregation` | Mean-logit pooling correctly aggregates 10 frame logits to 1 video | **PASSED** |
| `test_checkpoint_saving_and_safe_loading` | `weights_only=True` safe deserialization with exact weight fidelity | **PASSED** |
| `test_smoke_test_runner` | 5-stage training smoke test executes without error | **PASSED** |
| `test_image_preprocessing` | $224 \times 224$ normalization matches ImageNet $\mu, \sigma$ tensors | **PASSED** |

**Test Execution:** 8 passed, 0 failed in 1.72 seconds.

---

## 10. Reproduction & Inference Commands

### 10.1 Command to Reproduce Baseline Training
```powershell
python ai-engine/scripts/train_baseline.py --epochs 10 --batch-size 64 --lr 0.0001 --weight-decay 0.01 --patience 3 --seed 42 --device cpu
```

### 10.2 Command to Evaluate Checkpoint (Test Partition)
```powershell
python ai-engine/scripts/evaluate_baseline.py --checkpoint ai-engine/models/baseline/best_model.pth --features-dir data/cache/features --output ../data/phase6_metrics.json --device cpu
```

### 10.3 Command to Run Video Inference
```powershell
python ai-engine/scripts/predict_video.py --input data/extracted/hidf/fake/c00003_c11507.mp4 --checkpoint ai-engine/models/baseline/best_model.pth --frames 10
```

---

## 11. Forensic Disclaimers & Operational Limitations

1. **Baseline Classification Scope:** The trained model is an initial visual convolutional baseline (EfficientNet-B0). It detects statistical compression and synthesis artifacts present in the training distribution.
2. **Generalization Boundary:** While evaluated on unseen target identities under Protocol C, high performance on the HiDF benchmark does NOT guarantee equivalent detection rates against unseen diffusion models, out-of-distribution video codecs, or extreme social media re-compression.
3. **Forensic Use:** In accordance with forensic science standards, visual CNN predictions indicate statistical artifact likelihood only. They do not constitute definitive proof of media manipulation or biometric tampering, and should be evaluated alongside multimodal physiological (rPPG) and kinematic evidence.
4. **Operating Threshold Policy:** In operational deployment, decisions must use validation-derived operating thresholds (0.5000 standard or 0.5343 validation-optimal). The test-derived point (0.5321) must never be used operationally to prevent retrospective data contamination.
5. **Metric Distinction & Calibration:** Raw sigmoid outputs $\sigma(\bar{z}) \in [0, 1]$ represent uncalibrated heuristic scores, not empirical posterior probabilities. Temperature scaling or isotonic calibration must be applied before treating outputs as true probabilities (`is_calibrated: false`).
