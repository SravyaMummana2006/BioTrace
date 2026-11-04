# BIOTrace — Physiological Model & Detection Pipeline Audit Report

**Date:** 2026-11-04  
**Audit Scope:** Read-Only Forensic Architecture & Model Inspection  
**System Evaluated:** BIOTrace AI Engine (`ai-engine/src/biotrace/`) & Desktop Client  
**Auditor:** Senior Research Data Engineer & Digital Forensics Systems Architect  

---

## 1. Executive Summary

This read-only audit examined the codebase to determine whether BIOTrace's physiological detection system incorporates trained machine-learning models or relies solely on classical signal processing and heuristic rules.

### Key Audit Findings:
1. **Physiological Modules are Classical Signal Extractors (Zero ML/DL):**
   The rPPG, eye-blink dynamics, facial kinematics, and spatial consistency modules are implementations of classical photometric and geometric algorithms (CHROM, POS, MediaPipe 3D Mesh, planar EAR, Butterworth filters, and hysteresis state machines). None contain trained neural networks, SVMs, or learned classifiers.
2. **Authenticity Assessment is Intentionally Unassessed:**
   Across all physiological extractors, outputs explicitly hardcode:
   ```python
   authenticity_score = None
   is_authenticity_assessed = False
   anomaly_detected = False
   ```
   No real vs. fake classification is produced by any physiological module.
3. **Evidence Fusion is a Deterministic Decision Tree:**
   The fusion engine does not use machine learning or trained probabilistic weights. It applies an if-else heuristic rule tree to categorize extracted signals into qualitative states (`SUPPORTING_EVIDENCE`, `CONFLICTING_EVIDENCE`, `MIXED_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
4. **The Only Trained Model in BIOTrace is the Phase 6 Visual CNN Baseline:**
   The only trained model checkpoint in the entire repository is `ai-engine/models/baseline/best_model.pth` (EfficientNet-B0), which is a **spatial visual artifact detector** trained on $224 \times 224$ RGB video frame crops. It does not analyze physiological signals.

---

## 2. Modality-by-Modality Audit

### 2.1 Remote Photoplethysmography (rPPG)
- **Primary Source Files:**
  - [`ai-engine/src/biotrace/rppg/algorithms.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/algorithms.py)
  - [`ai-engine/src/biotrace/rppg/face_tracker.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/face_tracker.py)
  - [`ai-engine/src/biotrace/rppg/filtering.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/filtering.py)
  - [`ai-engine/src/biotrace/rppg/quality.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/quality.py)
  - [`ai-engine/src/biotrace/rppg/extractor.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/extractor.py)
- **Mathematical / Algorithmic Basis:**
  - **CHROM** (de Haan & Jeanne 2013): Linear projection of normalized color channels ($X = 3R_n - 2G_n$, $Y = 1.5R_n + G_n - 1.5B_n$, $S = X - \alpha Y$ with $\alpha = \sigma(X) / \sigma(Y)$).
  - **POS** (Wang et al. 2017): Plane-Orthogonal-to-Skin projection matrix to extract pulse signals invariant to specular reflection.
  - **Filtering:** 2nd-order zero-phase Butterworth bandpass filter (0.75 Hz – 2.50 Hz / 45 – 150 BPM).
  - **Quality & Spectral Density:** Welch periodogram FFT peak detection and Signal-to-Noise Ratio (SNR) in dB.
- **Threshold / Classification Logic:**
  - Uses fixed engineering cutoffs (SNR: $\ge 3.0\text{ dB}$ optimal, $\ge 0.0\text{ dB}$ acceptable, $< 0.0\text{ dB}$ poor; landmark frame ratio $\ge 0.70$).
- **Is there an actual trained ML/DL model?** **NO.**
- **Checkpoints / Training Scripts:** None exist.
- **Dataset / Labels Used:** None. (`UBFC-rPPG` is referenced in template configs, but no dataset files exist on disk).
- **Are real/fake predictions generated?** **NO.** [`RPPGSignalExtractor.extract()`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/extractor.py#L216-L224) returns `authenticity_score=None`, `is_authenticity_assessed=False`.
- **Video-level performance metrics?** **NO.** No ROC-AUC, EER, or detection metrics exist.

---

### 2.2 Eye-Blink Dynamics Detection & Facial Kinematics
- **Primary Source Files:**
  - [`ai-engine/src/biotrace/facial_dynamics/landmarks.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/landmarks.py)
  - [`ai-engine/src/biotrace/facial_dynamics/ear.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/ear.py)
  - [`ai-engine/src/biotrace/facial_dynamics/calibration.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/calibration.py)
  - [`ai-engine/src/biotrace/facial_dynamics/blink_detector.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/blink_detector.py)
  - [`ai-engine/src/biotrace/facial_dynamics/temporal_analysis.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/temporal_analysis.py)
  - [`ai-engine/src/biotrace/facial_dynamics/extractor.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/extractor.py)
- **Mathematical / Algorithmic Basis:**
  - **Landmark Tracking:** Google MediaPipe Face Mesh tracking 6 canonical contour points per eye.
  - **Eye Aspect Ratio (EAR):** Soukupová & Čech (2016) formulation in physical pixel Euclidean space:
    $$\text{EAR} = \frac{||P_2 - P_6|| + ||P_3 - P_5||}{2 ||P_1 - P_4||}$$
  - **Baseline Calibration:** Percentile-based adaptive thresholding (75th/90th percentile of valid openness observations).
  - **Kinematic State Machine:** Dual-threshold hysteresis state machine:
    $$\theta_{\text{close}} = 0.65 \times \text{baseline\_ear}, \quad \theta_{\text{open}} = 0.82 \times \text{baseline\_ear}$$
- **Threshold / Classification Logic:**
  - Fixed biological heuristic bounds: minimum blink duration 80 ms, maximum closure 500 ms, refractory period 120 ms.
- **Is there an actual trained ML/DL model?** **NO.**
- **Checkpoints / Training Scripts:** None exist.
- **Dataset / Labels Used:** None.
- **Are real/fake predictions generated?** **NO.** [`FacialDynamicsExtractor.extract()`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/facial_dynamics/extractor.py#L383-L391) returns `authenticity_score=None`, `is_authenticity_assessed=False`.
- **Video-level performance metrics?** **NO.** No detection metrics exist.

---

### 2.3 Spatial Biological Consistency Analysis (Complementary Optical Modality)
- **Primary Source Files:**
  - [`ai-engine/src/biotrace/spatial_consistency/spatial_analyzer.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/spatial_consistency/spatial_analyzer.py)
  - [`ai-engine/src/biotrace/spatial_consistency/quality.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/spatial_consistency/quality.py)
  - [`ai-engine/src/biotrace/spatial_consistency/extractor.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/spatial_consistency/extractor.py)
- **Mathematical / Algorithmic Basis:**
  - Signed Pearson cross-correlation between pulse signals extracted across facial regions (forehead vs. cheeks).
  - Cosine spectral similarity of power spectral densities.
  - Rayleigh physical spectral resolution gating ($\Delta\text{BPM}_{\text{phys}} = 60 / T_{\text{obs}}$).
- **Is there an actual trained ML/DL model?** **NO.** Pure signal statistics.
- **Are real/fake predictions generated?** **NO.** Returns correlation measurements and consistency flags only.

---

### 2.4 Physiological Evidence Fusion
- **Primary Source Files:**
  - [`ai-engine/src/biotrace/evidence_fusion/dependency_model.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/evidence_fusion/dependency_model.py)
  - [`ai-engine/src/biotrace/evidence_fusion/evidence_normalizer.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/evidence_fusion/evidence_normalizer.py)
  - [`ai-engine/src/biotrace/evidence_fusion/fusion_engine.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/evidence_fusion/fusion_engine.py)
- **Fusion Mechanism:**
  - Does NOT combine model probability predictions.
  - Combines **extracted signal quality flags** and **heuristic discrepancy checks**:
    - Groups rPPG + Spatial Consistency into `OPTICAL_CARDIOVASCULAR` family to prevent double-counting.
    - Groups Eye-Blink into `FACIAL_DYNAMICS` family.
    - Evaluates an if-else decision tree (Lines 134–209 of `fusion_engine.py`) to categorize findings into qualitative states:
      - `SUPPORTING_EVIDENCE`
      - `CONFLICTING_EVIDENCE`
      - `MIXED_EVIDENCE`
      - `INSUFFICIENT_EVIDENCE`
- **Is there an actual trained fusion model?** **NO.** There is no trained fusion network, logistic regression, or learned belief combination.
- **Does it output an authenticity score?** **NO.**
  In [`job_manager.py`](file:///C:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/services/job_manager.py#L348-L359):
  ```python
  assessed_scores = [v.authenticity_score for v in modality_verdicts if v.is_authenticity_assessed and v.authenticity_score is not None]
  overall_score = float(np.mean(assessed_scores)) if assessed_scores else None
  ```
  Since all physiological modalities return `authenticity_score=None`, the overall authenticity score is `None`.

---

## 3. Comprehensive Model & Artifact Inventory

A complete filesystem audit was performed across all directories for model weight extensions (`.pth`, `.pt`, `.onnx`, `.pkl`, `.h5`, `.joblib`, `.bin`, `.ckpt`):

| File Path | File Size | Modality / Purpose | Is It Physiological? |
|---|---|---|:---:|
| `ai-engine/models/baseline/best_model.pth` | 16,331,193 bytes (16.3 MB) | EfficientNet-B0 Visual Baseline Classifier | **NO** (Visual CNN only) |
| `data/cache/features/train_features.pt` | 314,805,533 bytes (314.8 MB) | EfficientNet-B0 Visual Feature Embeddings (Train) | **NO** (Visual Features) |
| `data/cache/features/val_features.pt` | 67,432,077 bytes (67.4 MB) | EfficientNet-B0 Visual Feature Embeddings (Val) | **NO** (Visual Features) |
| `data/cache/features/test_features.pt` | 67,432,469 bytes (67.4 MB) | EfficientNet-B0 Visual Feature Embeddings (Test) | **NO** (Visual Features) |

**Conclusion on Checkpoints:**
There are **zero model checkpoints, zero weight files, and zero trained parameters** for rPPG, eye-blink dynamics, facial kinematics, or evidence fusion.

---

## 4. Dataset Audit

| Dataset | Location on Disk | Contents | Biometric Ground Truth Present? |
|---|---|---|:---:|
| **HiDF** | `data/extracted/hidf/` | 4,361 authentic MP4s + 4,361 manipulated MP4s | **NO** (No contact PPG, ECG, or EOG sensors) |
| **UBFC-rPPG** | *Not present on disk* | None | N/A |
| **FaceForensics++** | *Not present on disk* | None | N/A |
| **Celeb-DF-v2** | *Not present on disk* | None | N/A |

The existing dataset infrastructure consists entirely of the **HiDF video dataset** partitioned under the frozen `protocol_c_split_manifest.csv` (SHA-256: `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`).

---

## 5. Audit Conclusion

Based on thorough inspection of source code, configurations, algorithms, and serialized storage, the audit concludes:

### **Conclusion: B. Physiological modules exist, but no trained detection models exist.**

*(Note: BIOTrace does possess one genuinely trained model—the Phase 6 EfficientNet-B0 visual deepfake detection baseline—but within the physiological and biometric domain specifically, all modules are unlearned signal extractors and rule-based heuristics.)*

---

## 6. Recommended Fastest Scientifically Valid Next Step

### **Recommendation: Train a Supervised Tabular Physiological Deepfake Classifier on Extracted Biometric Features under Protocol C**

Rather than attempting to train complex deep neural networks end-to-end (which would require external contact sensor datasets that are not present), BIOTrace can immediately leverage its existing signal processing pipelines to create its first genuine physiological deepfake detection model:

1. **Feature Extraction Pass on HiDF:**
   Run the existing extractors over the audited HiDF dataset (4,361 real and 4,361 fake videos) to construct a structured physiological feature tabular dataset:
   - **rPPG Features:** Peak pulse frequency (BPM), spectral SNR (dB), CHROM-to-POS cross-spectral coherence, regional cheek-forehead cross-correlation, pulse wave amplitude variance.
   - **Blink & Kinematic Features:** Mean blink rate (BPM), average blink duration (ms), eyelid closing velocity, reopening velocity, EAR bilateral asymmetry index, prolonged closure frequency.
2. **Train a Supervised Machine Learning Classifier:**
   Train an interpretable classifier (e.g., **Logistic Regression with ElasticNet penalty**, **Gradient Boosted Trees / LightGBM**, or an **MLP**) on the frozen Protocol C training partition ($N=6,106$ videos).
3. **Validation & Single-Pass Test Evaluation:**
   - Tune decision thresholds and hyperparameters exclusively on the validation partition ($N=1,308$ videos).
   - Evaluate exactly once on the frozen test partition ($N=1,308$ videos) to compute genuine physiological detection metrics: **ROC-AUC, PR-AUC, EER, Precision, Recall, and F1**.
4. **Integration with Evidence Fusion:**
   Feed the learned physiological probability output into the FastAPI backend and Java client to replace the `authenticity_score=None` placeholder with a scientifically verified physiological authenticity verdict.
