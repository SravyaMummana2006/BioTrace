# BIOTrace Phase 10 — Controlled Multimodal Fusion Report

**Date:** 2026-11-04  
**Author:** Senior Machine-Learning Research Engineer  
**Status:** Completed & Empirically Validated  
**Dataset Scope:** HiDF Full Protocol C Partitions (`../data/protocol_c_split_manifest.csv`)  
**Frozen Manifest SHA-256:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`  
**Fusion Model Checkpoint:** [`ai-engine/models/fusion/phase10_fusion.joblib`](../../ai-engine/models/fusion/phase10_fusion.joblib)  
**Multimodal Scores Dataset:** [`data/derived/fusion/phase10_multimodal_scores.csv`](../../data/derived/fusion/phase10_multimodal_scores.csv)  
**Metrics JSON:** [`../data/phase10_metrics.json`](../data/phase10_metrics.json)  

---

## Executive Summary

Phase 10 implemented a controlled multimodal fusion architecture in BIOTrace, combining the **Phase 6 Visual Spatial Baseline** (EfficientNet-B0 with video-level mean-logit pooling) and the **Phase 9 Physiological Baseline** (Random Forest trained on 10 continuous kinematic and chromatic consistency features).

The central research objective was to determine whether incorporating physiological signals into a visual deepfake detector yields measurable, statistically verifiable performance improvements on the frozen Protocol C benchmark.

### Key Empirical Findings

1. **Empirical Performance Improvement:** On the held-out test partition ($N=1,308$ videos: 654 authentic real, 654 manipulated fake), multimodal fusion achieved a **Test ROC-AUC of 0.7952** (95% CI: $[0.7713, 0.8188]$), representing an absolute gain of **$+0.0132$ ($+1.32\%$)** over the visual baseline (**0.7821**) and **$+0.1686$ ($+16.86\%$)** over the physiological baseline (**0.6267**).
2. **Precision-Recall & Calibration Gains:** Fusion improved PR-AUC by **$+0.0249$** (from $0.7550$ to $0.7799$) and reduced Brier score from **$0.1991$ to $0.1866$** ($-0.0125$), demonstrating superior probability calibration.
3. **Statistical Significance Assessment:** In paired bootstrap resampling ($B=2,000$), fusion was superior to the visual-only detector in **$96.9\%$** of iterations ($p = 0.031$ one-sided). However, the two-sided 95% bootstrap confidence interval of the difference $[-0.0007, +0.0274]$ marginally spans zero, indicating **marginal statistical significance** under strict two-sided hypothesis testing.
4. **Error Recovery & Complementarity:** Across 1,308 test videos, fusion successfully **recovered 104 visual false predictions** where the physiological detector correctly classified the video, while introducing 97 errors on visual true predictions (net gain: $+7$ correct classifications).
5. **Rigorous Research Integrity:** Zero test data was used during training, scaling, or threshold tuning. Imputers and scalers were fit exclusively on training data ($N=6,106$), operating thresholds were selected on validation data ($N=1,308$), and the test partition was evaluated strictly once on identical sample sets.

---

## 1. Comparative Test Evaluation Matrix

All three models were evaluated on the **exact same 1,308 held-out videos** under the frozen Protocol C bipartite graph split:

| Performance Metric | Model 1: Visual Spatial Only (Phase 6) | Model 2: Physiological Only (Phase 9) | Model 3: Multimodal Fusion (Phase 10) | Delta (Fusion vs. Visual) | Delta (Fusion vs. Phys) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **0.7821** (0.782059) | **0.6267** (0.626694) | **0.7952** (0.795247) | **+0.0132** (+1.32%) | **+0.1686** (+16.86%) |
| **95% Bootstrap CI** | $[0.7571, 0.8054]$ | $[0.5975, 0.6581]$ | **$[0.7713, 0.8188]$** | $[-0.0007, +0.0274]$ | $[+0.1394, +0.1963]$ |
| **PR-AUC (Avg Prec)** | 0.7550 | 0.6117 | **0.7799** | **+0.0249** (+2.49%) | **+0.1682** (+16.82%) |
| **Accuracy ($\tau^*$)** | 71.64% ($\tau=0.534$) | 58.33% ($\tau=0.513$) | **72.17%** ($\tau=0.500$) | **+0.54%** | **+13.84%** |
| **Accuracy ($\tau=0.50$)** | 72.25% | 57.95% | **72.09%** | $-0.15\%$ | **+14.14%** |
| **F1 Score ($\tau^*$)** | 0.7153 | 0.5837 | **0.7213** | **+0.0060** | **+0.1376** |
| **Precision ($\tau^*$)** | 0.7180 | 0.5832 | **0.7224** | **+0.0044** | **+0.1392** |
| **Recall ($\tau^*$)** | 0.7125 | 0.5841 | **0.7202** | **+0.0076** | **+0.1361** |
| **Equal Error Rate (EER)** | 0.2813 | 0.4167 | **0.2783** | **-0.0031** (-0.31%) | **-0.1384** (-13.84%) |
| **Brier Score (Calibration)** | 0.1991 | 0.2380 | **0.1866** | **-0.0125** (Better) | **-0.0515** (Better) |

> [!NOTE]
> Operating thresholds $\tau^*$ were derived strictly from the validation partition at the Equal Error Rate operating point: $\tau_{\text{visual}}^* = 0.534347$, $\tau_{\text{phys}}^* = 0.5128$, $\tau_{\text{fusion}}^* = 0.499895$. Standard uncalibrated thresholds ($\tau=0.50$) are also reported for transparency.

---

## 2. Test Confusion Matrices ($N=1,308$)

All confusion matrices were evaluated on identical test pairs (654 Real, 654 Fake) using validation-locked operating thresholds:

### 2.1 Model 1: Visual Spatial Baseline (Phase 6, $\tau^* = 0.534347$)
```
                        Predicted Authentic (Real)    Predicted Manipulated (Fake)
Actual Authentic (Real)            471 (72.0%)                   183 (28.0%)
Actual Manipulated (Fake)          188 (28.7%)                   466 (71.3%)
```
- **True Negatives:** 471 | **False Positives:** 183
- **False Negatives:** 188 | **True Positives:** 466

### 2.2 Model 2: Physiological Baseline (Phase 9, $\tau^* = 0.5128$)
```
                        Predicted Authentic (Real)    Predicted Manipulated (Fake)
Actual Authentic (Real)            381 (58.3%)                   273 (41.7%)
Actual Manipulated (Fake)          272 (41.6%)                   382 (58.4%)
```
- **True Negatives:** 381 | **False Positives:** 273
- **False Negatives:** 272 | **True Positives:** 382

### 2.3 Model 3: Multimodal Fusion (Phase 10, $\tau^* = 0.499895$)
```
                        Predicted Authentic (Real)    Predicted Manipulated (Fake)
Actual Authentic (Real)            473 (72.3%)                   181 (27.7%)
Actual Manipulated (Fake)          183 (28.0%)                   471 (72.0%)
```
- **True Negatives:** 473 ($+2$ vs Visual) | **False Positives:** 181 ($-2$ vs Visual)
- **False Negatives:** 183 ($-5$ vs Visual) | **True Positives:** 471 ($+5$ vs Visual)

---

## 3. Dataset & Protocol C Split Accounting

Strict data provenance and split disjointness were maintained throughout:

```mermaid
flowchart TD
    M["Frozen Manifest (../data/protocol_c_split_manifest.csv) SHA-256: c15300..."]
    M --> TR["Train Split: 6,106 videos (3,053 pairs)"]
    M --> VA["Val Split: 1,308 videos (654 pairs)"]
    M --> TE["Test Split: 1,308 videos (654 pairs) - FROZEN"]

    TR --> S6_TR["P6 Visual Logits & Probs"]
    TR --> S9_TR["P9 Physiological Probs"]
    S6_TR & S9_TR --> FIT["Fit StandardScaler + LogisticRegression"]

    VA --> S6_VA["P6 Visual Val Probs"]
    VA --> S9_VA["P9 Phys Val Probs"]
    S6_VA & S9_VA --> EVAL_VA["Transform & Evaluate Val"]
    FIT -.->|"Apply Model"| EVAL_VA
    EVAL_VA --> TAU["Determine Val EER Threshold (0.4999)"]

    TE --> S6_TE["P6 Visual Test Probs"]
    TE --> S9_TE["P9 Phys Test Probs"]
    S6_TE & S9_TE --> EVAL_TE["Single-Pass Test Evaluation"]
    FIT -.->|"Frozen Model"| EVAL_TE
    TAU -.->|"Frozen Threshold"| EVAL_TE
```

- **Manifest Checksum:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` (Verified untouched).
- **Video ID Alignment:** Verified 100% exact set equality across all 8,722 videos between Phase 6 frame caches, Phase 9 feature tables, and the Protocol C split manifest.

---

## 4. Multimodal Fusion Architecture & Model Parameters

### 4.1 Input Features & Pipeline
The input vector for video $i$ is:
$$\mathbf{x}_i = \begin{bmatrix} p_{\text{visual}, i} \\ p_{\text{phys}, i} \end{bmatrix} \in [0, 1]^2$$
where:
- $p_{\text{visual}, i} = \sigma\left(\frac{1}{10} \sum_{k=1}^{10} \text{logit}_{i, k}\right)$ is the video-level pooled probability from the Phase 6 EfficientNet-B0 backbone.
- $p_{\text{phys}, i} = P(\text{Fake} \mid \mathbf{f}_{\text{phys}, i})$ is the predicted probability from the Phase 9 Random Forest classifier over the 10 approved kinematic and chromatic features.

### 4.2 Fitted Preprocessor & Classifier
Preprocessing and classification were trained as a scikit-learn `Pipeline`:
$$\mathbf{z}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
$$\text{logit}_{\text{fusion}, i} = w_{\text{visual}} \cdot z_{\text{visual}, i} + w_{\text{phys}} \cdot z_{\text{phys}, i} + b$$
$$p_{\text{fusion}, i} = \sigma(\text{logit}_{\text{fusion}, i})$$

Fitted parameters (fit strictly on 6,106 training videos):

| Component | Parameter | Value | Interpretation |
| :--- | :--- | :---: | :--- |
| **StandardScaler** | $\mu_{\text{visual}}$ | $0.52758$ | Training mean visual probability |
| | $\sigma_{\text{visual}}$ | $0.15470$ | Training visual probability standard deviation |
| | $\mu_{\text{phys}}$ | $0.50019$ | Training mean physiological probability |
| | $\sigma_{\text{phys}}$ | $0.09857$ | Training physiological probability standard deviation |
| **LogisticRegression** | $w_{\text{visual}}$ | **$+1.3838$** | Standardized visual feature weight |
| | $w_{\text{phys}}$ | **$+0.7668$** | Standardized physiological feature weight |
| | Intercept $b$ | **$-0.0313$** | Near-zero calibration offset |

> [!IMPORTANT]
> **Modality Weight Interpretation:** Both weights are strongly positive, indicating that both modalities contribute genuine discriminating signal. The visual modality receives approximately $64.3\%$ of the total linear weighting ($1.3838 / (1.3838 + 0.7668)$) while the physiological modality receives $35.7\%$.

---

## 5. Statistical Significance & Hypothesis Testing

To assess whether the $+0.0132$ AUC improvement is statistically robust or an artifact of random sampling, paired bootstrap resampling ($B=2,000$ iterations) was performed on the held-out test predictions:

```
Bootstrap Metric Distribution (B=2,000 paired resamples):
  Visual ROC-AUC:   Mean = 0.7819, 95% CI = [0.7571, 0.8054]
  Fusion ROC-AUC:   Mean = 0.7951, 95% CI = [0.7713, 0.8188]
  Delta (Fus - Vis):Mean = +0.0132, 95% CI = [-0.0007, +0.0274]
  P(Fusion > Vis):  96.9% (1,938 / 2,000 resamples)
```

### Research Hypothesis Verdicts
- **One-Sided Hypothesis ($H_1: \text{AUC}_{\text{fusion}} > \text{AUC}_{\text{visual}}$):** **REJECT $H_0$ ($p = 0.031 < 0.05$).** Multimodal fusion produces superior AUC in $96.9\%$ of resampled distributions.
- **Two-Sided Hypothesis ($H_1: \text{AUC}_{\text{fusion}} \neq \text{AUC}_{\text{visual}}$):** **FAIL TO REJECT $H_0$ at $\alpha = 0.05$ ($p \approx 0.062$).** The 95% confidence interval $[-0.0007, +0.0274]$ slightly intersects zero.
- **Objective Scientific Conclusion:** The improvement is **empirically positive and marginally significant**. Physiological features provide measurable complementary signal, but the magnitude of the gain ($+1.32\%$ AUC) is modest on short clips where spatial artifacts are readily detectable.

---

## 6. Modality Interactions & Forensic Case Analysis

Cross-tabulating the predictions of the visual baseline and physiological baseline reveals the mechanism of fusion:

| Cohort | Description | Count ($N=1,308$) | Percentage |
| :--- | :--- | :---: | :---: |
| **Consensus Correct** | Both Visual and Phys correctly classified the video | 539 | 41.2% |
| **Consensus Incorrect** | Both Visual and Phys misclassified the video | 147 | 11.2% |
| **Visual Dominant** | Visual was correct, Phys was incorrect | 398 | 30.4% |
| **Phys Dominant** | Phys was correct, Visual was incorrect | 224 | 17.1% |

### Impact of Fusion on Disagreeing Cases
- **Recovered Cases:** Fusion correctly flipped **104 videos** that the visual baseline got wrong, guided by strong physiological signals.
- **Corrupted Cases:** Fusion incorrectly flipped **97 videos** that the visual baseline got right, misled by noisy physiological signals.
- **Net Forensic Gain:** $+7$ net correct classifications ($+104 - 97$).
- **Uncertainty Reduction:** For cases where visual predictions were borderline ($0.40 \le p_{\text{visual}} \le 0.60$), physiological agreement significantly shifted confidence toward the correct ground truth, resulting in a **$-0.0125$ reduction in Brier score** ($0.1991 \rightarrow 0.1866$).

---

## 7. Limitations & Technical Confounds

1. **Short Video Duration Physical Constraints:** HiDF clips are $2.96$ seconds (72 frames at 24 FPS). As established in Phase 8 and Phase 9, the Rayleigh frequency resolution is $\Delta f = \pm 20.3$ BPM. rPPG spectral peaks cannot achieve clinical precision over 3 seconds, limiting the physiological model's standalone AUC to $0.6267$.
2. **Spatial Feature Dominance:** Deepfake synthesis in HiDF creates visible boundary, warping, and texture discrepancies that EfficientNet-B0 detects with high sensitivity ($78.2\%$ AUC). Because spatial signals are stronger, linear fusion appropriately weights visual features higher ($1.38$ vs $0.77$).
3. **Double Error Risk:** In 97 test cases, noisy physiological estimates degraded an otherwise correct visual prediction. Nonlinear gating or uncertainty-weighted fusion (e.g., attending to physiological features only when visual confidence is low) may prevent these regressions in future work.

---

## 8. Artifact Deliverables Manifest

| Artifact Path | Format | Description |
| :--- | :--- | :--- |
| [`scripts/train_phase10_fusion.py`](../../scripts/train_phase10_fusion.py) | Python Script | Training pipeline with strict train-only fitting and validation tuning |
| [`scripts/evaluate_phase10_fusion.py`](../../scripts/evaluate_phase10_fusion.py) | Python Script | Single-pass test evaluation script with paired bootstrap analysis |
| [`ai-engine/models/fusion/phase10_fusion.joblib`](../../ai-engine/models/fusion/phase10_fusion.joblib) | Serialized Binary | Frozen Logistic Regression pipeline + scaler + validation threshold |
| [`data/derived/fusion/phase10_multimodal_scores.csv`](../../data/derived/fusion/phase10_multimodal_scores.csv) | CSV (8,722 rows) | Complete per-video predictions and metadata for train, val, and test |
| [`../data/phase10_metrics.json`](../data/phase10_metrics.json) | JSON | Machine-readable metrics, bootstrap distributions, and confusion matrices |
| [`ai-engine/tests/test_phase10_fusion.py`](../../ai-engine/tests/test_phase10_fusion.py) | Pytest Suite (6 tests) | Automated regression and validation tests for Phase 10 |
| [`phase10_fusion_report.md`](phase10_fusion_report.md) | Markdown Report | This comprehensive scientific research deliverable |

---

## 9. Final Phase 10 PASS/FAIL Checklist

| Check | Item Description | Result |
| :---: | :--- | :---: |
| **1** | Frozen Protocol C manifest checksum verified (`c15300...`) | **PASS** |
| **2** | Zero retraining of Phase 6 or Phase 9 baselines | **PASS** |
| **3** | Exact video ID and label alignment verified across all 8,722 videos | **PASS** |
| **4** | Preprocessor (StandardScaler) fit strictly on training partition | **PASS** |
| **5** | Operating threshold ($\tau^* = 0.4999$) derived strictly on validation data | **PASS** |
| **6** | Single-pass evaluation on frozen test partition ($N=1,308$) | **PASS** |
| **7** | Honest reporting of performance gains and bootstrap statistical significance | **PASS** |
| **8** | Phase 10 unit test suite passes (6 / 6 passed) | **PASS** |
| **9** | Full regression test suite passes (296 / 296 passed) | **PASS** |
| **10** | All required script, model, and report artifacts created | **PASS** |

**Conclusion:** Phase 10 is **100% COMPLETE**. Multimodal fusion empirically improves deepfake detection performance over the visual-only baseline on Protocol C ($0.7821 \rightarrow 0.7952$ ROC-AUC, $0.7550 \rightarrow 0.7799$ PR-AUC, $0.1991 \rightarrow 0.1866$ Brier score) with marginal statistical significance ($p = 0.031$ one-sided, $96.9\%$ bootstrap confidence).
