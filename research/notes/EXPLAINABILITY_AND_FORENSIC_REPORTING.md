# BIOTrace — Explainable AI and Forensic Reporting

**Technical & Scientific Research Note**  
**Module:** `biotrace.explainability`  
**Date:** November 2026  
**Status:** Implemented & Verified (Stage 7)

---

## 1. Executive Summary & Core Objective

The primary objective of BIOTrace Stage 7 is to build an explainability, provenance, and forensic reporting layer that makes the entire analysis pipeline transparent, traceable, auditable, and scientifically reproducible. 

In forensic and biometric computing, a machine-learning score or signal extraction algorithm must never act as an opaque "black box." An investigator, forensic examiner, or legal auditor must understand:
1. **What was measured:** The exact physical, kinematic, or optical substrate observed.
2. **What was usable:** Which segments or frames satisfied engineering quality gating.
3. **What was unreliable or unavailable:** Which modalities failed tracking, suffered from low SNR, or lacked model checkpoints.
4. **What visual detectors produced:** Uncalibrated logits, sigmoid scores, frame-level dispersion, and temporal autocorrelation.
5. **What physiological modules observed:** Remote photoplethysmography (rPPG) pulses, eye aspect ratios (EAR), blink velocities, and multi-region spatial concordance.
6. **Where evidence agrees or conflicts:** Explicit cross-modality concordance and divergence.
7. **What cannot scientifically be concluded:** Boundaries preventing over-interpretation, false attributions, or pseudo-probabilistic certainty.

---

## 2. Explanation Methodology & Architecture

BIOTrace adheres to strict architectural separation of concerns between **physiological biological evidence** and **visual deepfake classification**:

```
+-----------------------------------------------------------------------------------+
|                            BIOTrace Analysis Result                               |
+-----------------------------------------+-----------------------------------------+
|      Physiological Evidence Base        |      Visual Classification Base         |
|  - rPPG (Capillary Hemodynamics)        |  - Convolutional Baseline Head          |
|  - Facial Dynamics (Eyelid Kinematics)  |  - Frame-Level Raw Logits               |
|  - Spatial Multi-Region Consistency     |  - Sigmoid Heuristic Scores             |
|  - Biological Substrate Concordance     |  - Temporal Dispersion & Autocorrelation|
+-----------------------------------------+-----------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------------+
|                              Explainability Engine                                |
|  - Modality Explanations (What Measured, Usable, Unreliable, Caveats, Limits)    |
|  - Temporal Evidence Timeline (Synchronized Points & Explicit Tracking Gaps)      |
|  - Reproducibility Provenance (Hashes, Versions, Seeds, Redacted Paths)           |
|  - Responsible Grad-CAM Sensitivity (Target Layers, No Synthetic Fabrications)   |
+-----------------------------------------------------------------------------------+
                                     |
                                     v
+------------------------------------+----------------------------------------------+
|           JSON Audit Report        |              Vector PDF Case Report          |
|    - Deterministic Sorted Schema   |     - 12 Standard Forensic Sections          |
|    - Machine-Readable Audit Data   |     - Multi-Page Vector Waveforms & Tables   |
+------------------------------------+----------------------------------------------+
```

### 2.1 The Five-Tier Explanation Schema
Every modality explanation in `ModalityExplanation` implements five structured tiers:
1. **Observed Physical Measurement:** Explains the biological or physical phenomenon (e.g. sub-dermal capillary light absorption via POS/CHROM, palpebral aperture distance).
2. **Engineering Quality Assessment:** Assesses signal-to-noise ratio (SNR), tracking validity ratio, spectral resolution, and motion flags.
3. **Usable Observations:** Specific quantitative metrics that survived quality gating.
4. **Unreliable or Unavailable Observations:** Explicitly tags tracking loss, missing face crops, or uncalibrated weights.
5. **Scientific Caveats & Unsupported Conclusions:** Concrete guardrails prohibiting forensic over-reach.

---

## 3. Responsible Visual Explanation: Grad-CAM Methodology & Guardrails

Grad-CAM (Gradient-weighted Class Activation Mapping; Selvaraju et al., 2017) provides exploratory visual sensitivity maps for convolutional neural networks.

### 3.1 Mathematical Formulation
Let $y$ denote the model's raw uncalibrated scalar output logit for the target class. Let $A^k \in \mathbb{R}^{H \times W}$ represent the feature activation map of channel $k$ in the target convolutional layer.
The neuron importance weights $\alpha_k$ are calculated via global average pooling of gradients:
$$\alpha_k = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y}{\partial A^k_{i,j}}$$
The localized Grad-CAM heat-map $L_{\text{Grad-CAM}}$ is computed by taking a rectified linear combination of forward activation maps:
$$L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
$L_{\text{Grad-CAM}}$ is then normalized to $[0.0, 1.0]$ and bilinearly interpolated to the original facial crop resolution $(H_{\text{orig}}, W_{\text{orig}})$.

### 3.2 Target Convolutional Layers
- **Custom `LightweightVisualBaseline`:** Target layer is `stage4[-1]` (the final 384-channel residual convolution block before adaptive average pooling).
- **Torchvision ResNet:** Target layer is `layer4[-1]` (the terminal bottleneck/basic block).
- **Arbitrary Architectures:** Dynamically discovers the deepest `nn.Conv2d` module; if absent, gracefully returns `UNSUPPORTED_ARCHITECTURE`.

### 3.3 Forensic Safeguards Against Misleading Visualizations
1. **Model Weights Missing (`UNAVAILABLE_WEIGHTS_MISSING`):** If model weights are not loaded (`MODEL_UNAVAILABLE`), Grad-CAM returns an explicit unavailable status with zero heatmaps. Generating fake, random, or surrogate heatmaps is strictly forbidden under BIOTrace protocols.
2. **Sensitivity vs. Causal Attribution:** Grad-CAM illustrates where the convolutional kernels showed gradient responsiveness. It does **NOT** prove that a deepfake artifact, face-swap seam, or GAN fingerprint physically exists in the highlighted pixels.
3. **Non-Destructive Storage:** The original unmanipulated facial crop is preserved independently. Heatmaps are generated as standalone overlay representations.

---

## 4. Synchronized Temporal Evidence Explorer

Biometric and video integrity investigations require analyzing signals across time rather than collapsing entire media files into a single scalar average.

### 4.1 Synchronized Observation Points
The `TemporalTimeline` aligns multi-modal evidence across discrete timestamps:
- Left and right Eye Aspect Ratio (EAR) measurements and eyelid closure classifications (`CLOSED`, `CLOSING`, `OPENING`).
- rPPG optical Signal-to-Noise Ratio (SNR dB) and pulse frequency (BPM).
- Frame-level visual artifact detector raw logits and sigmoid scores.
- Spatial multi-region biological concordance states.

### 4.2 Explicit Tracking Dropouts (No Interpolation)
When head rotation, occlusion, or illumination loss degrades tracking:
- The interval is explicitly recorded in `TemporalGapInterval(start_seconds, end_seconds, duration_seconds, missing_modalities, reason)`.
- The system **never fabricates or interpolates** missing signal values across tracking gaps.
- Missing timestamps are rendered as highlighted dropout zones in forensic charts.

---

## 5. Technical Provenance & Path Sanitization

Scientific reproducibility requires recording the exact execution environment while respecting strict data privacy and security boundaries.

### 5.1 Privacy & Path Sanitization
- Local filesystem paths (e.g. `C:\Users\username\Desktop\...` or `/home/user/...`) are strictly redacted.
- Only the media basename (e.g. `evidence_recording_01.mp4`) and container format are exposed in reports.
- Environmental secrets, API credentials, and internal hostnames are completely excluded.

### 5.2 Cryptographic & Environmental Auditing
The `ForensicProvenance` schema records:
- Cryptographic SHA-256 digest of the input video stream.
- Exact module versions: `biotrace`, `python`, `torch`, `torchvision`, `opencv`, `numpy`, `matplotlib`.
- Deterministic random seeds (`torch_initial_seed`, `numpy_seed`).
- Architecture identifier and weights digest (SHA-256).
- Explicit `unavailable_metadata` list recording unobservable items (e.g. camera sensor serial number, lossless Bayer stream, calibrated probability prior).

---

## 6. Forensic Case Report Structure (JSON & PDF)

BIOTrace provides two deterministic report formats:

### 6.1 Deterministic JSON Report
- Adheres to the `ForensicReportData` Pydantic data contract.
- Serialized with sorted keys and uniform 2-space indentation.
- Machine-parseable for ingestion into laboratory information management systems (LIMS) and evidence repositories.

### 6.2 Multi-Page Vector PDF Report
Generated via `matplotlib.backends.backend_pdf.PdfPages` without external binary dependencies:
1. **Case & Asset Identification:** Title banner, Case Reference, Report ID, SHA-256 digest, video dimensions, container format, duration.
2. **Executive Summary:** Fused evidence state badge, sufficiency status, non-probabilistic synthesis narrative.
3. **Analysis Scope & Configuration:** Modality table listing assessed status, engineering confidence, and observed physical substrate.
4. **Individual Modality Findings:** Detailed breakdowns for rPPG, Facial Dynamics/EAR, Spatial Biological Consistency, and Visual Baseline.
5. **Synchronized Temporal Evidence Charts:** Multi-subplot vector waveforms displaying EAR kinematics, visual raw scores, and shaded tracking gaps.
6. **Quality Limitations & Gaps:** Comprehensive itemization of tracking dropouts, short duration warnings, and sensor noise.
7. **Reproducibility Audit Table:** Complete software environment versions, seeds, and architectural digests.
8. **Scientific Limitations & Legal Caveats:** Prominent non-probabilistic warnings and neutral conclusion statement.

---

## 7. Evidence Interpretation Rules & Scientific Limitations

Forensic examiners must adhere to the following mandatory interpretation principles:

1. **Physiological Consistency $\neq$ Proof of Authenticity:**  
   These modalities measure distinct physiological mechanisms and may provide complementary observations. Their statistical independence has not been established, and they share video-level confounders. Plausible rPPG waveforms and natural blink rates demonstrate that observed signals match human biological ranges. However, advanced diffusion and neural radiance field models can synthesize plausible motion. Physiological consistency supports biological plausibility, but does not prove genuine provenance.
2. **Physiological Degradation $\neq$ Proof of Manipulation:**  
   Noisy, weak, or absent rPPG signals commonly occur in genuine, unmanipulated videos due to ambient lighting, subject motion, camera sensor compression, or darker skin tones. An anomalous or absent physiological signal must **never** be cited as standalone proof of digital manipulation.
3. **Uncalibrated Scores $\neq$ Probabilities:**  
   The visual baseline classifier produces raw unbounded logits and uncalibrated sigmoid scores on $[0.0, 1.0]$. These scores reflect feature proximity in uncalibrated representation space. They are **not** probabilities of authenticity or manipulation. Calibrated probabilities require empirical Platt or isotonic calibration on verified operational distributions.
4. **Decoupled Synthesis:**  
   Visual classifier scores and physiological observations are separate evidence streams. Merging them into a monolithic "deepfake probability" is scientifically ungrounded and strictly prohibited.
5. **No Green/Red Binary Verdicts:**  
   BIOTrace outputs structured evidence states (`SUPPORTING_EVIDENCE`, `CONFLICTING_EVIDENCE`, `MIXED_EVIDENCE`, `INSUFFICIENT_EVIDENCE`). It does not issue binary authentic/fake labels or legal verdicts of guilt or innocence.

---

## 8. Verification & Test Evidence

Stage 7 has been comprehensively verified:
- **Python Test Suite:** 218 passing unit and integration tests (`ai-engine/tests`), including 12 targeted explainability tests covering provenance sanitization, missing modality handling, uncalibrated scores, Grad-CAM (available/unavailable/unsupported), timeline tracking gaps, report serialization, and FastAPI endpoints.
- **Java Client Test Suite:** 25 passing Maven tests (`java-client`), verifying JSON deserialization, asynchronous API clients, Swing navigation deck switching, and Explainable Evidence Console instantiation.
- **Zero Unauthorized Operations:** Zero datasets were downloaded, no external model training occurred, and no Git commits or pushes were executed.
