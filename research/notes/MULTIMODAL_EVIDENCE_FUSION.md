# Multimodal Evidence Fusion in Biological Media Forensics

**Module:** BIOTrace Stage 5 & 5.1 Evidence Fusion Engine  
**Package:** `biotrace.evidence_fusion`  
**Status:** Implemented, Corrected, and Verified  

---

## 1. Scientific Motivation & Problem Statement

### 1.1 The Vulnerability of Unimodal Forensics
Unimodal forensic systems—whether purely pixel-based convolutional detectors or single-modality physiological extractors (e.g., global rPPG alone)—suffer from severe vulnerability to domain shift, compression artifacts, and environmental noise:
1. **False Positives from Technical Degradation:** Motion blur, rolling shutter artifacts, low frame rates, and poor lighting can destroy the subtle diffuse reflectance changes ($\Delta I / I \approx 0.1\%$) utilized by rPPG. If an algorithm equates "no detectable pulse" with "deepfake," compressed authentic videos will be falsely classified as synthetic.
2. **False Negatives from Physical / Facial Reenactment:** Puppetry or face-swap deepfakes can preserve real physiological signals if the source donor video had strong cardiac signals or if lighting variations induce periodic reflectance.

### 1.2 The Role of Multimodal Evidence Fusion
Multimodal evidence fusion addresses unimodal fragility not by computing weighted score averages, but by organizing, comparing, and auditing cross-system biological observations:
- Evaluating whether autonomic cardiovascular dynamics and voluntary/spontaneous motor kinematics are both observable.
- Preserving measurement uncertainty and engineering limitations explicitly.
- Avoiding naive statistical independence assumptions between correlated modalities and shared sensors.

---

## 2. Multi-System Biological Architecture & Dependency Analysis

A foundational contribution of Stage 5.1 is the formal **tripartite distinction** between physiological mechanisms, measurement modalities, and statistical independence:

```
                          ┌─────────────────────────────────────────────────────────┐
                          │               Input Video Stream & Camera               │
                          │  Shared Confounders: Motion, Illumination, Compression, │
                          │             Face-Tracking Loss, Frame Rate              │
                          └────────────────────────────┬────────────────────────────┘
                                                       │
                     ┌─────────────────────────────────┴─────────────────────────────────┐
                     ▼                                                                   ▼
┌──────────────────────────────────────────────┐                   ┌──────────────────────────────────────────────┐
│        Optical Cardiovascular Family         │                   │            Facial Dynamics Family            │
│  - Remote Photoplethysmography (rPPG)        │                   │  - Eye Aspect Ratio (EAR) Kinematics         │
│  - Spatial Biological Consistency            │                   │  - Spontaneous Blink Temporal Rate           │
├──────────────────────────────────────────────┤                   ├──────────────────────────────────────────────┤
│ Physiological Mechanism: Cardiac Ventricular │                   │ Physiological Mechanism: Central Motor       │
│ Stroke Volume & Dermal Capillary Perfusion   │                   │ Pacemaking (Cranial Nerves III & VII)        │
├──────────────────────────────────────────────┤                   ├──────────────────────────────────────────────┤
│ Measurement Modality: Pixel Chrominance      │                   │ Measurement Modality: 2D/3D Geometric        │
│ Diffuse Reflectance (~0.1% intensity shift)  │                   │ Landmark Coordinates & Palpebral Distances   │
├──────────────────────────────────────────────┤                   ├──────────────────────────────────────────────┤
│ Intra-Family Dependency: COUPLED             │                   │ Intra-Family Dependency: COUPLED             │
│ Consolidated Optical Channel (Anti-Double)   │                   │ Descriptive Kinematic Observation Channel    │
└──────────────────────┬───────────────────────┘                   └──────────────────────┬───────────────────────┘
                       │                                                                  │
                       └─────────────────────────────────┬────────────────────────────────┘
                                                         │
                                                         ▼
                                       ┌───────────────────────────────────┐
                                       │ Multimodal Evidence Fusion Engine │
                                       │ - Dependency-Aware Consolidation  │
                                       │ - Quality Gating & Failure Decoup.│
                                       │ - Non-Probabilistic State Output  │
                                       └───────────────────────────────────┘
```

### 2.1 The Tripartite Distinction
1. **Different Physiological Mechanisms:**
   - *Optical Cardiovascular System:* Autonomous cardiac ventricular systole and diastole pump blood through facial dermal capillary networks, modulating subcutaneous blood volume.
   - *Facial Dynamics System:* Brainstem motor pacemaking and cortical networks stimulate Cranial Nerves III (Oculomotor, *levator palpebrae superioris*) and VII (Facial, *orbicularis oculi*) to execute voluntary and spontaneous eyelid closures.
2. **Different Measurement Modalities:**
   - *Optical Cardiovascular:* Analyzes subtle (~0.1%) RGB color variations across skin regions (CHROM, POS).
   - *Facial Dynamics:* Tracks the geometric distance between upper and lower eyelid landmarks relative to the horizontal palpebral fissure (EAR).
3. **Absence of Statistical Independence:**
   A critical scientific error is conflating different physiological mechanisms with statistical independence. In computer vision media forensics, **facial dynamics and optical cardiovascular measurements are NOT statistically independent conditioned on the video observations**:
   $$\mathbb{P}(\text{rPPG}, \text{Blink} \mid \text{Video}) \neq \mathbb{P}(\text{rPPG} \mid \text{Video}) \times \mathbb{P}(\text{Blink} \mid \text{Video})$$
   They share pervasive video-level confounders:
   - **Subject Rigid & Non-Rigid Motion:** Head turns, nodding, talking, and laughing disrupt landmark tracking (causing false blink events or tracking failure) and concurrently inject severe motion artifacts into skin-tone color channels.
   - **Illumination Changes:** Global lighting shifts, screen reflections, flicker, and shadows alter chromatic signals (rPPG noise) and change edge contrast around the eyes (degrading landmark detection accuracy).
   - **Video Compression Artifacts:** Macroblocking, temporal motion-vector quantization, and low bitrates blur eyelid contours and wash out high-frequency dermal pulsatile variations.
   - **Face-Tracking & ROI Stability:** If the face detector or 3D mesh loses the subject, both rPPG dermal skin sampling and EAR landmark tracking fail simultaneously.
   - **Camera Hardware & Acquisition Limits:** Sub-30 fps capture rates, exposure rolling shutter, and motion blur degrade temporal resolution across both modalities simultaneously.

### 2.2 Optical Cardiovascular Consolidation (Anti-Double-Counting)
- **Shared Physical Substrate:** Both global rPPG and multi-region spatial consistency observe the same microvascular blood volume pulsations through the same camera sensor.
- **Rule Against Double-Counting:** Agreeing rPPG and spatial consistency measurements represent mutual optical consistency, **not two independent affirmative votes**. They are consolidated into a single optical cardiovascular channel.
- **Intra-Family Discrepancies:** While agreeing optical modalities cannot double-count affirmative evidence, a genuine discrepancy between them (e.g., global rPPG reports 72 BPM while regional cheek pulses exhibit $>20$ BPM divergence or anti-phase correlation under uniform conditions) registers an internal contradiction (`EvidenceFamilyAgreement.DISAGREEMENT`).

### 2.3 Descriptive Nature of Blink Kinematics
- Blink rate, EAR baselines, blink duration, and eye-closure velocities are **descriptive physiological observations**.
- Normal spontaneous human blink rates vary widely (from 3 to 45 blinks/min) depending on cognitive load, visual concentration, ambient humidity, fatigue, or contact lenses.
- Consequently, a normal blink pattern is **never proof of authenticity**, and an atypical blink pattern is **never proof of manipulation**.
- **Stage 5.1 Mandate:** If only blink observations are available, the fusion engine returns `INSUFFICIENT_EVIDENCE` while preserving all extracted measurements, diagnostics, and limitations.

---

## 3. Removal of Arbitrary Numerical Family Weighting

Stage 5.1 eliminates all arbitrary numerical family weighting (e.g., $w = 0.5, 1.0$) and weighted score averaging:
1. **Rejection of Pseudo-Probabilities:** Weighted score combinations create a veneer of mathematical precision that is ungrounded in empirical calibrative priors.
2. **Transparent Dependency Consolidation:** Modalities are grouped into evidence families with explicit qualitative consensus states:
   - `AGREEMENT`: Modalities within the family exhibit mutually consistent measurements.
   - `DISAGREEMENT`: Modalities within the family exhibit contradictory measurements.
   - `SINGLE_SOURCE`: Only one modality was evaluated in the family.
   - `INDETERMINATE`: Signal quality is borderline, degraded, or uncalibrated.
   - `NO_DATA`: No modalities were available.
3. **Separate Preservation of Measurements:** Global rPPG heart rate, regional spatial consistency states, and facial dynamics blink rates are stored separately in `modality_evidence`, guaranteeing full forensic auditability.

---

## 4. Evidence State Taxonomy

BIOTrace produces one of four conservative evidence states describing physiological consistency:

| Evidence State | Formal Definition | Operational Criteria |
| :--- | :--- | :--- |
| `SUPPORTING_EVIDENCE` | Available modalities across biological families exhibit physiologically plausible, high-quality, mutually agreeing signals without unresolvable contradictions. | Optical cardiovascular channels observe consistent dermal blood-flow pulses (and, if facial dynamics is available, motor kinematics exhibit plausible spontaneous blinking). |
| `CONFLICTING_EVIDENCE` | High-quality, reliable signals across or within families exhibit genuine physiological contradictions. | High engineering confidence with unresolvable regional pulse divergence, incompatible cross-region frequencies, or severe physiological contradiction. |
| `MIXED_EVIDENCE` | Heterogeneous modalities yield both supporting observations and ambiguous or divergent findings. | E.g., facial dynamics observes spontaneous blinking, but optical cardiovascular channels exhibit internal regional frequency discordance. |
| `INSUFFICIENT_EVIDENCE` | Observations are unavailable, uncalibrated, below minimum observation duration, degraded by noise/tracking loss, or consist solely of blink observations. | Zero optical cardiovascular channels achieved acceptable quality; or only descriptive blink observations are available. |

---

## 5. Scientific Safeguards & Forensic Guardrails

### 5.1 Decoupling Quality Degradation from Authenticity
- Low SNR (due to dim lighting, sensor noise, or distance) is classified as an engineering limitation (`quality_limitations`), **never** as negative evidence of authenticity.
- Face tracking loss or eye occlusion is tagged as an engineering limitation.
- These technical limitations **NEVER** set `is_manipulated = True` or lower an authenticity score.

### 5.2 Explicit Absence of Binary Manipulation Labels
- `ModalityVerdict.authenticity_score` is strictly `None`.
- `ModalityVerdict.is_authenticity_assessed` is strictly `False`.
- `AnalysisResult.overall_authenticity_score` is strictly `None`.
- `AnalysisResult.is_manipulated` is strictly `None`.

### 5.3 Graceful Degradation & Failure Decoupling
- If an extractor encounters an unexpected exception or hardware fault, `JobManager` isolates the fault to that modality's verdict.
- Remaining modalities proceed through fusion normally.
- If only one modality is requested (e.g., rPPG alone), the fusion engine evaluates that modality's evidence without penalizing the unrequested channels.

---

## 6. What Evidence Fusion Is and Is Not

| What It Is | What It Is Not |
| :--- | :--- |
| A structured synthesis of observed biological and kinematic signals. | A deepfake detection classifier. |
| A transparent record of measurement agreements, discrepancies, and limitations. | A black-box probability generator. |
| A scientific tool that explicitly quantifies and preserves engineering limitations. | A system that equates degraded video quality with synthetic forgery. |
| A dependency-aware model that prevents double-counting coupled optical sensors. | A naive score-averaging algorithm. |
| A system that treats blink kinematics as descriptive physiological observations. | A system that claims blink rates prove video authenticity or manipulation. |
