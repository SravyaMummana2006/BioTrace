# BIOTrace Phase 8 — Scientific & Statistical Validation Report

**Date:** 2026-11-04  
**Evaluator:** Senior Research Data Engineer & Forensic Computer Vision Architect  
**Dataset Scope:** HiDF Training Partition (`protocol_c_split_manifest.csv`)  
**Sample Inspected:** 200 Videos (100 Authentic Real, 100 Manipulated Fake — 100 Frozen Protocol C Pairs)  
**Frozen Manifest SHA-256:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`  
**Final Decision:** **CONDITIONAL GO (with Mandatory Feature Exclusions & Protocol Guardrails)**  

---

## Executive Summary

Before approving Phase 9 full-dataset extraction and classifier training, this read-only scientific validation re-evaluated the Phase 8 feasibility experiment across the physics of photoplethysmography (rPPG), the signal-processing mechanics of spatial multi-region consistency, the statistical validity of class differences under multiple testing correction (Benjamini-Hochberg FDR), and dataset confounding factors.

### Critical Scientific Findings
1. **Physical Impossibility of Clinical rPPG at $T \approx 2.96$s:** In a 72-frame, 24 FPS video ($2.961$s presentation span), the fundamental Rayleigh physical frequency resolution limit is $\Delta f = \frac{1}{T} \approx 0.338\text{ Hz}$, corresponding to **$\pm 20.3\text{ BPM}$**. At normal resting heart rates (60–100 BPM), only **$2.5$ to $4.9$ cardiac cycles** occur. True physiological vitals estimation (e.g. heart rate in BPM) is physically impossible; reported sub-beat estimates are mathematical artifacts of FFT zero-padding sinc interpolation.
2. **Spatial Consistency Features Are Genuinely Invariant ($p > 0.06$, $q > 0.20$):** Although the 7 recommended spatial consistency features could be computed on the 72-frame POS pulse arrays, rigorous statistical testing reveals **zero discriminative power**. For example, bilateral cheek peak cross-correlation is identical between real and fake ($0.3603$ vs. $0.3611, p = 0.998$). All 7 spatial features must be **excluded**.
3. **Multiple Testing Correction Collapses Significance:** At standard FDR $\alpha = 0.05$, **zero biological features survive Benjamini-Hochberg correction**. Out of 52 candidate features tested, only container `file_size_bytes` ($q = 6.70 \times 10^{-5}$) achieves significance. At FDR 10% ($q < 0.10$), only two biological features survive as exploratory trends: `rppg_pos_chrom_corr` ($q = 0.0787$) and `ear_right_std` ($q = 0.0787$).
4. **Major Confounding by Video Re-compression:** Fake videos are systematically smaller in file size than real videos ($154.2\text{ KB}$ vs. $178.9\text{ KB}$, $p = 1.29 \times 10^{-6}$, Cohen's $d = -0.659$). Because real and fake samples are 1-to-1 paired from identical base donor videos, this difference is an artifact of secondary deepfake video re-encoding (compression blur), which also artificially depresses eyelid edge variance (`ear_right_std`).
5. **Final Recommendation — CONDITIONAL GO:** Full extraction is approved **only** under an exploratory forensic protocol that restricts the model to a refined set of **8 continuous facial kinematic and color-projection features**, completely discarding all clinical pulse vitals, all spatial consistency features, all discrete blink counts, and all container-level compression shortcuts.

---

## 1. rPPG Validity & Physical Spectral Resolution Audit

### 1.1 Temporal Duration & Number of Cardiac Cycles
All HiDF videos in the sample have 72 frames at 23.976–24.0 FPS. Presentation timestamps span:
$$\Delta T = t_{71} - t_0 = \frac{71}{23.976\text{ s}^{-1}} = 2.961\text{ seconds}$$

In human physiology, normal cardiac frequencies range from $0.75\text{ Hz}$ to $2.50\text{ Hz}$ (45 to 150 BPM):
- At $60\text{ BPM}$ ($1.00\text{ Hz}$): $\text{Cycles} = 2.961\text{ s} \times 1.00\text{ s}^{-1} = \mathbf{2.96\text{ cycles}}$.
- At $75\text{ BPM}$ ($1.25\text{ Hz}$): $\text{Cycles} = 2.961\text{ s} \times 1.25\text{ s}^{-1} = \mathbf{3.70\text{ cycles}}$.
- At $100\text{ BPM}$ ($1.67\text{ Hz}$): $\text{Cycles} = 2.961\text{ s} \times 1.67\text{ s}^{-1} = \mathbf{4.94\text{ cycles}}$.

A signal consisting of only 3 to 5 periods is heavily corrupted by window edge truncation (spectral leakage) and respiration/vasomotor low-frequency drift.

### 1.2 Rayleigh Physical Frequency Resolution Limit
The physical frequency resolution limit $\Delta f_{\text{physical}}$ of any finite observation window of length $T$ is governed by the Fourier uncertainty principle:
$$\Delta f_{\text{physical}} = \frac{1}{T} = \frac{1}{2.961\text{ s}} \approx \mathbf{0.3377\text{ Hz}}$$

Converting to physiological beats per minute:
$$\Delta \text{BPM}_{\text{physical}} = \Delta f_{\text{physical}} \times 60 = 0.3377 \times 60 \approx \mathbf{20.26\text{ BPM}}$$

> [!IMPORTANT]
> Components closer than $20.3\text{ BPM}$ cannot be physically resolved as distinct spectral peaks. If an individual has a true heart rate of $72\text{ BPM}$, the physical Rayleigh bin encompasses the entire range from $[62\text{ BPM}, 82\text{ BPM}]$. 

### 1.3 Physical Resolution vs. Zero-Padded DTFT Grid Interpolation
In `compute_welch_psd` ([`filtering.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/src/biotrace/rppg/filtering.py#L122)), the FFT length is zero-padded to `nfft = max(2048, nperseg * 4) = 2048`.
This zero-padding evaluates the Discrete-Time Fourier Transform (DTFT) on a dense grid with spacing:
$$\delta f_{\text{grid}} = \frac{f_s}{N_{\text{fft}}} = \frac{24.0\text{ Hz}}{2048} = 0.0117\text{ Hz} \equiv \mathbf{0.703\text{ BPM}}$$

**Scientific Distinction:** Zero-padding provides purely trigonometric sinc-interpolation between adjacent discrete Fourier bins. It interpolates the smooth continuous Fourier spectrum, but **generates zero physical resolving power**. Reporting pulse estimates with decimal precision (e.g. $82.4\text{ BPM}$) in a 3-second recording is a deceptive mathematical interpolation artifact.

### 1.4 POS/CHROM Correlation & De Haan SNR at 2.96s
- **De Haan SNR:** In `calculate_de_haan_snr`, the fundamental harmonic band $[f_p - \delta, f_p + \delta]$ with $\delta = 0.18\text{ Hz}$ has bandwidth $2\delta = 0.36\text{ Hz}$, which equals almost exactly one physical Rayleigh bin ($0.338\text{ Hz}$). Therefore, SNR in a 3-second window measures local energy concentration in a single Fourier bin relative to total variance, rather than a stable clinical SNR.
- **POS/CHROM Waveform Correlation:** POS (Plane-Orthogonal-to-Skin, Wang et al. 2017) and CHROM (Chrominance-based, de Haan & Jeanne 2013) project the 3-channel RGB skin trace onto two different color planes. Over 72 frames, the Pearson correlation between these two 72-point 1D signals measures whether spatial color variations project consistently across color spaces. This is an **exploratory color-space projection consistency metric**, not a clinically validated pulse measurement.

---

## 2. Spatial Feature Validity & Calculation Audit

The Phase 8 feasibility report noted that `SpatialConsistencyExtractor` assigned `GlobalConsistencyState.insufficient_evidence` to 100% of videos, yet recommended 7 spatial features. We audited the exact mathematical trace of these features:

### 2.1 Trace of Calculations
1. **Regional Pulse Extraction:** For each facial ROI (forehead, left cheek, right cheek), `assess_regional_signal` executed Wang et al.'s sliding-window POS projection ($L = 1.6\text{s} \approx 38\text{ frames}$, stride $= 1$), detrending, and zero-phase Butterworth filtering $[0.7, 3.0]\text{ Hz}$. Because $72\text{ frames} > 38\text{ frames}$, 72-sample 1D waveforms ($p_{\text{fore}}$, $p_{\text{left}}$, $p_{\text{right}}$) were successfully synthesized.
2. **Clinical Gating Rejection:** The high-level module (`SpatialBiologicalAnalyzer`) checks:
   ```python
   if overlap_duration_seconds < config.min_observation_seconds: # 2.96s < 3.00s
       return PairwiseComparison(state=PairConsistencyState.INSUFFICIENT_DATA, zero_lag_correlation=None, ...)
   ```
   Because $2.961\text{s} < 3.000\text{s}$, the clinical analysis rejected the comparison as `INSUFFICIENT_DATA`.
3. **Bypassing the Gate in the Extraction Script:** The Phase 8 feasibility script called the underlying mathematical helper `compute_lagged_cross_correlation(p_left, p_right, meta.fps, 0.25)` directly on the synthesized arrays, bypassing the high-level duration gate.
4. **Missing-Value Handling & Defaults:**
   - When a region had untracked skin pixels, $p_{\text{left}}$ or $p_{\text{right}}$ was `None`. In those cases (13 videos, $6.5\%$), the script correctly assigned `np.nan`.
   - **No invalid measurements were replaced with zeros or defaults.** The 187 successfully computed values were genuine Pearson correlations over the 72 frames.

### 2.2 Lack of Discriminative Power
Although mathematically computable, empirical evaluation demonstrates that **none of the 7 spatial features differ between authentic and manipulated videos**:

| Spatial Feature | Real Mean $\pm$ Std | Fake Mean $\pm$ Std | Cohen's $d$ | Mann-Whitney $p$ | FDR Corrected $q$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `spatial_snr_forehead` | $-2.66 \pm 1.84$ dB | $-2.51 \pm 1.77$ dB | $+0.087$ | $0.6130$ | **$0.9331$ (ns)** |
| `spatial_snr_left_cheek` | $-3.05 \pm 3.12$ dB | $-3.03 \pm 3.23$ dB | $+0.008$ | $0.4186$ | **$0.7256$ (ns)** |
| `spatial_snr_right_cheek`| $-3.57 \pm 3.79$ dB | $-2.79 \pm 2.99$ dB | $+0.226$ | $0.0640$ | **$0.2080$ (ns)** |
| `spatial_cheek_zero_lag_corr` | $+0.108 \pm 0.301$ | $+0.110 \pm 0.292$ | $+0.007$ | $0.8299$ | **$0.9331$ (ns)** |
| `spatial_cheek_max_cross_corr`| $+0.360 \pm 0.177$ | $+0.361 \pm 0.180$ | $+0.004$ | $0.9978$ | **$0.9978$ (ns)** |
| `spatial_cheek_lag_seconds` | $-0.006 \pm 0.169$ s | $+0.028 \pm 0.167$ s | $+0.204$ | $0.2928$ | **$0.5856$ (ns)** |
| `spatial_forehead_left_corr` | $+0.124 \pm 0.354$ | $+0.106 \pm 0.354$ | $-0.050$ | $0.7677$ | **$0.9331$ (ns)** |

`spatial_cheek_max_cross_corr` has a $p$-value of $0.9978$ (real $= 0.3603$, fake $= 0.3611$). Spatial biological consistency features provide **zero signal for deepfake classification** in this dataset and must be discarded.

---

## 3. Statistical Validity & Multiple Testing Correction (FDR)

In Phase 8, 52 candidate features were evaluated using uncorrected two-sample Mann-Whitney U tests ($p < 0.05$). We recalculated the exact statistics and applied the standard **Benjamini-Hochberg False Discovery Rate (FDR)** procedure ($m = 52$):

$$q_{(i)} = \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right)$$

### 3.1 Full Statistical Audit Table (Ranked by Significance)

| Rank | Feature | Category | Real Mean | Fake Mean | Cohen's $d$ | Raw $p$ (MWU) | Welch $t$ $p$ | FDR $q$-value | Status ($\alpha=0.05$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `file_size_bytes` | Container | $178,857$ | $154,191$ | $-0.659$ | $1.29 \times 10^{-6}$ | $6.10 \times 10^{-6}$ | **$6.70 \times 10^{-5}$** | **Significant** (Confounder) |
| 2 | `total_wall_time_s` | Execution | $1.8450$s | $1.6169$s | $-0.290$ | $0.0028$ | $0.0431$ | **$0.0726$** | Not Sig at $5\%$ (Trend) |
| 3 | `rppg_exec_time_s` | Execution | $0.6116$s | $0.5357$s | $-0.276$ | $0.0070$ | $0.0535$ | **$0.0787$** | Not Sig at $5\%$ (Trend) |
| 4 | `ear_right_std` | Kinematics | $0.0274$ | $0.0229$ | $-0.321$ | $0.0073$ | $0.0241$ | **$0.0787$** | Not Sig at $5\%$ (Trend) |
| 5 | `rppg_pos_chrom_corr` | rPPG | $-0.7599$ | $-0.6862$ | $+0.299$ | $0.0076$ | $0.0356$ | **$0.0787$** | Not Sig at $5\%$ (Trend) |
| 6 | `ear_bino_std` | Kinematics | $0.0269$ | $0.0225$ | $-0.364$ | $0.0119$ | $0.0108$ | **$0.1030$** | Not Significant |
| 7 | `ear_left_std` | Kinematics | $0.0285$ | $0.0240$ | $-0.294$ | $0.0187$ | $0.0401$ | **$0.1335$** | Not Significant |
| 8 | `rppg_pos_snr_db` | rPPG | $-0.0540$ | $+0.4960$ | $+0.290$ | $0.0205$ | $0.0405$ | **$0.1335$** | Not Significant |
| 9 | `ear_opening_vel_max` | Kinematics | $0.8956$ | $0.7623$ | $-0.296$ | $0.0293$ | $0.0381$ | **$0.1533$** | Not Significant |
| 10 | `ear_velocity_mean` | Kinematics | $0.2077$ | $0.1818$ | $-0.345$ | $0.0295$ | $0.0158$ | **$0.1533$** | Not Significant |
| 11 | `ear_closing_vel_max` | Kinematics | $1.1728$ | $0.9906$ | $-0.282$ | $0.0380$ | $0.0487$ | **$0.1550$** | Not Significant |
| 12 | `facial_exec_time_s` | Execution | $0.5848$s | $0.5135$s | $-0.289$ | $0.0411$ | $0.0573$ | **$0.1550$** | Not Significant |
| 13 | `rppg_pos_hr_bpm` | rPPG | $82.37$ | $76.77$ | $-0.254$ | $0.0447$ | $0.0763$ | **$0.1550$** | Not Significant |
| 14 | `rppg_pos_peak_freq_hz`| rPPG | $1.3728$ | $1.2795$ | $-0.254$ | $0.0447$ | $0.0763$ | **$0.1550$** | Not Significant (Duplicate) |
| 15 | `ear_velocity_max` | Kinematics | $1.2674$ | $1.0878$ | $-0.275$ | $0.0447$ | $0.0548$ | **$0.1550$** | Not Significant |
| 16 | `spatial_snr_right_cheek`| Spatial | $-3.5663$ | $-2.7940$ | $+0.226$ | $0.0640$ | $0.1132$ | **$0.2080$** | Not Significant |
| 17 | `rppg_pos_peak_prominence`| rPPG | $0.6825$ | $0.6331$ | $-0.183$ | $0.0946$ | $0.1989$ | **$0.2894$** | Not Significant |
| 18 | `spatial_exec_time_s` | Execution | $0.6081$s | $0.5252$s | $-0.311$ | $0.1150$ | $0.0435$ | **$0.3323$** | Not Significant |
| 19 | `spatial_bpm_left_cheek`| Spatial | $78.31$ | $75.00$ | $-0.124$ | $0.1274$ | $0.3831$ | **$0.3423$** | Not Significant |
| 20 | `ear_bino_min` | Kinematics | $0.2095$ | $0.2179$ | $+0.188$ | $0.1317$ | $0.1878$ | **$0.3423$** | Not Significant |

### 3.2 Critical Takeaways from Multiple Testing
1. **Zero Biological Significance at FDR 5%:** When controlling for the false discovery rate across 52 tested hypotheses, not a single biological or kinematic feature achieves $q < 0.05$.
2. **Exploratory Trends at FDR 10%:** Only two biological features achieve $q < 0.10$: `ear_right_std` ($q = 0.0787$) and `rppg_pos_chrom_corr` ($q = 0.0787$).
3. **Severe Multi-Collinearity / Feature Duplication:**
   - `rppg_pos_hr_bpm` and `rppg_pos_peak_freq_hz` are mathematically identical ($r = +1.0000, \text{BPM} = 60 \times \text{freq}$).
   - `ear_velocity_max` and `ear_closing_vel_max` have $r = +0.9107$.
   - `ear_bino_std` and `ear_left_std` have $r = +0.8784$.
   - Execution times (`rppg_exec_time_s`, `facial_exec_time_s`, `total_wall_time_s`) are all pairwise correlated at $r > 0.95$.

---

## 4. Dataset Pairing, Leakage & Confounding Audit

### 4.1 Sample Integrity & Pairing Verification
- **Manifest Verification:** The 200 videos in [`phase8_sample_features.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase8_sample_features.csv) strictly match the first 100 rows of the frozen `protocol_c_split_manifest.csv` filtered by `split == 'train'`.
- **Identity Disjointness:** No validation or test partition videos were included.
- **1-to-1 Base Identity Pairing:** The 100 real videos and 100 fake videos share the **exact same 100 base identities** (`c00003`, `c00004`, etc.). Each fake video is a face-swapped version of its corresponding real video using a donor target from the training set.

### 4.2 Confounding by Video Compression & Generation Artifacts
Because real and fake videos share identical base actors, scenes, camera sensors, lighting, and head poses, any observed macro-level differences must arise strictly from the deepfake generation and rendering pipeline:

```
[Real Base Video] ──(Deepfake Generation / H.264 Re-encode)──> [Fake Video]
   178.9 KB                                                        154.2 KB
   p = 1.29e-6                                                     Cohen's d = -0.659
```

1. **File Size Asymmetry:** Fake videos are on average **$24.7\text{ KB}$ smaller (13.8% smaller)** than real videos ($p = 1.288 \times 10^{-6}$). The neural face synthesis pipeline re-encoded the videos with higher compression / quantization, stripping high-frequency skin textures.
2. **Execution Time Asymmetry:** Real videos took $1.85$s to process, while fake videos took $1.62$s ($p = 0.0028$). OpenCV video frame decoding and skin patch mean pooling execute measurably faster on more heavily compressed, low-entropy frames.
3. **Eyelid Micro-Variability Confounding:** The reduction in `ear_right_std` ($0.0274 \rightarrow 0.0229$) and eyelid velocities in fake videos is partially caused by **temporal edge smoothing and frame blending** introduced by the reenactment model, which slightly blurs eyelid margins compared to the sharp, uncompressed native camera footage.

> [!CAUTION]
> **Shortcut Learning Warning:** A machine learning model trained naively on raw features could easily classify deepfakes based on compression bitrate or container properties rather than genuine biological signals. All container metrics (`file_size_bytes`) and execution timers must be strictly barred from the feature matrix.

---

## 5. Corrected Feature Recommendation & Taxonomy

Based on physical constraints, missingness rates, and FDR-adjusted statistical power, all evaluated features are classified into three strict categories:

```mermaid
flowchart TD
    AllFeatures["52 Feasibility Features Evaluated"] --> Unsuitable["Unsuitable (36 Features)\n- 15 Discrete Blink Metrics (100% NaN)\n- 7 Spatial Consistency Features (q > 0.20)\n- 8 Near-Zero Variance Constants\n- 4 Container / Timing Confounders\n- 2 Mathematical Duplicates"]
    AllFeatures --> Exploratory["Exploratory Short-Window (6 Features)\n- rPPG POS/CHROM Correlation\n- rPPG POS Peak Frequency\n- rPPG POS SNR\n- rPPG Peak Prominence\n- Bilateral Eyelid Asymmetry Mean & Std"]
    AllFeatures --> Reliable["Reliably Measurable & Discriminative (4 Features)\n- EAR Micro-Variability (ear_right_std)\n- EAR Binocular Dispersion (ear_bino_std)\n- Mean Eyelid Velocity (ear_velocity_mean)\n- Max Opening Velocity (ear_opening_vel_max)"]
```

### 5.1 Category Breakdown

| Feature Name | Modality | Status | Missing % | FDR $q$-value | Rationale / Forensic Caveat |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ear_right_std` | Facial | **Reliable** | $0.0\%$ | $0.0787$ | Captures eyelid micro-tremor attenuation in synthetic media; low missingness. |
| `ear_bino_std` | Facial | **Reliable** | $0.0\%$ | $0.1030$ | Robust binocular eyelid dispersion; $d = -0.364$. |
| `ear_velocity_mean` | Facial | **Reliable** | $0.0\%$ | $0.1533$ | Measures mean temporal dynamic movement of the palpebral fissure. |
| `ear_opening_vel_max` | Facial | **Reliable** | $0.0\%$ | $0.1533$ | Captures sluggish reopening velocity in synthetic blending; $d = -0.296$. |
| `rppg_pos_chrom_corr` | rPPG | **Exploratory** | $0.0\%$ | $0.0787$ | Evaluates color-projection consistency across POS/CHROM; exploratory short-window. |
| `rppg_pos_snr_db` | rPPG | **Exploratory** | $0.0\%$ | $0.1335$ | Spectral peak concentration in single Rayleigh bin; exploratory. |
| `rppg_pos_peak_freq_hz`| rPPG | **Exploratory** | $0.0\%$ | $0.1550$ | Dominant spectral peak ($\pm 20.3\text{ BPM}$ physical resolution); exploratory. |
| `rppg_pos_peak_prominence`| rPPG | **Exploratory** | $0.0\%$ | $0.2894$ | Relative peak prominence in cardiac band; exploratory. |
| `blink_asymmetry_mean` | Facial | **Exploratory** | $0.0\%$ | $0.8422$ | Measures monocular landmark discrepancy; physiologically grounded. |
| `blink_asymmetry_std` | Facial | **Exploratory** | $0.0\%$ | $0.7684$ | Temporal variability of bilateral asymmetry; physiologically grounded. |
| **All Spatial Features (7)** | Spatial | **Unsuitable** | $0 - 6.5\%$ | $0.20 - 0.99$ | **Zero discriminative power ($p > 0.06$ to $0.99$)**; discard completely. |
| **Blink Counts & Rates (15)**| Facial | **Unsuitable** | $100.0\%$ | N/A | **Uncomputable in 3-second clips**; calibration fails; discard completely. |
| **Duplicates (2)** | rPPG/Facial | **Unsuitable** | $0.0\%$ | N/A | `rppg_pos_hr_bpm` (exact copy of freq), `ear_velocity_max` (copy of close vel). |
| **Confounders (4)** | Container | **Unsuitable** | $0.0\%$ | $q < 0.08$ | `file_size_bytes`, `total_wall_time_s`, `rppg_exec_time_s`, `facial_exec_time_s`. |

---

## 6. Final Decision & Phase 9 Protocol Guardrails

### Decision: **CONDITIONAL GO**

Full-dataset extraction and baseline classifier training are approved under the following **strict conditions**:

1. **Feature Vector Restriction (Maximum 10 Features):**
   The feature matrix $\mathbf{X} \in \mathbb{R}^{N \times D}$ for Phase 9 must contain **at most 10 features**:
   - 4 Reliable Facial Kinematic features (`ear_right_std`, `ear_bino_std`, `ear_velocity_mean`, `ear_opening_vel_max`).
   - 4 Exploratory rPPG color-projection features (`rppg_pos_chrom_corr`, `rppg_pos_snr_db`, `rppg_pos_peak_freq_hz`, `rppg_pos_peak_prominence`).
   - 2 Exploratory Bilateral Asymmetry features (`blink_asymmetry_mean`, `blink_asymmetry_std`).
2. **Absolute Exclusion of Confounders and Spatial Features:**
   - All container metrics (`file_size_bytes`) and execution timers must be barred from model inputs to prevent shortcut learning.
   - All 7 spatial consistency features must be excluded to prevent adding non-discriminative noise.
   - All discrete blink rates and durations must be excluded.
3. **Scientific Terminology Governance:**
   - In all future reports and UI labels, rPPG features must be explicitly designated as **"Exploratory Short-Window Chromatic Consistency"** rather than "Cardiovascular Pulse Vitals" or "Clinical Heart Rate".
   - Clear disclaimers regarding the $20.3\text{ BPM}$ Rayleigh physical resolution limit must accompany any pulse frequency visualizations.
4. **Validation & Test Partition Isolation:**
   - Model development, hyperparameter selection, and threshold tuning must strictly use the frozen `train` partition features, evaluated via cross-validation or against the frozen `val` partition.
   - The frozen `test` partition must remain untouched until the final model checkpoint is locked.

---

## 7. Artifact Summary

The quantitative evidence supporting this validation is persisted in:
- Validation Statistics JSON: [`phase8_validation_stats.json`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase8_validation_stats.json)
- Validation Execution Script: [`scripts/validate_phase8_scientific.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/validate_phase8_scientific.py)
- Sample Features Dataset: [`phase8_sample_features.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase8_sample_features.csv)
- Phase 8 Feasibility Report: [`phase8_feasibility_report.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase8_feasibility_report.md)
