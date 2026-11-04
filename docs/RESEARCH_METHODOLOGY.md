# BIOTrace Research Methodology & Experimental Protocol

This document details the scientific foundations, dataset partitioning protocols, mathematical formulations, and validation procedures implemented in the BIOTrace media authenticity platform.

---

## 1. Research Hypothesis & Core Objectives

Generative face synthesis, diffusion reenactments, and deepfake face swaps optimize primarily for static pixel-level realism and spatial perceptual appearance. However, living human biology is governed by strict physiological and temporal constraints that are difficult to synthesize consistently across facial regions and time:

1. **Subcutaneous Blood Volume Pulse (rPPG)**: Periodic chrominance variations resulting from cardiac pulsation.
2. **Ocular & Facial Kinematics**: Rapid, coordinated muscular movements governing eyelid closure and opening trajectories.
3. **Bilateral Symmetry**: Coordinated muscular contractions between left and right facial compartments.

### Central Research Question:
> *Can multimodal physiological and temporal facial evidence provide complementary information to deep convolutional visual representations for detecting manipulated facial videos under strict, identity-isolated evaluation?*

---

## 2. Dataset Design & Protocol C Partitioning

Evaluations in deepfake literature frequently suffer from **identity leakage**, where facial features of the same actor appear in both training and test sets. When identity leakage occurs, machine learning classifiers memorize facial structure rather than learning generalizable synthetic artifacts.

### 2.1 Bipartite Actor Isolation
Face manipulation datasets involve pairs of actors: a **source actor** (whose facial expressions or identity are transferred) and a **target actor** (whose body or background is retained).

Let $\mathcal{I}$ represent the set of all human identities in the corpus. We enforce a strict bipartite partition:
$$\mathcal{I}_{\text{train}} \cap \mathcal{I}_{\text{val}} = \emptyset, \quad \mathcal{I}_{\text{train}} \cap \mathcal{I}_{\text{test}} = \emptyset, \quad \mathcal{I}_{\text{val}} \cap \mathcal{I}_{\text{test}} = \emptyset$$

No identity present in the training partition appears in the validation or held-out test partitions, either as a source or as a target actor.

### 2.2 Frozen Protocol C Split
BIOTrace benchmarks all models against the **HiDF** (High-Definition Face Manipulation) dataset under **Protocol C**:
- **Dataset Size**: 8,722 video clips (~2.96 seconds average duration per clip).
- **Frozen Manifest Path**: `datasets/protocol_c_manifest.json`
- **Frozen Manifest SHA-256**: `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`

#### Partition Distribution:
| Split | Authentic Videos ($N_{\text{real}}$) | Manipulated Videos ($N_{\text{fake}}$) | Total Videos ($N_{\text{total}}$) |
| :--- | :---: | :---: | :---: |
| **Train** | 3,053 | 3,053 | **6,106** |
| **Validation** | 654 | 654 | **1,308** |
| **Held-Out Test** | 654 | 654 | **1,308** |
| **Total Corpus** | **4,361** | **4,361** | **8,722** |

---

## 3. Mathematical Formulations & Feature Extraction

### 3.1 Facial Kinematics (Eye Aspect Ratio Dynamics)
Ocular motion is tracked across consecutive frames using 6 landmark coordinates per eye obtained from MediaPipe FaceMesh ($p_1, \dots, p_6$):

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$

From the temporal trajectory $\text{EAR}(t)$ over frames $t \in [1, T]$, we compute:
1. **Right Eye Standard Deviation (`ear_right_std`)**: $\sigma(\text{EAR}_{\text{right}})$
2. **Binocular Standard Deviation (`ear_bino_std`)**: $\sigma(\frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2})$
3. **Mean EAR Velocity (`ear_velocity_mean`)**:
   $$v(t) = \frac{\Delta \text{EAR}(t)}{\Delta t}, \quad \bar{v} = \frac{1}{T-1} \sum_{t=1}^{T-1} |v(t)|$$
4. **Maximum Opening Velocity (`ear_opening_vel_max`)**:
   $$v_{\text{open}}^{\max} = \max_{t} \left( \frac{\text{EAR}(t+1) - \text{EAR}(t)}{\Delta t} \right)$$

### 3.2 Exploratory Short-Window rPPG
Subcutaneous blood volume pulses are extracted from facial skin regions of interest (ROI: forehead, left cheek, right cheek) using two complementary projection algorithms:
- **CHROM (Chrominance-based)**:
  $$X_s = 3 R - 2 G, \quad Y_s = 1.5 R + G - 1.5 B$$
  $$S_{\text{CHROM}} = X_s - \frac{\sigma(X_s)}{\sigma(Y_s)} Y_s$$
- **POS (Plane-Orthogonal-to-Skin)**:
  Projects normalized temporal RGB traces onto a plane orthogonal to the skin-tone direction vector.

From the extracted pulse waveforms, we calculate:
1. **POS-CHROM Correlation (`rppg_pos_chrom_corr`)**: Pearson correlation coefficient between POS and CHROM waveforms over the temporal window.
2. **Signal-to-Noise Ratio (`rppg_pos_snr_db`)**: Power ratio of the cardiac band (0.75–2.5 Hz, corresponding to 45–150 BPM) to harmonic noise.
3. **Dominant Peak Frequency (`rppg_pos_peak_freq_hz`)**: Frequency corresponding to the maximum power spectral density within the physiological band.
4. **Peak Prominence (`rppg_pos_peak_prominence`)**: Prominence of the highest spectral peak relative to adjacent background power.

> [!WARNING]
> **Short-Window Limitation**: Because video clips average ~2.96 seconds in duration, spectral measurements are constrained by the **Rayleigh frequency resolution limit**:
> $$\Delta f = \frac{1}{T} \approx \frac{1}{2.96\,\text{s}} \approx 0.338\,\text{Hz} \approx 20.3\,\text{BPM}$$
> Consequently, extracted pulse frequencies represent exploratory short-window spectral heuristics rather than clinically validated vital signs.

### 3.3 Bilateral Facial Asymmetry
Measures inter-ocular movement disparity between left and right eyes:
$$\Delta_{\text{asym}}(t) = |\text{EAR}_{\text{left}}(t) - \text{EAR}_{\text{right}}(t)|$$
1. **Mean Asymmetry (`blink_asymmetry_mean`)**: $\frac{1}{T}\sum_{t=1}^T \Delta_{\text{asym}}(t)$
2. **Asymmetry Variance (`blink_asymmetry_std`)**: $\sigma(\Delta_{\text{asym}})$

### 3.4 Explicit Feature Exclusions
To maintain strict methodological validity, the following features were deliberately excluded:
- **Spatial Consistency Across Sub-regions**: Excluded due to insufficient skin patch coverage and high missingness (>25%) during extreme head rotations.
- **Discrete Blink Counts and Durations**: In a 2.96-second window, natural blink rates (12–18 blinks/min) produce 0 or at most 1 blink event per clip, yielding degenerate discrete histograms.
- **Container Metadata**: `file_size_bytes`, encoding bitrates, and compression tables are strictly stripped to prevent shortcut learning.

---

## 4. Model Architectures & Training Pipelines

### 4.1 Visual Baseline (Phase 6)
- **Backbone**: EfficientNet-B0 initialized from Torchvision ImageNet-1K pretrained weights.
- **Input Sampling**: 10 uniformly sampled frames per video:
  $$t_k = \left\lfloor k \cdot \frac{T - 1}{9} \right\rfloor, \quad k \in \{0, 1, \dots, 9\}$$
- **Preprocessing**: 224×224 spatial resolution, RGB format, ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Pooling**: Video-level mean-logit pooling:
  $$\bar{z} = \frac{1}{10} \sum_{k=0}^9 z_k, \quad \hat{y} = \sigma(\bar{z})$$
- **Training**: 10 epochs, AdamW optimizer, cosine annealing schedule, binary cross-entropy loss.
- **Checkpoint**: `ai-engine/models/baseline/best_model.pth` (16.3 MB).

### 4.2 Physiological Baseline (Phase 9)
- **Classifier**: Random Forest (100 estimators, max depth 8, balanced class weighting).
- **Feature Vector**: 10-dimensional feature vector ($4 \times \text{EAR kinematics} + 4 \times \text{rPPG} + 2 \times \text{asymmetry}$).
- **Preprocessing**: Median imputation and robust quantile scaling fit strictly on the training set.
- **Checkpoint**: `ai-engine/models/physiological/best_model.joblib`.

### 4.3 Multimodal Fusion Model (Phase 10)
- **Classifier**: L2-regularized Logistic Regression ($C = 1.0$).
- **Fusion Strategy**: Late decision-level fusion combining standardized visual logit and physiological probability scores:
  $$P(\text{Fake}) = \sigma(w_1 \tilde{z}_{\text{visual}} + w_2 \tilde{P}_{\text{phys}} + b)$$
  where learned parameters on Protocol C train are $w_1 \approx 1.3838$, $w_2 \approx 0.7668$, and $b \approx -0.0313$.
- **Validation Tuning**: Operating decision threshold $\theta^* = 0.499895$ selected on the validation partition.
- **Checkpoint**: `ai-engine/models/fusion/phase10_fusion.joblib`.
