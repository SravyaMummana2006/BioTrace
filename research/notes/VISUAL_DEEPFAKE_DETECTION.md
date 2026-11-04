# Visual Deepfake Detection: Architecture, Methodologies, and Forensic Integrity Guardrails

**Module:** BIOTrace Stage 6 & 6.1 Visual Deepfake Detection Baseline  
**Package:** `biotrace.visual_detection`  
**Status:** Implemented, Audited, and Integrated (Unvalidated Visual Baseline)  

---

## 1. Architectural Scope & Scientific Framing

### 1.1 Unvalidated Visual Baseline Framing
BIOTrace was founded on physiological forensics (Stages 2–5): remote photoplethysmography (rPPG), facial kinematics (blink kinetics and EAR), spatial biological consistency, and multimodal evidence fusion.

Stage 6 introduces a **modular visual deepfake detection baseline architecture**.
> [!IMPORTANT]
> **Scientific Status: Unvalidated Visual Baseline**  
> The visual detection module implemented herein provides a software architecture and convolutional baseline designed to process face crops and output structural scores. **It is an unvalidated visual baseline until comprehensively trained and evaluated on verified, real-world labeled deepfake datasets.** Unit tests in this repository verify software contract correctness, numerical stability, and algorithmic isolation; they do **NOT** constitute empirical scientific validation or establish real-world generalization.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   BIOTrace Video Pipeline                                   │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────────┐
                      ▼                                                 ▼
┌──────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│        Physiological Forensics Pipeline      │ │       Visual Deepfake Detection Baseline     │
│  - Stage 2: rPPG Extraction (CHROM / POS)    │ │  - Uniform Temporal Frame Sampling           │
│  - Stage 3: Facial Dynamics & Blink Kinetics │ │  - MediaPipe FaceMesh & 25% Margin Expansion │
│  - Stage 4: Spatial Biological Consistency   │ │  - Quality & Contrast Gating (Laplacian Var) │
│  - Stage 5: Multimodal Evidence Fusion       │ │  - Convolutional Backbone (Residual CNN)     │
│  ──────────────────────────────────────────  │ │  - Robust Temporal Aggregation (Trimmed/Med) │
│  OUTPUT: Fused Evidence & Biological State   │ │  ──────────────────────────────────────────  │
│          (Descriptive physiological record)  │ │  OUTPUT: Uncalibrated Visual Heuristic Score │
└──────────────────────┬───────────────────────┘ └──────────────────────┬───────────────────────┘
                       │                                                │
                       └───────────────────────┬────────────────────────┘
                                               ▼
                                 ┌───────────────────────────┐
                                 │      AnalysisResult       │
                                 │  - fused_evidence         │
                                 │  - visual_detection       │
                                 └───────────────────────────┘
```

### 1.2 Strict Decoupling Principle
A fundamental architectural and forensic principle governs Stage 6:
**Visual deepfake classification is an uncalibrated machine learning assessment and must remain completely decoupled from physiological evidence fusion.**

1. **Different Observation Domains:** Physiological modules measure biophysical processes (hemodynamic blood pulsation, voluntary and involuntary neuro-muscular blink cycles). Visual detection evaluates pixel-level boundary seams and synthesis artifacts.
2. **Preventing Error Cascades:** Deepfake classification networks are notoriously susceptible to domain shift, lossy compression codecs, and out-of-distribution generators. Mixing visual classifier scores into physiological evidence fusion would corrupt verifiable biophysical observations with black-box neural heuristics.
3. **Traceability:** In courtrooms and forensic investigations, evidence provenance must remain distinct. An analyst must be able to ascertain whether a finding originates from physiological failure or high-frequency pixel anomalies.

---

## 2. Image Preprocessing & Face Cropping Pipeline

### 2.1 Uniform Temporal Frame Sampling (`preprocessing.py`)
To prevent bias from uneven temporal sampling and limit computational overhead on high-duration recordings, `VideoFrameSampler` performs uniform temporal subsampling:
- **Configurable Frame Rate:** Default target rate of $2.0 \text{ fps}$ (configurable via `target_fps`), constrained by `max_frames` (default: 60 frames).
- **Exact Presentation Timestamps:** Each sampled frame retains its exact presentation timestamp ($t_{\text{sec}} = \text{index} / \text{fps}$ or container presentation timestamp) for temporal tracking.
- **Uniform Interval Calculation:**
  $$\text{interval} = \max\left(1, \operatorname{round}\left(\frac{\text{fps}_{\text{video}}}{\text{target\_fps}}\right)\right)$$
- **Memory Protection:** Frames are extracted sequentially; corrupt or unreadable frames are logged and isolated without failing the entire batch.

### 2.2 Face Detection & Boundary Margin Padding (`face_crops.py`)
Accurate examination of facial manipulation requires inspecting blending zones where synthetic faces are spliced into authentic donor frames:
1. **MediaPipe FaceMesh Landmark Detector:** Computes 468 landmark coordinates to establish the tight bounding box of facial features.
2. **25% Boundary Margin Expansion:**
   Tight bounding boxes exclude forehead hairlines, jawlines, and neck boundaries where blending seams and color mismatches are most prominent. `FaceCropExtractor` expands the bounding box by $25\%$ ($12.5\%$ on each side) while clamping strictly within image boundaries:
   $$\text{pad}_w = w \cdot \text{margin\_ratio}, \quad \text{pad}_h = h \cdot \text{margin\_ratio}$$
   $$x_1 = \max(0, x - \text{pad}_w), \quad y_1 = \max(0, y - \text{pad}_h)$$
   $$x_2 = \min(W, x + w + \text{pad}_w), \quad y_2 = \min(H, y + h + \text{pad}_h)$$
3. **Multi-Face Auditing Policy:**
   If multiple faces are detected in a single frame:
   - The primary face is selected based on maximum bounding box area and centrality.
   - An audit status `FaceDetectionStatus.MULTIPLE_FACES_DETECTED` is recorded.
   - Diagnostic warnings are logged so forensic examiners know secondary individuals were present.
4. **Crop Quality Metrics:**
   - **Resolution Verification:** Crops smaller than $16 \times 16$ pixels (or configured threshold) are flagged as degraded.
   - **Laplacian Variance Sharpness:** Sharpness is quantified via the variance of the 2D Laplacian operator:
     $$\text{sharpness} = \operatorname{Var}(\nabla^2 I_{\text{gray}})$$
     Crops with sharpness below a configurable threshold (e.g., $< 20.0$) are recorded as blurry.

---

## 3. Neural Architecture & Weight Management

### 3.1 Model Architectures (`model.py`)
The module supports convolutional baseline architectures designed for visual deepfake detection:
1. **`LightweightVisualBaseline`:**
   A custom, highly efficient 4-stage residual convolutional neural network designed for rapid CPU/GPU inference without external dependencies:
   - 4 Residual Conv Blocks with batch normalization, ReLU activations, and max pooling.
   - Global Average Pooling (GAP) collapsing spatial dimensions to 128 channels.
   - Fully connected classification head with Dropout ($p=0.3$) yielding a single linear logit $z$.
2. **`torchvision` Backbones:**
   Configurable support for standard backbones (`resnet18`, `resnet34`, `efficientnet_b0`) initialized with `weights=None` to ensure air-gapped security.
3. **Zero Automatic Downloads:**
   In compliance with forensic security and air-gapped environment standards, models are **never automatically downloaded** from remote third-party URLs at runtime.

### 3.2 Weight Availability & Integrity Verification
```
                 ┌──────────────────────────────────────┐
                 │        Visual Detection Config       │
                 │         - weights_path: Path         │
                 │         - expected_sha256: str       │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                          Is weights_path None?
                         /                     \
                      YES                       NO
                      /                           \
         ┌─────────────────────────┐     ┌────────────────────────┐
         │ MODEL_UNAVAILABLE       │     │ Does file exist?       │
         │ - No fabricated scores  │     └───────────┬────────────┘
         │ - Returns clean result  │                / \
         │ - Explicit limitations  │             NO     YES
         └─────────────────────────┘             /       \
                              ┌───────────────────────┐   ┌───────────────────────┐
                              │ WEIGHTS_NOT_FOUND     │   │ Compute SHA-256 Digest│
                              │ - Log path & error    │   │ Compare with expected │
                              └───────────────────────┘   └───────────┬───────────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │ torch.load()          │
                                                          │ (weights_only=True)   │
                                                          └───────────┬───────────┘
                                                                      │
                                                                      ▼
                                                          ┌───────────────────────┐
                                                          │ AVAILABLE             │
                                                          │ Ready for inference   │
                                                          └───────────────────────┘
```

1. **`weights_only=True` Deserialization:**
   PyTorch checkpoints are strictly loaded using `torch.load(..., weights_only=True)` to prevent arbitrary code execution vulnerabilities from pickled Python objects.
2. **SHA-256 Digest Provenance:**
   Every loaded weights file has its cryptographic SHA-256 digest computed and recorded in `VisualDetectionResult.weights_digest`. This establishes non-repudiation and exact scientific reproducibility.
3. **Zero-Fabrication Guardrail:**
   If weights are not found, unconfigured, or corrupt, the system returns `ModelAvailabilityStatus.MODEL_UNAVAILABLE` or `WEIGHTS_NOT_FOUND` with explicit diagnostic explanations. **No random, synthetic, or default scores are ever fabricated.**

---

## 4. Temporal Aggregation & Statistical Interpretation

### 4.1 Temporal Frame Autocorrelation
Frame-level convolutional models output predictions $\{z_t\}_{t=1}^T$ for individual video frames. Treating these predictions as independent identically distributed (i.i.d.) observations is an egregious statistical error:
- Adjacent video frames share identical camera optics, lighting, compression macroblocks, and facial geometry.
- Empirical lag-1 serial autocorrelation $\rho_1$ is typically high ($\rho_1 \approx 0.8 - 0.95$).

### 4.2 Aggregation Methods (`temporal_aggregation.py`)
`TemporalAggregator` provides robust video-level aggregation strategies:
1. **Trimmed Mean (Default):**
   Discards the top and bottom $10\%$ of frame logits before computing the mean, protecting the video-level verdict against brief tracking glitches, partial occlusions, or sudden motion blur.
2. **Median:**
   Computes the $50\text{th}$ percentile of frame logits, offering maximum breakdown point against outliers.
3. **Mean Logit:**
   Averages unconstrained logit values in $\mathbb{R}$ before applying the sigmoid function:
   $$\bar{z} = \frac{1}{N} \sum_{i=1}^N z_i, \quad s_{\text{agg}} = \sigma(\bar{z}) = \frac{1}{1 + e^{-\bar{z}}}$$
4. **Top-$K$ Percentile:**
   Examines the $85\text{th}$ percentile of frame anomaly scores to identify localized manipulation where synthetic faces appear only during specific sub-segments.

### 4.3 Statistical Rigor: Removal of Invalid Confidence Claims (Stage 6.1 Audit)
> [!WARNING]
> **Statistical Audit Notice: Descriptive Dispersion vs. Inferential Confidence Intervals**  
> While Bartlett's formula provides a descriptive effective sample size $N_{\text{eff}} = N \frac{1 - \rho_1}{1 + \rho_1}$ under an autoregressive AR(1) assumption, it does **NOT** validate frequentist hypothesis tests or establish a statistically valid 95% confidence interval on deep neural network logits.  
> - Stage 6.1 explicitly removes any claim of a statistically valid 95% confidence interval based solely on lag-1 autocorrelation.  
> - The engine reports **descriptive frame dispersion** (sample min-max score range, sample logit variance, lag-1 autocorrelation, and effective $N$).  
> - This range describes empirical sample variation across the examined video frames; it is **NOT** an inferential confidence interval.

---

## 5. Dataset Architecture & Leakage Prevention

### 5.1 Subject-Level Partitioning Priority (`dataset.py`)
In media forensics, partitioning data at the frame level causes catastrophic data leakage. If frames from the same video (or the same actor under the same background/lighting) appear in both the training and test sets:
- The network memorizes background textures, identity features, and camera sensor patterns rather than synthetic generation artifacts.
- The reported test accuracy is artificially inflated (e.g., $99\%$ test accuracy dropping to $60\%$ on unseen videos).

`VideoDataSplitter` enforces strict partitioning rules:
1. **Subject-Level Priority:**
   Whenever `subject_id` is available, grouping occurs by subject. All videos depicting the same individual are allocated strictly to the same partition.
2. **Handling Missing Subject IDs:**
   When `subject_id` is None, missing, or empty, the partitioning unit defaults to the unique `video_id` (`video_<id>`). This guarantees that unassigned videos are treated as independent video units and cannot be inappropriately grouped together.
3. **Mathematical Disjointness Assertion:**
   The splitter asserts mathematical disjointness across both video IDs and subject IDs:
   $$\mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{val}} = \emptyset, \quad \mathcal{V}_{\text{train}} \cap \mathcal{V}_{\text{test}} = \emptyset, \quad \mathcal{V}_{\text{val}} \cap \mathcal{V}_{\text{test}} = \emptyset$$
   $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$

### 5.2 Class Balancing via `WeightedRandomSampler`
To mitigate dataset imbalance (e.g., real videos outnumbering synthetic videos 4:1 or vice versa), `VideoFrameDataset` constructs inverse-frequency sampling weights:
$$w_c = \frac{N_{\text{total}}}{2 \cdot N_c}, \quad w_i = w_{c(i)}$$
These weights feed directly into PyTorch's `WeightedRandomSampler`, ensuring expected 50/50 batch-level class exposure without discarding observations.

---

## 6. Training Pipeline & Video-Level Evaluation

### 6.1 Video-Level Metric Evaluation (`training.py`)
Treating individual frames as independent test samples artificially inflates effective sample size and distorts AUROC, AP, and EER metrics. `VisualDetectionTrainer.evaluate()` aggregates frame logits to the unique **video level** (using median logit aggregation) before evaluating classification metrics.

### 6.2 Checkpoint Metadata & Provenance
Model checkpoints serialize complete reproducibility metadata:
- Training epoch and configuration.
- Model architecture class name.
- Random seed used across PyTorch, NumPy, and random runtimes.
- Optimizer state dict.
- Video-level evaluation metrics.
- Provenance notice:
  > *"Training performance reflects optimization on local dataset splits only. In accordance with forensic science standards, high training/validation metrics do NOT establish out-of-distribution generalization, real-world artifact detection, or cross-generator efficacy."*

---

## 7. Forensic Safeguards & Critical Caveats

### 7.1 Distinction: Raw Logits, Sigmoid Scores, and Calibrated Probabilities
1. **Raw Logits ($z \in \mathbb{R}$):** Unbounded outputs from the linear classification layer. Positive indicates manipulated tendency; negative indicates authentic tendency.
2. **Sigmoid-Transformed Raw Scores ($s = \sigma(z) \in [0, 1]$):** Monotonic mapping of logits to the unit interval. Uncalibrated heuristics. **MUST NEVER BE LABELED OR REPORTED AS PROBABILITIES OF AUTHENTICITY OR MANIPULATION.**
3. **Calibrated Probabilities ($P(\text{Manipulated} \mid \text{Video})$):** Requires empirical post-hoc calibration (e.g. Platt scaling or Isotonic Regression) evaluated on verified out-of-distribution media. The baseline model sets `is_calibrated = False` and `calibrated_probability = None`.

### 7.2 Vulnerability to Operational Confounders
Visual deepfake detection engines operate under well-documented real-world vulnerabilities:
1. **Lossy Codec Compression:**
   High compression (e.g., CRF > 28, H.264/H.265 streaming quantization) eliminates high-frequency facial textures. Deepfake artifacts can be completely smoothed out by macroblocking, leading to severe false negatives.
2. **Cross-Generator Domain Shift:**
   A detector trained on one generator family (e.g., GANs) typically fails when evaluated on different paradigms (e.g., Diffusion, Flow-Matching, or NeRF reenactment).
3. **Demographic and Illumination Bias:**
   Variations in subject skin tone, extreme lighting angles, motion blur, and low frame resolution can artificially trigger artifact detectors.

---

## 8. Summary of Architectural Interfaces

| Component | Class / Function | Purpose | Forensic Guardrail |
|:---|:---|:---|:---|
| **Sampling** | `VideoFrameSampler` | Uniform temporal extraction | Preserves frame presentation timestamps |
| **Cropping** | `FaceCropExtractor` | MediaPipe FaceMesh + 25% margin | Multi-face auditing, sharpness/blur metric |
| **Model** | `load_model_with_status` | Safe checkpoint loading | `weights_only=True`, SHA-256 digest, no auto-downloads |
| **Inference** | `VisualDeepfakeDetector` | End-to-end video analysis | Thread cancellation, error isolation, zero fabrication |
| **Aggregation**| `TemporalAggregator` | Video-level score aggregation | Descriptive dispersion; no invalid 95% CI claims |
| **Partitioning**| `VideoDataSplitter` | Video/subject-level dataset splitting | Subject-level priority, zero frame leakage assertion |
| **Metrics** | `compute_metrics` | Video-level multi-metric evaluation | AUROC, AP, EER, Brier score calibration |
| **Training** | `VisualDetectionTrainer` | Model training & checkpointing | Embeds full provenance; no generalization claims |
| **Integration**| `JobManager` | Async pipeline execution | Decoupled from Stage 5 physiological evidence |
