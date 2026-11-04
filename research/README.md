# BIOTrace Research Foundation

## Central Research Question

> **"Can multimodal physiological and temporal facial evidence provide complementary information for detecting manipulated videos?"**

Digital facial manipulation algorithms (deepfakes, diffusion head reenactments, face swaps) often optimize for perceptual visual realism and frame-level texture synthesis. However, human biological systems operate through continuous physiological dynamics that are difficult to synthesize consistently across space and time.

---

## Research Hypotheses

### Hypothesis 1: Biological Signal Incoherence (rPPG)
*Remote photoplethysmography (rPPG) signals extracted from spatially distributed regions of interest (ROIs) on genuine facial skin exhibit high mutual cross-correlation and characteristic cardiac harmonics (~0.75 Hz to 2.5 Hz / 45 to 150 BPM). In synthesized or reenacted faces, subtle blending boundaries, Poisson blending, and generative temporal jitter disrupt localized chrominance variations, resulting in phase shifts, attenuated signal-to-noise ratio (SNR), and spatial incoherence across facial regions.*

### Hypothesis 2: Temporal Dynamic Distortions
*Natural voluntary and involuntary facial dynamics—such as the Eye Aspect Ratio (EAR) kinematics during spontaneous blinking and micro-tremor frequencies—follow constrained biomechanical distributions. Generative video synthesis models often suffer from temporal phase artifacts, blink omission, or abnormal blink durations.*

### Hypothesis 3: Multimodal Complementarity
*No single biological signal provides infallible detection under real-world conditions (such as motion blur, low illumination, or lossy video compression like H.264/HEVC). However, fusing orthogonal physiological channels (spatial blood-volume pulse consistency) with kinematic channels (temporal landmark kinematics) yields complementary detection resilience.*

---

## Methodological Guidelines

1. **No Assumption of Perfection**: Biological signals can be degraded by head motion, uneven lighting, compression artifacts, and skin tone variations. All algorithms must report uncertainty and confidence metrics rather than definitive binary claims without statistical backing.
2. **Reproducibility First**: All experimental protocols must specify sampling frame rates, resolution thresholds, ROI bounding boxes, filter parameters (Butterworth bandpass cutoffs), and color-space representations (CHROM, POS, Green channel).
3. **Transparent Baseline Evaluation**: Baseline models will be benchmarked alongside classical frequency-domain and spatial CNN baselines on identical evaluation splits.
