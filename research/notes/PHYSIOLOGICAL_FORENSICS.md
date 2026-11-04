# Theoretical Foundations of Biological Media Forensics

## 1. Remote Photoplethysmography (rPPG)

Remote Photoplethysmography measures diffuse reflection of ambient light modulated by subcutaneous blood volume changes during the cardiac cycle. Hemoglobin absorbs green light (~520–580 nm) significantly more than red or blue light.

### Classical Extraction Algorithms

1. **Chrominance-based method (CHROM)**:
   Linear combination of normalized color channels designed to eliminate specular reflections:
   $$X = 3R_n - 2G_n$$
   $$Y = 1.5R_n + G_n - 1.5B_n$$
   $$S = X - \alpha Y$$
   where $\alpha = \sigma(X) / \sigma(Y)$.

2. **Plane-Orthogonal-to-Skin (POS)**:
   Projects temporal RGB traces onto a plane orthogonal to the skin-tone direction, separating pulsatile variations from specular motion noise.

---

## 2. Temporal Kinematics & Eye Aspect Ratio (EAR)

Spontaneous human eye blinks exhibit asymmetric kinematic curves: a rapid closing phase (~70–100 ms) followed by a slower reopening phase (~150–200 ms).
Using 6 localized 2D/3D landmarks per eye:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

In synthetic media, blinks are frequently unnatural in duration, shape, or synchrony with head movements.

---

## 3. Multimodal Evidence Fusion

Rather than simple score averaging, fusion combines features via:
- Feature-level concatenation with attention weighting.
- Late probabilistic fusion incorporating modality-specific confidence estimates.
- Evidential reasoning (Dempster-Shafer or Dirichlet distributions) to quantify epistemic uncertainty.
