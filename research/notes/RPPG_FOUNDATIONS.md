# BIOTrace Research Foundations: Remote Photoplethysmography (rPPG)

**Document Version:** 1.0.0  
**Domain:** Biomedical Signal Processing & Digital Media Forensics  
**Module:** `biotrace.rppg`

---

## 1. Biophysical and Optical Foundations

### 1.1 Skin Tissue Optical Model and the Modified Beer-Lambert Law
Remote photoplethysmography (rPPG) is an optical, non-contact measurement modality capable of capturing cardiovascular pulse waveforms from facial video streams. The physical phenomenon relies on time-varying light absorption within cutaneous vascular beds during the cardiac cycle.

Incident ambient light $I_0(\lambda, t)$ penetrates the stratum corneum and epidermis, entering the dermis where arterial and venous capillary beds reside. As the left ventricle contracts during systole, a pressure wave propagates through the arterial tree, transiently increasing arterial blood volume and red blood cell density in microvascular tissue beds. This periodic dilation alters the optical path length and localized absorption of hemoglobin.

Under the Modified Beer-Lambert Law, optical density (attenuation) $A(\lambda, t)$ through layered biological tissue is expressed as:
$$A(\lambda, t) = \ln \left( \frac{I_0(\lambda, t)}{I(\lambda, t)} \right) = \sum_{i} \epsilon_i(\lambda) C_i(t) d(t) \cdot \mathrm{DPF}(\lambda) + G(\lambda)$$
where:
- $\epsilon_i(\lambda)$ is the molar extinction coefficient of chromophore $i$ (primarily oxygenated hemoglobin $\mathrm{HbO}_2$, deoxygenated hemoglobin $\mathrm{Hb}$, and epidermal melanin).
- $C_i(t)$ represents the temporal concentration of chromophore $i$.
- $d(t) = d_0 + \Delta d(t)$ represents the effective vascular optical path length, decomposing into a baseline DC component $d_0$ and a pulsatile AC component $\Delta d(t)$ driven by cardiac ejection.
- $\mathrm{DPF}(\lambda)$ is the differential pathlength factor accounting for multiple scattering in the dermis.
- $G(\lambda)$ is a geometry-dependent attenuation constant accounting for tissue scatter and measurement configuration.

### 1.2 Spectral Selectivity of Hemoglobin Absorption
The molar extinction spectrum of hemoglobin demonstrates pronounced wavelength dependence:
- **Green spectrum ($\sim 520\text{--}580\text{ nm}$):** Peak absorption of oxygenated hemoglobin ($\mathrm{HbO}_2$) occurs near $540\text{ nm}$ and $575\text{ nm}$ ($\alpha$ and $\beta$ absorption bands). Because dermal scattering is moderate and hemoglobin absorption is high, green spectral channels exhibit the highest dynamic amplitude modulation (AC/DC ratio) in consumer RGB sensors.
- **Red spectrum ($\sim 600\text{--}700\text{ nm}$):** Extinction coefficients drop significantly. Photons penetrate deeper into hypodermal tissue with lower relative blood volume modulation, rendering the red channel primarily sensitive to gross surface reflection and motion.
- **Blue spectrum ($\sim 400\text{--}490\text{ nm}$):** Highly absorbed by epidermal melanin and superficial blood; Rayleigh and Mie scattering in superficial epidermis substantially attenuate depth penetration, yielding poor pulse contrast and high susceptibility to surface reflection artifacts.

---

## 2. Mathematical Formulations of rPPG Algorithms

In consumer cameras, recorded pixel intensities represent a spatial and spectral integration of specular reflectance and diffuse subsurface reflectance. For facial skin under ambient illumination, the time-varying RGB observation vector $\mathbf{C}(t) = [R(t), G(t), B(t)]^T$ can be decomposed into:
$$\mathbf{C}(t) = I(t) \cdot (\mathbf{u}_s \cdot s(t) + \mathbf{u}_d \cdot d_0 + \mathbf{u}_p \cdot p(t)) + \mathbf{n}(t)$$
where:
- $I(t)$ is overall illumination intensity.
- $\mathbf{u}_s$ is the unit direction of specular (Fresnel) reflection (mirror-like surface scatter, carrying no physiological modulation).
- $\mathbf{u}_d$ is the static diffuse skin reflection vector.
- $\mathbf{u}_p$ is the physiological color-change vector induced by blood volume changes.
- $p(t)$ is the scalar physiological pulsatile signal.
- $\mathbf{n}(t)$ represents camera sensor noise and quantization error.

To extract $p(t)$ without invasive skin calibration, standard rPPG algorithms apply mathematical projections designed to eliminate specular reflections $\mathbf{u}_s$ and suppress intensity variations $I(t)$.

### 2.1 CHROM (Chrominance-Based Method; de Haan & Jeanne, 2013)
The chrominance-based method establishes a standard skin-color vector under standardized white illumination to project RGB traces into two orthogonal chrominance signals where specular reflections cancel out.

#### Formulation:
1. **Temporal Normalization:**
   For a temporal analysis window of length $L$, compute mean-normalized color signals:
   $$R_n(t) = \frac{R(t)}{\mu_R}, \quad G_n(t) = \frac{G(t)}{\mu_G}, \quad B_n(t) = \frac{B(t)}{\mu_B}$$
   where $\mu_R, \mu_G, \mu_B$ are the temporal means over window $L$.

2. **Orthogonal Chrominance Projections:**
   Two orthogonal chrominance signals $X_s(t)$ and $Y_s(t)$ are defined based on the human skin spectral model:
   $$X_s(t) = 3 R_n(t) - 2 G_n(t)$$
   $$Y_s(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$

3. **Adaptive Ratio Estimation:**
   The residual specular component is eliminated by taking a linear combination steered by their respective standard deviations $\sigma(X_s)$ and $\sigma(Y_s)$:
   $$\alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$$
   The local rPPG estimate $S_{\mathrm{CHROM}}(t)$ across window $L$ is:
   $$S_{\mathrm{CHROM}}(t) = X_s(t) - \alpha \cdot Y_s(t)$$

4. **Overlap-Add Synthesis:**
   Sliding windows with stride $S$ and Hanning window tapering are summed to form the continuous synthesized waveform:
   $$S_{\mathrm{final}}(t) = \sum_{k} w(t - kS) \cdot S_k(t)$$

### 2.2 POS (Plane-Orthogonal-to-Skin Method; Wang et al., 2017)
Wang et al. recognized that skin tones cluster along a predictable tone axis in RGB space and derived an exact mathematical projection onto a plane orthogonal to the skin-tone vector, completely eliminating specular reflection without empirical skin color calibration.

#### Formulation:
1. **Temporal Normalization:**
   Within sliding window $L$, compute normalized signals:
   $$C_n(t) = \left[ \frac{R(t)}{\mu_R}, \frac{G(t)}{\mu_G}, \frac{B(t)}{\mu_B} \right]^T$$

2. **Projection onto Orthogonal Plane:**
   Project normalized RGB vectors onto two orthogonal planes perpendicular to the normalized skin vector:
   $$P = \begin{bmatrix} 0 & 1 & -1 \\ -2 & 1 & 1 \end{bmatrix}$$
   Yielding signals:
   $$S_1(t) = G_n(t) - B_n(t)$$
   $$S_2(t) = -2 R_n(t) + G_n(t) + B_n(t)$$

3. **Spatial/Temporal Tuning:**
   The exact physiological pulse direction within the projection plane is selected by matching the standard deviation ratio:
   $$\alpha = \frac{\sigma(S_1)}{\sigma(S_2)}$$
   The windowed pulse signal is:
   $$H(t) = S_1(t) + \alpha \cdot S_2(t)$$

4. **Sliding-Window Synthesis:**
   Crucially, Wang et al. specify that $\alpha$ must be computed **per temporal window** ($L \approx 1.6\text{ s}$, stride $S \approx 0.1\text{ s}$) rather than globally. Head rotation and blood volume changes modulate skin geometry dynamically; a global projection fails under dynamic subject motion. Successive windowed signals are overlap-added with Hanning weighting:
   $$H_{\mathrm{POS}}(t) = \sum_k w(t - kS) \cdot \tilde{H}_k(t)$$

---

## 3. Sampling Rates and Temporal Synchronization

### 3.1 Nyquist-Shannon Sampling Constraints
Human resting heart rates typically span $40\text{--}180\text{ beats per minute (BPM)}$, corresponding to fundamental frequencies:
$$f_{\mathrm{cardiac}} \in [0.67\text{ Hz}, 3.0\text{ Hz}]$$
Harmonics (dicrotic notch, second harmonic) span $1.3\text{--}6.0\text{ Hz}$.

- **Theoretical Minimum ($f_s \ge 15\text{ Hz}$):**
  While $15\text{ Hz}$ exceeds the Nyquist limit for the fundamental cardiac frequency ($2 \times 3.0\text{ Hz} = 6.0\text{ Hz}$), it leaves only 3--5 temporal samples per cardiac cycle, inducing severe phase quantization and frequency leakage in spectral estimation.
- **Recommended Minimum ($f_s \ge 30\text{ Hz}$):**
  $30\text{ Hz}$ provides 10--30 samples per cardiac cycle, permitting reliable peak detection, robust Welch PSD calculation with $0.05\text{ Hz}$ bin resolution, and inter-beat interval (IBI) analysis.
- **High-Fidelity Research Standard ($f_s \ge 60\text{ Hz}$):**
  Required for accurate Heart Rate Variability (HRV) analysis, pulse transit time (PTT) estimation, and dicrotic notch morphology analysis.

### 3.2 Handling Actual Frame Timestamps vs Video Metadata FPS
Real-world video capture suffers from dropped frames, encoder rate control jitter, and non-constant frame rates (VFR). Relying on container metadata FPS (e.g., `30.00`) introduces severe spectral distortion:
1. **Timestamp Monotonicity:** Frame timestamps $t_i$ must be strictly monotonically increasing ($t_{i+1} > t_i$). Duplicate timestamps ($\Delta t = 0$) indicate frame repetition or decoding anomalies and must be eliminated.
2. **Effective Sampling Frequency:**
   Calculated from real duration:
   $$f_{s,\mathrm{eff}} = \frac{N - 1}{t_N - t_1}$$
3. **Uniform Resampling:**
   Digital filtering (Butterworth SOS) and Welch PSD assume uniformly sampled time series $x[n] = x(n \cdot \Delta t)$. BIOTrace uses linear interpolation onto a strictly uniform grid $t_{\mathrm{grid}} = t_0 + k / f_{s,\mathrm{eff}}$ prior to filtering and spectral analysis.

---

## 4. Signal Degradation and Physical Limitations

### 4.1 Video Compression (H.264 / HEVC / AV1)
Forensic video analysis operates predominantly on compressed web video. Standard video codecs introduce severe artifacts:
- **Chroma Subsampling (YUV 4:2:0):** Color difference channels ($U, V$) are downsampled by a factor of 2 both horizontally and vertically, retaining only 25% of color spatial resolution. Microvascular pulse modulations (typically 0.1% to 1% of total pixel value) are frequently obliterated.
- **Quantization and Block DCT:** Macroblock quantizers treat subtle temporal skin color fluctuations as high-frequency noise and discard them at low bitrates (CRF > 28).
- **Temporal Motion Compensation:** Codec macroblock vectors interpolate across frames, smearing physiological variations into spatial motion vectors.

### 4.2 Subject Motion Artifacts
- **Rigid Motion:** Head rotation, translation, and nodding shift facial regions across varying ambient light gradients, introducing large baseline drift orders of magnitude larger than cardiac AC signals.
- **Non-Rigid Motion:** Facial expressions, smiling, talking, and swallowing alter skin tension and muscle blood perfusion locally, causing temporary loss of cardiovascular pulsatile coherence.
- **Ballistocardiographic (BCG) Motion:** Cardiac ejection physically vibrates the head and torso ($0.1\text{--}0.5\text{ mm}$ displacement). While BCG can be tracked, optical motion vectors frequently couple into color channels.

### 4.3 Melanin Concentration and Fitzpatrick Skin Phototypes
Melanin in the epidermis acts as an optical broadband neutral density filter with high absorption across both blue and green wavelengths. In darker skin phototypes (Fitzpatrick IV–VI), a greater fraction of incident green light is absorbed before reaching dermis capillaries, reducing the AC/DC signal-to-noise ratio by $6\text{--}12\text{ dB}$. Algorithms must maintain sensitivity without mistaking low melanin transmittance for artificial video tampering.

---

## 5. Forensic Interpretation: The Asymmetry of rPPG Evidence

In digital media forensics, a fundamental scientific tenet is that **physiological absence does not imply synthetic manipulation**.

```
                           ┌──────────────────────────────┐
                           │   Observed Video Evidence    │
                           └──────────────┬───────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │  Robust rPPG Pulse Found  │               │  Weak / No Pulse Detected │
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │                                           │
         Biophysical Consistency                      Ambiguous Forensics
         Supports biological authenticity             Cannot distinguish:
         (unless replay attack or                      • Deepfake / Face Swap
          biometric spoofing present)                  • Heavy H.264 compression
                                                       • Poor sensor / low bitrate
                                                       • Dark skin / dim lighting
                                                       • Excessive motion
```

### 5.1 Why rPPG Alone Cannot Declare a Video "Fake"
A low-quality rPPG signal (`LOW_QUALITY` or `INSUFFICIENT_SIGNAL`) can stem from benign physical factors:
1. High video compression (e.g., social media re-encoding).
2. Poor illumination ($< 100\text{ lux}$).
3. Excessive subject or camera motion.
4. Camera auto-exposure or auto-white balance hunting.
5. High melanin concentration without adequate photon density.

Treating low rPPG quality as a deepfake indicator generates unacceptable false positive rates on real, low-quality surveillance or mobile recordings.

### 5.2 Forensic Role of rPPG: Evidence Extraction Decoupled from Authenticity
In the BIOTrace multimodal architecture:
- **Physiological Evidence vs Authenticity Probability:** rPPG extracts cardiovascular blood-volume modulation traces, pulse frequency, and signal quality metrics. It does **not** estimate an authenticity probability. In the data contracts, `authenticity_score` is explicitly `None` (`is_authenticity_assessed = False`).
- **Absence of Proof is Not Proof of Absence:** A low-quality signal (`LOW_QUALITY` or `INSUFFICIENT_SIGNAL`) simply documents that reliable cardiovascular waveforms could not be recovered under the given recording conditions. Forensic decision-making must never treat an unassessed physiological module as a 50% probability of a deepfake.

---

## 6. Experimental Validation Roadmap

BIOTrace rPPG algorithms will be systematically benchmarked across three standard open-access research benchmarks:

| Dataset | Modality & Resolution | Ground Truth | Primary Research Focus |
|---|---|---|---|
| **UBFC-rPPG** | Uncompressed RGB ($640 \times 480$, $30\text{ fps}$) | Pulse Oximeter (CMS50E) PPG waveform & HR | Baseline mathematical accuracy under natural illumination and mild movement |
| **PURE** | Uncompressed RGB ($640 \times 480$, $30\text{ fps}$) | Finger pulse oximeter & ECG | Head motion robustness (talking, rotating, nodding) |
| **FaceForensics++** | Compressed H.264 (raw, c23, c40) | Manipulated masks (Deepfakes, Face2Face, FaceSwap, NeuralTextures) | Compression resilience and forensic signal degradation analysis |

### Evaluation Metrics
- **Mean Absolute Error (MAE):** $\mathrm{MAE} = \frac{1}{M}\sum_{i=1}^M |\mathrm{HR}_{\mathrm{pred}, i} - \mathrm{HR}_{\mathrm{gt}, i}|$ (in BPM).
- **Root Mean Square Error (RMSE):** $\mathrm{RMSE} = \sqrt{\frac{1}{M}\sum_{i=1}^M (\mathrm{HR}_{\mathrm{pred}, i} - \mathrm{HR}_{\mathrm{gt}, i})^2}$.
- **Pearson Correlation ($r$):** Linear correlation between predicted and reference pulse rates across sliding temporal windows.
- **de Haan Signal-to-Noise Ratio ($\mathrm{SNR}_{\mathrm{dB}}$):** Energy in fundamental and first harmonic cardiac bins relative to background spectral noise within $[0.75, 3.50]\text{ Hz}$.

---

## 7. Configuration and Provisional Engineering Parameters

The following engineering defaults are implemented in `biotrace.rppg.domain.QualityConfig` and `FilterConfig`:

```python
# Provisional Engineering Defaults (Configurable)
MIN_DURATION_SECONDS = 4.0          # Minimum temporal window for Welch PSD (0.25 Hz physical resolution)
MIN_SAMPLING_RATE_HZ = 15.0         # Minimum temporal rate for cardiac Nyquist
MIN_VALID_FRAME_RATIO = 0.70        # Minimum fraction of frames with stable landmarks
PROVISIONAL_MIN_SNR_DB = 1.0        # de Haan SNR boundary for MODERATE vs LOW quality
PROVISIONAL_HIGH_SNR_DB = 3.0       # de Haan SNR boundary for HIGH quality
MIN_PEAK_PROMINENCE = 0.15          # Normalized spectral peak prominence threshold
CARDIAC_FREQ_RANGE = (0.75, 2.50)   # 45 to 150 BPM physiological bandpass
SNR_INTEGRATION_PASSBAND = (0.75, 3.50) # 45 to 210 BPM SNR evaluation band
FILTER_ORDER = 4                    # 4th-order Butterworth bandpass (SOS, zero-phase)
WINDOW_LENGTH_SECONDS = 1.6         # POS/CHROM temporal window length
WINDOW_STRIDE_SECONDS = 0.1         # Sliding window hop size
```

These parameters serve as starting baselines for experimental tuning and must not be cited as rigid physiological constants.
