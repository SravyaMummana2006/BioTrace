# BIOTrace Phase 5 — Frame Extraction & Sampling Policy Specification

**Author:** Senior Computer Vision Engineer & Deepfake Detection Researcher  
**Status:** Approved & Frozen for Baseline Dataset Preparation  
**Target Architecture:** CNN Baseline (EfficientNet-B0 / ResNet18)  
**Input Media:** HiDF Digital Video Archives (Protocol C Bipartite Split)  

---

## 1. Scientific Objectives & Principles

To prepare a research-grade, leak-free image dataset from the 8,722 audited HiDF videos ($4,361$ real and $4,361$ fake) for immediate binary classification model training, we establish a deterministic, reproducible, and balanced frame extraction policy.

### Core Principles
1. **Zero Partition Leakage:** Frames are strictly partitioned according to `../data/protocol_c_split_manifest.csv` (`c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`).
2. **Deterministic Uniform Temporal Coverage:** Rather than clustering frames in the first few seconds or taking random crops, frames are sampled across the entire temporal trajectory of each video clip.
3. **Exact Class & Pair Balance:** Every real video and its corresponding deepfake derivative are sampled with identical temporal targets ($K = 10$ frames per clip), maintaining an exact 1:1 real-to-fake ratio across train, validation, and test splits.
4. **Standardized Spatial Dimensions:** Frames are resized to $224 \times 224$ pixels using area-averaging anti-aliasing interpolation (`cv2.INTER_AREA`), matching standard CNN input tensors (ResNet18, EfficientNet-B0) and eliminating on-the-fly resizing latency during GPU/CPU training.
5. **Full Cryptographic Lineage:** Every extracted frame is hashed (SHA-256) and mapped back to its source video, base donor identity, target donor identity, component ID, and exact timestamp.

---

## 2. Deterministic Sampling Policy

### 2.1 Uniform Temporal Striding Formula
For a video $V$ with $N$ total decoded frames and target sample count $K = 10$:

$$\text{index}(i) = \text{round}\left( i \cdot \frac{N - 1}{K - 1} \right), \quad \text{for } i \in \{0, 1, 2, \dots, K-1\}$$

- When $N \ge K$: Exactly $K$ unique, monotonically increasing frame indices are sampled.
  - *Example ($N = 72, K = 10$):* Indices sampled are `[0, 8, 16, 24, 32, 39, 47, 55, 63, 71]`.
- When $N < K$ (corrupted or anomalously short video): All available $N$ frames are extracted, and a status flag `WARNING_SHORT_CLIP` is recorded in the metadata manifest.

### 2.2 Timestamp Tracking
The exact timestamp $t_i$ for each sampled frame is calculated using the container frame rate:

$$t_i = \frac{\text{index}(i)}{\text{FPS}}$$

Since matched real and fake HiDF video pairs share the same temporal structure (~3.00 seconds at 23.98–24.00 FPS), their sampled frames align temporally across both biological and synthetic counterparts.

---

## 3. Storage, Geometry, and Encoding

| Property | Value | Rationale |
| :--- | :--- | :--- |
| **Output Resolution** | $224 \times 224$ pixels | Standard input dimensions for PyTorch CNN backbones; eliminates runtime resizing bottlenecks. |
| **Interpolation Algorithm** | `cv2.INTER_AREA` (downsampling)<br>`cv2.INTER_LINEAR` (upsampling) | Area interpolation prevents Moiré artifacts and aliasing in facial textures. |
| **Storage Format** | JPEG (`cv2.IMWRITE_JPEG_QUALITY = 95`) | High-fidelity compression; preserves facial boundary textures while requiring ~20 KB/frame. |
| **Total Frames** | $87,220$ frames | $8,722 \text{ videos} \times 10 \text{ frames/video}$. |
| **Estimated Disk Space** | $\approx 1.74 \text{ GB}$ | Well within the ~196 GB available disk capacity. |

---

## 4. Directory Structure & Partition Organization

Extracted frames are organized into a dedicated derived dataset root `data/derived/frames/` with partitioned subdirectories:

```
data/derived/frames/
├── train/
│   ├── real/    # 30,530 frames (3,053 real videos * 10)
│   └── fake/    # 30,530 frames (3,053 fake videos * 10)
├── val/
│   ├── real/    #  6,540 frames (654 real videos * 10)
│   └── fake/    #  6,540 frames (654 fake videos * 10)
└── test/
    ├── real/    #  6,540 frames (654 real videos * 10)
    └── fake/    #  6,540 frames (654 fake videos * 10)
```

### Deterministic Filename Schema
- **Real Frames:** `{base_id}_f{frame_idx:04d}.jpg` (e.g., `c00003_f0008.jpg`)
- **Fake Frames:** `{base_id}_{target_id}_f{frame_idx:04d}.jpg` (e.g., `c00003_c11507_f0008.jpg`)

---

## 5. Metadata Schema (`../data/phase5_frame_extraction_manifest.csv`)

Every extracted frame produces an immutable manifest entry with the following 16 attributes:

1. `frame_id`: Deterministic unique identifier (`{video_id}_f{frame_idx:04d}`).
2. `video_id`: Source video filename stem (e.g., `c00003` or `c00003_c11507`).
3. `base_id`: Base donor identifier from HiDF pairing.
4. `target_id`: Target donor identifier (empty string for real videos).
5. `label`: Binary authenticity label (`real` or `fake`).
6. `split`: Assigned partition (`train`, `val`, or `test`).
7. `component_id`: Bipartite graph connected component ID (from Protocol C).
8. `source_video_path`: Relative path to source MP4.
9. `frame_path`: Relative path to extracted JPEG frame.
10. `timestamp_seconds`: Temporal position of frame within video clip.
11. `original_width`: Native container frame width (e.g., 512, 848).
12. `original_height`: Native container frame height.
13. `output_width`: Extracted frame width ($224$).
14. `output_height`: Extracted frame height ($224$).
15. `frame_sha256`: Cryptographic digest of the extracted frame file.
16. `extraction_status`: Status indicator (`SUCCESS`, `WARNING_SHORT_CLIP`, or error details).

---

## 6. Known Policy Limitations & Mitigation

1. **Spatial Downsampling:** Downsampling from native $512 \times 512$ or $848 \times 848$ to $224 \times 224$ attenuates high-frequency sub-pixel camera noise patterns.
   *Mitigation:* $224 \times 224$ retains all macroscopic deepfake facial artifacts (blending boundaries, eye warping, mouth synthesis errors) and is standard in published literature. Original dimensions are recorded for provenance.
2. **Temporal Sparsity for Continuous Physiological Modeling:** Sampling 10 frames over 3 seconds (~3.3 FPS) does not satisfy the Nyquist criterion for continuous heart rate extraction (requiring $\ge 15\text{--}30\text{ FPS}$).
   *Mitigation:* Frame extraction serves the **visual CNN deepfake detection baseline** (Phase 6). Continuous rPPG and eye-blink physiological models operate directly on the raw extracted MP4 files as implemented in the AI engine.
