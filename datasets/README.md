# Forensic Media Datasets & Benchmark Guidelines

> **IMPORTANT FORENSIC NOTICE**: Video files, raw datasets, and biometric captures are excluded by `.gitignore` and must **never** be committed to this version control repository.

## Supported Benchmark Datasets

For academic evaluation and publication-grade empirical research, the following standard datasets will be targeted:

1. **FaceForensics++ (FF++)**:
   - Contains 1,000 pristine video sequences and corresponding manipulations across 4 algorithms (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
   - Evaluated across three compression rates: raw (c0), light compression (c23), heavy compression (c40).

2. **Celeb-DF (v2)**:
   - High-quality deepfake video dataset with reduced visual boundary artifacts, serving as a rigorous benchmark for subtle manipulation detection.

3. **UBFC-rPPG / PURE**:
   - Ground-truth physiological datasets with contact photoplethysmography (finger clip PPG) and synchronized facial videos to benchmark rPPG signal extraction fidelity.

## Directory Structure for Local Datasets

When downloading datasets locally, organize them as follows:

```
datasets/
├── FaceForensics++/
│   ├── original_sequences/
│   │   └── c23/
│   │       └── videos/
│   └── manipulated_sequences/
│       ├── Deepfakes/
│       │   └── c23/
│       └── FaceSwap/
│           └── c23/
└── Celeb-DF-v2/
    ├── Celeb-real/
    ├── Celeb-synthesis/
    └── YouTube-real/
```

## Privacy, Ethical & Institutional Compliance

- All biometrics data must comply with institutional academic review and ethical handling guidelines.
- Video files must be stored on local encrypted drives.
