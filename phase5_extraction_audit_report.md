# BIOTrace Phase 5 — Frame Extraction & Training Dataset Preparation Audit Report

**Author:** Senior Computer Vision Engineer & Deepfake Detection Researcher  
**Date:** November 4, 2026  
**Status:** Complete, Verified & Frozen for Phase 6 Training  
**Manifest SHA-256:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`  

---

## 1. Executive Summary

Phase 5 of the BIOTrace project successfully extracted, standardized, and verified a balanced, leak-free image dataset of **87,220 frames** from all **8,722 audited HiDF videos** ($4,361$ real and $4,361$ deepfake).

The extraction strictly followed the frozen Protocol C bipartite graph partition, guaranteeing **0.0% identity and donor leakage** across train, validation, and test splits. Every frame was resized using area-averaging anti-aliasing (`cv2.INTER_AREA`) to standard CNN dimensions ($224 \times 224$ pixels) and recorded with cryptographic provenance (SHA-256) in [`phase5_frame_extraction_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_frame_extraction_manifest.csv).

### Key Performance & Dataset Indicators
| Metric | Value |
| :--- | :--- |
| **Total Source Videos Processed** | **8,722** ($4,361$ authentic base + $4,361$ deepfake derivatives) |
| **Total Frames Extracted** | **87,220** |
| **Failed Videos** | **0** (100.0% extraction success rate) |
| **Average Frames per Video** | Exactly **10.0** across all splits and classes |
| **Output Image Resolution** | $224 \times 224 \times 3$ RGB JPEG (Quality 95) |
| **Average File Size** | $13.21\text{ KB}$ per frame |
| **Total Storage Consumed** | **1,124.89 MB (1.10 GB)** |
| **Extraction Wall-Clock Time** | **413.21 seconds (6.88 minutes)** |
| **Throughput** | **21.1 videos/sec** (~211 frames/sec across 8 CPU worker processes) |
| **Automated Test Suite** | **260 / 260 tests passed (100%)** |

---

## 2. Ingestion & Frozen Split Integrity

Prior to frame extraction, all baseline dataset assets were independently verified on disk:
1. **Frozen Split Manifest (`protocol_c_split_manifest.csv`):**
   - SHA-256: `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` (bit-for-bit identical to Phase 3/4).
   - Rows: $4,361$ real/fake pairs ($3,053$ train, $654$ validation, $654$ test).
2. **Authentic Base Videos (`data/extracted/hidf/real/`):**
   - Exactly $4,361$ valid MP4 files confirmed on disk. Zero missing, zero corrupted.
3. **Deepfake Derivative Videos (`data/extracted/hidf/fake/`):**
   - Exactly $4,361$ valid MP4 files confirmed on disk. Zero missing, zero corrupted.

---

## 3. Pilot Extraction & Validation

Before launching full extraction across all 8,722 videos, a multi-split pilot run was executed:
- **Configuration:** 2 pairs per split = 6 pairs = 12 videos total (6 real, 6 fake).
- **Execution:** $120$ frames extracted in $2.04$ seconds ($6.3\text{ videos/sec}$, 0 failures).
- **Inspection Findings:**
  - Decoded resolutions, aspect ratios, and exposure levels confirmed valid (average pixel intensity 26.6–83.9, non-blank).
  - Spatial dimensions exactly $224 \times 224$ pixels.
  - Pilot integrity audit passed 100% with zero duplicate timestamps or split leakage.
- Following successful pilot validation, full dataset extraction was immediately executed.

---

## 4. Full Extraction Statistics & Class Balance

Extracted frames are partitioned in `data/derived/frames/` by split and ground-truth label:

| Split | Videos Processed | Frames Extracted | Authentic (Real) | Synthetic (Fake) | Class Ratio | Identity Leakage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | $6,106$ | $61,060$ | $30,530$ | $30,530$ | **1.0000** ($50.0\% / 50.0\%$) | **0.0%** |
| **Validation** | $1,308$ | $13,080$ | $6,540$ | $6,540$ | **1.0000** ($50.0\% / 50.0\%$) | **0.0%** |
| **Test** | $1,308$ | $13,080$ | $6,540$ | $6,540$ | **1.0000** ($50.0\% / 50.0\%$) | **0.0%** |
| **Total** | **8,722** | **87,220** | **43,610** | **43,610** | **1.0000** ($50.0\% / 50.0\%$) | **0.0%** |

Full summary statistics are recorded in [`phase5_extraction_statistics.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_extraction_statistics.csv).

---

## 5. Methodological & Integrity Verification (`scripts/audit_phase5_frames.py`)

An exhaustive automated audit was executed using [`scripts/audit_phase5_frames.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/audit_phase5_frames.py). All 7 integrity criteria were satisfied:

1. **Frozen Manifest Immutability:** `protocol_c_split_manifest.csv` hash verified bit-for-bit (`c15300c15...`).
2. **Source Video Traceability:** 100% of the 87,220 frames map to verified existing MP4 videos on disk.
3. **Uniqueness:** Zero duplicate `frame_id`s and zero duplicate `frame_path`s across the entire dataset.
4. **Zero Partition Leakage:**
   - 0 videos cross splits.
   - 0 base donor identities cross splits.
   - 0 target donor identities cross splits.
   - 0 bipartite connected components cross splits.
5. **Split Consistency:** 100% of frame split labels match the frozen Protocol C assignments.
6. **Class Balance:** Exactly 1:1 real-to-fake parity in train ($30,530 / 30,530$), validation ($6,540 / 6,540$), and test ($6,540 / 6,540$).
7. **On-Disk Integrity & Checksums:** Stratified random sample of $4,361$ on-disk frame files (5% of total dataset) verified: 0 missing files, 0 zero-byte files, and 100% bit-for-bit SHA-256 checksum matches.

---

## 6. Automated Test Suite Outcomes

The automated test suite in `ai-engine/tests` was expanded to include dedicated Phase 5 tests ([`test_phase5_frames.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/tests/test_phase5_frames.py)):
- `test_sampling_indices_uniform_spacing`: **PASSED** (Monotonic uniform coverage).
- `test_sampling_indices_short_video`: **PASSED** (Graceful handling of short clips).
- `test_sampling_indices_edge_cases`: **PASSED** (0 and 1-frame boundaries).
- `test_frozen_manifest_hash_unaltered`: **PASSED** (Manifest cryptographic lockdown).
- `test_source_video_directories_intact`: **PASSED** (4,361 real and 4,361 fake videos intact).
- `test_frame_manifest_schema_if_present`: **PASSED** (All 16 schema columns and zero leakage).

**Overall Pytest Result:** `260 passed, 15 warnings in 17.71s (100% passing)`.

---

## 7. Deliverables Index

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **Frame Metadata Manifest** | [`phase5_frame_extraction_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_frame_extraction_manifest.csv) | Full 87,220-row metadata table with cryptographic SHA-256 digests and Protocol C lineage. |
| **Extraction Statistics** | [`phase5_extraction_statistics.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_extraction_statistics.csv) | Partition-level breakdown of video counts, frame counts, and class balance. |
| **Sampling Policy Specification** | [`phase5_sampling_policy.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_sampling_policy.md) | Mathematical formulation, timestamp calculations, and methodological limitations. |
| **Extraction Audit Report** | [`phase5_extraction_audit_report.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase5_extraction_audit_report.md) | This document. |
| **Extraction Script** | [`scripts/extract_hidf_frames.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/extract_hidf_frames.py) | High-throughput, multiprocessing, resumable frame extraction utility. |
| **Audit Script** | [`scripts/audit_phase5_frames.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/audit_phase5_frames.py) | Automated integrity, partition containment, and cryptographic hash verification script. |
| **Automated Tests** | [`ai-engine/tests/test_phase5_frames.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/tests/test_phase5_frames.py) | Pytest regression and schema verification tests. |
| **Derived Frames Directory** | `data/derived/frames/` | 87,220 standardized $224 \times 224$ JPEG images arranged by `split/label/`. |

---

## 8. Phase 6 Model Training Readiness & Recommendations

The dataset in `data/derived/frames/` is fully verified and **ready for immediate model training**.

### Recommended Phase 6 Baseline Setup:
1. **Framework:** PyTorch 2.10 with Torchvision.
2. **Backbone Architecture:** **EfficientNet-B0** pretrained on ImageNet-1K (`weights=EfficientNet_B0_Weights.DEFAULT`), or **ResNet18** (`weights=ResNet18_Weights.DEFAULT`).
3. **Classification Head:** Single linear projection layer (`num_classes=1`) with `BCEWithLogitsLoss`.
4. **Data Input:** Direct tensor loading of pre-extracted $224 \times 224$ images, eliminating runtime resizing latency.
5. **Class Balance:** Exact 50/50 balance in train, validation, and test eliminates any need for class-weighted loss functions.
6. **Clip Aggregation:** Mean logit pooling across the 10 frames per video to compute clip-level authenticity scores.
7. **Evaluation Metrics:** Video-level AUC-ROC, Equal Error Rate (EER), and Accuracy at default ($0.5$) and calibrated thresholds.
8. **Checkpoints & Early Stopping:** Save best validation AUC-ROC checkpoint; patience = 3 epochs.
