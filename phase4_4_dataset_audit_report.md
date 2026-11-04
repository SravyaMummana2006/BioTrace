# BIOTrace Phase 4.4 — Final Dataset Integrity, Pairing and Protocol C Audit Report

**Auditor / Engineer:** Senior Research Data Engineer  
**Dataset Targets:** HiDF Authentic Base Videos & Deepfake Derivative Videos  
**Official Repository:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Audit & Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** COMPLETE / ALL INTEGRITY GATES PASSED (11 / 11)

---

## Executive Summary

Phase 4.4 conducted an exhaustive, independent scientific audit of the complete HiDF dataset acquired across Phases 4.1 through 4.3. This audit verified the frozen Protocol C experimental design, resolved historical narrative reporting discrepancies, inspected all 4,361 authentic and 4,361 deepfake video files at the bit level, confirmed mathematical zero-leakage partition isolation, and evaluated subgroup demographic distributions.

All 11 audit requirements passed with zero unresolved discrepancies:
1. **Frozen Manifest Integrity:** [`protocol_c_split_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/protocol_c_split_manifest.csv) SHA-256 verified as `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` (untouched, identical to Phase 3.1 baseline).
2. **True Partition Counts:** Recomputed directly from manifest:
   - **Train:** Exactly $3,053$ pairs ($3,053$ real, $3,053$ fake = $6,106$ videos, $70.007\%$)
   - **Validation:** Exactly $654$ pairs ($654$ real, $654$ fake = $1,308$ videos, $14.997\%$)
   - **Test:** Exactly $654$ pairs ($654$ real, $654$ fake = $1,308$ videos, $14.997\%$)
   - **Total:** Exactly $4,361$ pairs ($8,722$ videos, $100.000\%$)
3. **Discrepancy Resolution:** Fully resolved. The figures $3,061 / 651 / 649$ appearing in the narrative markdown table of `phase4_2_real_extraction_report.md` were clerical transcription errors. The manifest generated during Phase 4.2 ([`real_video_extraction_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_video_extraction_manifest.csv)) and the frozen split manifest have always contained the true $3,053 / 654 / 654$ counts.
4. **Pair-by-Pair Validation:** Exactly 4,361 pairs verified. Every real video (`<base_id>.mp4`) and fake video (`<base_id>_<target_id>.mp4`) exists on disk, shares the identical split and connected component, and matches archive CRC32 checksums and file sizes with 0 discrepancies.
5. **Zero Leakage:** Proved mathematically and empirically: exactly 0 base IDs, 0 target donors, and 0 connected components overlap across train, validation, or test partitions.
6. **Demographic Invariants:** Full distribution characterized across Race, Gender, and Age. Minority cohort statistical limitations (Indian $N=13$, Child $N=10$ in test) documented without modifying the split.
7. **Automated Testing:** Dedicated audit test suite added to [`ai-engine/tests/test_hidf_dataset_audit.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/tests/test_hidf_dataset_audit.py); complete project test suite passed ($254 / 254$ tests passing in 64.97s).
8. **Deliverables Generated:**
   - [`phase4_4_dataset_audit_report.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_dataset_audit_report.md)
   - [`phase4_4_pairing_audit.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_pairing_audit.csv) ($4,361$ rows)
   - [`phase4_4_split_statistics.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_split_statistics.csv) ($4$ summary rows)

---

## 1. Frozen Manifest Verification & True Partition Counts

The experimental partitioning manifest ([`protocol_c_split_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/protocol_c_split_manifest.csv)) was parsed and audited:

- **Filesystem Path:** `protocol_c_split_manifest.csv`
- **Verified SHA-256 Hash:** `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`
- **Total Data Rows:** $4,361$ rows (+ header)
- **Unique Base Video IDs:** $4,361$ (0 duplicates, 0 nulls)
- **Unique Target Donor IDs:** $4,236$ (0 duplicates in 1:1 components; reused donors contained strictly within components)
- **Unique Connected Components:** $4,236$
- **Total Data Cells:** $30,527$ data cells (0 empty, 0 malformed)

### Recomputed True Split Allocations

| Partition | Pair Count | Real Videos | Fake Videos | Total Videos | Manifest Proportion | Component Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 3,053 | 3,053 | 3,053 | 6,106 | 70.007% | 2,969 |
| **Validation** | 654 | 654 | 654 | 1,308 | 14.997% | 633 |
| **Test** | 654 | 654 | 654 | 1,308 | 14.997% | 634 |
| **Total** | **4,361** | **4,361** | **4,361** | **8,722** | **100.000%** | **4,236** |

---

## 2. Resolution of Real-Video Split Count Discrepancy

A focal requirement of Phase 4.4 was resolving the numerical discrepancy between the Phase 4.2 report and the Phase 4.3 report:

- **Reported in `phase4_2_real_extraction_report.md` Section 3 Table:**
  `Train: 3,061 | Val: 651 | Test: 649 (Total: 4,361)`
- **Reported in `phase4_3_fake_extraction_report.md` Section 4 Table:**
  `Train: 3,053 | Val: 654 | Test: 654 (Total: 4,361)`

### Root Cause Analysis & Forensic Resolution
1. **Source Manifest Audit:**
   The master experimental split file, [`protocol_c_split_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/protocol_c_split_manifest.csv), was generated during Phase 3.1 with fixed random seed $42$. Its cryptographic hash (`c15300c1...`) was verified prior to Phase 4.1 download, prior to Phase 4.2 real extraction, prior to Phase 4.3 fake extraction, and in this audit. At no point has the file been regenerated or modified.
2. **Extraction Manifest Inspection:**
   The programmatic manifest written during Phase 4.2 real extraction ([`real_video_extraction_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_video_extraction_manifest.csv)) was loaded and its split column tabulated:
   ```python
   # Split counts in real_video_extraction_manifest.csv:
   {"train": 3053, "val": 654, "test": 654}
   ```
3. **Conclusion:**
   The underlying real extraction data generated in Phase 4.2 **always conformed to $3,053 / 654 / 654$**. The values $3,061 / 651 / 649$ in the markdown prose of `phase4_2_real_extraction_report.md` were an isolated clerical transcription error in documentation, not an engineering or data error. Both real and fake datasets are perfectly symmetric, identical in split counts, and 100% aligned with Protocol C.

---

## 3. Cryptographic Archive & Filesystem Integrity

Both Zenodo source archives and their extracted contents were audited:

| Archive / Asset | Expected Size | Measured Size | Expected MD5 | Measured MD5 | Measured SHA-256 | CRC32 Defective | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Real-vid.zip`** | $842,885,152\text{ B}$ | $842,885,152\text{ B}$ | `ee53b758...` | `ee53b758...` | `d7ecd565...` | 0 | **PASS** |
| **`Fake-vid.zip`** | $715,291,222\text{ B}$ | $715,291,222\text{ B}$ | `708890cc...` | `708890cc...` | `a5833c23...` | 0 | **PASS** |
| **Extracted Real Videos** | $4,361\text{ files}$ | $4,361\text{ files}$ | N/A | N/A | $854,731,837\text{ B}$ | 0 vs ZIP | **PASS** |
| **Extracted Fake Videos** | $4,361\text{ files}$ | $4,361\text{ files}$ | N/A | N/A | $726,240,367\text{ B}$ | 0 vs ZIP | **PASS** |

- **Double-Hash Verification:**
  - `Real-vid.zip` SHA-256 logged in [`real_archive_sha256.txt`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_archive_sha256.txt): `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c`.
  - `Fake-vid.zip` SHA-256 logged in [`fake_archive_sha256.txt`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/fake_archive_sha256.txt): `a5833c23e0fe98285146f313bcc3ecd9f0af6b098de87aeb775af8960655cd6c`.
- **Bit-Level Payload Validation:**
  - 100% of real video files on disk match their respective `Real-vid.zip` central directory member uncompressed size and CRC32 (0 mismatches across 4,361 files).
  - 100% of fake video files on disk match their respective `Fake-vid.zip` central directory member uncompressed size and CRC32 (0 mismatches across 4,361 files).

---

## 4. Pair-by-Pair Validation & Graph Containment

Every single one of the 4,361 rows in `protocol_c_split_manifest.csv` was audited against the physical filesystem:

- **Real Video Member:** `<base_id>.mp4` verified present in `data/extracted/hidf/real/`.
- **Fake Video Member:** `<base_id>_<target_id>.mp4` verified present in `data/extracted/hidf/fake/`.
- **Component Containment:** Verified that the authentic base video and its deepfake derivative belong to the identical connected component ID.
- **Split Consistency:** Verified that the authentic base video and its deepfake derivative are assigned to the identical experimental partition.
- **Anomalies / Corruptions Detected:**
  - Missing real files: **0**
  - Missing fake files: **0**
  - Unexpected / extraneous files: **0**
  - Duplicate base IDs: **0**
  - Duplicate fake pairs: **0**
  - Malformed filenames: **0**

The complete pair-by-pair validation ledger was generated and saved to [`phase4_4_pairing_audit.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_pairing_audit.csv) ($4,361$ rows).

---

## 5. Strict Zero-Leakage Partition Disjointness Audit

Protocol C mandates strict isolation across all three partitions ($V_{\text{train}}, V_{\text{val}}, V_{\text{test}}$) in the bipartite graph:

| Leakage Dimension | Intersection Evaluated | Intersecting Count | Audit Status |
| :--- | :--- | :---: | :---: |
| **Base Video Identity** | $V_{\text{base}}^{\text{train}} \cap V_{\text{base}}^{\text{val}}$ | **0** | **STRICT PASS** |
| **Base Video Identity** | $V_{\text{base}}^{\text{train}} \cap V_{\text{base}}^{\text{test}}$ | **0** | **STRICT PASS** |
| **Base Video Identity** | $V_{\text{base}}^{\text{val}} \cap V_{\text{base}}^{\text{test}}$ | **0** | **STRICT PASS** |
| **Target Donor Identity** | $V_{\text{target}}^{\text{train}} \cap V_{\text{target}}^{\text{val}}$ | **0** | **STRICT PASS** |
| **Target Donor Identity** | $V_{\text{target}}^{\text{train}} \cap V_{\text{target}}^{\text{test}}$ | **0** | **STRICT PASS** |
| **Target Donor Identity** | $V_{\text{target}}^{\text{val}} \cap V_{\text{target}}^{\text{test}}$ | **0** | **STRICT PASS** |
| **Connected Component** | $C^{\text{train}} \cap C^{\text{val}}$ | **0** | **STRICT PASS** |
| **Connected Component** | $C^{\text{train}} \cap C^{\text{test}}$ | **0** | **STRICT PASS** |
| **Connected Component** | $C^{\text{val}} \cap C^{\text{test}}$ | **0** | **STRICT PASS** |

**Graph Disjointness Invariant:**
$$\sum_{s \in \{\text{train}, \text{val}, \text{test}\}} |C^s| = 2,969 + 633 + 634 = 4,236 = |C_{\text{total}}|$$
$$\sum_{s \in \{\text{train}, \text{val}, \text{test}\}} |V_{\text{target}}^s| = 2,969 + 633 + 634 = 4,236 = |V_{\text{target}}|$$
$$\sum_{s \in \{\text{train}, \text{val}, \text{test}\}} |V_{\text{base}}^s| = 3,053 + 654 + 654 = 4,361 = |V_{\text{base}}|$$

Partition leakage is mathematically zero. Neither background environment, facial geometry, nor donor identity crosses split boundaries.

---

## 6. Demographic Balance & Subgroup Distribution Audit

Demographic distributions across partitions were tabulated into [`phase4_4_split_statistics.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_split_statistics.csv):

### 6.1 Racial Cohort Distribution

| Racial Cohort | Dataset Total ($N=4,361$) | Train ($N=3,053$) | Val ($N=654$) | Test ($N=654$) | Stratification Delta ($\Delta_{\text{max}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **White** | 3,406 (78.10%) | 2,385 (78.12%) | 508 (77.68%) | 513 (78.44%) | $+0.34\%$ |
| **Black** | 456 (10.46%) | 317 (10.38%) | 71 (10.86%) | 68 (10.40%) | $+0.40\%$ |
| **Asian** | 215 (4.93%) | 152 (4.98%) | 31 (4.74%) | 32 (4.89%) | $-0.19\%$ |
| **Latino** | 194 (4.45%) | 136 (4.45%) | 30 (4.59%) | 28 (4.28%) | $+0.14\%$ |
| **Indian** | 90 (2.06%) | 63 (2.06%) | 14 (2.14%) | 13 (1.99%) | $+0.08\%$ |

### 6.2 Gender & Age Cohort Distributions

| Demographic Dimension | Cohort | Dataset Total ($N=4,361$) | Train ($N=3,053$) | Val ($N=654$) | Test ($N=654$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gender** | Female | 1,694 (38.84%) | 1,184 (38.78%) | 268 (40.98%) | 242 (37.00%) |
| | Male | 2,667 (61.16%) | 1,869 (61.22%) | 386 (59.02%) | 412 (63.00%) |
| **Age** | Adult | 3,923 (89.96%) | 2,742 (89.81%) | 596 (91.13%) | 585 (89.45%) |
| | Elderly | 371 (8.51%) | 265 (8.68%) | 47 (7.19%) | 59 (9.02%) |
| | Child | 67 (1.54%) | 46 (1.51%) | 11 (1.68%) | 10 (1.53%) |

### 6.3 Critical Statistical Limitations & Imbalances
1. **Severe Dataset Population Skew:**
   The underlying HiDF dataset reflects severe representation bias: White subjects constitute $78.10\%$, while Indian subjects represent only $2.06\%$ ($N=90$ total). Adults constitute $89.96\%$, while Children represent only $1.54\%$ ($N=67$ total).
2. **Test Subgroup Statistical Power Warning:**
   In the test partition ($N=654$ pairs):
   - **Indian Cohort:** $N = 13$ pairs ($13$ real, $13$ fake). Under an assumed AUROC of $0.85$, the asymptotic standard error is $\text{SE} \approx 0.086$, producing a $95\%$ confidence interval width of $\pm 0.17$ ($[0.68, 1.00]$).
   - **Child Cohort:** $N = 10$ pairs ($10$ real, $10$ fake). The standard error is $\text{SE} \approx 0.098$ ($95\%\text{ CI width} \approx \pm 0.19$).
   - **Auditor Mandate:** Standalone subgroup fairness or AUROC parity claims on Indian and Child cohorts cannot achieve statistical significance without external multi-dataset pooling. In accordance with Phase 3 recommendations, subgroup evaluation on these cohorts must be reported with exact sample sizes and Wilson/bootstrap intervals rather than asymptotic Wald intervals.
3. **Gender Variance in Test Partition:**
   Female representation in the test set is $37.00\%$ ($N=242$ pairs), a slight negative deviation of $-1.84\%$ from the dataset expectation ($38.84\%$). As established in Phase 3.2, this minor delta is the mathematical consequence of component-level greedy bin packing where multi-donor connected components could not be fractured. It is accepted without altering the split.

---

## 7. Storage Utilization & Operational Footprint

| Storage Category | Physical Path | File Count | Byte Count | Human Readable |
| :--- | :--- | :---: | :---: | :---: |
| **Compressed Real Archive** | `data/raw/hidf/Real-vid.zip` | 1 archive | 842,885,152 B | ~842.89 MB |
| **Compressed Fake Archive** | `data/raw/hidf/Fake-vid.zip` | 1 archive | 715,291,222 B | ~715.29 MB |
| **Extracted Real Videos** | `data/extracted/hidf/real/` | 4,361 MP4s | 854,731,837 B | ~815.14 MB |
| **Extracted Fake Videos** | `data/extracted/hidf/fake/` | 4,361 MP4s | 726,240,367 B | ~692.60 MB |
| **Total HiDF Dataset Footprint** | `data/` | **8,724 files** | **3,139,148,578 B** | **~2.923 GiB (~3.14 GB)** |
| **Available Space on Drive C:** | `C:\` | N/A | **211,140,292,608 B** | **~196.64 GB** |

Over $196\text{ GB}$ of physical free space remains on volume `C:\`, confirming ample headroom for downstream pipeline artifacts, rPPG signal extractions, and model weights.

---

## 8. Automated Test Suite Validation

The AI Engine test suite was augmented with Phase 4.4 dataset audit unit tests ([`ai-engine/tests/test_hidf_dataset_audit.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/tests/test_hidf_dataset_audit.py)):
- `test_split_manifest_integrity`: Validates exact row count ($4,361$) and partition allocations ($3,053 / 654 / 654$).
- `test_zero_leakage_invariants`: Mathematically validates zero base ID, target donor, and component overlap across all three splits.
- `test_pairing_audit_invariants`: Verifies all $4,361$ pairs in `phase4_4_pairing_audit.csv` have status `VERIFIED_VALID_PAIR` with $100\%$ size and CRC32 match.
- `test_split_statistics_consistency`: Validates that partition component sums match total unique components ($4,236$).

**Full Test Suite Execution:**
```text
================= 254 passed, 15 warnings in 64.97s (0:01:04) =================
pytest exited with code: 0
```
$100\%$ test pass rate ($254 / 254$ tests passing) confirmed.

---

## 9. Deliverables Inventory

1. [`phase4_4_dataset_audit_report.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_dataset_audit_report.md) — This formal comprehensive audit and discrepancy resolution document.
2. [`phase4_4_pairing_audit.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_pairing_audit.csv) — $4,361$ rows mapping every authentic/deepfake pair to base_id, target_id, component_id, split, demographics, file sizes, CRC32s, and verification status.
3. [`phase4_4_split_statistics.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_4_split_statistics.csv) — Summary statistics per partition and dataset total, detailing pair counts, video counts, component counts, uncompressed byte counts, and full demographic breakdowns.
4. [`scripts/audit_phase4_4.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/audit_phase4_4.py) — Standalone reproducible audit script.
5. [`ai-engine/tests/test_hidf_dataset_audit.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/ai-engine/tests/test_hidf_dataset_audit.py) — Invariant verification test suite.

---

## 10. Strict Boundaries & Awaiting Approval

In accordance with strict operational boundaries:
- **No dataset, archive, manifest, or split assignment was modified.**
- **Zero video frames were decoded.**
- **Zero models were trained.**
- **Zero experimental benchmarks were executed.**

Phase 4.4 is complete with all gates **`PASS`**. Awaiting user approval before proceeding to any downstream pipeline phase.
