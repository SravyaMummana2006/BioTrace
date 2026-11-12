# BIOTrace Phase 4.3 — Fake Video Acquisition and Validation Report

**Auditor / Engineer:** Senior Research Data Engineer  
**Dataset Target:** HiDF Deepfake Derivative Videos Archive (`Fake-vid.zip`)  
**Official Repository:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Audit & Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** COMPLETE / ALL INTEGRITY GATES PASSED (10 / 10)

---

## Executive Summary

Phase 4.3 executed the controlled acquisition, cryptographic auditing, and safe extraction of the deepfake derivative video archive (`Fake-vid.zip`) from Zenodo record `16140829`.

All acquisition, security, and experimental alignment gates passed without discrepancy:
- **Official Source Provenance:** Verified directly against the Zenodo REST API for record `16140829`:
  - Official File Size: `715,291,222` bytes (~715.29 MB)
  - Official Published MD5: `708890ccaedc3a6cccee0c1d3e64a711`
  - Computed SHA-256: `a5833c23e0fe98285146f313bcc3ecd9f0af6b098de87aeb775af8960655cd6c`
  - Downloaded byte count and hashes matched Zenodo official records bit-for-bit.
- **ZIP Archive Health:** `ZipFile.testzip()` completed with 0 corrupt members ($100\%$ CRC32 integrity across all 4,362 entries).
- **Defensive Extraction Protocol:** Path traversal (ZipSlip) defense active; members extracted into an isolated staging directory (`.fake_staging`), verified for uncompressed size and CRC32 before atomic promotion to [`data/extracted/hidf/fake/`](../../data/extracted/hidf/fake/).
- **Member Alignment:** Exactly 4,361 deepfake `.mp4` video files extracted ($726,240,367\text{ bytes} \approx 692.60\text{ MB}$). Every extracted file (`<base_id>_<target_id>.mp4`) maps 1:1 to the frozen experimental split manifest ([`../data/protocol_c_split_manifest.csv`](../data/protocol_c_split_manifest.csv)) with 0 missing, 0 unexpected, and 0 duplicate files.
- **Bit-Level Integrity:** 100% of extracted files match their respective ZIP member CRC32 checksums and file sizes (0 mismatches across 4,361 files).
- **Split Allocation:** Extracted fake videos strictly adhere to Protocol C split boundaries:
  - **Train:** 3,053 pairs ($70.01\%$)
  - **Validation:** 654 pairs ($15.00\%$)
  - **Test:** 654 pairs ($15.00\%$)
- **Preservation of Existing Assets:** Real-video archive (`Real-vid.zip`, 842,885,152 bytes) and extracted real videos (`data/extracted/hidf/real/`, 4,361 files) remain untouched. Frozen split manifest SHA-256 (`c15300c1...`) remains untouched.
- **Strict Boundary Adherence:** Zero video frames decoded, zero models trained, and zero experiments initiated.

---

## 1. Official Metadata & Pre-Download Verifications

Before download, official metadata was queried directly from the Zenodo Record 16140829 REST endpoint (`https://zenodo.org/api/records/16140829`):

| Pre-Acquisition Check | Expected Reference | Actual Verified Value | Status |
| :--- | :--- | :--- | :---: |
| **Zenodo Record** | Record `16140829` | HiDF: A Human-Indistinguishable Deepfake Dataset | **PASS** |
| **Target URL** | `https://zenodo.org/records/16140829/files/Fake-vid.zip` | Confirmed direct file endpoint | **PASS** |
| **Expected Size** | `715,291,222` bytes | `715,291,222` bytes | **PASS** |
| **Official Published MD5** | `708890ccaedc3a6cccee0c1d3e64a711` | `708890ccaedc3a6cccee0c1d3e64a711` | **PASS** |
| **Available Disk Space** | $\ge 2.0\text{ GB}$ required | $198.81\text{ GB}$ available on volume `C:\` | **PASS** |
| **Frozen Split Integrity** | `../data/protocol_c_split_manifest.csv` SHA-256 | `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` | **PASS** |

*(Note: Earlier project planning notes referenced an unverified draft checksum `1e5057b5...`; authoritative API query of the live Zenodo repository confirmed `md5:708890ccaedc3a6cccee0c1d3e64a711`, eliminating discrepancy.)*

---

## 2. Acquisition Telemetry & Archive Verification

The download was executed via [`scripts/acquire_hidf.py`](../../scripts/acquire_hidf.py) with automatic HTTP Range resume and chunk streaming:

- **Downloaded File:** `data/raw/hidf/Fake-vid.zip`
- **Total Downloaded Bytes:** `715,291,222` bytes (100.00% complete)
- **Calculated MD5:** `708890ccaedc3a6cccee0c1d3e64a711` (Exact match)
- **Calculated SHA-256:** `a5833c23e0fe98285146f313bcc3ecd9f0af6b098de87aeb775af8960655cd6c`
- **Logged Hashes:** Saved to [`fake_archive_sha256.txt`](../data/fake_archive_sha256.txt) and `data/raw/hidf/acquisition_log.json`.
- **Pre-Extraction Archive Test (`scripts/verify_fake_archive.py`):**
  - `ZipFile.testzip()`: `None` (0 corrupted members).
  - Directory entries: 1 (`Fake-vid/`).
  - Video member entries: 4,361 (`Fake-vid/<base_id>_<target_id>.mp4`).
  - Filename cross-referencing against frozen split: 4,361 matched, 0 missing, 0 unexpected, 0 malformed.
  - Verification report generated: [`../data/fake_archive_integrity_report.csv`](../data/fake_archive_integrity_report.csv) ($4,361\text{ rows}$).

---

## 3. Extraction Architecture & Isolation Controls

Extraction was conducted using [`scripts/extract_hidf.py`](../../scripts/extract_hidf.py) enforcing strict security and atomicity:

1. **Path Traversal Guard:** `is_safe_path()` prevented any member directory traversal or escaping; stripped archive prefix (`Fake-vid/c00003_c11507.mp4` $\rightarrow$ `c00003_c11507.mp4`).
2. **Temporary Staging Directory:** All 4,361 files decompressed into `data/extracted/hidf/.fake_staging/`.
3. **Double Verification per Member:** Each file was written to `.tmp`, checked for exact `file_size` and `CRC32` checksum before internal staging rename.
4. **Pre-Promotion Staging Audit:** Verified exactly 4,361 files in staging matching expected `<base_id>_<target_id>.mp4` pairs from `../data/protocol_c_split_manifest.csv`.
5. **Atomic Promotion:** Promoted `.fake_staging` to production [`data/extracted/hidf/fake/`](../../data/extracted/hidf/fake/).

---

## 4. Post-Extraction Independent Validation Audit

| Validation Metric | Target Specification | Actual Measured Value | Status |
| :--- | :---: | :---: | :---: |
| **Extracted File Count** | 4,361 files | **4,361 files** | **PASS** |
| **File Format Compliance** | 100% `.mp4` format | **4,361 / 4,361 `.mp4` files** | **PASS** |
| **Matched Frozen Fake Pairs** | 4,361 | **4,361 / 4,361 pairs** | **PASS** |
| **Missing Fake Pairs** | 0 | **0** | **PASS** |
| **Unexpected Extraneous Files** | 0 | **0** | **PASS** |
| **Duplicate Filenames** | 0 | **0** | **PASS** |
| **Total Uncompressed Bytes** | 726,240,367 bytes (~692.60 MB) | **726,240,367 bytes** | **PASS** |
| **Size Discrepancies vs ZIP** | 0 | **0** | **PASS** |
| **CRC32 Discrepancies vs ZIP** | 0 | **0** | **PASS** |
| **Real Archive Preserved** | 842,885,152 bytes | **842,885,152 bytes** | **PASS** |
| **Real Extracted Files Preserved** | 4,361 files | **4,361 files** | **PASS** |
| **Frozen Split SHA-256 Preserved** | `c15300c1...` | **`c15300c1...`** (Untouched) | **PASS** |

### Protocol C Split Distribution of Extracted Fake Videos

Cross-referencing extracted fake videos with [`../data/protocol_c_split_manifest.csv`](../data/protocol_c_split_manifest.csv):

| Split Partition | Extracted Fake Videos | Extracted Real Videos | Uncompressed Bytes (Fake) | Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,053 | 3,053 | 508,443,158 bytes (~484.89 MB) | 70.01% |
| **Val** | 654 | 654 | 109,211,489 bytes (~104.15 MB) | 15.00% |
| **Test** | 654 | 654 | 108,585,720 bytes (~103.55 MB) | 15.00% |
| **Total** | **4,361** | **4,361** | **726,240,367 bytes (~692.60 MB)** | **100.00%** |

---

## 5. Storage Footprint & System Headroom

| Storage Resource | Size / Bytes | Human Readable |
| :--- | :--- | :--- |
| **Real Archive** (`data/raw/hidf/Real-vid.zip`) | 842,885,152 bytes | ~842.89 MB |
| **Real Extracted** (`data/extracted/hidf/real/`) | 854,731,837 bytes | ~815.14 MB |
| **Fake Archive** (`data/raw/hidf/Fake-vid.zip`) | 715,291,222 bytes | ~715.29 MB |
| **Fake Extracted** (`data/extracted/hidf/fake/`) | 726,240,367 bytes | ~692.60 MB |
| **Total HiDF Dataset On Disk** | **3,139,148,578 bytes** | **~2.923 GB (~3.14 GB)** |
| **Free Disk Space Remaining (`C:\`)** | **211,803,602,944 bytes** | **~197.26 GB** |

Storage headroom remains vast (>197 GB free), confirming full capacity for subsequent features, models, and evaluation outputs.

---

## 6. Automated Test Suite Validation

The test suite in [`ai-engine/tests/test_hidf_extraction.py`](../../ai-engine/tests/test_hidf_extraction.py) was expanded to validate both real and fake extraction logic:
- `test_is_safe_path_valid` & `test_is_safe_path_traversal_attacks`: Validates ZipSlip defense.
- `test_compute_file_crc32` & `test_compute_hashes`: Validates bit-level verification.
- `test_load_split_manifest`: Validates dual indexing by base ID and pair key `(base_id, target_id)`.
- `test_mock_safe_real_extraction`: Validates staging, CRC check, and atomic promotion for real videos.
- `test_mock_safe_fake_extraction`: Validates staging, pair parsing, CRC check, and atomic promotion for fake videos.

**Full Test Suite Execution:** **`250 passed, 0 failures`** in 19.44s.

---

## 7. Deliverables Summary

1. [`../data/fake_video_extraction_manifest.csv`](../data/fake_video_extraction_manifest.csv) — 4,361 rows (+ header) mapping every extracted fake video to base_id, target_id, file_size, crc32, split, demographic attributes, and filesystem path.
2. [`data/extracted/hidf/fake_extraction_manifest.csv`](../../data/extracted/hidf/fake_extraction_manifest.csv) — In-tree dataset compatibility manifest.
3. [`../data/fake_archive_integrity_report.csv`](../data/fake_archive_integrity_report.csv) — Archive member-level audit report.
4. [`fake_archive_sha256.txt`](../data/fake_archive_sha256.txt) — Cryptographic SHA-256 hash record.
5. [`phase4_3_fake_extraction_report.md`](phase4_3_fake_extraction_report.md) — This formal Phase 4.3 engineering and audit report.

---

## 8. Strict Boundaries & Stop Directive

- [x] **`../data/protocol_c_split_manifest.csv` untouched:** Verified SHA-256 `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`.
- [x] **Real-video directory untouched:** Verified 4,361 files in `data/extracted/hidf/real/`.
- [x] **Real-video archive untouched:** Verified 842,885,152 bytes in `data/raw/hidf/Real-vid.zip`.
- [x] **No video frames decoded:** Zero frame decoding executed.
- [x] **No model training or experiments started:** Strict halt enforced.
- [x] **Stop after Phase 4.3:** Execution halted awaiting user approval.
