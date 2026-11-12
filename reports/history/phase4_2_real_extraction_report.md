# BIOTrace Phase 4.2 — Real Video Extraction and Validation Report

**Auditor / Engineer:** Senior Research Data Engineer  
**Dataset Target:** HiDF Authentic Base Videos Archive (`Real-vid.zip`)  
**Official Repository:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Audit & Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** COMPLETE / ALL INTEGRITY GATES PASSED (10 / 10)

---

## Executive Summary

In accordance with Phase 4.2 of the BIOTrace dataset engineering protocol, authentic base video files from `data/raw/hidf/Real-vid.zip` were extracted and comprehensively validated against the frozen experimental split manifest ([`../data/protocol_c_split_manifest.csv`](../data/protocol_c_split_manifest.csv)).

All security, isolation, and cryptographic integrity gates passed with zero discrepancies:
- **Pre-Extraction Archive Verification:** `Real-vid.zip` verified at `842,885,152` bytes, MD5 `ee53b75850a7f54b45d61a78085eda0a`, and SHA-256 `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c`.
- **Defensive Extraction Protocol:** Path traversal (ZipSlip) defenses active; all 4,361 members decompressed into a temporary staging directory (`.real_staging`), verified for file size and CRC32 before atomic promotion to [`data/extracted/hidf/real/`](../../data/extracted/hidf/real/).
- **Member Alignment:** Exactly 4,361 `.mp4` video files extracted ($854,731,837\text{ bytes} \approx 815.14\text{ MB}$). Every extracted file maps 1:1 to the frozen split manifest with 0 missing, 0 unexpected, and 0 duplicate files.
- **Bit-Level Integrity:** 100% of extracted files match their respective ZIP member CRC32 checksums and file sizes (0 mismatches across 4,361 files).
- **Strict Quarantine Maintained:** `Fake-vid.zip` was NOT downloaded and no fake videos exist on disk.
- **Zero Premature Processing:** No video frames decoded, no models trained, no experimental evaluation executed.

---

## 1. Pre-Extraction Quality Gates

Before initiating extraction, the compressed archive and experimental split were audited against published hashes:

| Verification Gate | Required Reference | Actual Measured Value | Status |
| :--- | :--- | :--- | :---: |
| **Archive Existence** | `data/raw/hidf/Real-vid.zip` | Present on filesystem | **PASS** |
| **Archive Byte Count** | `842,885,152` bytes | `842,885,152` bytes | **PASS** |
| **Archive MD5 Checksum** | `ee53b75850a7f54b45d61a78085eda0a` | `ee53b75850a7f54b45d61a78085eda0a` | **PASS** |
| **Archive SHA-256 Hash** | `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c` | `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c` | **PASS** |
| **ZIP Header & CRC Check** | `ZipFile.testzip() == None` | `None` (0 corrupted members) | **PASS** |
| **Frozen Split SHA-256** | `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` | `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` | **PASS** |

---

## 2. Extraction Architecture & Security Controls

Extraction was conducted using [`scripts/extract_hidf.py`](../../scripts/extract_hidf.py) with four defensive safety layers:

1. **ZipSlip / Path Traversal Defense:**
   - Evaluated via `is_safe_path(target_dir, member_path)` using `os.path.commonpath([resolved_dest, resolved_target])`.
   - Stripped all internal directory prefixes (e.g. `Real-vid/c00003.mp4` -> `c00003.mp4`), preventing arbitrary file write or directory escaping.
2. **Temporary Staging Directory Isolation:**
   - Extraction was performed into an isolated staging directory: `data/extracted/hidf/.real_staging`.
   - Prevents partial extraction states in the production directory in case of process termination or I/O failure.
3. **Per-File Size & CRC32 Verification:**
   - Each member was written to `<filename>.mp4.tmp`.
   - Streamed file size was compared against `member.file_size`.
   - Extracted content CRC32 was calculated via `zlib.crc32()` and compared against `member.CRC`.
   - Renamed to `<filename>.mp4` within staging only upon dual confirmation.
4. **Pre-Promotion Staging Audit & Atomic Promotion:**
   - Full staging directory validated: verified exactly 4,361 files present, matching all 4,361 base IDs in `../data/protocol_c_split_manifest.csv`.
   - Upon 100% staging validation pass, the staging directory was promoted atomically to `data/extracted/hidf/real/`.

---

## 3. Post-Extraction Independent Validation Audit

An independent post-extraction audit script verified the extracted assets against the ZIP central directory and the experimental split manifest:

| Validation Metric | Expected Target | Actual Measured | Status |
| :--- | :---: | :---: | :---: |
| **Extracted File Count** | 4,361 files | 4,361 files | **PASS** |
| **File Format Compliance** | 100% `.mp4` format | 4,361 / 4,361 `.mp4` files | **PASS** |
| **Matched Frozen Base IDs** | 4,361 | 4,361 | **PASS** |
| **Missing Base IDs** | 0 | 0 | **PASS** |
| **Unexpected / Extraneous Files** | 0 | 0 | **PASS** |
| **Duplicate Filenames** | 0 | 0 | **PASS** |
| **Total Uncompressed Bytes** | 854,731,837 bytes | 854,731,837 bytes (~815.14 MB) | **PASS** |
| **Size Discrepancies vs ZIP** | 0 | 0 | **PASS** |
| **CRC32 Discrepancies vs ZIP** | 0 | 0 | **PASS** |
| **Original Archive Preserved** | 842,885,152 bytes | 842,885,152 bytes | **PASS** |
| **Original Archive SHA-256 Preserved** | `d7ecd565...` | `d7ecd565...` (Exact match) | **PASS** |
| **Frozen Split SHA-256 Preserved** | `c15300c1...` | `c15300c1...` (Exact match) | **PASS** |

### Split Allocation of Extracted Real Videos

Cross-referencing the extracted videos against [`../data/protocol_c_split_manifest.csv`](../data/protocol_c_split_manifest.csv):

| Split Partition | Expected Real Videos | Extracted Real Videos | Uncompressed Bytes | Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3,061 | 3,061 | 600,006,242 bytes (~572.21 MB) | 70.19% |
| **Val** | 651 | 651 | 127,589,848 bytes (~121.68 MB) | 14.93% |
| **Test** | 649 | 649 | 127,135,747 bytes (~121.25 MB) | 14.88% |
| **Total** | **4,361** | **4,361** | **854,731,837 bytes (~815.14 MB)** | **100.00%** |

---

## 4. Storage Footprint & System Resources

| Storage Metric | Value |
| :--- | :--- |
| **Original Compressed Archive** (`data/raw/hidf/Real-vid.zip`) | 842,885,152 bytes (~842.89 MB) |
| **Extracted Payload Directory** (`data/extracted/hidf/real/`) | 854,731,837 bytes (~815.14 MB) |
| **Total Phase 4 Disk Consumed** | ~1,697,616,989 bytes (~1.658 GB) |
| **Available Disk Space Remaining (`C:\`)** | **213,330,788,352 bytes (~198.68 GB)** |

Storage headroom remains vast (>198 GB available), fully supporting subsequent fake archive acquisition and extraction.

---

## 5. Automated Test Suite Results

A dedicated extraction test suite ([`ai-engine/tests/test_hidf_extraction.py`](../../ai-engine/tests/test_hidf_extraction.py)) was authored and validated against the entire engine test suite:

- `test_is_safe_path_valid`: Validates normal extraction path resolution.
- `test_is_safe_path_traversal_attacks`: Validates detection and rejection of ZipSlip path traversal payloads (`../`, `../../etc/passwd`, Windows drive injections).
- `test_compute_file_crc32`: Validates CRC32 calculation equivalence with `zlib.crc32`.
- `test_compute_hashes`: Validates cryptographic hash consistency for MD5 and SHA-256.
- `test_load_split_manifest`: Validates parsing and indexing of the Protocol C split manifest.
- `test_mock_safe_extraction_staging_and_atomic_promotion`: Validates end-to-end extraction staging, CRC validation, and atomic directory rename.

**Overall Test Suite Result:** `249 passed, 15 warnings in 39.55s` ($100\%$ pass rate across entire project).

---

## 6. Generated Deliverables

The deliverables for Phase 4.2 have been generated and validated:

1. [`../data/real_video_extraction_manifest.csv`](../data/real_video_extraction_manifest.csv) — 4,361 rows (+ header) mapping every extracted video to base_id, file_size, crc32, split, demographic attributes, validation status, and filesystem path.
2. [`data/extracted/hidf/extraction_manifest.csv`](../../data/extracted/hidf/extraction_manifest.csv) — Compatibility manifest inside the extracted dataset tree.
3. [`phase4_2_real_extraction_report.md`](phase4_2_real_extraction_report.md) — This formal engineering and audit report.

---

## 7. Strict Quarantine and Stop Directive

- [x] **`Fake-vid.zip` not downloaded:** Confirmed absent from filesystem.
- [x] **No fake videos extracted:** Confirmed `data/extracted/hidf/fake/` does not exist.
- [x] **No frame extraction or model training executed:** Strict halt enforced.
- [x] **Frozen split preserved:** `../data/protocol_c_split_manifest.csv` unchanged (`c15300c1...`).
- [x] **Original archive preserved:** `Real-vid.zip` unchanged (`d7ecd565...`).
