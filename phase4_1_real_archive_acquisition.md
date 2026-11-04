# BIOTrace Phase 4.1 — Authorized Real-Video Archive Acquisition Report

**Auditor / Engineer:** Senior Research Data Engineer  
**Dataset Target:** HiDF Authentic Base Videos Archive (`Real-vid.zip`)  
**Official Repository:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Audit & Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** COMPLETE / ALL INTEGRITY GATES PASSED (7 / 7)

---

## Executive Summary

Phase 4.1 executed the authorized, controlled acquisition and non-extracting cryptographic audit of the authentic video archive (`Real-vid.zip`) from Zenodo record `16140829`.

All acquisition and integrity gates passed without exception:
- **Download Authorization:** Strictly scoped to `Real-vid.zip` (User explicitly authorized; `Fake-vid.zip` untouched).
- **Physical Size:** Exactly `842,885,152` bytes (~842.89 MB), matching published metadata bit-for-bit.
- **Cryptographic Hash Verification:**
  - **MD5:** `ee53b75850a7f54b45d61a78085eda0a` (Exact match to Zenodo official record).
  - **SHA-256:** `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c`.
- **ZIP Archive Health:** `ZipFile.testzip()` verified 100% CRC32 checksums with 0 corrupt members.
- **Experimental Alignment:** All 4,361 authentic base video members match `protocol_c_split_manifest.csv` and `remote_fake_pair_manifest.csv` with 0 missing, 0 unexpected, 0 duplicates, and 0 malformed filenames.
- **Zero-Payload Extraction Guard:** 100% compliant. No video frames, audio, or MP4 files were decompressed or written to disk.

---

## 1. Pre-Download Verifications & Authorization

In strict adherence to the research project charter, acquisition commenced ONLY after explicit user authorization for `Real-vid.zip`. All pre-flight safety gates passed:

| Pre-Download Check Item | Required Specification | Verified Actual Value | Status |
| :--- | :--- | :--- | :---: |
| **User Authorization** | Explicit approval for `Real-vid.zip` | Granted by user | **PASS** |
| **Target Isolation** | `Real-vid.zip` ONLY (`Fake-vid.zip` prohibited) | Confirmed: target is strictly `real` | **PASS** |
| **Frozen Manifest Integrity** | `protocol_c_split_manifest.csv` SHA-256 | `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7` | **PASS** |
| **Available Disk Space** | $\ge 2.0\text{ GB}$ free on volume | $201.42\text{ GB}$ free ($216,275,542,016\text{ B}$) | **PASS** |
| **Destination Path** | `data/raw/hidf/Real-vid.zip` | Pre-created and verified clean | **PASS** |
| **Existing Archive Check** | Skip re-download if valid archive present | Verified initial state (clean start) | **PASS** |

---

## 2. Acquisition Telemetry & Download Architecture

The download was executed via [`scripts/acquire_hidf.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/acquire_hidf.py) with the following parameters:

- **Source URL:** `https://zenodo.org/records/16140829/files/Real-vid.zip`
- **Expected Byte Count:** `842,885,152` bytes (~842.89 MB)
- **Expected MD5 Checksum:** `ee53b75850a7f54b45d61a78085eda0a`
- **Download Staging:** Streaming directly to `data/raw/hidf/Real-vid.zip.part` using $1\text{ MB}$ chunks with automatic HTTP Range resume support (`Range: bytes={existing_bytes}-`).
- **Network Resilience:** Automatic retry loop handling upstream connection drops from CERN/Zenodo nginx proxy.
- **Atomic Promotion:** The `.part` file was promoted to final `Real-vid.zip` ONLY after byte count and MD5 verification succeeded.

Acquisition metadata was recorded in `data/raw/hidf/acquisition_log.json`.

---

## 3. Post-Acquisition Verification Protocol

Post-acquisition verification was conducted by [`scripts/verify_real_archive.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/verify_real_archive.py) using non-extracting methods:

### 3.1 File Size & Hashes

| Metric | Target Specification | Verified Value | Match Status |
| :--- | :--- | :--- | :---: |
| **File Path** | `data/raw/hidf/Real-vid.zip` | `data/raw/hidf/Real-vid.zip` | **PASS** |
| **Exact Bytes** | `842,885,152` | `842,885,152` | **PASS** |
| **Size (MB)** | ~`842.89 MB` | `842.89 MB` | **PASS** |
| **MD5 Hash** | `ee53b75850a7f54b45d61a78085eda0a` | `ee53b75850a7f54b45d61a78085eda0a` | **PASS** |
| **SHA-256 Hash** | Calculated & logged | `d7ecd565b638e699ac7ec928eae954cc206ca68837a1939587696dbee9ae560c` | **PASS** |

The SHA-256 hash was permanently recorded in [`real_archive_sha256.txt`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_archive_sha256.txt).

### 3.2 ZIP Archive Structural Integrity & CRC32

- **Archive Test (`ZipFile.testzip()`):** Returns `None` (0 corrupted members, 100% CRC32 integrity).
- **Total Central Directory Entries:** 4,362 members.
  - Root directory member: `Real-vid/` (1 entry).
  - MP4 video files: 4,361 entries.
  - Non-MP4 or unexpected file types: 0 entries.

### 3.3 Member Alignment with Frozen Manifest

Every video member in the archive was cross-referenced with `protocol_c_split_manifest.csv` and `remote_fake_pair_manifest.csv`:

| Verification Metric | Expected Count | Archive Actual Count | Status |
| :--- | :---: | :---: | :---: |
| **Total MP4 Video Files** | 4,361 | 4,361 | **PASS** |
| **Matched Frozen Base IDs** | 4,361 | 4,361 | **PASS** |
| **Matched Remote Pair Manifest** | 4,361 | 4,361 | **PASS** |
| **Missing Expected Base IDs** | 0 | 0 | **PASS** |
| **Unexpected Extraneous Base IDs** | 0 | 0 | **PASS** |
| **Duplicate Filenames** | 0 | 0 | **PASS** |
| **Malformed Filenames** | 0 | 0 | **PASS** |

The item-by-item verification log was written to [`real_archive_integrity_report.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_archive_integrity_report.csv) ($4,361$ rows).

---

## 4. Storage Footprint & Capacity Audit

| Storage Resource | Before Phase 4.1 | After Phase 4.1 | Delta / Consumed |
| :--- | :--- | :--- | :--- |
| **Archive Footprint** | $0.00\text{ MB}$ | $842.89\text{ MB}$ ($842,885,152\text{ B}$) | $+842.89\text{ MB}$ |
| **Free Space (Drive C:)** | $201.42\text{ GB}$ | $199.62\text{ GB}$ ($214,339,403,776\text{ B}$) | $-1.80\text{ GB}$ (incl. temp/logs) |

Remaining capacity is more than sufficient for downstream acquisition of `Fake-vid.zip` (~715.3 MB) and selective extraction (~1.6 GB total uncompressed).

---

## 5. Summary of Phase 4.1 Deliverables

The following artifacts have been generated and validated:

1. [`data/raw/hidf/Real-vid.zip`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/data/raw/hidf/Real-vid.zip) — The cryptographically verified authentic archive ($842,885,152\text{ bytes}$).
2. [`real_archive_sha256.txt`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_archive_sha256.txt) — Official SHA-256 hash record.
3. [`real_archive_integrity_report.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/real_archive_integrity_report.csv) — 4,361-row audit tracking every member file, uncompressed size, compressed size, CRC32, and manifest alignment.
4. [`phase4_1_real_archive_acquisition.md`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/phase4_1_real_archive_acquisition.md) — This formal audit report.
5. `data/raw/hidf/acquisition_log.json` — Telemetry and audit run record.
6. `scratch/real_archive_summary.json` — Machine-readable summary metrics.

---

## 6. Phase 4.1 Compliance Checklist

- [x] **Download completed:** PASS (100% complete, atomic promotion).
- [x] **Downloaded size:** PASS (842,885,152 bytes).
- [x] **MD5 & SHA-256 verification:** PASS (MD5 matches Zenodo, SHA-256 recorded).
- [x] **ZIP integrity result:** PASS (0 corrupt members via `testzip()`).
- [x] **Filename matching result:** PASS (4,361 / 4,361 matched, 0 missing/unexpected).
- [x] **Remaining storage reported:** PASS (199.62 GB free).
- [x] **Deliverables created:** PASS (all 3 required files in workspace).
- [x] **Zero extraction maintained:** PASS (0 video frames decompressed).
- [x] **Fake archive strictly quarantined:** PASS (`Fake-vid.zip` not acquired).
- [x] **Experimental split intact:** PASS (`protocol_c_split_manifest.csv` untouched).
