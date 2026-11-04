# BIOTrace Phase 2.2 — Remote ZIP Central Directory Inspection (Zero Download) Audit Report

**Date:** March 2026  
**Auditor:** BIOTrace Senior Dataset Engineering & Methodology Team  
**Dataset Record:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Methodology:** Remote HTTP Range Parsing (Zero Video Download, Zero Extraction)  
**Execution Status:** COMPLETE & VERIFIED  

---

## Executive Summary

Using standard HTTP/1.1 Range requests directed at Zenodo's official CERN Invenio storage endpoints, we successfully parsed the complete End of Central Directory (EOCD) records and Central Directory File Headers (CDFH) for both **`Real-vid.zip`** and **`Fake-vid.zip`** without downloading a single video payload or extracting any archive contents.

### Key Audit Metrics at a Glance

| Metric | Target / Budget | Real-vid.zip | Fake-vid.zip | Combined Total | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Archive Size** | N/A | 842,885,152 B (~842.89 MB) | 715,291,222 B (~715.29 MB) | 1,558,176,374 B (~1.56 GB) | **VERIFIED** |
| **Data Transferred** | < 10,485,760 B (10 MB) | 452,982 B (~0.43 MB) | 483,509 B (~0.46 MB) | **936,491 B (~0.89 MB)** | **VERIFIED** |
| **Budget Utilization** | 100.00% max | 4.32% | 4.61% | **8.93%** | **VERIFIED** |
| **Video Payloads Downloaded** | 0 B | 0 B | 0 B | **0 B (Zero)** | **VERIFIED** |
| **Central Directory Records** | N/A | 4,362 (1 dir + 4,361 vids) | 4,362 (1 dir + 4,361 vids) | 8,724 entries | **VERIFIED** |
| **Valid Video Files (.mp4)** | N/A | 4,361 | 4,361 | 8,722 files | **VERIFIED** |
| **Unique Base Video IDs** | N/A | 4,361 | 4,361 | 4,361 | **VERIFIED** |
| **Unique Target Donor IDs**| N/A | N/A | 4,236 | 4,236 | **VERIFIED** |
| **Verified Valid Pairs** | 100% target | N/A | 4,361 / 4,361 (100.00%) | **4,361 pairs** | **VERIFIED** |
| **Cluster Degree (Fakes/Base)**| N/A | N/A | Exactly 1 for all bases | Degree 1: 100.00% | **VERIFIED** |
| **Malformed / Corrupt Names**| 0 target | 0 | 0 | **0** | **VERIFIED** |
| **Duplicate Filenames** | 0 target | 0 | 0 | **0** | **VERIFIED** |

---

## Phase 1: HTTP Support & Endpoint Verification

### Request & Endpoint Telemetry

Both archives were probed via HTTP `HEAD` and test `GET` Range requests following redirects:

1. **`Real-vid.zip`**
   - **Request URL:** `https://zenodo.org/records/16140829/files/Real-vid.zip`
   - **Final Response URL:** `https://zenodo.org/records/16140829/files/Real-vid.zip` (no redirects, no query params)
   - **HTTP Status:** `200 OK`
   - **Content-Length:** `842,885,152` bytes
   - **Content-Type:** `application/octet-stream`
   - **Accept-Ranges:** Declared `none` on HEAD, but returned `206 Partial Content` on test range probe (`bytes=0-0`).
   - **ETag:** `none`
   - **Classification:** **VERIFIED**

2. **`Fake-vid.zip`**
   - **Request URL:** `https://zenodo.org/records/16140829/files/Fake-vid.zip`
   - **Final Response URL:** `https://zenodo.org/records/16140829/files/Fake-vid.zip` (no redirects, no query params)
   - **HTTP Status:** `200 OK`
   - **Content-Length:** `715,291,222` bytes
   - **Content-Type:** `application/octet-stream`
   - **Accept-Ranges:** Declared `none` on HEAD, but returned `206 Partial Content` on test range probe (`bytes=0-0`).
   - **ETag:** `none`
   - **Classification:** **VERIFIED**

---

## Phase 2: ZIP Footer & EOCD Parsing

Standard ZIP format specifications specify that the End of Central Directory (EOCD) record (`b"PK\x05\x06"`, signature `0x06054b50`) resides within the terminal 65,557 bytes of the archive (accounting for the 22-byte fixed EOCD record and up to 65,535 bytes of variable archive comment).

### Footer Range Queries

1. **`Real-vid.zip` Footer:**
   - **Byte Range Requested:** `bytes=842819595-842885151` (exactly 65,557 bytes).
   - **HTTP Status:** `206 Partial Content`.
   - **EOCD Location:** Byte offset `842,885,130` (exactly 22 bytes before EOF).
   - **EOCD Attributes:**
     - Disk number: `0`
     - Central directory start disk: `0`
     - Central directory records on this disk: `4,362`
     - Total central directory records: `4,362`
     - Central directory byte size: `387,424` bytes
     - Central directory start offset: `842,497,706`
     - Archive comment length: `0`
     - ZIP64 Locator present: **No** (`is_zip64 = False`, standard ZIP32).
   - **Classification:** **VERIFIED**

2. **`Fake-vid.zip` Footer:**
   - **Byte Range Requested:** `bytes=715225665-715291221` (exactly 65,557 bytes).
   - **HTTP Status:** `206 Partial Content`.
   - **EOCD Location:** Byte offset `715,291,200` (exactly 22 bytes before EOF).
   - **EOCD Attributes:**
     - Disk number: `0`
     - Central directory start disk: `0`
     - Central directory records on this disk: `4,362`
     - Total central directory records: `4,362`
     - Central directory byte size: `417,951` bytes
     - Central directory start offset: `714,873,249`
     - Archive comment length: `0`
     - ZIP64 Locator present: **No** (`is_zip64 = False`, standard ZIP32).
   - **Classification:** **VERIFIED**

---

## Phase 3: Central Directory Retrieval & Parsing

With the exact byte offsets and lengths extracted from the EOCD records, targeted HTTP Range requests were dispatched to retrieve only the Central Directory byte arrays:

### Central Directory Range Queries

1. **`Real-vid.zip` Central Directory:**
   - **Byte Range Requested:** `bytes=842497706-842885129` (387,424 bytes).
   - **HTTP Status:** `206 Partial Content`.
   - **Bytes Transferred:** `387,424` bytes.
   - **Headers Parsed:** 4,362 CDFH entries (`0x02014b50`).
   - **Structure:** 1 directory record (`Real-vid/`) + 4,361 video files (`Real-vid/<id>.mp4`).
   - **Classification:** **VERIFIED**

2. **`Fake-vid.zip` Central Directory:**
   - **Byte Range Requested:** `bytes=714873249-715291199` (417,951 bytes).
   - **HTTP Status:** `206 Partial Content`.
   - **Bytes Transferred:** `417,951` bytes.
   - **Headers Parsed:** 4,362 CDFH entries (`0x02014b50`).
   - **Structure:** 1 directory record (`Fake-vid/`) + 4,361 video files (`Fake-vid/<base_id>_<target_id>.mp4`).
   - **Classification:** **VERIFIED**

### Network Budget Audit

- **Enforced Budget:** $10\text{ MB} = 10,485,760\text{ bytes}$.
- **Total Network Transfer:**
  - Real-vid.zip HEAD: 0 B
  - Real-vid.zip Range test: 1 B
  - Real-vid.zip Footer: 65,557 B
  - Real-vid.zip Central Directory: 387,424 B
  - Fake-vid.zip HEAD: 0 B
  - Fake-vid.zip Range test: 1 B
  - Fake-vid.zip Footer: 65,557 B
  - Fake-vid.zip Central Directory: 417,951 B
  - **Grand Total:** **936,491 bytes** ($0.893\text{ MB}$).
- **Headroom Remaining:** $9,549,269\text{ bytes}$ ($9.107\text{ MB}$, 91.07% headroom).
- **Classification:** **VERIFIED**

---

## Phase 4: Pair Analysis & Metadata Cross-Verification

All member filenames extracted from the Central Directories were joined and verified against the authentic base video and image donor rows in `datasets/metadata.csv` (1,647,437 bytes, 42,901 rows).

### 1. Real Video Verification
- **Total real files:** 4,361.
- **File extension:** 100% `.mp4`.
- **Base ID extraction:** Stem parsed directly (e.g., `c00003.mp4` $\to$ ID `c00003`).
- **Metadata match:** All 4,361 IDs match authentic base video rows (`Type == 'Vid'` and `Base == 'Base'`) in `metadata.csv`.
- **Unmatched real videos:** `0`.
- **Missing authentic metadata rows:** `0`.
- **Classification:** **VERIFIED**

### 2. Fake Video Verification
- **Total fake files:** 4,361.
- **Naming convention:** All 4,361 filenames strictly match `<base_id>_<target_id>.mp4`.
- **Malformed filenames:** `0` (zero multi-underscore, empty, or non-mp4 names).
- **Unique base IDs in fake archive:** 4,361.
- **Cross-archive Base verification:** Every base ID in `Fake-vid.zip` exists in `Real-vid.zip` (4,361 / 4,361, 100.00%).
- **Metadata Base verification:** Every base ID in `Fake-vid.zip` exists in `metadata.csv` `(Vid, Base)` (4,361 / 4,361, 100.00%).
- **Metadata Target verification:** Every target ID in `Fake-vid.zip` exists in `metadata.csv` `(Img)` (4,361 / 4,361, 100.00%).
- **Unique target donor IDs:** 4,236 unique IDs (drawn from the 38,540 image donor candidates in `metadata.csv`).
- **Unverified fake pairs:** `0`.
- **Classification:** **VERIFIED**

### 3. Cluster Degree Distribution
The empirical cluster degree distribution reveals an exact 1-to-1 pairing structure across the entire dataset:

| Cluster Degree ($k$ fakes / base) | Number of Base Clusters | Total Fake Derivatives | % of Total Base Clusters | Status |
| :---: | :---: | :---: | :---: | :---: |
| **0** (No fake derivative) | 0 | 0 | 0.00% | Verified |
| **1** (Exactly one fake) | **4,361** | **4,361** | **100.00%** | **VERIFIED** |
| **> 1** (Multiple fakes) | 0 | 0 | 0.00% | Verified |

> **Scientific Insight:** In Phase 1, the presence of 38,540 image donor rows in `metadata.csv` suggested that base videos might have variable numbers of fake derivatives (e.g. 1 to 10 swaps per base). This empirical Phase 2.2 audit resolves that uncertainty definitively: **HiDF contains exactly 4,361 authentic base videos and exactly 4,361 fake derivative videos, forming 4,361 strictly isolated 1:1 pairs.**

---

## Phase 5: Identity Leakage Assessment

A core requirement of experimental rigor in deepfake detection is distinguishing **source-video content independence** from **subject person-identity independence**.

### Video-Level vs. Person-Level Separation

1. **Available Metadata Schema:**
   The official `metadata.csv` provides only:
   `ID, Img/Vid, Base/Swap, Race, Gender, Age`
   There is **no** `subject_id`, `person_id`, `actor_name`, or facial identity descriptor column.

2. **Source-Video Level Separation (VERIFIED):**
   - Each base video cluster consists of exactly one real video (`<base_id>.mp4`) and one fake video (`<base_id>_<target_id>.mp4`).
   - By partitioning splits strictly by `base_id` (placing both the real video and its fake swap into the same partition), **zero base-video content leakage** across train, validation, and test splits is guaranteed.
   - Ground truth physiological signals (rPPG pulses) originate from the base video subject's vascular perfusion; grouping by `base_id` ensures that rPPG models are evaluated on unseen video recordings.

3. **Subject Person-Identity Leakage (PARTIALLY VERIFIED / LIMITATION):**
   - Because `metadata.csv` does not provide an explicit subject identifier linking different video clips of the same individual, we cannot mathematically guarantee that two distinct base IDs (e.g., `c00003` and `c00004`) do not depict the same human actor recorded under different lighting, camera angles, or sessions.
   - **Target Donor Cross-Contamination:** 4,236 unique image donors were used across the 4,361 fakes. A small subset (125 target donors) is reused across multiple base videos. If target image donor $T$ is used in training set fakes, a deep learning visual model could potentially memorize visual features of donor $T$ if $T$ were also present in test set fakes.
   - **Mitigation Protocol for Experiment Splitting:**
     When constructing subsets for Experiments B, D, and E, the subset selection protocol must enforce **bi-bipartite isolation**: splits must be disjoint on both `base_id` AND `target_id`. Because 4,236 donors are available for 4,361 videos, selecting mutually disjoint subsets where no target donor crosses partition boundaries is trivial to enforce.

---

## Phase 6: Deliverables Inventory

The Phase 2.2 audit produced the following persistent, reproducible deliverables:

| Deliverable File | Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **Audit Report** | `phase2_2_remote_audit.md` | ~8 KB | This comprehensive audit report. |
| **Fake Pair Manifest** | `remote_fake_pair_manifest.csv` | 295,109 B | 4,361 verified pair rows mapping every fake video to its base and donor IDs. |
| **Cluster Degree Distribution** | `remote_cluster_degree_distribution.csv` | 90 B | Frequency table demonstrating 100.00% Degree-1 clustering. |
| **Unmatched Files Log** | `remote_unmatched_files.csv` | 54 B | Error catalog (0 errors, 0 unmatched files). |
| **Audit Summary JSON** | `scratch/remote_audit_summary.json` | ~4 KB | Machine-readable log of all HTTP queries and telemetry. |
| **Inspection Tool** | `scratch/remote_zip_audit.py` | ~16 KB | Fully reproducible Python script utilizing HTTP Range requests. |

*Note: All original Phase 1 and Phase 2 files (`phase2_archive_audit.md`, `datasets/fake_pair_manifest.csv`, `datasets/cluster_degree_distribution.csv`, `datasets/unmatched_archive_files.csv`, `datasets/metadata.csv`) remain completely untouched and preserved.*

---

## Final Classification Summary

| Component | Status | Empirical Verification Notes |
| :--- | :---: | :--- |
| **HTTP Range Support on Zenodo** | **VERIFIED** | Server returned `206 Partial Content` across all byte-range queries. |
| **Network Budget (< 10 MB)** | **VERIFIED** | Total network transfer was $936,491\text{ bytes}$ ($0.893\text{ MB}$, < 9% of budget). |
| **Zero Video Payload Download** | **VERIFIED** | Only EOCD (65 KB) and CDFH (387 KB / 418 KB) headers retrieved; 0 video frames fetched. |
| **Total Real Video Count** | **VERIFIED** | Exactly 4,361 authentic MP4 video members in `Real-vid.zip`. |
| **Total Fake Video Count** | **VERIFIED** | Exactly 4,361 fake MP4 video members in `Fake-vid.zip`. |
| **Base-to-Fake Pairing Integrity** | **VERIFIED** | Exactly 4,361 pairs (100.00% valid; 0 corrupt, 0 malformed, 0 unmatched). |
| **Cluster Degree Distribution** | **VERIFIED** | Exactly 1 fake derivative per base video (100.00% Degree 1). |
| **Video-Level Split Independence** | **VERIFIED** | Fully supported via strict grouping by `base_id`. |
| **Person-Level Split Independence** | **PARTIALLY VERIFIED** | Metadata lacks subject ID annotations; donor leakage can be mitigated via disjoint `target_id` filtering. |
