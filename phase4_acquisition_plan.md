# BIOTrace Phase 4 — Dataset Acquisition Plan

**Document:** `phase4_acquisition_plan.md`  
**Author:** Senior Research Data Engineer  
**Dataset Target:** High-Resolution Interpretable Deepfake Dataset (HiDF)  
**Official Repository:** Zenodo Record `16140829` (DOI: `10.5281/zenodo.16140829`)  
**Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** READY FOR STAGED ACQUISITION (AWAITING USER APPROVAL)  

---

## 1. Official Target Repository & Archive Specifications

The official HiDF dataset is hosted on CERN Invenio / Zenodo under Open Access. The acquisition target encompasses strictly the two video payload archives:

| Target Archive | Expected File Size | Published MD5 Checksum | Official Zenodo Download URL | Contents |
| :--- | :---: | :---: | :--- | :--- |
| **`Real-vid.zip`** | **842,885,152 bytes** (~842.9 MB) | `ee53b75850a7f54b45d61a78085eda0a` | `https://zenodo.org/records/16140829/files/Real-vid.zip` | 4,361 authentic base videos (`.mp4`) |
| **`Fake-vid.zip`** | **715,291,222 bytes** (~715.3 MB) | `1e5057b545d1f8872e42502c38baaa07` | `https://zenodo.org/records/16140829/files/Fake-vid.zip` | 4,361 deepfake derivative videos (`.mp4`) |

- **Total Combined Download Footprint:** **1,558,176,374 bytes** (~1,485.99 MB / ~1.45 GiB).
- **License & Legal Restriction:** Creative Commons Attribution-NonCommercial 4.0 International (**CC BY-NC 4.0**).
  - Allowed: Non-commercial academic research, adaptation, analysis.
  - Prohibited: Commercial exploitation, unauthorized redistribution.

---

## 2. Staged Controlled Acquisition Workflow

To adhere to the user's explicit approval safeguard and prevent uncontrolled network consumption, acquisition is strictly segmented into sequential gates:

```
                          Phase 4 Acquisition Plan
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
     [Gate 1: Real-vid.zip]                             [Gate 2: Fake-vid.zip]
  • User Explicit Approval                            • User Explicit Approval
  • Streaming Download to .part                       • Streaming Download to .part
  • HTTP Range Resumability                           • HTTP Range Resumability
  • Size & MD5 Verification                           • Size & MD5 Verification
  • Atomic Rename to Final File                       • Atomic Rename to Final File
           │                                                   │
           └─────────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                      [Gate 3: Archive Verification]
                   • Non-extracting ZIP integrity test
                   • Frozen manifest cross-verification
```

### Staged Execution Gates:
1. **Gate 1 — Acquisition of `Real-vid.zip`:**
   - Command: `python scripts/acquire_hidf.py --target real`
   - Downloads 842.9 MB to `data/raw/hidf/Real-vid.zip.part`.
   - Validates size ($842,885,152\text{ B}$) and MD5 (`ee53b75850a7f54b45d61a78085eda0a`).
   - Promotes atomically to `data/raw/hidf/Real-vid.zip`.
2. **Gate 2 — Acquisition of `Fake-vid.zip`:**
   - Command: `python scripts/acquire_hidf.py --target fake`
   - Downloads 715.3 MB to `data/raw/hidf/Fake-vid.zip.part`.
   - Validates size ($715,291,222\text{ B}$) and MD5 (`1e5057b545d1f8872e42502c38baaa07`).
   - Promotes atomically to `data/raw/hidf/Fake-vid.zip`.
3. **Gate 3 — Checksum & Manifest Archiving:**
   - Calculates SHA-256 hashes for both archives.
   - Logs acquisition metadata to `data/raw/hidf/acquisition_log.json`.

---

## 3. Network, Resumption, and Error Policies

- **Chunked Streaming:** Chunks are read and written in $1\text{ MB}$ blocks (`1024 * 1024 bytes`) to keep RAM consumption $< 50\text{ MB}$.
- **HTTP Range / Resumption:** If network disconnection occurs, the script reads existing bytes from the `.part` file and issues an HTTP `Range: bytes={existing_bytes}-` request. If Zenodo returns `206 Partial Content`, download resumes from the exact interruption offset.
- **Fail-Safe Integrity:** If a downloaded file fails the MD5 check, the corrupted `.part` file is quarantined/deleted and never promoted to the final filename.
