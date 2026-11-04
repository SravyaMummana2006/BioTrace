# BIOTrace Phase 4 — Acquisition Script Report

**Document:** `phase4_acquisition_script_report.md`  
**Script Path:** [`scripts/acquire_hidf.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/acquire_hidf.py)  
**Author:** Senior Research Data Engineer  
**Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** IMPLEMENTED, DRY-RUN VERIFIED, AWAITING EXECUTION AUTHORIZATION  

---

## 1. Architectural Design & Safety Safeguards

[`scripts/acquire_hidf.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/acquire_hidf.py) was engineered to meet the highest scientific reproducibility and data integrity standards:

1. **Staged Single-Archive Selection:**
   - Accepts `--target real`, `--target fake`, or `--target all`. Allows isolating downloads into separate user-approved steps.
2. **Streaming & Chunking:**
   - Reads data via `requests.get(..., stream=True)` with $1\text{ MB}$ chunks (`1024 * 1024 bytes`). Memory utilization is fixed at $< 50\text{ MB}$.
3. **HTTP Range / Resume:**
   - Detects partial downloads (`.part`). Computes `existing_bytes = os.path.getsize(part_path)` and dispatches `Range: bytes={existing_bytes}-` to avoid re-downloading existing chunks upon transient network disconnects.
4. **Two-Stage Checksum Verification:**
   - In a single streaming pass through the completed file, computes:
     - **MD5:** Compared against official Zenodo publication hashes (`ee53b75850a7f54b45d61a78085eda0a` for Real, `1e5057b545d1f8872e42502c38baaa07` for Fake).
     - **SHA-256:** Logged for research provenance and institutional audit.
5. **Atomic Promotion Pattern:**
   - Downloads are written to `Real-vid.zip.part`.
   - File is renamed to `Real-vid.zip` ONLY after exact size AND MD5 checksum match $100\%$.
6. **Overwrite Protection:**
   - If `Real-vid.zip` already exists and satisfies MD5 verification, the script skips re-downloading to conserve bandwidth.

---

## 2. Dry-Run Verification Log

Pre-flight dry-run execution was performed via `python scripts/acquire_hidf.py --dry-run`:

```text
======================================================================
BIOTrace HiDF Acquisition — DRY-RUN PRE-FLIGHT VERIFICATION
======================================================================
Destination Directory: C:\Users\Sravya mummana\OneDrive\BioTrace\data\raw\hidf
Target Archives Selected: real, fake
Total Required Size: 1,558,176,374 bytes (1485.99 MB)
Available Disk Space: 216,237,105,152 bytes (201.39 GB)
Disk Space Check: PASS

Target: REAL (HiDF Authentic Base Videos Archive)
  URL: https://zenodo.org/records/16140829/files/Real-vid.zip
  Final Destination: data\raw\hidf\Real-vid.zip
  Expected Size: 842,885,152 bytes
  Expected MD5: ee53b75850a7f54b45d61a78085eda0a
  Current Status: NOT YET DOWNLOADED

Target: FAKE (HiDF Deepfake Derivative Videos Archive)
  URL: https://zenodo.org/records/16140829/files/Fake-vid.zip
  Final Destination: data\raw\hidf\Fake-vid.zip
  Expected Size: 715,291,222 bytes
  Expected MD5: 1e5057b545d1f8872e42502c38baaa07
  Current Status: NOT YET DOWNLOADED

Dry-run completed successfully. Zero bytes downloaded.
```
