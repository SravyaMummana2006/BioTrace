# BIOTrace Phase 4 — Extraction Design & Security Protocol

**Document:** `phase4_extraction_design.md`  
**Script Path:** [`scripts/extract_hidf.py`](../../scripts/extract_hidf.py)  
**Author:** Senior Research Data Engineer  
**Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** IMPLEMENTED, DRY-RUN VERIFIED, AWAITING DOWNSTREAM EXECUTION  

---

## 1. Security & Integrity Architecture

Extracting large compressed archives in automated pipelines exposes systems to ZipSlip vulnerabilities, partial extraction corruption, and silent disk pollution. [`scripts/extract_hidf.py`](../../scripts/extract_hidf.py) enforces 5 defensive controls:

1. **Path Traversal Defense (ZipSlip Prevention):**
   - Implements `is_safe_path(target_dir, member_path)` using `os.path.commonpath([resolved_dest, resolved_target])`.
   - Rejects any member attempting directory traversal (`../`, absolute paths, drive letters).
   - Normalizes all paths to pure basenames (`pure_filename = os.path.basename(member.filename)`), stripping archive root folders.
2. **Pre-Extraction Structure Validation:**
   - Runs `zipfile.ZipFile.testzip()` to inspect the local ZIP header and payload before uncompressing any files.
3. **Double Verification per Member (Size & CRC32):**
   - For every extracted file, validates:
     - Uncompressed size matches `member.file_size`.
     - CRC32 checksum matches `member.CRC`.
4. **Atomic Write & Restartability:**
   - Members are extracted to `<filename>.mp4.tmp`.
   - File is atomically renamed to `<filename>.mp4` ONLY after CRC32 and size verification pass.
   - If an extraction is interrupted and restarted, existing files matching size and CRC32 are verified and skipped, enabling zero-waste resumption.
5. **Extraction Manifest Generation:**
   - Generates `data/extracted/hidf/extraction_manifest.csv` logging `archive, target_type, filename, file_size, crc32, status, extracted_path`.

---

## 2. Extraction Target Specifications

| Target | Source Archive | Target Extraction Directory | Expected Member Files | Payload Format |
| :--- | :--- | :--- | :---: | :---: |
| **Real** | `data/raw/hidf/Real-vid.zip` | `data/extracted/hidf/real/` | $4,361$ | `<base_id>.mp4` |
| **Fake** | `data/raw/hidf/Fake-vid.zip` | `data/extracted/hidf/fake/` | $4,361$ | `<base_id>_<target_id>.mp4` |

---

## 3. Dry-Run Verification Log

Pre-flight dry-run execution was performed via `python scripts/extract_hidf.py --dry-run`:

```text
======================================================================
BIOTrace HiDF Extraction — DRY-RUN PREPARATION
======================================================================
Raw Archive Directory:      C:\Users\Sravya mummana\OneDrive\BioTrace\data\raw\hidf
Target Extraction Root:     C:\Users\Sravya mummana\OneDrive\BioTrace\data\extracted\hidf

Target: REAL (Authentic base videos)
  Archive Path:    data\raw\hidf\Real-vid.zip
  Destination Dir: data\extracted\hidf\real
  Expected Files:  4361
  Archive Status:  ARCHIVE NOT FOUND (Awaiting download in Phase 4)

Target: FAKE (Deepfake derivative videos)
  Archive Path:    data\raw\hidf\Fake-vid.zip
  Destination Dir: data\extracted\hidf\fake
  Expected Files:  4361
  Archive Status:  ARCHIVE NOT FOUND (Awaiting download in Phase 4)

Dry-run completed successfully. Zero files extracted.
```
