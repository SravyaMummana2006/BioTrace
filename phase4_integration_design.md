# BIOTrace Phase 4 — Dataset Integration Design

**Document:** `phase4_integration_design.md`  
**Script Path:** [`scripts/integrate_hidf.py`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/scripts/integrate_hidf.py)  
**Author:** Senior Research Data Engineer  
**Execution Date:** November 4, 2026 (`2026-11-04`)  
**Status:** IMPLEMENTED, DRY-RUN VERIFIED, AWAITING DOWNSTREAM EXECUTION  

---

## 1. Frozen Experimental Split Integration

To ensure strict scientific reproducibility and prevent data leakage, dataset integration binds the extracted video files exclusively to the frozen Protocol C split manifest:

- **Frozen Manifest Path:** [`protocol_c_split_manifest.csv`](file:///c:/Users/Sravya%20mummana/OneDrive/BioTrace/protocol_c_split_manifest.csv)
- **Immutable Cryptographic Hash (SHA-256):**
  ```text
  c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7
  ```
- **Integrity Rule:** `scripts/integrate_hidf.py` computes the SHA-256 hash of `protocol_c_split_manifest.csv` prior to processing. If any bit has changed, the script raises an immediate fatal exception.

---

## 2. Filename Mapping & Integrity Verification Rules

1. **Authentic Base Videos:**
   - Mapped from `base_id` column: `data/extracted/hidf/real/{base_id}.mp4`
   - Exact count: $4,361$ files.
2. **Deepfake Derivative Videos:**
   - Mapped from `base_id` and `target_id` columns: `data/extracted/hidf/fake/{base_id}_{target_id}.mp4`
   - Exact count: $4,361$ files.
3. **Partition Assignments (Zero Random Splitting):**
   - **Train Partition:** $3,053$ real + $3,053$ fake = $6,106$ videos ($70.01\%$)
   - **Validation Partition:** $654$ real + $654$ fake = $1,308$ videos ($15.00\%$)
   - **Test Partition:** $654$ real + $654$ fake = $1,308$ videos ($15.00\%$)
4. **Leakage Enforcement:**
   - Real and fake pairs are permanently bound.
   - Partitions are strictly disjoint on `base_id` AND `target_id` (0 base leakage, 0 donor leakage).

---

## 3. Output Deliverable Schema: `hidf_integrity_manifest.csv`

The integration script outputs `data/manifests/hidf_integrity_manifest.csv` containing:
- `base_id`: Unique base video recording ID
- `target_id`: Unique face-swap image donor ID
- `component_id`: Bipartite star component ID (`COMP_0001` to `COMP_4236`)
- `split`: `train`, `val`, or `test`
- `race, gender, age`: Base subject demographic attributes
- `real_filename, fake_filename`: Exact relative filenames
- `real_exists, fake_exists`: Boolean flags confirming presence on disk
- `real_size_bytes, fake_size_bytes`: Physical file sizes on disk
- `status`: `VALID` or `MISSING_PAYLOAD`

---

## 4. Dry-Run Verification Log

Pre-flight dry-run execution was performed via `python scripts/integrate_hidf.py --dry-run`:

```text
======================================================================
BIOTrace HiDF Dataset Integration — DRY-RUN VERIFICATION
======================================================================
Checking frozen split manifest: protocol_c_split_manifest.csv...
Frozen Split Manifest Verified: SHA-256 MATCH (c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7)
Total Verified Split Rows: 4,361
Expected Real Video Directory: data\extracted\hidf\real
Expected Fake Video Directory: data\extracted\hidf\fake
Manifest Output Directory:     data\manifests

Extracted Files Currently Present:
  Real Videos: 0 / 4,361 expected
  Fake Videos: 0 / 4,361 expected
  Status: Awaiting Phase 4 archive download and extraction.

Dry-run completed successfully.
```
