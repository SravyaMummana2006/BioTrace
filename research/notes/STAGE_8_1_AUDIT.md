# BIOTrace Stage 8.1 — Independent Scientific and Engineering Audit Report

**Audit Date:** November 3, 2026  
**Auditor:** Independent Senior ML Researcher & Computer-Vision Experimental Systems Auditor  
**Scope:** Stage 8 Experimental Validation Framework (`ai-engine/src/biotrace/experiments/`, tests, and associated research notes)  
**Repository State:** master branch, git clean, zero uncommitted or pushed artifacts  

---

## 1. Overall Verdict

### Summary Assessment
The Stage 8 experimental framework represents a **scientifically disciplined, mathematically rigorous, and architecturally complete software engineering framework** for reproducible benchmark evaluation. 

However, **it is critical to distinguish software engineering correctness from empirical scientific validation:**
- **Software Framework:** Fully verified, typed, integrated with BIOTrace extraction engines, free of leakage bugs, and validated by 239 unit/integration tests in Python and 25 tests in Java.
- **Empirical Scientific Validation:** **PENDING REAL-WORLD DATASET ACCESS.** No real video benchmark data (UBFC-rPPG, FaceForensics++, Celeb-DF v2, DFDC) has been downloaded or evaluated. All tests and local demonstrations strictly execute against synthetic test fixtures or mock pipeline data.

> [!IMPORTANT]
> **No Empirical Accuracy Claims:** BIOTrace has **not** established empirical AUROC, EER, or rPPG Pearson $r$ benchmarks against real human subjects or deepfake videos. Any numerical values observed during testing are artifacts of deterministic synthetic fixtures used to verify metric calculation pipelines.

---

## 2. Findings by Experiment (A through E)

Every experiment runner in `ai-engine/src/biotrace/experiments/` was audited for execution flow, inputs, extractor invocation, video handling, per-sample outputs, and implementation completeness.

| Experiment | Module Name | Primary Objective | Extractor Invoked | Input Requirements | Real vs. Mock Handling | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | `experiment_a.py` | rPPG physiological fidelity vs. contact PPG ground truth | `RPPGSignalExtractor` (`biotrace.rppg.extractor`) | Video files + ground truth BVP/pulse waveforms (`.txt`, `.csv`, `.mat`) + `DatasetManifest` | Analyzes actual video frames when available; accepts precomputed BVP signals for mock evaluation; flags missing ground truth | **Implemented & Wired** *(Awaiting external video datasets)* |
| **B** | `experiment_b.py` | Visual deepfake detection in-distribution baseline | `VisualDeepfakeDetector` (`biotrace.visual_detection.detector`) | Video files + binary class labels (`PRISTINE` / `MANIPULATED`) + `DatasetSplit` | Extracts video frames, crops faces, runs neural backbone inference (`EfficientNet`/`ResNet`); handles uncalibrated scores; gracefully emits `MODEL_UNAVAILABLE` when weights missing | **Implemented & Wired** *(Awaiting external video datasets)* |
| **C** | `experiment_c.py` | Cross-dataset generalisation & domain transfer gap | `VisualDeepfakeDetector` (`biotrace.visual_detection.detector`) | Multiple dataset manifests (e.g. FF++ source, Celeb-DF target) | Runs multi-domain inference, isolates domain transfer degradation ($\Delta\text{AUROC}$, $\Delta\text{EER}$); logs dataset distribution shifts | **Implemented & Wired** *(Awaiting external video datasets)* |
| **D** | `experiment_d.py` | Compression & perturbation robustness stress tests | `VisualDeepfakeDetector` + `RPPGSignalExtractor` | Video files + synthetic perturbation generator (`perturbations.py`) | Dynamically applies synthetic blur, H.264 CRF compression, downsampling, and additive noise; measures performance degradation curves across degradation levels | **Implemented & Wired** *(Awaiting external video datasets)* |
| **E** | `experiment_e.py` | Multimodal interaction & quality-gated stratification | `VisualDeepfakeDetector` + `RPPGSignalExtractor` + `MultimodalEvidenceFusionEngine` | Video files + multimodal ground truth + split partitions | Performs dual extraction; feeds outputs to `MultimodalEvidenceFusionEngine`; analyzes visual detector accuracy stratified by physiological signal quality (observable vs. degraded) | **Implemented & Corrected** *(Awaiting external video datasets)* |

### Detailed Execution Verification
1. **Direct Component Invocation:** Experiments do not use synthetic stubs for extraction when run on video paths; they directly instantiate `RPPGSignalExtractor`, `VisualDeepfakeDetector` with `VisualDetectionConfig`, and `MultimodalEvidenceFusionEngine`.
2. **Per-Video Granularity:** Every experiment emits a typed `SamplePrediction` or `PerVideoResult` containing video ID, file hash, ground truth, predicted value, confidence, and quality metrics before computing global cohort statistics.
3. **Graceful Degradation:** When model weights are absent (`weights_path=None`), the system cleanly yields `MODEL_UNAVAILABLE` without crashing or inventing synthetic confidence scores.

---

## 3. Metric Validity Findings

The statistical metric calculation routines in `ai-engine/src/biotrace/experiments/metrics.py` and experiment modules were audited for edge cases, numerical stability, and statistical soundess.

### Defect 1: Masking Single-Class Undefined AUROC/AP (FIXED)
- **Previous State:** In `experiment_b.py`, if a split contained only a single class (all authentic or all manipulated), the evaluation fallback caught exceptions and defaulted to `auroc = 0.5` and `ap = 0.5`.
- **Scientific Flaw:** A receiver operating characteristic curve and average precision are mathematically undefined when only one class is present. Imputing $0.5$ masks fatal dataset partition errors or misconfigured splits.
- **Resolution:** Replaced fallback with `None` (strictly undefined). Downstream summaries and tests explicitly recognize that AUROC cannot be computed on single-class partitions.

### Defect 2: Pearson Correlation P-Values on Insufficient Sample Sizes (FIXED)
- **Previous State:** `experiment_a.py` computed Pearson $r$ p-values via Student's $t$-distribution approximation whenever $N \ge 3$.
- **Scientific Flaw:** The asymptotic normal approximation of the $t$-test for Pearson correlation is statistically invalid and severely biased for extremely small sample sizes ($N < 5$).
- **Resolution:** Updated gating to enforce $N \ge 5$ before reporting asymptotic p-values. For $N < 5$, the correlation coefficient $r$ is reported descriptively while `p_value` is set to `None`.

### Defect 3: Ambiguity Between Synthetic and Empirical Provenance (FIXED)
- **Previous State:** Metrics models stored numerical values without an immutable record of whether the underlying data originated from a mock test fixture or a real benchmark.
- **Resolution:** Introduced `DataSourceType` enum (`SYNTHETIC_TEST_FIXTURE`, `MOCK_PIPELINE_FIXTURE`, `EMPIRICAL_BENCHMARK`). All metric objects, runner logs, and markdown summaries now explicitly embed this attribution.

---

## 4. Leakage and Splitting Findings

Data partitioning routines in `ai-engine/src/biotrace/experiments/splitting.py` were audited to prevent data leakage between training, validation, and test splits.

```
+-------------------------------------------------------------------------+
|                  TRANSITIVE IDENTITY CLUSTERING (UNION-FIND)             |
+-------------------------------------------------------------------------+
|                                                                         |
|  Pristine Sample A                    Pristine Sample B                 |
|  [subject_id: "subj_01"]             [subject_id: "subj_01"]           |
|  [source_video_id: "v_orig_1"]       [source_video_id: "v_orig_2"]     |
|          |                                   |                          |
|          | (Union by subject_id)             |                          |
|          +-----------------+-----------------+                          |
|                            |                                            |
|                   Cluster: {"subj_01"}                                  |
|                            |                                            |
|                            | (Union by source_video_id)                 |
|          +-----------------+-----------------+                          |
|          |                                   |                          |
|  Manipulated Sample C                 Manipulated Sample D              |
|  [subject_id: None]                  [subject_id: None]                 |
|  [source_video_id: "v_orig_1"]       [source_video_id: "v_orig_2"]     |
|                                                                         |
|  FINAL TRANSITIVE CLUSTER: {Sample A, Sample B, Sample C, Sample D}     |
|  => Entire cluster allocated atomically to ONE split partition.        |
+-------------------------------------------------------------------------+
```

### Defect 4: Transitive Identity Leakage on Mixed Identifiers (FIXED)
- **Previous State:** Splitting grouped records sequentially. If an authentic video had `subject_id = "subj_01"` and `source_video_id = "v_orig"`, but its manipulated counterpart had `subject_id = None` and `source_video_id = "v_orig"`, naive grouping could split the manipulated video into Test while the authentic video was assigned to Train.
- **Resolution:** Replaced sequential grouping with a formal **Disjoint Set Union (Union-Find)** data structure. Every video index is transitively clustered across `subject_id`, `source_video_id`, and `pair_id`.
- **Partition Allocation:** Implemented deficit-based greedy cluster allocation that maintains target split ratios while guaranteeing that no cluster is ever bifurcated across partitions.
- **Guarantee Level Attribution:** The split result dynamically checks identifier completeness:
  - If all records have `subject_id`: `SplitGuaranteeLevel.SUBJECT_LEVEL`
  - If records lack `subject_id` but possess `source_video_id`: `SplitGuaranteeLevel.SOURCE_VIDEO_LEVEL`
  - Otherwise: `SplitGuaranteeLevel.VIDEO_LEVEL`

---

## 5. Experiment E Scientific Validity Audit

### Defect 5: Unprincipled Composite Authenticity Scores (FIXED)
- **Previous State:** `experiment_e.py` previously attempted to compute an empirical "fused_score" using an ad-hoc linear combination:
  $$\text{fused\_score} = 0.50 + 0.50 \cdot (\text{vis\_score} - 0.50)$$
- **Scientific Flaw:** This formula directly violated Stage 5's fundamental scientific guardrail:
  > *"Do not fabricate an arbitrary weighted composite authenticity score. Visual probabilities and physiological SNR cannot be linearly combined without a joint generative or statistical model."*
- **Resolution:**
  1. Completely excised the ad-hoc composite formula.
  2. Aligned Experiment E with the core Stage 5 mission: **Quality-Gated Stratification and Evidence Agreement**.
  3. Experiment E now evaluates:
     - Visual detector performance (AUROC, AP, EER) on samples with **observable physiological signals** ($\text{SNR} \ge \text{threshold}$).
     - Visual detector performance on samples with **degraded physiological signals** ($\text{SNR} < \text{threshold}$).
     - Distribution of categorical fusion outcomes via `MultimodalEvidenceFusionEngine` (`CONSISTENT`, `INCONGRUENT`, `INSUFFICIENT_EVIDENCE`).
  4. This provides scientifically defensible insight: determining whether visual detectors perform worse under conditions of physiological degradation without inventing uncalibrated probabilities.

---

## 6. Dataset Governance Findings

The candidate dataset registry (`ai-engine/src/biotrace/experiments/dataset_registry.py`) and dataset documentation were audited against academic licensing and research distribution ethics.

| Dataset | Verified Access Mechanism | Academic Use Allowed | Commercial Redistribution | Local Hosting Requirements | Current BioTrace Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UBFC-rPPG** | Request via University of Bourgogne academic portal / institutional contact | Yes (Research only) | Strictly Prohibited | Local secured drive; raw `.avi` and ground truth `.txt` | `AWAITING_DATASET_ACCESS` |
| **FaceForensics++** | Request form / academic email verification to Technical University of Munich (TUM) | Yes (Research only) | Strictly Prohibited | Local script download; H.264 raw, c23, c40 variants | `AWAITING_DATASET_ACCESS` |
| **Celeb-DF v2** | Request via SUNY Albany Computer Vision and Machine Learning Lab (CVML) | Yes (Research only) | Strictly Prohibited | Local download; 590 pristine, 5639 deepfake YouTube videos | `AWAITING_DATASET_ACCESS` |
| **DFDC** | Kaggle account agreement / DeepFake Detection Challenge rules | Yes (Research only) | Strictly Prohibited | Local download; 470 GB raw split archives | `AWAITING_DATASET_ACCESS` |

### Governance Rules Enforced:
1. **Zero External Network Fetching in Sandbox:** BIOTrace does not auto-download external video corpora during testing or runtime.
2. **Explicit User Approval Required:** Dataset acquisition requires the end-user to execute official academic agreements with the dataset creators.
3. **No Repository Storage:** Video corpora must reside outside source control in local, un-tracked storage paths.

---

## 7. Reproducibility Findings

Reproducibility mechanisms were audited in `manifest.py`, `splitting.py`, and `runner.py`:

1. **Cryptographic Content Hashing:**
   - Every video record is indexed by its SHA-256 hash.
   - `DatasetManifest` computes a manifest-level SHA-256 fingerprint from sorted sample hashes.
   - `DatasetSplit` computes a split-level SHA-256 hash verifying train/val/test allocation immutability.
   - Model weights paths are verified against expected SHA-256 checkpoints.
2. **Deterministic Pseudorandom Seeds:**
   - Partitioning and synthetic perturbations utilize an explicit `seed` parameter (default `42`).
   - Re-running partition generation produces bit-for-bit identical splits.
3. **System Runtime Provenance:**
   - `get_system_provenance()` records Python version (`3.11.9`), PyTorch version, OpenCV version, platform OS (`Windows`), and UTC timestamp.
4. **Path Sanitization:**
   - The runner replaces absolute local user paths (e.g. `C:\Users\Sravya mummana\...`) with sanitized logical identifiers in exported research artifacts, preventing private directory leakage.

---

## 8. Fixes and Regression Tests Added

To permanently prevent regressions, targeted test cases were integrated into `ai-engine/tests/test_experiments.py`:

```python
# ai-engine/tests/test_experiments.py

def test_mixed_availability_subject_ids_no_leakage():
    """Verify Union-Find clustering prevents leakage when pristine samples
    have subject IDs but manipulated samples only carry source video IDs."""
    ...

def test_single_class_undefined_auroc_and_eer():
    """Verify single-class splits strictly yield None for AUROC/AP rather
    than defaulting to an arbitrary 0.5."""
    ...

def test_multimodal_experiment_e_quality_stratification_no_arbitrary_weights():
    """Verify Experiment E operates via quality-gated stratification and
    evidence states, without linear composite authenticity formulas."""
    ...

def test_system_provenance_and_data_source_tagging():
    """Verify runtime provenance captures Python/OS info and correctly
    tags data sources as SYNTHETIC_TEST_FIXTURE or MOCK_PIPELINE_FIXTURE."""
    ...
```

All 4 audit regression tests pass deterministically.

---

## 9. Verification Test Results

Independent automated test runs were executed across the entire BIOTrace repository:

### Python Test Suite (`pytest`)
```text
Command: cmd.exe /c "set PYTHONPATH=ai-engine\src& python -m pytest ai-engine/tests -W ignore"
Results: 239 passed, 0 failed, 0 errors, 1 warning (deprecation from scipy.signal)
Duration: 15.79 seconds
Status: PASSED (100% pass rate)
```

### Java Client Suite (`mvn test`)
```text
Command: cmd.exe /c "set JAVA_HOME=C:\Program Files\Java\jdk-26.0.2&& cd java-client&& ..\.tools\apache-maven-3.9.6\bin\mvn.cmd test -o"
Results: 25 tests run, 21 passed, 4 skipped (live server integration tests requiring backend daemon), 0 failures, 0 errors
Status: BUILD SUCCESS
```

---

## 10. Explicit List of Work Requiring Authorized Real Datasets

The following tasks cannot and should not be completed using synthetic mocks; they require the user to obtain authorized access to real datasets:

1. **Obtain Academic Data Approvals:**
   - Sign institutional user agreements for UBFC-rPPG, FaceForensics++, Celeb-DF v2, and DFDC.
2. **Ingest Real Video Archives:**
   - Place extracted video archives in a dedicated local directory (e.g. `data/benchmarks/`).
   - Run `biotrace.experiments.manifest.build_manifest_from_directory()` to generate production manifest JSON files with real SHA-256 hashes and metadata.
3. **Train Production Visual Detection Backbone:**
   - Train `VisualDeepfakeDetector` backbone weights on FaceForensics++ c23 training split using `biotrace.visual_detection.training`.
   - Export and verify model weight checkpoint hash.
4. **Execute Empirical Experimentation:**
   - Execute Experiment A on UBFC-rPPG to obtain the real physiological heart-rate MAE and Pearson correlation $r$.
   - Execute Experiment B on FaceForensics++ to establish real in-distribution baseline AUROC and EER.
   - Execute Experiment C on Celeb-DF v2 to quantify the real cross-dataset generalization drop ($\Delta\text{AUROC}$).
   - Execute Experiment D to map real compression degradation curves across H.264 CRF 0, 23, 32, 40.
   - Execute Experiment E to measure visual detector error rates stratified by real human rPPG signal-to-noise ratio.
5. **Publish Empirical Research Report:**
   - Once empirical data is generated, update `research/notes/STAGE_8_EXPERIMENTAL_VALIDATION.md` with true empirical measurements, strictly marked with `DataSourceType.EMPIRICAL_BENCHMARK`.
