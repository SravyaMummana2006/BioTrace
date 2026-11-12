# BIOTrace Phase 3.2 — Final Reproducibility and Statistical Audit Report

**Auditor:** Independent Senior ML Research Auditor  
**Audit Scope:** Manifest verification failure root-cause analysis, bipartite component reconstruction, demographic balance mechanism, statistical power limits, UBFC-Phys feasibility, and provenance calibration  
**Audit Execution Date:** November 3, 2026 (`2026-11-03`)  
**Status:** COMPLETE — ALL ASSERTIONS VERIFIED & VALIDATED  
**Primary Deliverables:**
- [`phase3_2_final_audit.md`](phase3_2_final_audit.md) (This report)
- [`../../reports/data/phase3_2_manifest_validation.csv`](../../reports/data/phase3_2_manifest_validation.csv) (Machine-checkable validation matrix)
- [`../../reports/history/phase3_2_test_results.txt`](../../reports/history/phase3_2_test_results.txt) (Deterministic test execution log)

---

## 1. Investigation of Failed Manifest Verification

During Phase 3.1, an automated check command executed against `../../reports/data/protocol_c_split_manifest.csv` returned exit code 1. We audited the exact terminal command, environment, and output to establish the root cause:

### 1.1 Root-Cause Analysis
- **Exact Command Executed:**
  ```powershell
  python -c "import csv; manifest = list(csv.DictReader(open('../../reports/data/protocol_c_split_manifest.csv'))); print('Total manifest rows:', len(manifest)); print('Columns:', manifest[0].keys()); print('Split counts:', Counter(r['split'] for r in manifest))"
  ```
- **Exit Code:** `1`
- **Stderr Traceback:**
  ```text
  NameError: name 'Counter' is not defined
  ```
- **Audit Finding:** The failure was caused by an unimported Python class (`Counter` from `collections`) in an inline one-liner shell string. The file `../../reports/data/protocol_c_split_manifest.csv` was completely written and intact on disk. When `from collections import Counter` was added, the command succeeded immediately with code 0.

### 1.2 Programmatic Manifest File Validation
To ensure that no latent data defects existed within `../../reports/data/protocol_c_split_manifest.csv`, we executed an exhaustive schema and record verification:
- **Encoding & Delimiter:** UTF-8 encoded text with standard comma `,` delimiters and CRLF line terminators.
- **Header Columns:** Exactly `base_id,target_id,component_id,split,race,gender,age` (7 columns).
- **Row Count:** Exactly $4,361$ data rows ($4,362$ lines including header).
- **Null / Missing Values:** Exactly $0$ across all $30,527$ data cells.
- **Duplicate Base IDs:** Exactly $0$ duplicate `base_id` entries.
- **Pair Identity Verification:** Every `(base_id, target_id)` pair maps $1:1$ to `../../reports/data/remote_fake_pair_manifest.csv`.
- **Authentic Metadata Cross-Check:** Every `base_id` exists in `datasets/metadata.csv` under `(Vid, Base)`, and demographic fields (`race`, `gender`, `age`) exhibit $100.00\%$ concordance.
- **Verdict:** The manifest is structurally, semantically, and forensically defect-free. No replacement manifest is required.

---

## 2. Independent Split Integrity Verification

We independently reconstructed the bipartite graph $G = (V_{\text{base}}, V_{\text{target}}, E)$ directly from `../../reports/data/remote_fake_pair_manifest.csv`:

### 2.1 Graph Reconstruction Metrics
- **Base Nodes ($V_{\text{base}}$):** $4,361$ unique nodes, each with $\deg(b) = 1$.
- **Target Donor Nodes ($V_{\text{target}}$):** $4,236$ unique nodes with degrees $\deg(t) \in \{1, 2, 3, 4\}$.
- **Connected Components ($C$):** Exactly $4,236$ star components.
  - $4,114$ components of size $1$ base video ($4,114$ pairs)
  - $120$ components of size $2$ base videos ($240$ pairs)
  - $1$ component of size $3$ base videos ($3$ pairs)
  - $1$ component of size $4$ base videos ($4$ pairs)

### 2.2 Machine-Checkable Partition Isolation Assertions

| Assertion | Checked Condition | Observed Value | Verdict |
| :--- | :--- | :---: | :---: |
| **Complete Pair Inclusion** | $\sum |S_i| == 4,361$ | $3,053 + 654 + 654 = 4,361$ | **PASS** |
| **No Base ID Leakage (Train vs Val)** | $V_{\text{base}}^{\text{train}} \cap V_{\text{base}}^{\text{val}} == \emptyset$ | $0$ overlap | **PASS** |
| **No Base ID Leakage (Train vs Test)** | $V_{\text{base}}^{\text{train}} \cap V_{\text{base}}^{\text{test}} == \emptyset$ | $0$ overlap | **PASS** |
| **No Base ID Leakage (Val vs Test)** | $V_{\text{base}}^{\text{val}} \cap V_{\text{base}}^{\text{test}} == \emptyset$ | $0$ overlap | **PASS** |
| **No Donor Leakage (Train vs Val)** | $V_{\text{target}}^{\text{train}} \cap V_{\text{target}}^{\text{val}} == \emptyset$ | $0$ overlap | **PASS** |
| **No Donor Leakage (Train vs Test)** | $V_{\text{target}}^{\text{train}} \cap V_{\text{target}}^{\text{test}} == \emptyset$ | $0$ overlap | **PASS** |
| **No Donor Leakage (Val vs Test)** | $V_{\text{target}}^{\text{val}} \cap V_{\text{target}}^{\text{test}} == \emptyset$ | $0$ overlap | **PASS** |
| **Component Split Consistency** | Components allocated to $\le 1$ partition | $0$ component splits | **PASS** |

---

## 3. Reassessment of Demographic Stratification

### 3.1 Unit of Stratification & Percentage Denominators
- **Accounting Clarification:** All reported split counts ($3,053$ train, $654$ val, $654$ test) represent **REAL/FAKE VIDEO PAIRS**.
- Because every pair consists of exactly one authentic base video and exactly one deepfake video synthesized from that base video, demographic proportions are identical whether computed across **pairs ($N=4,361$)**, **authentic base videos ($N=4,361$)**, or **total video files ($N=8,722$)**.

### 3.2 Explanation of the Female Test Set Variance
In [`../../reports/data/protocol_c_demographic_validation.csv`](../../reports/data/protocol_c_demographic_validation.csv), the proportion of female subjects across partitions is:
- **Global Population:** $1,694 / 4,361 = \mathbf{38.84\%}$
- **Train Partition:** $1,184 / 3,053 = \mathbf{38.78\%}$ ($-0.06\%$ delta)
- **Validation Partition:** $268 / 654 = \mathbf{40.98\%}$ ($+2.14\%$ delta)
- **Test Partition:** $242 / 654 = \mathbf{37.00\%}$ ($-1.84\%$ delta, $-12$ pairs)

#### Algorithmic Mechanism:
1. **Multi-Criterion Conflict:** Component-level greedy bin packing was constrained primarily by the **5-class racial dimension** to prevent ethnic collapse in small groups (Indian: $N=90$, Asian: $N=215$, Latino: $N=194$).
2. **Indivisible Star Components:** $122$ components contain $2$ to $4$ base videos. Within these multi-base components, $42$ components have heterogeneous demographics across their constituent base videos.
3. **Natural Variance:** Because components cannot be split without violating donor-disjointness, packing entire components to satisfy racial quotas produces a minor natural residual in the secondary gender dimension ($\pm 1.8\%$). Perfect balance across all demographic intersections simultaneously is mathematically unattainable without splitting star components.

---

## 4. Audit of Statistical Methods

### 4.1 Reproduction of Hanley–McNeil Standard Errors & Confidence Intervals
We reproduced all calculations assuming an expected true $\text{AUROC} = 0.85$:

| Cohort / Regime | Sample Size ($n_1 = n_2$) | Exact SE ($\theta = 0.85$) | $95\%$ Wald CI | Margin of Error | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Test Set (15%)** | **654 pairs** | **0.010730** | **[0.8290, 0.8710]** | **$\pm 0.0210$** | **Robust ($\ge 99\%$ power)** |
| **350-Component Subset** | **52 pairs** | **0.038285** | **[0.7750, 0.9250]** | **$\pm 0.0750$** | **Severely Degraded** |
| **Subgroup: White** | 513 pairs | 0.012117 | [0.8263, 0.8737] | $\pm 0.0237$ | High Precision |
| **Subgroup: Black** | 68 pairs | 0.033427 | [0.7845, 0.9155] | $\pm 0.0655$ | Moderate Precision |
| **Subgroup: Elderly** | 59 pairs | 0.035914 | [0.7796, 0.9204] | $\pm 0.0704$ | Moderate Precision |
| **Subgroup: Asian** | 32 pairs | 0.049004 | [0.7540, 0.9460] | $\pm 0.0960$ | Low Precision |
| **Subgroup: Latino** | 28 pairs | 0.052467 | [0.7472, 0.9528] | $\pm 0.1028$ | Low Precision |
| **Subgroup: Indian** | 13 pairs | 0.078070 | [0.6970, 1.0000] | $\pm 0.1530$ | **Severely Underpowered** |
| **Subgroup: Child** | 10 pairs | 0.089689 | [0.6742, 1.0000] | $\pm 0.1758$ | **Severely Underpowered** |

### 4.2 Limitations of Asymptotic Wald Intervals for Small Cohorts
- For Indian ($N=13$) and Child ($N=10$) cohorts, standard Wald asymptotic intervals produce unclipped upper bounds of $1.003$ and $1.026$, respectively.
- For small sample sizes ($N < 50$), asymptotic normality fails. BIOTrace must report **Logit-transformed intervals** or **percentile bootstrap intervals**:
  $$\text{logit}(\hat{\theta}) \pm 1.96 \cdot \frac{\text{SE}}{\hat{\theta}(1 - \hat{\theta})}, \qquad \theta = \frac{1}{1 + e^{-\text{logit}(\theta)}}$$

### 4.3 Clustered Bootstrap & DeLong for Clustered Donors
- In the test partition ($N=654$ pairs), $635$ components exist: $616$ singletons ($k=1$) and $19$ pairs ($k=2$, comprising $38$ video pairs).
- The $38$ pairs sharing a donor within the test set violate the i.i.d. assumption of classical DeLong et al. (1988).
- **Approved Comparative Procedure:**
  1. Primary: **Clustered DeLong test** (Geng et al., 2020) adjusting the Mann-Whitney covariance matrix by cluster size $m_i$.
  2. Secondary: **Stratified paired cluster-bootstrap** ($B=2,000$ replicates resampled at the component level).

### 4.4 Experiment D Paired Differential Analysis
- Experiment D computes $\Delta_i = f(\text{fake}_i) - f(\text{real}_i)$ on identical base video subjects.
- For the $19$ multi-pair components, shared donor artifacts could induce mild intra-cluster correlation.
- **Approved Analysis:** Primary evaluation via **Wilcoxon signed-rank test** supplemented by a **Linear Mixed-Effects Model** ($\Delta_{ij} = \mu + u_j + \epsilon_{ij}$, where $u_j \sim \mathcal{N}(0, \sigma_u^2)$ is the random effect of donor $j$).

---

## 5. Audit of UBFC-Phys Feasibility

An audit of the official documentation and specifications for UBFC-Phys (Meziati et al., IEEE TBME 2021) establishes its operational parameters:

### 5.1 Dataset Modalities & Synchronization
- **Video Modality (`vid.avi`):**
  - Uncompressed/lightly compressed RGB video recorded at nominal **$35\text{ fps}$** (resolution: $1024 \times 1024$).
  - Per-frame timestamps are recorded to monitor camera frame jitter.
- **Reference Contact BVP (`bvp.csv`):**
  - Acquired via medical-grade Empatica E4 wristband at a fixed sampling rate of **$64\text{ Hz}$**.
- **Temporal Alignment Feasibility:**
  - Video and BVP acquisition were initiated with a hardware trigger.
  - To align $35\text{ fps}$ rPPG waveforms with $64\text{ Hz}$ wrist BVP, bandlimited cubic spline interpolation must be used to resample both signals to a unified clock (e.g., $64\text{ Hz}$).

### 5.2 Feasible vs. High-Risk Metrics on UBFC-Phys

| Metric | Modality | Feasibility Status | Methodological Constraints & Safeguards |
| :--- | :--- | :---: | :--- |
| **Heart Rate MAE (bpm)** | rPPG vs. BVP | **APPROVED** | Fast Fourier Transform (FFT) peak detection over sliding 8s windows ($[45, 180]\text{ bpm}$). |
| **Heart Rate RMSE (bpm)** | rPPG vs. BVP | **APPROVED** | Measures large frequency estimation outliers; standard benchmark standard. |
| **Pearson Correlation ($r$)** | rPPG vs. BVP | **APPROVED** | Evaluates overall instantaneous pulse morphology agreement across windows. |
| **Pulse Rate Variability (SDNN)** | Inter-Beat Intervals | **CONDITIONALLY APPROVED** | Requires parabolic sub-frame peak interpolation to mitigate $35\text{ fps}$ ($28.6\text{ ms}$) quantization jitter. |
| **Pulse Rate Variability (RMSSD)** | Inter-Beat Intervals | **HIGH RISK / EXPERIMENTAL** | Short-term successive difference metrics are highly sensitive to wrist pulse transit time (PTT) jitter. |
| **Deepfake Detection AUROC** | Binary Classification | **PROHIBITED** | **UBFC-Phys contains 0 deepfakes.** AUROC cannot be calculated on single-class data. |

---

## 6. Report Provenance & Artifact Verification

To ensure reproducibility, all provenance metadata has been audited:

- **Actual Audit Execution Date:** **November 3, 2026** (`2026-11-03T23:55:00+05:30`). *(Corrected from inherited placeholder date `March 2026`).*
- **Python Runtime Environment:** Python `3.11.9` on `Windows 10 / 11 (build 10.0.26200)`.
- **Deterministic Random Seed:** `42` (used for component-level stratified greedy bin-packing).
- **Cryptographic Input Hashes (SHA-256):**
  - `../../reports/data/protocol_c_split_manifest.csv`: `c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7`
  - `datasets/metadata.csv`: `62d49f5abebdefbbe100184ae1408cdc10d824019800881fcf6745493d358db1`
  - `../../reports/data/remote_fake_pair_manifest.csv`: `45e85a3ef456a992d2628c5a07484df203338a925bf222d1395b497e922ff3f4`
  - `../../reports/data/protocol_c_demographic_validation.csv`: `183ce52e9fcf4dc9622d0577a7d425b0cb5aa33ea3e43ee3165b699925695029`

---

## Final Review Decisions & Authorizations

### 1. Is the split manifest trustworthy?
**YES.** [`../../reports/data/protocol_c_split_manifest.csv`](../../reports/data/protocol_c_split_manifest.csv) is mathematically verified. It contains $4,361$ rows, exactly matching all base and donor pairs with zero missing values, zero duplicates, zero base leakage, and zero donor leakage.

### 2. Is Protocol C ready for use?
**YES.** Protocol C guarantees simultaneous isolation of both base video content and target image donor features. It maintains representative demographic stratification across train ($3,053$ pairs), validation ($654$ pairs), and test ($654$ pairs) partitions without discarding any data.

### 3. Which statistical procedures are approved?
- **Approved for Benchmark Detection:** Full test set AUROC ($N=654$ pairs) with Hanley-McNeil SE ($\text{SE} \approx 0.0107$).
- **Approved for Model Comparisons:** Clustered DeLong test and stratified paired cluster-bootstrap ($B=2,000$).
- **Approved for Experiment D:** Paired Wilcoxon signed-rank test and Cohen's $d_z$, supplemented by linear mixed-effects modeling.
- **Approved for UBFC-Phys (Track E.1):** HR-MAE, HR-RMSE, Pearson $r$, and bandlimited cubic spline interpolation against 64 Hz BVP.
- **Prohibited:** Reporting deepfake AUROC on UBFC-Phys; claiming confirmatory demographic fairness on Indian ($N=13$) or Child ($N=10$) cohorts.

### 4. Is BIOTrace ready for Phase 4 acquisition?
**DECISION: GO FOR PHASE 4.**  
All theoretical, architectural, graph-theoretic, and statistical validation criteria are fully satisfied. The project may proceed to Phase 4 dataset acquisition and extractor integration following the approved protocols.
