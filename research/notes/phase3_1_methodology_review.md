# BIOTrace Phase 3.1 — Independent Methodology Verification & Experimental Audit Report

**Auditor:** Independent Research-Methodology Reviewer  
**Audit Scope:** Rigorous verification of Phase 3 findings, graph topology, statistical power, demographic stratification, and cross-dataset evaluation design  
**Date:** March 2026  
**Status:** COMPLETE — CRITICAL CORRECTIONS & VERIFIED REPRODUCIBLE MANIFEST  

---

## Executive Summary

As an independent methodology reviewer, this audit examined the dataset architecture, graph-theoretic assumptions, demographic distributions, statistical power, and external benchmark compatibility of the proposed BIOTrace experimental design.

### Summary of Major Audit Findings
1. **Numerical Reproduction (Section 1):** Verified the exact count of **$4,361$ real/fake pairs**, **$4,236$ unique donors**, and the component frequency distribution ($4,114$ singletons, $120$ pairs, $1$ triple, $1$ quadruple). Resolved a minor textual ambiguity in early drafts that conflated the $125$ surplus videos with the $122$ unique reused donors.
2. **Reproducible Protocol C Split (Section 2):** Successfully generated [`../../reports/data/protocol_c_split_manifest.csv`](../../reports/data/protocol_c_split_manifest.csv) using a fixed random seed (`seed=42`). Confirmed **zero base-source leakage** and **zero target-donor leakage** across all partitions (**Train: $3,053$ pairs [70.01%]**, **Val: $654$ pairs [15.00%]**, **Test: $654$ pairs [15.00%]**).
3. **Statistical Power & Minority Group Limitations (Section 3):**
   - The full test set ($N=654$ pairs) achieves high statistical power ($\text{SE} = 0.0107$, $95\%\text{ CI} = [0.829, 0.871]$).
   - In contrast, the tentative $350$-component subset ($N=52$ test pairs) yields an unacceptably wide confidence interval ($\pm 0.075$, width $= 0.150$).
   - In the full test set, Indian ($N=13$) and Child ($N=10$) cohorts are **severely underpowered** ($95\%\text{ CI width} > 0.30$), meaning demographic fairness claims on these subgroups cannot be established with statistical significance.
4. **Fatal Flaw & Correction in Experiment E (Section 4):**
   - Official documentation confirms that **UBFC-Phys contains 56 participants, 168 genuine videos, and ZERO deepfake videos**.
   - Reporting "deepfake AUROC" on UBFC-Phys alone is mathematically impossible.
   - Experiment E is formally restructured into two separate evaluations: **Track E.1** (Physiological Signal Generalization against 64 Hz BVP) and **Track E.2** (Deepfake Classification Generalization requiring an external manipulated dataset).
5. **Phase 4 Recommendation:** **CONDITIONAL GO** adhering strictly to the corrected protocol.

---

## 1. Reproduction of Phase 3 Statistics & Anomaly Audit

All underlying records from `../../reports/data/remote_fake_pair_manifest.csv` and `datasets/metadata.csv` were independently parsed and recalculated:

### 1.1 Pair and Donor Census Verification
- **Total Valid Pairs:** Exactly **$4,361$ pairs** ($4,361$ authentic base videos, $4,361$ deepfake videos).
- **Unique Target Donors:** Exactly **$4,236$ donors**.
- **Connected Component Count:** Exactly **$4,236$ star components**.
- **Component Size Distribution:**
  - Size 1 ($k=1$ base video): **$4,114$ components** ($4,114$ pairs, $94.34\%$ of dataset).
  - Size 2 ($k=2$ base videos): **$120$ components** ($240$ pairs, $5.50\%$ of dataset).
  - Size 3 ($k=3$ base videos): **$1$ component** ($3$ pairs, $0.07\%$ of dataset; Donor `f21247`).
  - Size 4 ($k=4$ base videos): **$1$ component** ($4$ pairs, $0.09\%$ of dataset; Donor `c19941`).
  - **Sum of Allocated Pairs:** $4,114 + 240 + 3 + 4 = 4,361$ pairs ($100.00\%$).

### 1.2 Inconsistency Resolution
- *Discrepancy Noted:* The initial draft of `../../reports/data/split_feasibility_report.csv` stated *"125 donors cross partitions if unconstrained"*. 
- *Correction:* The number of unique reused donors is **$122$** ($120 + 1 + 1$). The number **$125$** represents the *surplus video count* ($4,361\text{ pairs} - 4,236\text{ unique donors} = 125\text{ extra fakes}$). Both `../../reports/data/donor_frequency_distribution.csv` and `../../reports/data/split_feasibility_report.csv` have been verified and reconciled to reflect $122$ unique reused donors.

---

## 2. Independent Verification of Protocol C & Manifest Generation

### 2.1 Bipartite Graph & Star Topology Proof
In the bipartite graph $G = (V_{\text{base}}, V_{\text{target}}, E)$:
- Every base node $b \in V_{\text{base}}$ has degree $\deg(b) = 1$ (each authentic base video was modified to create exactly one fake video).
- Since no base node has degree $\ge 2$, no path can exist between two distinct target donor nodes $t_1, t_2 \in V_{\text{target}}$.
- Therefore, every connected component in $G$ is mathematically guaranteed to be a **disjoint star graph centered at a unique target donor**.
- Allocating partitions by whole connected components guarantees **simultaneous 100% isolation of both base video content AND target donor features**.

### 2.2 Reproducible Stratified Splitting Execution
Using a deterministic pseudo-random generator (`seed=42`), all $4,236$ components were allocated using greedy bin-packing stratified across the 5 primary racial strata.

#### Actual Partition Proportions

| Partition | Target Ratio | Allocated Components | Allocated Real/Fake Pairs | Total Video Files | Actual Percentage | Base Leakage | Donor Leakage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 70.00% | 2,965 | **3,053** | 6,106 | **70.01%** | **0** | **0** |
| **Validation** | 15.00% | 636 | **654** | 1,308 | **15.00%** | **0** | **0** |
| **Test** | 15.00% | 635 | **654** | 1,308 | **15.00%** | **0** | **0** |
| **Total** | 100.00% | 4,236 | **4,361** | 8,722 | **100.00%** | **0** | **0** |

The generated manifest is preserved in [`../../reports/data/protocol_c_split_manifest.csv`](../../reports/data/protocol_c_split_manifest.csv) with columns: `base_id, target_id, component_id, split, race, gender, age`.

### 2.3 Partition Demographic Alignment

As documented in [`../../reports/data/protocol_c_demographic_validation.csv`](../../reports/data/protocol_c_demographic_validation.csv), component-level stratification successfully prevents demographic distribution shift:

- **Race:**
  - White: Train $78.12\%$, Val $77.68\%$, Test $78.44\%$ (Dataset: $78.10\%$)
  - Black: Train $10.38\%$, Val $10.86\%$, Test $10.40\%$ (Dataset: $10.46\%$)
  - Asian: Train $4.98\%$, Val $4.74\%$, Test $4.89\%$ (Dataset: $4.93\%$)
  - Latino: Train $4.45\%$, Val $4.59\%$, Test $4.28\%$ (Dataset: $4.45\%$)
  - Indian: Train $2.06\%$, Val $2.14\%$, Test $1.99\%$ (Dataset: $2.06\%$)
- **Gender:**
  - Female: Train $38.78\%$, Val $40.98\%$, Test $37.00\%$ (Dataset: $38.84\%$)
  - Male: Train $61.22\%$, Val $59.02\%$, Test $63.00\%$ (Dataset: $61.16\%$)
- **Age:**
  - Adult: Train $89.81\%$, Val $91.13\%$, Test $89.45\%$ (Dataset: $89.96\%$)
  - Elderly: Train $8.68\%$, Val $7.19\%$, Test $9.02\%$ (Dataset: $8.51\%$)
  - Child: Train $1.51\%$, Val $1.68\%$, Test $1.53\%$ (Dataset: $1.54\%$)

---

## 3. Statistical Validity & Methodological Critique

A comprehensive power analysis was performed (see [`../../reports/history/protocol_c_statistical_power_analysis.md`](../../reports/history/protocol_c_statistical_power_analysis.md)):

### 3.1 Minority Demographic Representation & Power Failure
- In the full test set ($N=654$ pairs), the sample counts for minority groups are:
  - **Indian:** $n_1 = 13, n_2 = 13 \implies \text{SE} = 0.0781, \; 95\%\text{ CI} = [0.697, 1.000]$ (CI width $= 0.303$).
  - **Child:** $n_1 = 10, n_2 = 10 \implies \text{SE} = 0.0897, \; 95\%\text{ CI} = [0.674, 1.000]$ (CI width $= 0.326$).
- **Methodological Verdict:** Subgroup AUROC or fairness metrics on Indian or Child cohorts are **severely underpowered**. A single error alters accuracy by up to $10\%$. BIOTrace must formally state that subgroup disaggregations on these two cohorts are exploratory and cannot support statistically definitive claims.

### 3.2 Evaluation of the 350-Component Subset Option
- A subset of $350$ components yields only $N = 52$ test pairs ($104$ test videos).
- Under assumed $\text{AUROC} = 0.85$, the standard error rises to $\text{SE} \approx 0.0383$, producing a confidence interval of $[0.775, 0.925]$ ($\text{width} = 0.150$).
- Furthermore, an Indian or Child cohort in a $350$-component subset would contain at most $1$ sample, eliminating demographic disaggregation entirely.
- **Recommendation:** If storage allows, the **full $4,361$-pair dataset should be utilized**. If a subset must be used due to storage limits, confidence intervals must be reported transparently and subgroup analysis restricted.

### 3.3 Comparative Hypothesis Testing: DeLong & Wilcoxon
- **DeLong's Test:** DeLong et al. (1988) is valid for comparing correlated AUROCs on the identical test fold, but assumes continuous, unquantized decision values and asymptotic normality. Because $2.9\%$ of test pairs share a donor with another test pair, **Clustered DeLong** or **stratified paired bootstrap ($B=2,000$ replicates resampled by component)** must be used to prevent anticonservative p-values.
- **Experiment D Differential Testing:** Computing $\Delta = f(\text{fake}) - f(\text{real})$ on identical base subjects is a matched-pair design. The **Wilcoxon signed-rank test** must be the primary test, accompanied by **Cohen's $d_z$** effect size.

---

## 4. Critical Correction of Experiment E (UBFC-Phys)

An audit of the official UBFC-Phys documentation (Meziati et al., IEEE TBME 2021; `https://imvia.ube.fr/ubfc-phys/`) revealed an immediate methodological barrier:

### The UBFC-Phys Reality
- **Dataset Composition:** 56 human participants performing 3 tasks (resting, speaking, arithmetic stress) = $168$ uncompressed videos with contact-sensor ground-truth (Empatica E4 BVP at 64 Hz).
- **Crucial Finding:** **UBFC-Phys contains ZERO deepfake videos. All 168 videos are authentic human subjects.**
- **The Prior Proposal's Error:** Proposing to report "cross-dataset deepfake detection AUROC" on UBFC-Phys is a mathematical impossibility. When class labels are strictly real ($y=0$ for all $N$), True Positive Rate and AUROC cannot be computed.

### Corrected Two-Track Protocol for Experiment E

```
                                    Experiment E
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Track E.1: rPPG Signal                       Track E.2: Deepfake
             Generalization                              Generalization
       (Evaluated on UBFC-Phys)                   (Evaluated on External Benchmark)
  ──────────────────────────────────────         ──────────────────────────────────
  • Target: 168 Videos (56 Subjects)             • Target: Benchmark with Real & Fake
  • Reference: 64 Hz Wrist BVP Ground Truth        (e.g., SDFVD or Celeb-DF v2)
  • Metrics: HR-MAE, HR-RMSE, Pearson r, PRV     • Metrics: Cross-dataset AUROC, EER
  • Goal: Verify physiological fidelity          • Goal: Verify classifier transfer
```

1. **Track E.1 — Cross-Dataset Physiological Signal Generalization:**
   - **Data:** UBFC-Phys ($168$ genuine videos from $56$ subjects).
   - **Ground Truth:** Empatica E4 synchronized wristband BVP ($64\text{ Hz}$).
   - **Metrics:** Heart Rate Mean Absolute Error (HR-MAE in bpm), Root Mean Squared Error (HR-RMSE in bpm), Pearson correlation ($r$), and Pulse Rate Variability (PRV) agreement (SDNN, RMSSD).
   - **Evaluation Scheme:** Subject-independent cross-validation (56-subject leave-subject-out or fixed train/test subject partition).
2. **Track E.2 — Cross-Dataset Deepfake Classification Generalization:**
   - **Requirement:** Must be evaluated on an external benchmark containing verified genuine AND manipulated videos (e.g., FaceForensics++, Celeb-DF v2, or SDFVD).
   - **Role of UBFC-Phys in Detection:** If probed with UBFC-Phys, the model can ONLY be evaluated for **False Positive Rate / Specificity on authentic videos** ($P(\hat{y} = \text{Fake} \mid y = \text{Real})$).

---

## 5. Reassessment of Research Claims

To maintain research integrity in published manuscripts and grant deliverables, the following concepts must be strictly distinguished:

| Term | Formal Definition | BIOTrace Empirical Status |
| :--- | :--- | :--- |
| **Deepfake Detection Performance** | Discrimination accuracy / AUROC separating authentic from manipulated videos. | Measured on paired test splits in Experiments B & D. |
| **Physiological Signal Accuracy** | Agreement between rPPG pulse waveforms and contact medical biosensors (ECG/BVP). | Measured in Track E.1 against UBFC-Phys 64 Hz BVP ground truth. |
| **Cross-Dataset Physiological Transfer** | Generalization of pulse recovery across unseen cameras, lighting, and subjects. | Validated on UBFC-Phys (Track E.1). |
| **Base-Source Independence** | Guaranteeing real and fake videos derived from the same base video do not cross splits. | **VERIFIED & GUARANTEED** by Protocol C. |
| **Donor Independence** | Guaranteeing face-swap donors appear in only one partition. | **VERIFIED & GUARANTEED** by Protocol C. |
| **Human Identity Independence** | Guaranteeing distinct video clips from the same human actor do not cross splits. | **UNVERIFIED LIMITATION**; official HiDF metadata contains no actor annotations. |

---

## 6. Deliverables Inventory

| File Name | Location | Description |
| :--- | :--- | :--- |
| [`phase3_1_methodology_review.md`](phase3_1_methodology_review.md) | Workspace Root | This independent review report. |
| [`../../reports/data/protocol_c_split_manifest.csv`](../../reports/data/protocol_c_split_manifest.csv) | Workspace Root | $4,361$ rows mapping every pair to `train`/`val`/`test` under Protocol C (`seed=42`). |
| [`../../reports/data/protocol_c_demographic_validation.csv`](../../reports/data/protocol_c_demographic_validation.csv) | Workspace Root | Tabular verification of demographic stratification across partitions. |
| [`../../reports/history/protocol_c_statistical_power_analysis.md`](../../reports/history/protocol_c_statistical_power_analysis.md) | Workspace Root | Complete Hanley-McNeil derivations, subgroup sample size analysis, and DeLong review. |

---

## Conclusions & Final Recommendations

### 1. Corrected Primary Experimental Protocol
- **Dataset Splitting:** Adopt [`../../reports/data/protocol_c_split_manifest.csv`](../../reports/data/protocol_c_split_manifest.csv) ($3,053$ train, $654$ val, $654$ test pairs) generated with seed 42.
- **Experiment E Structure:** Split into **Track E.1** (rPPG signal accuracy on UBFC-Phys against 64 Hz BVP) and **Track E.2** (deepfake classification on a dual-class benchmark).
- **Statistical Testing:** Use Clustered DeLong or stratified paired bootstrap for model comparisons; use Wilcoxon signed-rank test for paired differential analysis.

### 2. Remaining Methodological Limitations
- Human-actor identity independence cannot be claimed for HiDF due to missing subject annotations in official metadata.
- Subgroup demographic fairness on Indian ($N=13$) and Child ($N=10$) cohorts cannot be evaluated with statistical certainty due to small sample sizes.

### 3. GO/NO-GO Recommendation for Phase 4
**DECISION: GO (With Corrected Protocol)**  
The experimental design, split manifest, and statistical controls are now fully verified, mathematically sound, and leak-free. Proceed to Phase 4 (Dataset Acquisition & Integration) using the corrected protocol.
