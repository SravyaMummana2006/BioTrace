# BIOTrace Phase 3 — Experimental Dataset Design and Leakage Audit Report

**Date:** March 2026  
**Auditor:** Senior Machine Learning Research Scientist (Computer Vision & Physiological Signal Forensics)  
**Project:** BIOTrace Deepfake Detection Framework  
**Scope:** HiDF Official Dataset Architecture, Donor Graph Topology, Demographic Distribution, and Experimental Partitioning  
**Status:** COMPLETE — RIGOROUS AUDIT & RECOMMENDATIONS  

---

## Executive Summary

Following the completion of the Phase 2.2 zero-download remote archive inspection, which established the existence of **4,361 authentic base videos** and **4,361 fake derivative videos** with strict 1-to-1 pairing, this Phase 3 audit evaluates target donor reuse, demographic balance, and graph-theoretic split designs.

### Key Audit Findings
1. **Target Donor Reuse (Task 1):** Of the **4,236 unique target donors**, **4,114 (97.12%)** are used exactly once. Only **122 donors (2.88%)** are reused across multiple base videos, accounting for **247 fake videos (5.66%)**. The maximum donor reuse is $4$.
2. **Demographic Imbalance & Cross-Swaps (Task 2):** 
   - Base video subjects are heavily skewed toward **White (78.10%)**, **Male (61.16%)**, and **Adults (89.96%)**.
   - Target donor demographics closely mirror the base distribution but reveal significant cross-demographic synthesis: **25.38% of fakes are cross-race swaps**, **12.22% are cross-age swaps**, and **3.00% are cross-gender swaps**. Only **63.38%** match all three categories.
3. **Split Feasibility & Bipartite Graph (Task 3):** Because every base video has degree 1 in the fake dataset, the bipartite graph between bases and donors decomposes into **4,236 disjoint star components** centered at each donor. Strict source-video AND target-donor disjoint partitioning (**Protocol C**) is **100% practical**, requiring **0 discarded samples** ($4,361 / 4,361$ pairs usable).
4. **Identity Annotations (Task 4):** Official HiDF metadata records **clip-level identifiers only**, with **zero subject/actor identity annotations**. While source-video and donor independence are mathematically provable, human-identity independence cannot be claimed without external face verification.
5. **Clear Recommendation:** **CONDITIONAL GO** for dataset acquisition and curation using **Protocol C (Bi-Bipartite Component-Disjoint Splitting)** with demographic stratification.

---

## Task 1: Audit of Target Donor Reuse

Analyzing the verified pairs in `../../reports/data/remote_fake_pair_manifest.csv` reveals the exact frequency distribution of target image donor reuse:

### Donor Reuse Distribution

| Donor Frequency ($k$ times used) | Number of Donors | % of Unique Donors | Total Fake Videos | % of Total Fake Videos | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** (Single use) | **4,114** | **97.12%** | **4,114** | **94.34%** | Disjoint |
| **2** (Reused twice) | **120** | **2.83%** | **240** | **5.50%** | Star ($k=2$) |
| **3** (Reused three times) | **1** | **0.02%** | **3** | **0.07%** | Star ($k=3$) |
| **4** (Reused four times) | **1** | **0.02%** | **4** | **0.09%** | Star ($k=4$) |
| **Total** | **4,236** | **100.00%** | **4,361** | **100.00%** | Complete |

### Highly Reused Donors Detail

1. **Donor `c19941` ($k=4$ fakes):**
   - Paired Base Videos: `c04154`, `c12745`, `c12795`, `c12802`
   - Generated Fakes: `c04154_c19941.mp4`, `c12745_c19941.mp4`, `c12795_c19941.mp4`, `c12802_c19941.mp4`
2. **Donor `f21247` ($k=3$ fakes):**
   - Paired Base Videos: `c08256`, `c12797`, `c12803`
   - Generated Fakes: `c08256_f21247.mp4`, `c12797_f21247.mp4`, `c12803_f21247.mp4`

All 122 reused donors are fully cataloged in [`../../reports/data/donor_frequency_distribution.csv`](../../reports/data/donor_frequency_distribution.csv).

---

## Task 2: Demographic Balance & Cross-Concordance Analysis

Demographics are recorded separately for base video subjects (`(Vid, Base)`) and image donors (`Img`). Base-subject and donor demographics must never be conflated:

### Demographic Distributions

| Demographic Dimension | Attribute Value | Base Video Subjects ($N=4,361$) | Unique Donors ($N=4,236$) | Fake Videos (by Donor) ($N=4,361$) |
| :--- | :--- | :---: | :---: | :---: |
| **Gender** | Male | 2,667 (61.16%) | 2,525 (59.61%) | 2,617 (60.01%) |
| | Female | 1,694 (38.84%) | 1,711 (40.39%) | 1,744 (39.99%) |
| **Race / Ethnicity** | White | 3,406 (78.10%) | 3,384 (79.89%) | 3,484 (79.89%) |
| | Black | 456 (10.46%) | 234 (5.52%) | 241 (5.53%) |
| | Latino | 194 (4.45%) | 354 (8.36%) | 365 (8.37%) |
| | Asian | 215 (4.93%) | 202 (4.77%) | 206 (4.72%) |
| | Indian | 90 (2.06%) | 62 (1.46%) | 65 (1.49%) |
| **Age Group** | Adult | 3,923 (89.96%) | 3,797 (89.64%) | 3,908 (89.61%) |
| | Elderly | 371 (8.51%) | 371 (8.76%) | 385 (8.83%) |
| | Child | 67 (1.54%) | 68 (1.61%) | 68 (1.56%) |

### Cross-Demographic Swapping Concordance

In physiological rPPG detection, skin melanin density and facial tissue geometry dictate light absorption and pulsatile signal strength (SNR). A cross-demographic swap creates an optical chimera:

- **Gender Concordance:** **4,230 pairs (97.00%)** share identical gender; **131 pairs (3.00%)** are cross-gender swaps.
- **Race Concordance:** **3,254 pairs (74.62%)** share identical race; **1,107 pairs (25.38%)** are cross-race swaps.
  - *Critical bias observation:* Black base subjects comprise $10.46\%$ of authentic videos, but Black donors comprise only $5.52\%$ of target donors. Consequently, nearly half of Black base videos were swapped with non-Black donor faces.
- **Age Concordance:** **3,828 pairs (87.78%)** share identical age category; **533 pairs (12.22%)** are cross-age swaps.
- **Full Demographic Match (Gender + Race + Age):** **2,764 pairs (63.38%)**.
- **Cross-Demographic Transformation:** **1,597 pairs (36.62%)** involve at least one demographic transformation.

---

## Task 3: Design of Leakage-Resistant Splits

We evaluated three candidate partitioning protocols on the bipartite graph $G = (V_{\text{base}}, V_{\text{target}}, E)$:

```
  Protocol A: Naive Random Split           Protocol B: Source-Video Grouped
     [Real Video]    [Fake Video]             [Real Video] + [Fake Video]
           │               │                                │
    Train Partition   Test Partition                   Single Partition
    (CRITICAL CONTENT LEAKAGE)                    (Reused Donors May Cross Splits)
  ─────────────────────────────────────────────────────────────────────────────
                    Protocol C: Component-Disjoint Split (RECOMMENDED)
                    ┌──────────────────────────────────────────────┐
                    │  Connected Component (Star Graph around T_j) │
                    │   {Donor T_j, Base B_1, Base B_2, ...}       │
                    │   -> Real_1, Fake_1, Real_2, Fake_2, ...     │
                    └──────────────────────┬───────────────────────┘
                                           ▼
                                   Single Partition
                  (Zero Base Leakage AND Zero Donor Leakage)
```

### Graph-Theoretic Connected Component Analysis

- **Nodes:** $4,361$ base nodes ($V_{\text{base}}$), $4,236$ donor nodes ($V_{\text{target}}$).
- **Edges:** $4,361$ fake video pairings ($E$).
- **Degree Properties:**
  - $\forall b \in V_{\text{base}}, \deg(b) = 1$ (each base video has exactly 1 fake swap).
  - $\forall t \in V_{\text{target}}, \deg(t) \in \{1, 2, 3, 4\}$.
- **Connected Components ($C$):** Exactly **4,236 star components**.
  - No two donor nodes can share an edge or path through a base node because every base node has degree 1.
  - Therefore, the connected components are **strictly disjoint star graphs** $S_j = \{t_j\} \cup \mathcal{N}(t_j)$, where $|\mathcal{N}(t_j)| \le 4$.
- **Usability & Discard Rate:**
  - Because the maximum component size is only 4 base pairs (8 videos), greedy or linear-programming bin packing can partition all 4,236 components across train (70%), validation (15%), and test (15%) partitions **without discarding a single sample**.
  - **Usable sample count:** **$4,361$ pairs ($8,722$ videos, 100.00%)**.

### Protocol Comparison Summary

| Protocol | Grouping Level | Source-Video Leakage Prevented? | Target-Donor Leakage Prevented? | Usable Samples | Practicality | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Protocol A** (Random) | Video file level | **NO** (Fatal) | **NO** | 4,361 pairs | High | **REJECTED** |
| **Protocol B** (Source-Grouped) | Real/Fake base pair | **YES** | **NO** (122 donors cross) | 4,361 pairs | High | **SECONDARY** |
| **Protocol C** (Component-Disjoint) | Bipartite Star Component | **YES** | **YES** | **4,361 pairs** | **High (0 discarded)** | **STRONGLY RECOMMENDED** |

---

## Task 4: Identity Limitations & Scientific Rigor

A key failure mode in computer vision research is conflating different levels of experimental independence. We formally define and delineate these boundaries:

1. **Video-Level Independence:** Treating individual video files as i.i.d. samples.
   - *Status:* Invalid in deepfake research. Real and fake videos derived from the same recording share backgrounds, head motion, camera sensor noise, and facial geometry.
2. **Base-Source Independence (VERIFIED):** Grouping all derivatives of a base video recording clip into a single partition.
   - *Status:* Strictly enforced by Protocols B and C. Guarantees that the underlying cardiovascular perfusion signal from a specific recording clip never crosses partition boundaries.
3. **Donor Independence (VERIFIED):** Ensuring image donors used to synthesize fake faces appear in only one partition.
   - *Status:* Strictly enforced by Protocol C. Prevents deep neural networks from memorizing visual artifacts or idiosyncratic facial landmarks belonging to specific donor photos.
4. **Human-Identity Independence (UNVERIFIED / LIMITATION):**
   - *Official HiDF Documentation Audit:* The official repository and `metadata.csv` provide clip labels (`c00003`, `c00004`), but **zero actor identity annotations** (`person_id`, `actor_name`).
   - *Scientific Boundary:* We cannot prove whether two different base video clips were recorded from the same human subject on different occasions or camera setups.
   - *Audit Rule:* In research reports, BIOTrace must explicitly state: **"Evaluations enforce base-source video independence and donor independence, but cannot claim verified human-subject identity independence due to lack of actor annotations in official HiDF metadata."**

---

## Task 5: Recommended Experimental Protocols

To balance scientific validity, statistical power, and computational feasibility, the following experimental configuration is established:

### 1. Primary Evaluation Split: Protocol C (Component-Disjoint Stratified Split)
- **Train Partition (70%):** $\sim 3,052$ pairs ($3,052$ real, $3,052$ fake = $6,104$ videos).
- **Validation Partition (15%):** $\sim 654$ pairs ($654$ real, $654$ fake = $1,308$ videos).
- **Test Partition (15%):** $\sim 655$ pairs ($655$ real, $655$ fake = $1,310$ videos).
- **Curated Subset Option (Strategy C from Stage 8.2):** If storage/runtime constraints require a smaller download, a random sample of $350$ star components ($350$ real, $350$ fake = $700$ videos) can be extracted under Protocol C with zero leakage.

### 2. Demographic Stratification Strategy
Components must be allocated to partitions using greedy bin-packing stratified across the 5 racial categories (`White`, `Black`, `Asian`, `Latino`, `Indian`) to prevent distribution drift between training and testing.

### 3. Leakage Controls by BIOTrace Experiment

| Experiment | Focus | Required Protocol | Specific Leakage Control |
| :--- | :--- | :--- | :--- |
| **Experiment A** | Pipeline Verification | Synthetic Fixtures | Mock extractors; zero real-video dependency. |
| **Experiment B** | Physiological vs Visual Detection | **Protocol C** | Test set ($N=655$ pairs) strictly disjoint on base & donor. Real/fake evaluated jointly. |
| **Experiment C** | Feature Ablation | **Protocol C** | Evaluated on the identical test fold as Experiment B for paired DeLong significance. |
| **Experiment D** | Pairwise Delta Analysis | **Protocol C** | Computes $\Delta = f(\text{fake}) - f(\text{real})$ on identical base video subjects in test fold. |
| **Experiment E** | Cross-Dataset Generalization | **HiDF $\to$ UBFC-Phys** | Zero leakage: completely disjoint datasets, hardware, sensors, and human subjects. |

---

## Task 6: Deliverables Inventory

The following Phase 3 deliverables have been generated and validated:

1. [`phase3_dataset_design.md`](phase3_dataset_design.md) (This audit report)
2. [`../../reports/data/donor_frequency_distribution.csv`](../../reports/data/donor_frequency_distribution.csv) ($4$ rows detailing donor reuse rates and percentages)
3. [`../../reports/data/demographic_distribution.csv`](../../reports/data/demographic_distribution.csv) ($10$ rows detailing Gender, Race, and Age proportions)
4. [`../../reports/data/split_feasibility_report.csv`](../../reports/data/split_feasibility_report.csv) ($3$ candidate protocols evaluated against leakage criteria)

---

## Final Recommendation: CONDITIONAL GO

The dataset architecture and experimental design are scientifically sound:
- **Zero code changes were introduced.**
- **Zero video archives were downloaded.**
- **All 4,361 pairs are 100% usable under Protocol C.**

**Condition for Acquisition:** When downloading or curating HiDF videos, splits must be generated using **Protocol C (Star Component-Disjoint Splitting)** to guarantee simultaneous base-source and target-donor isolation. Human identity claims must be transparently qualified as clip-level base-source independence.
