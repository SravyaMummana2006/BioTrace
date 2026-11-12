# BIOTrace Protocol C — Statistical Power and Sample Size Analysis

**Author:** Independent Research-Methodology Reviewer  
**Date:** March 2026  
**Document Type:** Formal Statistical Audit & Power Assessment  
**Reference Dataset:** HiDF Zenodo Record `16140829` (4,361 Verified Real/Fake Pairs)  

---

## 1. Mathematical Formulation

### 1.1 AUROC Variance Estimation (Hanley-McNeil Formulation)
For a binary classification task evaluating continuous prediction scores on $n_1$ positive (Fake) and $n_2$ negative (Real) samples with an expected true AUROC of $\theta$, the asymptotic variance $\text{Var}(\theta)$ is given by the Hanley & McNeil (1982) formulation:

$$\text{Var}(\theta) = \frac{\theta(1 - \theta) + (n_1 - 1)(Q_1 - \theta^2) + (n_2 - 1)(Q_2 - \theta^2)}{n_1 n_2}$$

Where $Q_1$ is the probability that two randomly selected positive instances will both be ranked higher than a randomly selected negative instance, and $Q_2$ is the probability that one randomly selected positive instance will be ranked higher than two randomly selected negative instances:

$$Q_1 = \frac{\theta}{2 - \theta}, \qquad Q_2 = \frac{2\theta^2}{1 + \theta}$$

The Standard Error is $\text{SE}(\theta) = \sqrt{\text{Var}(\theta)}$, and the two-sided $95\%$ Wald Confidence Interval is:

$$\text{CI}_{95\%} = \left[ \max\left(0, \theta - 1.96 \cdot \text{SE}\right), \; \min\left(1, \theta + 1.96 \cdot \text{SE}\right) \right]$$

---

## 2. Statistical Power & Confidence Interval Comparisons

We evaluate three test set regimes under the assumption of baseline classifier performance $\theta = 0.85$:

| Evaluation Regime | Test Positive ($n_1$) | Test Negative ($n_2$) | Total Test Videos | Standard Error ($\text{SE}$) | $95\%$ Confidence Interval | Margin of Error (Half-Width) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Protocol C Full Test Set (15%)** | **654** | **654** | **1,308** | **0.0107** | **[0.829, 0.871]** | **$\pm 0.0210$** |
| **350-Component Subset Test Set (15%)** | **52** | **52** | **104** | **0.0383** | **[0.775, 0.925]** | **$\pm 0.0750$** |
| **Stage 8.2 Tentative Subset (27 pairs)** | **27** | **27** | **54** | **0.0534** | **[0.745, 0.955]** | **$\pm 0.1047$** |

### Critical Statistical Finding
- The **Protocol C Full Test Set** ($N=654$ pairs = $1,308$ videos) provides **exceptional statistical precision** ($\text{SE} \approx 0.011$, CI width $= 0.042$), easily meeting the rigorous $\pm 0.025$ margin-of-error standard for peer-reviewed machine learning benchmarks.
- The **350-Component Subset** ($N=52$ test pairs = $104$ test videos) yields an unacceptably wide confidence interval ($\pm 0.075$, width $= 0.150$). A measured AUROC of $0.85$ cannot be distinguished statistically from an AUROC of $0.78$ or $0.92$.
- The **27-Pair Subset** is severely underpowered ($\pm 0.105$, width $> 0.20$), rendering performance benchmarks statistically indistinguishable from marginal baselines.

---

## 3. Demographic Subgroup Statistical Power Analysis

In the **Protocol C Full Test Set** ($N=654$ pairs), demographic allocation mirrors the global dataset distribution:

| Demographic Stratum | Test Positive ($n_1$) | Test Negative ($n_2$) | Proportional Share | Standard Error ($\text{SE}$) | $95\%$ Confidence Interval | Width of CI | Power Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Race: White** | 513 | 513 | 78.44% | 0.0121 | [0.826, 0.874] | 0.0475 | **High Power** |
| **Race: Black** | 68 | 68 | 10.40% | 0.0334 | [0.784, 0.916] | 0.1310 | **Moderate Power** |
| **Race: Elderly** | 59 | 59 | 9.02% | 0.0359 | [0.780, 0.920] | 0.1408 | **Moderate Power** |
| **Race: Asian** | 32 | 32 | 4.89% | 0.0490 | [0.754, 0.946] | 0.1921 | **Low Power** |
| **Race: Latino** | 28 | 28 | 4.28% | 0.0525 | [0.747, 0.953] | 0.2057 | **Low Power** |
| **Race: Indian** | 13 | 13 | 1.99% | 0.0781 | [0.697, 1.000] | 0.3030 | **Severely Underpowered** |
| **Age: Child** | 10 | 10 | 1.53% | 0.0897 | [0.674, 1.000] | 0.3258 | **Severely Underpowered** |

### Methodological Limitation
While aggregate detection metrics are statistically robust, **demographic subgroup fairness claims on Indian ($N=13$) and Child ($N=10$) cohorts cannot be substantiated** within standard confidence boundaries ($95\%\text{ CI width} > 0.30$). Subgroup disaggregation must be framed as exploratory rather than confirmatory.

---

## 4. Hypothesis Testing & Comparative Methodology

### 4.1 DeLong's Test for Correlated ROC Curves
When comparing two distinct classifiers (e.g., $M_{\text{rPPG}}$ vs. $M_{\text{Visual}}$) on the **same test partition**, the resulting ROC curves are correlated. DeLong, DeLong, & Clarke-Pearson (1988) formulated a non-parametric test using generalized U-statistics:

$$z = \frac{\hat{\theta}_1 - \hat{\theta}_2}{\sqrt{\text{Var}(\hat{\theta}_1) + \text{Var}(\hat{\theta}_2) - 2\text{Cov}(\hat{\theta}_1, \hat{\theta}_2)}}$$

- **Appropriateness:** Valid when prediction scores are continuous and well-calibrated.
- **Critical Caveat:** If classifiers output discrete, quantized, or heavily thresholded scores, DeLong variance estimates underestimate true variance. In such scenarios, a **stratified paired bootstrap** ($B = 2,000$ replicates resampled at the component level) is required.
- **Clustered DeLong:** In Protocol C, $97.1\%$ of test components are singletons ($k=1$), but $2.9\%$ contain $k=2$ base videos sharing a donor. Clustered DeLong (Geng et al., 2020) must be employed to adjust for intra-cluster correlation.

### 4.2 Paired Differential Testing (Experiment D)
Experiment D computes the paired differential response $\Delta = f(\text{fake}) - f(\text{real})$ on identical base video subjects:
- **Design:** Matched pairs ($N=654$ paired observations).
- **Statistical Test:** **Wilcoxon signed-rank test** (distribution-free) supplemented by **Paired Student's t-test** if skewness $\le 1.0$.
- **Effect Size Metric:** **Cohen's $d_z = \bar{\Delta} / s_{\Delta}$** and Matched-Pairs Rank Biserial Correlation ($r_{\text{rb}}$).
