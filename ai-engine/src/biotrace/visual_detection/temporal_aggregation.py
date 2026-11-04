"""Temporal aggregation of frame-level predictions into video-level visual forensic summaries.

Scientific Principles & Forensic Guardrails (Stage 6.1 Audit):
    1. Rejection of the Independence Fallacy & Invalid Inferential Claims:
       Adjacent video frames share identical camera optics, illumination, and generative synthesis
       artifacts, exhibiting substantial serial autocorrelation.
       CRITICAL SCIENTIFIC PRINCIPLE: An autocorrelation-adjusted formula (such as Bartlett's
       effective sample size) provides a descriptive metric of temporal redundancy. It does NOT
       establish exchangeability, independent identically distributed sampling, or Gaussian normality.
       Therefore, this engine strictly REFRAINS from claiming a statistically valid 95% inferential
       confidence interval based solely on lag-1 autocorrelation and sample size.
    2. Descriptive Frame Statistics & Variance:
       The engine reports descriptive sample statistics across valid video frames:
       - Sample logit variance (inter-frame instability)
       - Empirical lag-1 serial autocorrelation
       - Bartlett effective sample size (descriptive reduction factor)
       - Frame raw score dispersion range [min, max]
       These are explicitly documented as descriptive sample variations, NOT confidence intervals.
    3. Outlier Resistance:
       Occasional face detector failures, profile turns, or motion blur can produce momentary
       spikes in frame logits. Trimmed mean and median aggregation provide robust central tendency
       estimators without being distorted by localized anomalies.
"""

from typing import List, Optional, Tuple
import numpy as np

from biotrace.visual_detection.domain import (
    FramePrediction,
    TemporalAggregationMethod,
)


class TemporalAggregator:
    """Combines frame-level visual classifier predictions into robust video-level summaries."""

    def __init__(
        self,
        method: TemporalAggregationMethod = TemporalAggregationMethod.TRIMMED_MEAN,
        trim_ratio: float = 0.1,
    ):
        self.method = method
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        predictions: List[FramePrediction],
    ) -> Tuple[Optional[float], Optional[float], float, Optional[Tuple[float, float]]]:
        """Convenience method aggregating frame predictions into (raw_score, logit, variance, descriptive_range)."""
        score, logit, var, desc_range, _, _, _ = self.aggregate_predictions(
            predictions, method=self.method, trim_ratio=self.trim_ratio
        )
        return score, logit, var, desc_range

    @staticmethod
    def calculate_lag1_autocorrelation(values: np.ndarray) -> float:
        """Calculates empirical lag-1 serial autocorrelation of a 1D sequence.

        Args:
            values: 1D NumPy array of sequential observations (e.g. frame logits).

        Returns:
            rho_1 bounded in [-1.0, 1.0], or 0.0 if variance is negligible or N < 3.
        """
        n = len(values)
        if n < 3:
            return 0.0

        var = float(np.var(values))
        if var < 1e-12:
            return 0.0

        mean_val = float(np.mean(values))
        centered = values - mean_val
        cov1 = float(np.sum(centered[:-1] * centered[1:]) / (n - 1))
        rho1 = cov1 / var
        return float(np.clip(rho1, -1.0, 1.0))

    @staticmethod
    def calculate_effective_sample_size(n: int, rho1: float) -> float:
        """Calculates descriptive Bartlett effective sample size: N_eff = N * (1 - rho1) / (1 + rho1).

        Scientific Note:
            This metric provides a descriptive estimate of information redundancy under an AR(1)
            assumption. It does NOT validate frequentist hypothesis testing or inferential confidence
            intervals on deep learning logits.
        """
        if n <= 1:
            return float(n)

        if abs(rho1) >= 0.999:
            return 1.0

        factor = (1.0 - rho1) / (1.0 + rho1)
        n_eff = float(n) * factor
        return float(np.clip(n_eff, 1.0, float(n)))

    @classmethod
    def aggregate_predictions(
        cls,
        predictions: List[FramePrediction],
        method: TemporalAggregationMethod = TemporalAggregationMethod.TRIMMED_MEAN,
        trim_ratio: float = 0.1,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        float,
        Optional[Tuple[float, float]],
        List[str],
        Optional[float],
        Optional[float],
    ]:
        """Aggregates frame predictions into robust summary statistics and descriptive dispersion.

        Returns:
            (aggregated_raw_score, aggregated_logit, temporal_variance, descriptive_score_range,
             diagnostics, lag1_autocorrelation, effective_sample_size)
        """
        valid_preds = [p for p in predictions if p.is_valid and p.face_detected]
        if not valid_preds:
            return (
                None,
                None,
                0.0,
                None,
                ["Zero valid face frames available for temporal aggregation."],
                None,
                None,
            )

        logits = np.array([p.raw_logit for p in valid_preds], dtype=np.float64)
        raw_scores = np.array([p.raw_score for p in valid_preds], dtype=np.float64)
        n = len(logits)
        diagnostics: List[str] = []

        # 1. Calculate inter-frame logit variance
        variance = float(np.var(logits)) if n > 1 else 0.0

        # 2. Calculate lag-1 serial autocorrelation and descriptive effective sample size
        rho_1 = cls.calculate_lag1_autocorrelation(logits)
        n_eff = cls.calculate_effective_sample_size(n, rho_1)

        # 3. Compute aggregated logit according to chosen strategy
        if method == TemporalAggregationMethod.TRIMMED_MEAN:
            if n >= 5:
                trim_count = max(1, int(round(trim_ratio * n)))
                sorted_logits = np.sort(logits)
                trimmed = sorted_logits[trim_count : n - trim_count]
                agg_logit = float(np.mean(trimmed))
                diagnostics.append(
                    f"Trimmed mean aggregation: discarded {trim_count} highest and {trim_count} "
                    f"lowest frame logits ({int(trim_ratio*100)}% per tail across N={n} frames)."
                )
            else:
                agg_logit = float(np.mean(logits))
                diagnostics.append(f"Trimmed mean fallback to arithmetic mean (sample size N={n} < 5).")

        elif method == TemporalAggregationMethod.MEDIAN:
            agg_logit = float(np.median(logits))
            diagnostics.append("Median logit aggregation: selected 50th percentile across valid frames.")

        elif method == TemporalAggregationMethod.MEAN_LOGIT:
            agg_logit = float(np.mean(logits))
            diagnostics.append("Arithmetic mean logit aggregation.")

        elif method == TemporalAggregationMethod.TOP_K_PERCENTILE:
            agg_logit = float(np.percentile(logits, 85))
            diagnostics.append("85th percentile logit aggregation: evaluated upper tail for localized artifact spikes.")

        elif method == TemporalAggregationMethod.MAX_POOLING:
            agg_logit = float(np.max(logits))
            diagnostics.append("Max-pooling logit aggregation: captured peak frame anomaly.")

        else:
            agg_logit = float(np.mean(logits))
            diagnostics.append(f"Default mean logit aggregation applied for method '{method.value}'.")

        # 4. Transform aggregated logit via sigmoid to [0.0, 1.0] uncalibrated raw score
        # CRITICAL FORENSIC GUARDRAIL: Uncalibrated heuristic score, NOT an authenticity probability
        agg_score = float(1.0 / (1.0 + np.exp(-agg_logit)))

        # 5. Descriptive frame score dispersion (min-max range)
        # Explicitly represents intra-video sample variation, NOT an inferential confidence interval
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))
        descriptive_range = (round(min_score, 4), round(max_score, 4))

        diagnostics.append(
            f"Descriptive frame dispersion note: Frame score range is [{min_score:.4f}, {max_score:.4f}] "
            f"across N={n} frames (temporal logit variance = {variance:.4f}, serial lag-1 autocorrelation = {rho_1:.3f}, "
            f"Bartlett descriptive effective N = {n_eff:.1f}). This dispersion describes sample frame variation only; "
            f"it is explicitly NOT a frequentist or Bayesian inferential confidence interval."
        )

        return agg_score, agg_logit, variance, descriptive_range, diagnostics, rho_1, n_eff
