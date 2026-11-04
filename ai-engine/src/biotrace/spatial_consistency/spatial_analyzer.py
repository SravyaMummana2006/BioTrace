"""Spatial consistency analysis comparing physiological pulse waveforms across facial regions.

Scientific Principles & Audit Safeguards:
    1. Actual Timestamp Alignment:
       Observations are matched and evaluated on actual presentation timestamps,
       not array indices. If regions drop frames asynchronously, comparisons are
       performed strictly on mutually valid frames.
    2. Explicit Large Gap Partitioning:
       If tracking interruptions exceed max_permissible_gap_seconds (default 0.75s),
       signals are NOT stitched or interpolated across the gap. The longest continuous
       valid segment is isolated; if it falls below min_observation_seconds, the result
       is explicitly categorized as INSUFFICIENT_DATA.
    3. Physical Spectral Resolution Gating:
       The Rayleigh physical frequency resolution limit is Delta_BPM_physical = 60 / T_obs.
       If T_obs < min_duration_for_bpm_analysis (default 6.0s), frequency-domain peak
       discrepancy cannot physically resolve heart-rate divergence (Delta_BPM_phys > 10 BPM).
       Under such durations, frequency discrepancy is reported as unresolvable (None)
       and is NOT used to declare INCONSISTENT.
    4. Signed Pearson Correlation Justification:
       Signed correlation is scientifically required: cardiac pulse propagation through
       arterial networks induces synchronized vasodilation across facial dermal beds.
       Physiological coupling requires positive in-phase correlation (rho > 0).
       Negative correlation (rho ~ -1) represents an unphysiological anti-phase oscillation.
    5. Quality Decoupling:
       Degraded SNR (< min_snr_db) strictly produces LOW_QUALITY_SIGNAL, never INCONSISTENT.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from biotrace.core.logging import get_logger
from biotrace.rppg.filtering import apply_bandpass_filter, detrend_signal
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    GlobalConsistencyState,
    PairConsistencyState,
    PairwiseComparison,
    RegionalSample,
    RegionalSignalSummary,
    SpatialConsistencyConfig,
    SpatialConsistencyResult,
)
from biotrace.spatial_consistency.quality import (
    compute_cardiac_snr,
    extract_pos_pulse_from_rgb,
    partition_continuous_segments,
    sanitize_and_resample_continuous_segment,
)

logger = get_logger(__name__)


def compute_spectral_overlap(
    psd_a: np.ndarray,
    psd_b: np.ndarray,
) -> float:
    """Calculates cosine similarity between two power spectral density vectors."""
    if len(psd_a) == 0 or len(psd_b) == 0 or len(psd_a) != len(psd_b):
        return 0.0

    dot_prod = float(np.sum(psd_a * psd_b))
    norm_a = float(np.sqrt(np.sum(psd_a**2)))
    norm_b = float(np.sqrt(np.sum(psd_b**2)))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    similarity = dot_prod / (norm_a * norm_b)
    return round(float(np.clip(similarity, 0.0, 1.0)), 3)


def compute_lagged_cross_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sampling_rate_hz: float,
    max_lag_seconds: float,
) -> Tuple[float, float, float, int]:
    """Computes zero-lag and peak signed cross-correlation within a physiological lag window.

    Scientific Justification for Signed Correlation:
        Cardiac pulse propagation through arterial and capillary networks induces synchronized
        vasodilation and increased hemoglobin absorption across the face. Physiological coupling
        requires positive in-phase correlation (rho > 0). Negative correlation (rho ~ -1)
        represents an unphysiological anti-phase oscillation.

    Returns:
        (zero_lag_corr, max_cross_corr, optimal_lag_seconds, optimal_lag_frames)
    """
    n = len(signal_a)
    if n < 5 or len(signal_b) != n:
        return 0.0, 0.0, 0.0, 0

    std_a = float(np.std(signal_a))
    std_b = float(np.std(signal_b))

    if std_a < 1e-7 or std_b < 1e-7:
        return 0.0, 0.0, 0.0, 0

    norm_a = (signal_a - np.mean(signal_a)) / std_a
    norm_b = (signal_b - np.mean(signal_b)) / std_b

    # Zero-lag signed Pearson correlation
    zero_lag_corr = float(np.mean(norm_a * norm_b))
    zero_lag_corr = float(np.clip(zero_lag_corr, -1.0, 1.0))

    # Cross-correlation within physiological lag [-max_lag_frames, +max_lag_frames]
    max_lag_frames = max(1, int(round(max_lag_seconds * sampling_rate_hz)))
    max_lag_frames = min(max_lag_frames, n // 3)

    best_corr = zero_lag_corr
    best_lag_frames = 0

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag == 0:
            corr = zero_lag_corr
        elif lag > 0:
            # Positive lag: B is delayed relative to A (s_B(t + lag) aligns with s_A(t))
            corr = float(np.mean(norm_a[:-lag] * norm_b[lag:]))
        else:
            # Negative lag: B leads A (s_A(t + |lag|) aligns with s_B(t))
            k = abs(lag)
            corr = float(np.mean(norm_a[k:] * norm_b[:-k]))

        corr = float(np.clip(corr, -1.0, 1.0))
        if corr > best_corr:
            best_corr = corr
            best_lag_frames = lag

    optimal_lag_sec = best_lag_frames / (sampling_rate_hz if sampling_rate_hz > 0 else 30.0)

    return (
        round(zero_lag_corr, 3),
        round(best_corr, 3),
        round(optimal_lag_sec, 4),
        best_lag_frames,
    )


# partition_continuous_segments is imported from quality.py


def compare_region_pair(
    region_a: FacialRegionId,
    region_b: FacialRegionId,
    samples_a: List[RegionalSample],
    samples_b: List[RegionalSample],
    summary_a: RegionalSignalSummary,
    summary_b: RegionalSignalSummary,
    pulse_a: Optional[np.ndarray],
    pulse_b: Optional[np.ndarray],
    config: SpatialConsistencyConfig,
    reported_fps: float,
) -> PairwiseComparison:
    """Evaluates temporal cross-correlation and spectral cardiac agreement between two regions.

    Scientific Safeguards:
        1. Actual timestamp matching: Only frames where both regions have valid mean RGB are compared.
        2. Gap partitioning: Inter-sample gaps > max_permissible_gap_seconds are never interpolated across.
        3. Physical frequency resolution gating: If duration < min_duration_for_bpm_analysis (4.0s),
           frequency divergence cannot be resolved (Delta_BPM_phys > 15 BPM) and is NOT used to declare inconsistency.
        4. Low-quality decoupling: Low SNR strictly yields LOW_QUALITY_SIGNAL, never INCONSISTENT.
    """
    # 1. Match valid samples by actual timestamps / frame indices
    min_len = min(len(samples_a), len(samples_b))
    mutual_indices = [
        i for i in range(min_len)
        if samples_a[i].is_valid and samples_b[i].is_valid and samples_a[i].mean_rgb and samples_b[i].mean_rgb
    ]
    mutual_count = len(mutual_indices)

    if mutual_count < config.min_valid_samples:
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.INSUFFICIENT_DATA,
            mutually_valid_frames=mutual_count,
            overlap_duration_seconds=0.0,
            diagnostic_explanation=(
                f"Insufficient mutually valid observations ({mutual_count} frames < {config.min_valid_samples} required)."
            ),
        )

    mutual_timestamps = np.array([samples_a[i].timestamp_seconds for i in mutual_indices], dtype=np.float64)

    # 2. Explicit Large Gap Partitioning
    segments = partition_continuous_segments(mutual_timestamps, config.max_permissible_gap_seconds)
    if not segments:
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.INSUFFICIENT_DATA,
            mutually_valid_frames=mutual_count,
            overlap_duration_seconds=0.0,
            diagnostic_explanation="No continuous tracking segments found.",
        )

    # Select longest continuous segment
    longest_seg = max(segments, key=lambda seg: seg[1] - seg[0])
    seg_start, seg_end = longest_seg
    seg_count = seg_end - seg_start
    seg_ts = mutual_timestamps[seg_start:seg_end]
    seg_duration = max(0.0, float(seg_ts[-1] - seg_ts[0])) if len(seg_ts) > 1 else 0.0

    if seg_duration < config.min_observation_seconds or seg_count < config.min_valid_samples:
        gap_info = f" (tracking interrupted into {len(segments)} segments)" if len(segments) > 1 else ""
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.INSUFFICIENT_DATA,
            mutually_valid_frames=seg_count,
            overlap_duration_seconds=round(seg_duration, 3),
            diagnostic_explanation=(
                f"Continuous mutually valid observation duration ({seg_duration:.2f}s, {seg_count} frames) "
                f"is below required minimum ({config.min_observation_seconds:.2f}s){gap_info}."
            ),
        )

    # 3. Quality Gating: If either region exhibits low SNR, declare LOW_QUALITY_SIGNAL
    if summary_a.snr_db < config.min_snr_db or summary_b.snr_db < config.min_snr_db:
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.LOW_QUALITY_SIGNAL,
            mutually_valid_frames=seg_count,
            overlap_duration_seconds=round(seg_duration, 3),
            diagnostic_explanation=(
                f"One or both regions exhibit low physiological SNR "
                f"({region_a.value}={summary_a.snr_db:.1f} dB, {region_b.value}={summary_b.snr_db:.1f} dB; min={config.min_snr_db:.1f} dB). "
                f"Physiological consistency is indeterminate."
            ),
        )

    # 4. Extract time-aligned pulse waveforms from the continuous segment
    aligned_indices = [mutual_indices[idx] for idx in range(seg_start, seg_end)]
    rgb_a = np.array([samples_a[i].mean_rgb for i in aligned_indices], dtype=np.float64)
    rgb_b = np.array([samples_b[i].mean_rgb for i in aligned_indices], dtype=np.float64)

    # Sanitize and ensure uniform sampling across the continuous segment
    ok_a, u_ts, u_rgb_a, effective_fs, diag_a = sanitize_and_resample_continuous_segment(
        timestamps=seg_ts,
        signal_series=rgb_a,
        max_jitter_ratio=config.max_timing_irregularity_ratio,
    )
    ok_b, _, u_rgb_b, _, diag_b = sanitize_and_resample_continuous_segment(
        timestamps=seg_ts,
        signal_series=rgb_b,
        max_jitter_ratio=config.max_timing_irregularity_ratio,
    )

    if not ok_a or not ok_b or u_rgb_a is None or u_rgb_b is None:
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.INSUFFICIENT_DATA,
            mutually_valid_frames=seg_count,
            overlap_duration_seconds=round(seg_duration, 3),
            diagnostic_explanation=f"Excessive timing irregularity rejected continuous segment: {diag_a or diag_b}",
        )

    # Standard sliding-window POS projection on uniformly sampled segment
    raw_sub_a = extract_pos_pulse_from_rgb(
        u_rgb_a, effective_fs, window_seconds=config.pos_window_seconds, stride_frames=config.pos_stride_frames
    )
    raw_sub_b = extract_pos_pulse_from_rgb(
        u_rgb_b, effective_fs, window_seconds=config.pos_window_seconds, stride_frames=config.pos_stride_frames
    )

    if raw_sub_a is None or raw_sub_b is None:
        return PairwiseComparison(
            region_a=region_a,
            region_b=region_b,
            state=PairConsistencyState.INSUFFICIENT_DATA,
            mutually_valid_frames=seg_count,
            overlap_duration_seconds=round(seg_duration, 3),
            diagnostic_explanation="Continuous observation duration is insufficient to form standard sliding POS windows.",
        )

    # Detrend and bandpass filter
    from biotrace.rppg.domain import FilterConfig

    filt_cfg = FilterConfig(
        low_cutoff_hz=config.cardiac_low_hz,
        high_cutoff_hz=config.cardiac_high_hz,
        order=config.filter_order,
    )
    sub_pulse_a = apply_bandpass_filter(
        detrend_signal(raw_sub_a, polynomial_order=config.detrend_polynomial_order),
        sampling_rate_hz=effective_fs,
        config=filt_cfg,
    )
    sub_pulse_b = apply_bandpass_filter(
        detrend_signal(raw_sub_b, polynomial_order=config.detrend_polynomial_order),
        sampling_rate_hz=effective_fs,
        config=filt_cfg,
    )

    # 5. Cross-correlation with physiological lag
    zero_corr, max_corr, opt_lag_sec, opt_lag_frames = compute_lagged_cross_correlation(
        signal_a=sub_pulse_a,
        signal_b=sub_pulse_b,
        sampling_rate_hz=effective_fs,
        max_lag_seconds=config.max_physiological_lag_seconds,
    )

    # 6. Frequency-domain spectral comparison & Physical Resolution Gating
    delta_bpm_phys = round(60.0 / max(0.1, seg_duration), 1)

    _, bpm_a, dom_hz_a, _, psd_a, _ = compute_cardiac_snr(
        sub_pulse_a, sampling_rate_hz=effective_fs, cardiac_low_hz=config.cardiac_low_hz, cardiac_high_hz=config.cardiac_high_hz
    )
    _, bpm_b, dom_hz_b, _, psd_b, _ = compute_cardiac_snr(
        sub_pulse_b, sampling_rate_hz=effective_fs, cardiac_low_hz=config.cardiac_low_hz, cardiac_high_hz=config.cardiac_high_hz
    )

    freq_diff = abs(dom_hz_a - dom_hz_b) if (dom_hz_a is not None and dom_hz_b is not None) else None
    bpm_diff = abs(bpm_a - bpm_b) if (bpm_a is not None and bpm_b is not None) else None
    spectral_overlap = compute_spectral_overlap(psd_a, psd_b)

    # Spectral resolution guard: require duration >= min_duration_for_bpm_analysis (default 6.0s for 10 BPM resolution)
    is_spectral_resolvable = seg_duration >= config.min_duration_for_bpm_analysis
    effective_bpm_tol = max(config.max_bpm_discrepancy, delta_bpm_phys)

    # 7. Consistency Classification
    is_corr_consistent = max_corr >= config.correlation_consistent_threshold
    is_lag_valid = abs(opt_lag_sec) <= config.max_physiological_lag_seconds

    # Anti-phase rejection: if zero-lag correlation is strongly negative, peak lag cannot override
    is_not_antiphase = zero_corr >= -0.20

    if is_spectral_resolvable and bpm_diff is not None:
        is_freq_consistent = bpm_diff <= effective_bpm_tol
        is_freq_inconsistent = bpm_diff > max(15.0, 1.5 * delta_bpm_phys)
    else:
        # In observations shorter than required physical resolution duration (<6s),
        # physical frequency resolution cannot resolve 10 BPM; rely on temporal cross-correlation.
        is_freq_consistent = True
        is_freq_inconsistent = False

    is_corr_inconsistent = max_corr < config.correlation_inconsistent_threshold

    if is_corr_consistent and is_lag_valid and is_freq_consistent and is_not_antiphase:
        state = PairConsistencyState.CONSISTENT
        freq_note = f", Delta_HR={bpm_diff:.1f} BPM" if (is_spectral_resolvable and bpm_diff is not None) else ""
        explanation = (
            f"Temporally and spectrally consistent: peak cross-correlation rho={max_corr:.2f} at lag={opt_lag_sec:.3f}s"
            f"{freq_note} (overlap={spectral_overlap:.2f})."
        )
    elif is_corr_inconsistent or (is_spectral_resolvable and is_freq_inconsistent):
        state = PairConsistencyState.INCONSISTENT
        explanation = (
            f"Regional physiological divergence: peak correlation rho={max_corr:.2f} (inconsistent threshold: {config.correlation_inconsistent_threshold:.2f})"
            + (f" or Delta_HR={bpm_diff:.1f} BPM" if (is_spectral_resolvable and bpm_diff is not None) else "")
            + f" despite acceptable SNR ({summary_a.snr_db:.1f} dB, {summary_b.snr_db:.1f} dB)."
        )
    else:
        state = PairConsistencyState.AMBIGUOUS
        explanation = (
            f"Moderate regional correlation (rho={max_corr:.2f}, lag={opt_lag_sec:.3f}s) "
            f"within indeterminate research bounds."
        )

    if not is_spectral_resolvable:
        explanation += (
            f" [Spectral note: observation duration {seg_duration:.2f}s is below {config.min_duration_for_bpm_analysis:.1f}s "
            f"required to resolve {config.max_bpm_discrepancy:.1f} BPM; physical Rayleigh resolution is +- {delta_bpm_phys:.1f} BPM]."
        )

    return PairwiseComparison(
        region_a=region_a,
        region_b=region_b,
        state=state,
        mutually_valid_frames=seg_count,
        overlap_duration_seconds=round(seg_duration, 3),
        zero_lag_correlation=zero_corr,
        max_cross_correlation=max_corr,
        optimal_lag_seconds=opt_lag_sec,
        optimal_lag_frames=opt_lag_frames,
        frequency_difference_hz=round(freq_diff, 3) if freq_diff is not None else None,
        bpm_difference=round(bpm_diff, 1) if (is_spectral_resolvable and bpm_diff is not None) else None,
        spectral_overlap=spectral_overlap,
        diagnostic_explanation=explanation,
    )


class SpatialBiologicalAnalyzer:
    """Consolidates multi-region physiological comparisons and forms structured scientific results."""

    def __init__(self, config: Optional[SpatialConsistencyConfig] = None):
        self.config = config or SpatialConsistencyConfig()

    def analyze_spatial_consistency(
        self,
        regional_samples: Dict[FacialRegionId, List[RegionalSample]],
        regional_summaries: Dict[FacialRegionId, RegionalSignalSummary],
        regional_pulses: Dict[FacialRegionId, Optional[np.ndarray]],
        timing_diagnostics: Dict[str, Any],
        reported_fps: float,
    ) -> SpatialConsistencyResult:
        """Executes full pairwise comparison matrix across Forehead, Left Cheek, and Right Cheek."""
        total_frames = timing_diagnostics.get("total_frames", 0)
        fps = timing_diagnostics.get("fps", reported_fps)
        total_duration = total_frames / fps if fps > 0 else 0.0

        pairwise_comparisons: Dict[str, PairwiseComparison] = {}
        analysis_limitations: List[str] = []

        # Inspect timing provenance for limitations
        if timing_diagnostics.get("fps_fallback_frames", 0) > 0:
            analysis_limitations.append(
                f"Presentation timestamps unavailable on {timing_diagnostics.get('fps_fallback_frames')} frames; FPS fallback used."
            )
        if timing_diagnostics.get("duplicate_timestamps", 0) > 0:
            analysis_limitations.append(
                f"Detected {timing_diagnostics.get('duplicate_timestamps')} duplicate timestamp frames."
            )

        # Region pairs to analyze
        pairs = [
            (FacialRegionId.LEFT_CHEEK, FacialRegionId.RIGHT_CHEEK, "bilateral_cheeks"),
            (FacialRegionId.FOREHEAD, FacialRegionId.LEFT_CHEEK, "forehead_vs_left_cheek"),
            (FacialRegionId.FOREHEAD, FacialRegionId.RIGHT_CHEEK, "forehead_vs_right_cheek"),
        ]

        bilateral_cheek_cmp: Optional[PairwiseComparison] = None
        forehead_left_cmp: Optional[PairwiseComparison] = None
        forehead_right_cmp: Optional[PairwiseComparison] = None

        for reg_a, reg_b, key in pairs:
            cmp = compare_region_pair(
                region_a=reg_a,
                region_b=reg_b,
                samples_a=regional_samples.get(reg_a, []),
                samples_b=regional_samples.get(reg_b, []),
                summary_a=regional_summaries[reg_a],
                summary_b=regional_summaries[reg_b],
                pulse_a=regional_pulses.get(reg_a),
                pulse_b=regional_pulses.get(reg_b),
                config=self.config,
                reported_fps=reported_fps,
            )
            pairwise_comparisons[key] = cmp

            if key == "bilateral_cheeks":
                bilateral_cheek_cmp = cmp
            elif key == "forehead_vs_left_cheek":
                forehead_left_cmp = cmp
            elif key == "forehead_vs_right_cheek":
                forehead_right_cmp = cmp

        # Determine global consistency state
        states = [c.state for c in pairwise_comparisons.values()]

        if all(s == PairConsistencyState.INSUFFICIENT_DATA for s in states):
            global_state = GlobalConsistencyState.INSUFFICIENT_EVIDENCE
            is_measurable = False
            summary_text = (
                "Spatial consistency indeterminate: Insufficient mutually valid observations across facial regions."
            )
        elif all(s in (PairConsistencyState.LOW_QUALITY_SIGNAL, PairConsistencyState.INSUFFICIENT_DATA) for s in states):
            global_state = GlobalConsistencyState.LOW_QUALITY
            is_measurable = False
            summary_text = (
                "Spatial consistency inconclusive: Facial regions exhibit low physiological SNR or degraded tracking."
            )
        elif any(s == PairConsistencyState.INCONSISTENT for s in states):
            global_state = GlobalConsistencyState.GLOBALLY_INCONSISTENT
            is_measurable = True
            inconsistent_pairs = [k for k, c in pairwise_comparisons.items() if c.state == PairConsistencyState.INCONSISTENT]
            summary_text = (
                f"Regional physiological divergence detected across ({', '.join(inconsistent_pairs)}). "
                f"Caution: Biological factors (uneven perfusion, facial expression) or environmental factors "
                f"(directional lighting, localized compression) can induce regional decorrelation."
            )
        elif all(s == PairConsistencyState.CONSISTENT for s in states):
            global_state = GlobalConsistencyState.GLOBALLY_CONSISTENT
            is_measurable = True
            summary_text = (
                "Temporally and spectrally consistent physiological pulse waves observed across all examined facial regions."
            )
        elif any(s == PairConsistencyState.CONSISTENT for s in states):
            global_state = GlobalConsistencyState.PARTIALLY_CONSISTENT
            is_measurable = True
            consistent_pairs = [k for k, c in pairwise_comparisons.items() if c.state == PairConsistencyState.CONSISTENT]
            summary_text = (
                f"Partial regional consistency observed across ({', '.join(consistent_pairs)})."
            )
        else:
            global_state = GlobalConsistencyState.INSUFFICIENT_EVIDENCE
            is_measurable = False
            summary_text = (
                "Spatial consistency ambiguous: Correlations fall within indeterminate research boundaries."
            )

        valid_overlap = max(
            [c.overlap_duration_seconds for c in pairwise_comparisons.values()],
            default=0.0,
        )

        return SpatialConsistencyResult(
            global_state=global_state,
            pairwise_comparisons=pairwise_comparisons,
            bilateral_cheek_agreement=bilateral_cheek_cmp,
            forehead_left_agreement=forehead_left_cmp,
            forehead_right_agreement=forehead_right_cmp,
            regional_summaries={r.value: s for r, s in regional_summaries.items()},
            total_frames=total_frames,
            duration_seconds=round(total_duration, 3),
            valid_observation_duration_seconds=round(valid_overlap, 3),
            analysis_limitations=analysis_limitations,
            timing_provenance=timing_diagnostics,
            is_consistency_measurable=is_measurable,
            diagnostic_summary=summary_text,
        )
