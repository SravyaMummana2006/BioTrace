"""Comprehensive synthetic tests verifying the Spatial Biological Consistency research module.

Scientific Verification Protocol:
    1. Tests verify mathematical behaviour (cross-correlation, lag recovery, spectral coherence,
       quality gating, gap handling, and invariants), NOT real-world detection claims.
    2. Zero Authenticity Predication:
       authenticity_score must be strictly None, and is_authenticity_assessed must be False.
    3. Noise, occlusion, tracking failure, and short durations must produce explicit neutral
       diagnostic states (INSUFFICIENT_DATA, LOW_QUALITY_SIGNAL), never false manipulation flags.
"""

from pathlib import Path
import threading
import numpy as np
import pytest

from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage
from biotrace.domain.models import VideoMetadata
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    GlobalConsistencyState,
    PairConsistencyState,
    RegionalSample,
    RegionTrackingStatus,
    SpatialConsistencyConfig,
)

from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor
from biotrace.spatial_consistency.quality import (
    assess_regional_signal,
    compute_cardiac_snr,
    extract_pos_pulse_from_rgb,
)
from biotrace.spatial_consistency.spatial_analyzer import (
    SpatialBiologicalAnalyzer,
    compare_region_pair,
    compute_lagged_cross_correlation,
    compute_spectral_overlap,
)

SYNTHETIC_VIDEO = Path(__file__).resolve().parents[2] / "datasets" / "test_synthetic.mp4"


def _make_synthetic_rgb_stream(
    duration_seconds: float = 4.0,
    fps: float = 30.0,
    pulse_freq_hz: float = 1.2,  # 72 BPM
    amplitude: float = 2.0,
    phase_offset_seconds: float = 0.0,
    noise_std: float = 0.05,
    base_rgb: tuple = (180.0, 140.0, 110.0),
    is_valid: bool = True,
) -> tuple:
    """Generates synthetic RGB time series and timestamps with known physiological pulse."""
    n_frames = int(round(duration_seconds * fps))
    timestamps = np.linspace(0.0, duration_seconds, n_frames, endpoint=False)

    pulse = amplitude * np.sin(2.0 * np.pi * pulse_freq_hz * (timestamps - phase_offset_seconds))
    noise = np.random.normal(0.0, noise_std, (n_frames, 3))

    # Blood volume pulse manifests primarily in the green channel (-G) and to a lesser extent blue
    rgb = np.zeros((n_frames, 3), dtype=np.float64)
    rgb[:, 0] = base_rgb[0] + noise[:, 0]
    rgb[:, 1] = base_rgb[1] - pulse + noise[:, 1]
    rgb[:, 2] = base_rgb[2] - 0.3 * pulse + noise[:, 2]
    rgb = np.clip(rgb, 0.0, 255.0)

    samples = [
        RegionalSample(
            frame_index=i,
            timestamp_seconds=float(timestamps[i]),
            region_id=FacialRegionId.FOREHEAD,
            status=RegionTrackingStatus.VALID if is_valid else RegionTrackingStatus.LOW_SKIN_PIXELS,
            mean_rgb=[float(rgb[i, 0]), float(rgb[i, 1]), float(rgb[i, 2])] if is_valid else None,
            pixel_count=150 if is_valid else 0,
            skin_pixel_ratio=0.9 if is_valid else 0.0,
            is_valid=is_valid,
        )
        for i in range(n_frames)
    ]

    return timestamps, rgb, samples


def _make_dummy_metadata(duration_seconds: float = 4.0, fps: float = 30.0) -> VideoMetadata:
    return VideoMetadata(
        file_path="synthetic_test.mp4",
        file_name="synthetic_test.mp4",
        width=640,
        height=480,
        fps=fps,
        frame_count=int(round(duration_seconds * fps)),
        duration_seconds=duration_seconds,
        codec="h264",
        file_size_bytes=10240,
    )


# ---------------------------------------------------------------------------
# 1. Mathematical Behavior Tests
# ---------------------------------------------------------------------------


def test_identical_periodic_signals():
    """Identical pulses across all regions must yield maximum correlation and global consistency."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig(min_observation_seconds=3.0, min_duration_for_bpm_analysis=6.0)
    analyzer = SpatialBiologicalAnalyzer(config=cfg)

    ts, rgb, samples = _make_synthetic_rgb_stream(duration_seconds=7.0, pulse_freq_hz=1.2, noise_std=0.01)

    reg_samples = {
        FacialRegionId.FOREHEAD: [s.model_copy(update={"region_id": FacialRegionId.FOREHEAD}) for s in samples],
        FacialRegionId.LEFT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.LEFT_CHEEK}) for s in samples],
        FacialRegionId.RIGHT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.RIGHT_CHEEK}) for s in samples],
    }

    summaries = {}
    pulses = {}
    for r_id, s_list in reg_samples.items():
        summary, _, pulse = assess_regional_signal(s_list, r_id, cfg, reported_fps=30.0)
        summaries[r_id] = summary
        pulses[r_id] = pulse

    diagnostics = {"total_frames": len(samples), "fps": 30.0}
    res = analyzer.analyze_spatial_consistency(reg_samples, summaries, pulses, diagnostics, 30.0)

    assert res.global_state == GlobalConsistencyState.GLOBALLY_CONSISTENT
    assert res.is_consistency_measurable is True

    # Bilateral cheek check
    assert res.bilateral_cheek_agreement is not None
    assert res.bilateral_cheek_agreement.state == PairConsistencyState.CONSISTENT
    assert res.bilateral_cheek_agreement.max_cross_correlation >= 0.95
    assert abs(res.bilateral_cheek_agreement.optimal_lag_seconds) <= 0.05
    assert res.bilateral_cheek_agreement.bpm_difference == pytest.approx(0.0, abs=1.0)


def test_signals_with_known_phase_offsets():
    """Known lag within physiological tolerance must be accurately recovered."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig(max_physiological_lag_seconds=0.25)
    known_lag_sec = 0.10  # 3 frames at 30 fps

    ts, rgb_a, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.0, phase_offset_seconds=0.0, noise_std=0.01)
    _, rgb_b, samples_b = _make_synthetic_rgb_stream(duration_seconds=4.0, phase_offset_seconds=known_lag_sec, noise_std=0.01)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.CONSISTENT
    assert cmp.max_cross_correlation >= 0.90
    assert cmp.optimal_lag_seconds == pytest.approx(known_lag_sec, abs=0.04)


def test_different_frequencies_yield_inconsistency():
    """High SNR signals with divergent cardiac frequencies must be flagged INCONSISTENT."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig()
    analyzer = SpatialBiologicalAnalyzer(config=cfg)

    # Left cheek: 1.2 Hz (72 BPM), Right cheek: 2.0 Hz (120 BPM) across 7.0s (>= 6.0s for 10 BPM resolution)
    _, _, samples_fh = _make_synthetic_rgb_stream(duration_seconds=7.0, pulse_freq_hz=1.2, noise_std=0.01)
    _, _, samples_lc = _make_synthetic_rgb_stream(duration_seconds=7.0, pulse_freq_hz=1.2, noise_std=0.01)
    _, _, samples_rc = _make_synthetic_rgb_stream(duration_seconds=7.0, pulse_freq_hz=2.0, noise_std=0.01)

    reg_samples = {
        FacialRegionId.FOREHEAD: [s.model_copy(update={"region_id": FacialRegionId.FOREHEAD}) for s in samples_fh],
        FacialRegionId.LEFT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.LEFT_CHEEK}) for s in samples_lc],
        FacialRegionId.RIGHT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.RIGHT_CHEEK}) for s in samples_rc],
    }

    summaries = {}
    pulses = {}
    for r_id, s_list in reg_samples.items():
        summary, _, pulse = assess_regional_signal(s_list, r_id, cfg, 30.0)
        summaries[r_id] = summary
        pulses[r_id] = pulse

    res = analyzer.analyze_spatial_consistency(reg_samples, summaries, pulses, {"total_frames": len(samples_fh)}, 30.0)

    assert res.global_state == GlobalConsistencyState.GLOBALLY_INCONSISTENT
    assert res.bilateral_cheek_agreement.state == PairConsistencyState.INCONSISTENT
    assert res.bilateral_cheek_agreement.bpm_difference >= 30.0


def test_different_amplitudes_scale_invariance():
    """Pearson correlation and POS must remain invariant to regional amplitude differences."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig()

    # Forehead amplitude 3.0, cheek amplitude 0.5
    _, _, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=3.0, noise_std=0.01)
    _, _, samples_b = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=0.5, noise_std=0.01)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.FOREHEAD, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.LEFT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.FOREHEAD,
        FacialRegionId.LEFT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.CONSISTENT
    assert cmp.max_cross_correlation >= 0.90


def test_noisy_signals_trigger_low_quality_not_inconsistency():
    """Heavy noise degrading SNR must produce LOW_QUALITY_SIGNAL, never false INCONSISTENT."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig(min_snr_db=0.0)

    # Heavy noise destroying pulse waveform
    _, _, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=0.5, noise_std=5.0)
    _, _, samples_b = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=0.5, noise_std=5.0)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.LOW_QUALITY_SIGNAL
    assert "low physiological snr" in cmp.diagnostic_explanation.lower()


def test_illumination_changes_handled_by_pos_and_detrending():
    """Slow baseline drift or illumination changes should be removed by polynomial detrending."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig()

    ts, rgb_base, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.0, pulse_freq_hz=1.2, noise_std=0.01)

    # Add strong illumination ramp (+30 over 4 seconds) to second signal
    samples_b = []
    for i, s in enumerate(samples_a):
        ramp = float(30.0 * (i / len(samples_a)))
        orig_rgb = s.mean_rgb
        drifted_rgb = [min(255.0, c + ramp) for c in orig_rgb]
        samples_b.append(s.model_copy(update={"region_id": FacialRegionId.LEFT_CHEEK, "mean_rgb": drifted_rgb}))

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.FOREHEAD, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.LEFT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.FOREHEAD,
        FacialRegionId.LEFT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.CONSISTENT
    assert cmp.max_cross_correlation >= 0.80


def test_missing_and_occluded_regions():
    """When a region is missing or occluded, it must return INSUFFICIENT_DATA and partial consistency."""
    np.random.seed(42)
    cfg = SpatialConsistencyConfig()
    analyzer = SpatialBiologicalAnalyzer(config=cfg)

    _, _, samples_fh = _make_synthetic_rgb_stream(duration_seconds=4.0, is_valid=False)  # Hair fringe occlusion
    _, _, samples_lc = _make_synthetic_rgb_stream(duration_seconds=4.0, is_valid=True, noise_std=0.01)
    _, _, samples_rc = _make_synthetic_rgb_stream(duration_seconds=4.0, is_valid=True, noise_std=0.01)

    reg_samples = {
        FacialRegionId.FOREHEAD: [s.model_copy(update={"region_id": FacialRegionId.FOREHEAD}) for s in samples_fh],
        FacialRegionId.LEFT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.LEFT_CHEEK}) for s in samples_lc],
        FacialRegionId.RIGHT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.RIGHT_CHEEK}) for s in samples_rc],
    }

    summaries = {}
    pulses = {}
    for r_id, s_list in reg_samples.items():
        summary, _, pulse = assess_regional_signal(s_list, r_id, cfg, 30.0)
        summaries[r_id] = summary
        pulses[r_id] = pulse

    res = analyzer.analyze_spatial_consistency(reg_samples, summaries, pulses, {"total_frames": len(samples_fh)}, 30.0)

    assert res.global_state == GlobalConsistencyState.PARTIALLY_CONSISTENT
    assert res.bilateral_cheek_agreement.state == PairConsistencyState.CONSISTENT
    assert res.forehead_left_agreement.state == PairConsistencyState.INSUFFICIENT_DATA


def test_insufficient_observation_duration():
    """Observation duration below min_observation_seconds must trigger INSUFFICIENT_DATA."""
    cfg = SpatialConsistencyConfig(min_observation_seconds=3.0)

    # Only 1.0 second of data
    _, _, samples_a = _make_synthetic_rgb_stream(duration_seconds=1.0, noise_std=0.01)
    _, _, samples_b = _make_synthetic_rgb_stream(duration_seconds=1.0, noise_std=0.01)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.INSUFFICIENT_DATA
    assert "below required minimum" in cmp.diagnostic_explanation.lower()


def test_irregular_and_duplicate_timestamps():
    """Duplicate and non-monotonic timestamps must be detected and documented in diagnostics."""
    cfg = SpatialConsistencyConfig()
    _, _, samples = _make_synthetic_rgb_stream(duration_seconds=3.0)

    # Inject duplicate and irregular timestamps
    samples[10].timestamp_seconds = samples[9].timestamp_seconds
    samples[20].timestamp_seconds = samples[19].timestamp_seconds - 0.05

    timing_diag = {
        "total_frames": len(samples),
        "fps": 30.0,
        "fps_fallback_frames": 10,
        "duplicate_timestamps": 1,
        "non_monotonic_timestamps": 1,
    }

    analyzer = SpatialBiologicalAnalyzer(config=cfg)
    reg_samples = {r: samples for r in FacialRegionId}
    summaries = {r: assess_regional_signal(samples, r, cfg, 30.0)[0] for r in FacialRegionId}
    pulses = {r: assess_regional_signal(samples, r, cfg, 30.0)[2] for r in FacialRegionId}

    res = analyzer.analyze_spatial_consistency(reg_samples, summaries, pulses, timing_diag, 30.0)

    assert len(res.analysis_limitations) >= 2
    assert any("duplicate timestamp" in lim.lower() for lim in res.analysis_limitations)
    assert any("fps fallback" in lim.lower() for lim in res.analysis_limitations)


# ---------------------------------------------------------------------------
# 2. Extractor Interface Contract & Forensic Guarantees
# ---------------------------------------------------------------------------


def test_spatial_consistency_extractor_contract():
    """Verifies that SpatialConsistencyExtractor strictly preserves forensic guardrails."""
    cfg = SpatialConsistencyConfig()
    extractor = SpatialConsistencyExtractor(config=cfg)

    assert extractor.modality == ForensicModality.SPATIAL_CONSISTENCY

    _, _, samples = _make_synthetic_rgb_stream(duration_seconds=3.5, noise_std=0.01)
    reg_samples = {
        FacialRegionId.FOREHEAD: [s.model_copy(update={"region_id": FacialRegionId.FOREHEAD}) for s in samples],
        FacialRegionId.LEFT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.LEFT_CHEEK}) for s in samples],
        FacialRegionId.RIGHT_CHEEK: [s.model_copy(update={"region_id": FacialRegionId.RIGHT_CHEEK}) for s in samples],
    }

    meta = _make_dummy_metadata(duration_seconds=3.5)
    diag = {"total_frames": len(samples), "fps": 30.0}

    verdict = extractor.extract_from_samples(reg_samples, diag, meta)

    assert verdict is not None
    assert verdict.modality == ForensicModality.SPATIAL_CONSISTENCY
    # Critical Guardrails
    assert verdict.authenticity_score is None
    assert verdict.is_authenticity_assessed is False
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.raw_signal is not None
    assert verdict.raw_signal.modality == ForensicModality.SPATIAL_CONSISTENCY
    assert len(verdict.raw_signal.points) > 0
    assert "experimental physiological measure" in verdict.explanation.lower()
    assert "not proof of video manipulation" in verdict.explanation.lower()


def test_cancellation_during_extraction():
    """Cancellation event must cause immediate cooperative return of None."""
    extractor = SpatialConsistencyExtractor()
    cancel_event = threading.Event()
    cancel_event.set()

    meta = _make_dummy_metadata()
    verdict = extractor.extract("dummy_path.mp4", meta, cancel_event=cancel_event)
    assert verdict is None

    verdict_from_samples = extractor.extract_from_samples({}, {}, meta, cancel_event=cancel_event)
    assert verdict_from_samples is None


def test_config_invariants():
    """Config validation must enforce mathematical bounds."""
    with pytest.raises(ValueError, match="cardiac_low_hz"):
        SpatialConsistencyConfig(cardiac_low_hz=3.5, cardiac_high_hz=2.0)

    with pytest.raises(ValueError, match="correlation_inconsistent_threshold"):
        SpatialConsistencyConfig(
            correlation_inconsistent_threshold=0.8,
            correlation_consistent_threshold=0.5,
        )


# ---------------------------------------------------------------------------
# 3. Pipeline Integration & Fault Isolation Tests
# ---------------------------------------------------------------------------


def test_job_manager_integration_spatial_consistency():
    """JobManager executes spatial consistency modality and records verdicts."""
    from biotrace.core.config import Settings
    from biotrace.domain.models import JobCreationRequest
    from biotrace.services.job_manager import JobManager
    from biotrace.services.video_inspector import VideoInspector
    import time

    if not SYNTHETIC_VIDEO.exists():
        pytest.skip(f"Test video {SYNTHETIC_VIDEO} not found")

    settings = Settings()
    inspector = VideoInspector(settings)
    jm = JobManager(settings, inspector)

    req = JobCreationRequest(
        video_path=str(SYNTHETIC_VIDEO),
        requested_modalities=[ForensicModality.SPATIAL_CONSISTENCY],
    )
    job = jm.submit_job(req)

    for _ in range(40):
        time.sleep(0.1)
        st = jm.get_job(job.job_id)
        if st.status.is_terminal:
            break

    assert st.status == JobStatus.COMPLETED
    assert st.progress_percentage == 100.0
    assert st.result is not None
    assert st.result.overall_authenticity_score is None
    assert st.result.is_manipulated is None

    mod_verdicts = st.result.modality_verdicts
    assert len(mod_verdicts) == 1
    assert mod_verdicts[0].modality == ForensicModality.SPATIAL_CONSISTENCY
    assert mod_verdicts[0].authenticity_score is None
    assert mod_verdicts[0].is_authenticity_assessed is False


def test_spatial_consistency_failure_isolation():
    """An unrecoverable exception in spatial consistency does not fail other modalities."""
    from unittest.mock import MagicMock
    from biotrace.core.config import Settings
    from biotrace.domain.models import JobCreationRequest
    from biotrace.services.job_manager import JobManager
    from biotrace.services.video_inspector import VideoInspector
    import time

    if not SYNTHETIC_VIDEO.exists():
        pytest.skip(f"Test video {SYNTHETIC_VIDEO} not found")

    mock_spatial = MagicMock()
    mock_spatial.extract.side_effect = RuntimeError("Simulated spatial module failure")

    settings = Settings()
    inspector = VideoInspector(settings)
    jm = JobManager(settings, inspector, spatial_consistency_extractor=mock_spatial)

    req = JobCreationRequest(
        video_path=str(SYNTHETIC_VIDEO),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.SPATIAL_CONSISTENCY],
    )
    job = jm.submit_job(req)

    for _ in range(40):
        time.sleep(0.1)
        st = jm.get_job(job.job_id)
        if st.status.is_terminal:
            break

    assert st.status == JobStatus.COMPLETED
    assert st.result is not None
    verdicts = {v.modality: v for v in st.result.modality_verdicts}
    assert ForensicModality.RPPG in verdicts
    assert ForensicModality.SPATIAL_CONSISTENCY in verdicts
    assert "Simulated spatial module failure" in verdicts[ForensicModality.SPATIAL_CONSISTENCY].explanation
    assert verdicts[ForensicModality.SPATIAL_CONSISTENCY].confidence == 0.0


# ---------------------------------------------------------------------------
# 4. Final Scientific Audit Tests
# ---------------------------------------------------------------------------


def test_short_duration_spectral_resolution_safeguard():
    """In short observations (<6.0s), frequency resolution is too coarse to resolve a 10 BPM discrepancy."""
    cfg = SpatialConsistencyConfig(min_observation_seconds=3.0, min_duration_for_bpm_analysis=6.0)

    # 4.5s observation: Delta_BPM_physical = 60 / 4.5 = 13.3 BPM > 10.0 BPM
    _, _, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.5, pulse_freq_hz=1.2, noise_std=0.01)
    _, _, samples_b = _make_synthetic_rgb_stream(duration_seconds=4.5, pulse_freq_hz=1.6, noise_std=0.01)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    # bpm_difference must be unresolvable (None) due to physical Rayleigh limit
    assert cmp.bpm_difference is None
    assert "rayleigh" in cmp.diagnostic_explanation.lower() or "physical resolution" in cmp.diagnostic_explanation.lower()


def test_large_gap_partitioning_safeguard():
    """Inter-sample gaps > max_permissible_gap_seconds (0.75s) must not be stitched or interpolated."""
    cfg = SpatialConsistencyConfig(min_observation_seconds=3.0, max_permissible_gap_seconds=0.75)

    _, _, full_samples = _make_synthetic_rgb_stream(duration_seconds=6.0, noise_std=0.01)

    # Introduce a 1.5s tracking gap between frame 60 (2.0s) and frame 105 (3.5s)
    # Segment 1: frames 0..59 (2.0s duration < 3.0s min)
    # Segment 2: frames 105..179 (2.5s duration < 3.0s min)
    gapped_samples = [s for s in full_samples if s.frame_index < 60 or s.frame_index >= 105]

    sum_a, _, pulse_a = assess_regional_signal(gapped_samples, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(gapped_samples, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        gapped_samples,
        gapped_samples,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.INSUFFICIENT_DATA
    assert "interrupted by large gap" in cmp.diagnostic_explanation.lower() or "below required minimum" in cmp.diagnostic_explanation.lower()


def test_zero_and_negative_snr_denominator_safeguards():
    """compute_cardiac_snr must never crash on zero-variance, flat, or degenerate signals."""
    fs = 30.0

    # 1. Constant flat signal (zero variance)
    flat = np.full(100, 42.0)
    snr, bpm, hz, _, _, phys_res = compute_cardiac_snr(flat, fs)
    assert snr == -20.0
    assert bpm is None
    assert hz is None

    # 2. Pure sine wave where noise power (p_total - p_signal) -> 0
    t = np.linspace(0, 5.0, 150)
    pure_sine = np.sin(2.0 * np.pi * 1.2 * t)
    snr_sine, bpm_sine, hz_sine, _, _, _ = compute_cardiac_snr(pure_sine, fs)
    assert np.isfinite(snr_sine)
    assert snr_sine > 0.0  # Acceptable SNR for pure periodic oscillation
    assert bpm_sine == pytest.approx(72.0, abs=2.0)

    # 3. Non-finite array with NaNs and Infs
    corrupt = np.full(100, np.nan)
    snr_nan, _, _, _, _, _ = compute_cardiac_snr(corrupt, fs)
    assert snr_nan == -20.0

    # 4. Empty array
    empty = np.array([])
    snr_empty, _, _, _, _, _ = compute_cardiac_snr(empty, fs)
    assert snr_empty == -20.0


def test_weak_signals_lagged_correlation_safeguard():
    """Weak/noisy signals with low SNR must strictly yield LOW_QUALITY_SIGNAL, never INCONSISTENT."""
    cfg = SpatialConsistencyConfig(min_snr_db=0.0)

    # Signals dominated by heavy noise
    _, _, samples_a = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=0.2, noise_std=10.0)
    _, _, samples_b = _make_synthetic_rgb_stream(duration_seconds=4.0, amplitude=0.2, noise_std=10.0)

    sum_a, _, pulse_a = assess_regional_signal(samples_a, FacialRegionId.LEFT_CHEEK, cfg, 30.0)
    sum_b, _, pulse_b = assess_regional_signal(samples_b, FacialRegionId.RIGHT_CHEEK, cfg, 30.0)

    cmp = compare_region_pair(
        FacialRegionId.LEFT_CHEEK,
        FacialRegionId.RIGHT_CHEEK,
        samples_a,
        samples_b,
        sum_a,
        sum_b,
        pulse_a,
        pulse_b,
        cfg,
        30.0,
    )

    assert cmp.state == PairConsistencyState.LOW_QUALITY_SIGNAL
    assert cmp.state != PairConsistencyState.INCONSISTENT


def test_standard_pos_windowing_insufficient_frames():
    """When frames are fewer than standard POS window (1.6s = 48 frames at 30Hz), POS returns None."""
    # 30 frames = 1.0s at 30Hz (< 1.6s window)
    rgb_short = np.ones((30, 3), dtype=np.float64) * 100.0

    pulse = extract_pos_pulse_from_rgb(rgb_short, sampling_rate_hz=30.0, window_seconds=1.6)
    assert pulse is None

    # 60 frames = 2.0s at 30Hz (> 1.6s window)
    rgb_valid = np.ones((60, 3), dtype=np.float64) * 100.0
    rgb_valid[:, 1] += np.sin(np.linspace(0, 4 * np.pi, 60))
    pulse_valid = extract_pos_pulse_from_rgb(rgb_valid, sampling_rate_hz=30.0, window_seconds=1.6)
    assert pulse_valid is not None
    assert len(pulse_valid) == 60


def test_physical_resolution_boundary_invariants():
    """Config must enforce min_duration_for_bpm_analysis >= 60 / max_bpm_discrepancy."""
    # Default 6.0s duration resolves 10.0 BPM discrepancy
    cfg = SpatialConsistencyConfig()
    assert cfg.min_duration_for_bpm_analysis == 6.0
    assert cfg.max_bpm_discrepancy == 10.0

    # Contradiction: 4.0s duration cannot physically resolve 10.0 BPM (Rayleigh limit is 15.0 BPM)
    with pytest.raises(ValueError, match="Rayleigh resolution"):
        SpatialConsistencyConfig(min_duration_for_bpm_analysis=4.0, max_bpm_discrepancy=10.0)

    # Coarser resolution policy: 4.0s is valid if discrepancy threshold is at least 15.0 BPM
    cfg_coarse = SpatialConsistencyConfig(min_duration_for_bpm_analysis=4.0, max_bpm_discrepancy=15.0)
    assert cfg_coarse.min_duration_for_bpm_analysis == 4.0
    assert cfg_coarse.max_bpm_discrepancy == 15.0


def test_timestamp_irregularity_resampling_and_rejection():
    """Verify that minor jitter is resampled to uniform grid while excessive jitter is rejected."""
    from biotrace.spatial_consistency.quality import sanitize_and_resample_continuous_segment

    # 1. Minor timestamp jitter (jitter ratio ~ 0.05 < 0.25 tolerance)
    np.random.seed(42)
    n = 90  # 3.0 seconds at ~30 fps
    base_t = np.linspace(0.0, 3.0, n)
    jitter = np.random.uniform(-0.003, 0.003, n)
    jitter[0] = 0.0
    jitter[-1] = 0.0
    jitter_t = np.sort(base_t + jitter)
    sig = np.sin(2.0 * np.pi * 1.2 * jitter_t)

    ok, u_ts, u_sig, fs, diag = sanitize_and_resample_continuous_segment(
        timestamps=jitter_t,
        signal_series=sig,
        max_jitter_ratio=0.25,
    )
    assert ok is True
    assert u_ts is not None and u_sig is not None
    assert len(u_ts) == n
    assert u_ts[0] == pytest.approx(jitter_t[0])
    assert u_ts[-1] == pytest.approx(jitter_t[-1])
    # Resampled timestamps must be strictly uniform
    dt_resampled = np.diff(u_ts)
    assert float(np.std(dt_resampled)) < 1e-12

    # 2. Extreme timestamp jitter (jitter ratio > 0.25)
    bad_t = np.copy(base_t)
    bad_t[10:30] += 0.50  # severe timing disruption
    bad_t = np.sort(bad_t)
    ok_bad, _, _, _, diag_bad = sanitize_and_resample_continuous_segment(
        timestamps=bad_t,
        signal_series=sig,
        max_jitter_ratio=0.25,
    )
    assert ok_bad is False
    assert "exceeds tolerance" in diag_bad.lower()


def test_hann_overlap_add_edge_coverage():
    """Every sample in windowed POS overlap-add must receive positive accumulated Hann weight."""
    n_frames = 80
    rgb = np.ones((n_frames, 3), dtype=np.float64) * 120.0
    rgb[:, 1] += 5.0 * np.sin(np.linspace(0, 8 * np.pi, n_frames))

    for stride in (1, 2, 5):
        pulse = extract_pos_pulse_from_rgb(
            rgb_series=rgb,
            sampling_rate_hz=30.0,
            window_seconds=1.6,
            stride_frames=stride,
        )
        assert pulse is not None
        assert len(pulse) == n_frames
        assert np.isfinite(pulse).all()
        # Edge samples must not be flat or zero-clipped
        assert pulse[0] != 0.0 or pulse[1] != 0.0
        assert pulse[-1] != 0.0 or pulse[-2] != 0.0



