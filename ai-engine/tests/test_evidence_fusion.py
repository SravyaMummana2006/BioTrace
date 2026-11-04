"""Comprehensive tests for Stage 5 & 5.1 Multimodal Evidence Fusion Engine.

Verifies the core scientific and architectural requirements:
    1. All three modalities available with supporting observations (SUPPORTING_EVIDENCE).
    2. rPPG and spatial consistency agreeing without double-counting (consolidated optical channel).
    3. rPPG and spatial consistency disagreeing (DISAGREEMENT within family -> MIXED_EVIDENCE or CONFLICTING_EVIDENCE).
    4. Blink evidence alone is descriptive -> INSUFFICIENT_EVIDENCE with preserved measurements.
    5. Optical evidence available while blink evidence is missing (SUPPORTING_EVIDENCE from optical alone).
    6. All modalities unavailable (INSUFFICIENT_EVIDENCE).
    7. One modality throwing an exception (other modalities preserved, error tagged in missing/limitations).
    8. Low-quality evidence not becoming negative authenticity evidence (quality limitation, not is_manipulated=True).
    9. Contradictory evidence producing MIXED_EVIDENCE or CONFLICTING_EVIDENCE.
    10. Cancellation before fusion (cancel_event.set() before Stage 5).
    11. Empty modality selection (requested_modalities=[] -> INSUFFICIENT_EVIDENCE).
    12. No authenticity score or manipulated label being generated (authenticity_score=None, is_manipulated=None).
    13. Atypical blink rates preserved as biological variation without negative authenticity inferences.
    14. Shared video-level confounders documented across families.
"""

from pathlib import Path
import threading
import time
from typing import Dict, List, Optional
import pytest

from biotrace.core.config import Settings
from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage, SignalType
from biotrace.domain.models import (
    AnalysisResult,
    ForensicSignal,
    JobCreationRequest,
    ModalityVerdict,
    SignalPoint,
    VideoMetadata,
)
from biotrace.evidence_fusion.dependency_model import EvidenceDependencyGraph
from biotrace.evidence_fusion.domain import (
    EvidenceFamilyAgreement,
    EvidenceFamilyId,
    EvidenceFamilySummary,
    EvidenceQualityStatus,
    FusedEvidenceResult,
    FusedEvidenceState,
    ModalityEvidenceRecord,
)
from biotrace.evidence_fusion.evidence_normalizer import EvidenceNormalizer
from biotrace.evidence_fusion.fusion_engine import MultimodalEvidenceFusionEngine
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector


# ---------------------------------------------------------------------------
# Helper Mock Constructors
# ---------------------------------------------------------------------------

def _dummy_metadata(duration_seconds: float = 5.0, fps: float = 30.0) -> VideoMetadata:
    return VideoMetadata(
        file_name="test_subject.mp4",
        file_path="C:/biotrace/data/test_subject.mp4",
        width=640,
        height=480,
        fps=fps,
        frame_count=int(duration_seconds * fps),
        duration_seconds=duration_seconds,
        codec="h264",
        file_size_bytes=1024 * 1024,
    )


def _make_dummy_signal(
    modality: ForensicModality,
    signal_type: SignalType,
    sampling_rate_hz: float = 30.0,
    num_points: int = 150,
    metadata: Optional[Dict] = None,
) -> ForensicSignal:
    points = [
        SignalPoint(timestamp_seconds=i / sampling_rate_hz, frame_index=i, value=0.5, confidence=1.0)
        for i in range(num_points)
    ]
    return ForensicSignal(
        modality=modality,
        signal_type=signal_type,
        sampling_rate_hz=sampling_rate_hz,
        points=points,
        metadata=metadata or {},
    )


def _make_rppg_verdict(
    hr_bpm: float = 72.0,
    snr_db: float = 6.0,
    confidence: float = 0.88,
    grade: str = "A",
    is_available: bool = True,
    explanation: Optional[str] = None,
) -> ModalityVerdict:
    if not is_available:
        return ModalityVerdict(
            modality=ForensicModality.RPPG,
            authenticity_score=None,
            is_authenticity_assessed=False,
            anomaly_detected=False,
            confidence=0.0,
            raw_signal=None,
            explanation=explanation or "rPPG signal unobservable: tracking failure or no face detected.",
        )

    meta = {
        "heart_rate_bpm": str(hr_bpm),
        "snr_db": str(snr_db),
        "quality_grade": grade,
        "algorithm": "POS",
    }
    sig = _make_dummy_signal(
        modality=ForensicModality.RPPG,
        signal_type=SignalType.RPPG_POS,
        metadata=meta,
    )
    return ModalityVerdict(
        modality=ForensicModality.RPPG,
        authenticity_score=None,
        is_authenticity_assessed=False,
        anomaly_detected=False,
        confidence=confidence,
        raw_signal=sig,
        explanation=explanation or f"rPPG cardiac rhythm identified: {hr_bpm:.1f} BPM, SNR={snr_db:.1f} dB.",
    )


def _make_spatial_verdict(
    global_state: str = "globally_consistent",
    confidence: float = 0.85,
    regional_snr: Optional[Dict[str, float]] = None,
    regional_bpms: Optional[Dict[str, float]] = None,
    anomaly_detected: bool = False,
    is_available: bool = True,
    explanation: Optional[str] = None,
) -> ModalityVerdict:
    if not is_available:
        return ModalityVerdict(
            modality=ForensicModality.SPATIAL_CONSISTENCY,
            authenticity_score=None,
            is_authenticity_assessed=False,
            anomaly_detected=False,
            confidence=0.0,
            raw_signal=None,
            explanation=explanation or "Spatial consistency unavailable: insufficient valid regions.",
        )

    snr_map = regional_snr or {"forehead": 5.2, "left_cheek": 4.8, "right_cheek": 5.0}
    bpm_map = regional_bpms or {"forehead": 72.0, "left_cheek": 72.0, "right_cheek": 72.0}

    # Construct regional lines in explanation to match regex if parsed
    reg_lines = [
        f"{r}: 150/150 frames (100.0%), SNR={snr_map.get(r, 5.0):.1f} dB, HR={bpm_map.get(r, 72.0):.1f} BPM"
        for r in snr_map.keys()
    ]
    expl = explanation or (
        f"Spatial consistency analysis: state={global_state}, valid_regions=3/3. "
        + "; ".join(reg_lines)
    )

    meta = {
        "global_state": global_state,
        "primary_region": "forehead",
        "regional_snr": snr_map,
        "pairwise_states": {"forehead_left_cheek": "consistent", "forehead_right_cheek": "consistent"},
    }
    sig = _make_dummy_signal(
        modality=ForensicModality.SPATIAL_CONSISTENCY,
        signal_type=SignalType.SPATIAL_RPPG_MULTI_REGION,
        metadata=meta,
    )
    return ModalityVerdict(
        modality=ForensicModality.SPATIAL_CONSISTENCY,
        authenticity_score=None,
        is_authenticity_assessed=False,
        anomaly_detected=anomaly_detected,
        confidence=confidence,
        raw_signal=sig,
        explanation=expl,
    )


def _make_facial_dynamics_verdict(
    blink_rate: float = 16.0,
    confidence: float = 0.90,
    calibration_state: str = "calibrated",
    obs_duration: float = 5.0,
    anomaly_detected: bool = False,
    is_available: bool = True,
    explanation: Optional[str] = None,
) -> ModalityVerdict:
    if not is_available:
        return ModalityVerdict(
            modality=ForensicModality.FACIAL_DYNAMICS,
            authenticity_score=None,
            is_authenticity_assessed=False,
            anomaly_detected=False,
            confidence=0.0,
            raw_signal=None,
            explanation=explanation or "Facial dynamics unavailable: no face detected.",
        )

    meta = {
        "calibration_state": calibration_state,
        "bilateral_blinks_per_minute": blink_rate,
        "left_blinks_per_minute": blink_rate,
        "right_blinks_per_minute": blink_rate,
        "total_complete_blinks": int((blink_rate / 60.0) * obs_duration),
        "complete_bilateral_blinks": int((blink_rate / 60.0) * obs_duration),
        "mean_complete_duration_ms": 220.0,
        "valid_observation_duration_seconds": obs_duration,
        "timing_source": "presentation_timestamp",
        "has_timestamp_fallback": False,
        "duplicate_timestamps_count": 0,
        "non_monotonic_timestamps_count": 0,
    }
    sig = _make_dummy_signal(
        modality=ForensicModality.FACIAL_DYNAMICS,
        signal_type=SignalType.EYE_BLINK_EAR,
        metadata=meta,
    )
    return ModalityVerdict(
        modality=ForensicModality.FACIAL_DYNAMICS,
        authenticity_score=None,
        is_authenticity_assessed=False,
        anomaly_detected=anomaly_detected,
        confidence=confidence,
        raw_signal=sig,
        explanation=explanation or f"Facial dynamics: {blink_rate:.1f} blinks/min, calibrated.",
    )


# ---------------------------------------------------------------------------
# Test Suite: 12 Core Requirements
# ---------------------------------------------------------------------------

def test_01_all_three_modalities_supporting():
    """Requirement 1: All three modalities available with supporting observations -> SUPPORTING_EVIDENCE."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(hr_bpm=72.0, snr_db=6.5, confidence=0.90),
        _make_spatial_verdict(global_state="globally_consistent", confidence=0.88),
        _make_facial_dynamics_verdict(blink_rate=16.0, confidence=0.92),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[
            ForensicModality.RPPG,
            ForensicModality.SPATIAL_CONSISTENCY,
            ForensicModality.FACIAL_DYNAMICS,
        ],
    )

    assert result.overall_state == FusedEvidenceState.SUPPORTING_EVIDENCE
    assert result.is_evidence_sufficient is True
    assert "optical_cardiovascular" in result.family_summaries
    assert "facial_dynamics" in result.family_summaries

    opt_sum = result.family_summaries["optical_cardiovascular"]
    assert opt_sum.agreement_state == EvidenceFamilyAgreement.AGREEMENT
    assert opt_sum.quality_status == EvidenceQualityStatus.OPTIMAL

    dyn_sum = result.family_summaries["facial_dynamics"]
    assert dyn_sum.quality_status == EvidenceQualityStatus.OPTIMAL

    # Concordant cross-system findings
    assert len(result.agreements) >= 2
    assert any("Cross-modality physiological concordance" in a for a in result.agreements)
    assert len(result.disagreements) == 0


def test_02_optical_cardiovascular_no_double_counting():
    """Requirement 2: rPPG and spatial consistency agreeing without double-counting (consolidated optical channel)."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(hr_bpm=70.0, snr_db=7.0, confidence=0.95),
        _make_spatial_verdict(global_state="globally_consistent", confidence=0.90),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG, ForensicModality.SPATIAL_CONSISTENCY],
    )

    opt_sum = result.family_summaries["optical_cardiovascular"]
    # Check that both members are accounted for
    assert len(opt_sum.member_modalities) == 2
    assert ForensicModality.RPPG in opt_sum.member_modalities
    assert ForensicModality.SPATIAL_CONSISTENCY in opt_sum.member_modalities

    # Check dependency properties: single consolidated channel, no arbitrary numerical weights
    assert opt_sum.is_statistically_dependent is True
    assert opt_sum.is_single_consolidated_channel is True
    assert opt_sum.consolidated_evidence_weight is None
    assert len(opt_sum.shared_video_confounders) > 0
    assert "dermal_skin_reflectance" in opt_sum.shared_video_confounders

    # Distinct measurements preserved in modality evidence
    assert "rppg" in result.modality_evidence
    assert "spatial_consistency" in result.modality_evidence
    assert result.modality_evidence["rppg"].measurements["heart_rate_bpm"] == 70.0
    assert result.modality_evidence["spatial_consistency"].measurements["global_state"] == "globally_consistent"


def test_03_optical_disagreement_within_family():
    """Requirement 3: rPPG and spatial consistency disagreeing -> DISAGREEMENT within family."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    # Global rPPG observes 72 BPM, but spatial consistency detects regional pulse divergence (>20 BPM diff)
    verdicts_with_dynamics = [
        _make_rppg_verdict(hr_bpm=72.0, snr_db=6.0, confidence=0.88),
        _make_spatial_verdict(
            global_state="globally_inconsistent",
            confidence=0.85,
            regional_bpms={"forehead": 72.0, "left_cheek": 98.0, "right_cheek": 99.0},
            anomaly_detected=True,
        ),
        _make_facial_dynamics_verdict(blink_rate=15.0, confidence=0.90),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts_with_dynamics,
        video_metadata=meta,
        requested_modalities=[
            ForensicModality.RPPG,
            ForensicModality.SPATIAL_CONSISTENCY,
            ForensicModality.FACIAL_DYNAMICS,
        ],
    )

    opt_sum = result.family_summaries["optical_cardiovascular"]
    assert opt_sum.agreement_state == EvidenceFamilyAgreement.DISAGREEMENT
    assert len(opt_sum.intra_family_discrepancies) > 0
    # Because facial dynamics is intact and supporting while optical has internal divergence -> MIXED_EVIDENCE
    assert result.overall_state == FusedEvidenceState.MIXED_EVIDENCE
    assert len(result.disagreements) > 0

    # Case B: Without facial dynamics (only the disagreeing optical family is present)
    result_optical_alone = engine.fuse(
        modality_verdicts=[verdicts_with_dynamics[0], verdicts_with_dynamics[1]],
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG, ForensicModality.SPATIAL_CONSISTENCY],
    )
    assert result_optical_alone.overall_state == FusedEvidenceState.CONFLICTING_EVIDENCE


def test_04_blink_evidence_alone_when_optical_missing():
    """Requirement 4: Blink evidence alone is descriptive -> INSUFFICIENT_EVIDENCE with preserved measurements."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_facial_dynamics_verdict(blink_rate=14.0, confidence=0.88),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[ForensicModality.FACIAL_DYNAMICS],
    )

    # Blink observations alone must NOT be classified as SUPPORTING_EVIDENCE
    assert result.overall_state == FusedEvidenceState.INSUFFICIENT_EVIDENCE
    assert result.is_evidence_sufficient is False

    # Measurements and diagnostics must be preserved
    assert "facial_dynamics" in result.modality_evidence
    dyn_rec = result.modality_evidence["facial_dynamics"]
    assert dyn_rec.measurements["bilateral_blinks_per_minute"] == 14.0
    assert "facial_dynamics.bilateral_blinks_per_minute" in result.supporting_measurements
    assert result.supporting_measurements["facial_dynamics.bilateral_blinks_per_minute"] == 14.0

    # Reasons explain descriptive nature of blink observations
    assert any("Blink observations alone are descriptive" in r for r in result.evidence_state_reasons)
    assert any("Optical cardiovascular evidence is unavailable" in r for r in result.evidence_state_reasons)


def test_05_optical_evidence_alone_when_blink_missing():
    """Requirement 5: Optical evidence available while blink evidence is missing -> SUPPORTING_EVIDENCE from optical."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(hr_bpm=74.0, snr_db=6.2, confidence=0.90),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG],
    )

    assert result.overall_state == FusedEvidenceState.SUPPORTING_EVIDENCE
    assert result.is_evidence_sufficient is True
    assert result.family_summaries["optical_cardiovascular"].quality_status in (
        EvidenceQualityStatus.OPTIMAL,
        EvidenceQualityStatus.ACCEPTABLE,
    )
    assert result.family_summaries["facial_dynamics"].quality_status == EvidenceQualityStatus.UNAVAILABLE
    assert any("Supporting physiological pulse observed within optical cardiovascular family" in r for r in result.evidence_state_reasons)


def test_06_all_modalities_unavailable_or_insufficient():
    """Requirement 6: All modalities unavailable -> INSUFFICIENT_EVIDENCE."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(is_available=False),
        _make_spatial_verdict(is_available=False),
        _make_facial_dynamics_verdict(is_available=False),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[
            ForensicModality.RPPG,
            ForensicModality.SPATIAL_CONSISTENCY,
            ForensicModality.FACIAL_DYNAMICS,
        ],
    )

    assert result.overall_state == FusedEvidenceState.INSUFFICIENT_EVIDENCE
    assert result.is_evidence_sufficient is False
    assert len(result.missing_or_insufficient_modalities) == 3


def test_07_one_modality_exception_handling(sample_video: Path, test_settings: Settings):
    """Requirement 7: One modality throwing an exception (other modalities preserved, error tagged in missing/limitations)."""
    class FaultyExtractor:
        def extract(self, video_path: str, metadata: VideoMetadata, cancel_event=None):
            raise RuntimeError("Synthetic simulated hardware fault in rPPG extraction!")

        def close(self):
            pass

    inspector = VideoInspector(test_settings)
    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        rppg_extractor=FaultyExtractor(),  # Will raise
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS],
    )
    job = manager.submit_job(req)

    # Wait for completion
    for _ in range(50):
        current = manager.get_job(job.job_id)
        if current.status == JobStatus.COMPLETED:
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None

    res = final.result
    # Check that rPPG failed gracefully and recorded the error
    rppg_v = next((v for v in res.modality_verdicts if v.modality == ForensicModality.RPPG), None)
    assert rppg_v is not None
    assert rppg_v.confidence == 0.0
    assert "error" in rppg_v.explanation.lower()

    # Check that facial dynamics was preserved
    dyn_v = next((v for v in res.modality_verdicts if v.modality == ForensicModality.FACIAL_DYNAMICS), None)
    assert dyn_v is not None

    # Check fusion output
    fused: FusedEvidenceResult = res.fused_evidence
    assert fused is not None
    assert "rppg" in fused.missing_or_insufficient_modalities
    manager.shutdown()


def test_08_low_quality_evidence_not_penalized_as_manipulated():
    """Requirement 8: Low-quality evidence not becoming negative authenticity evidence (quality limitation, not is_manipulated=True)."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    # Severely degraded rPPG (low SNR, poor grade)
    degraded_rppg = _make_rppg_verdict(
        hr_bpm=75.0,
        snr_db=-4.5,
        confidence=0.15,
        grade="D",
        explanation="rPPG signal degraded by excessive motion blur and low SNR.",
    )
    # Facial dynamics with uncalibrated state
    degraded_dynamics = _make_facial_dynamics_verdict(
        blink_rate=12.0,
        confidence=0.30,
        calibration_state="uncalibrated",
        obs_duration=1.5,  # Short
        explanation="Eye tracking intermittent due to severe head rotation.",
    )

    result = engine.fuse(
        modality_verdicts=[degraded_rppg, degraded_dynamics],
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS],
    )

    # Must classify as INSUFFICIENT_EVIDENCE
    assert result.overall_state == FusedEvidenceState.INSUFFICIENT_EVIDENCE
    assert result.is_evidence_sufficient is False

    # Limitations must be captured explicitly
    assert len(result.quality_limitations) >= 2

    # Decoupling check: ModalityEvidenceRecord must NOT have authenticity scores
    for rec in result.modality_evidence.values():
        assert rec.authenticity_score is None
        assert rec.is_authenticity_assessed is False


def test_09_contradictory_evidence_producing_mixed_or_conflicting():
    """Requirement 9: Contradictory evidence producing MIXED_EVIDENCE or CONFLICTING_EVIDENCE."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    # Optical channels with conflicting heart rates (70 BPM vs 105 BPM)
    verdicts = [
        _make_rppg_verdict(hr_bpm=70.0, snr_db=6.5, confidence=0.90),
        _make_spatial_verdict(
            global_state="globally_inconsistent",
            confidence=0.88,
            regional_bpms={"forehead": 70.0, "left_cheek": 105.0, "right_cheek": 105.0},
            anomaly_detected=True,
        ),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG, ForensicModality.SPATIAL_CONSISTENCY],
    )

    assert result.overall_state == FusedEvidenceState.CONFLICTING_EVIDENCE
    assert result.is_evidence_sufficient is True
    assert len(result.disagreements) > 0


def test_10_cancellation_before_fusion(sample_video: Path, test_settings: Settings):
    """Requirement 10: Cancellation before fusion (cancel_event.set() before Stage 5)."""
    inspector = VideoInspector(test_settings)
    manager = JobManager(test_settings, inspector)

    req = JobCreationRequest(video_path=str(sample_video))
    job = manager.submit_job(req)

    # Cancel immediately
    success = manager.cancel_job(job.job_id)
    assert success is True

    # Poll for cancelled terminal state
    for _ in range(50):
        current = manager.get_job(job.job_id)
        if current.status == JobStatus.CANCELLED:
            break
        time.sleep(0.1)

    cancelled = manager.get_job(job.job_id)
    assert cancelled.status == JobStatus.CANCELLED
    # Must NOT have completed or created an AnalysisResult
    assert cancelled.result is None
    manager.shutdown()


def test_11_empty_modality_selection():
    """Requirement 11: Empty modality selection (requested_modalities=[] -> INSUFFICIENT_EVIDENCE)."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    result = engine.fuse(
        modality_verdicts=[],
        video_metadata=meta,
        requested_modalities=[],
    )

    assert result.overall_state == FusedEvidenceState.INSUFFICIENT_EVIDENCE
    assert result.is_evidence_sufficient is False
    assert len(result.modality_evidence) == 0
    assert any("Zero biological evidence families achieved acceptable measurement quality" in r for r in result.evidence_state_reasons)


def test_12_no_authenticity_score_or_manipulated_label():
    """Requirement 12: No authenticity score or manipulated label being generated (authenticity_score=None, is_manipulated=None)."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(hr_bpm=72.0, snr_db=6.0, confidence=0.85),
        _make_spatial_verdict(global_state="globally_consistent", confidence=0.80),
        _make_facial_dynamics_verdict(blink_rate=15.0, confidence=0.90),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[
            ForensicModality.RPPG,
            ForensicModality.SPATIAL_CONSISTENCY,
            ForensicModality.FACIAL_DYNAMICS,
        ],
    )

    # Inspect all records in FusedEvidenceResult
    for mod_name, rec in result.modality_evidence.items():
        assert rec.authenticity_score is None, f"Modality {mod_name} must not have authenticity score"
        assert rec.is_authenticity_assessed is False, f"Modality {mod_name} must have is_authenticity_assessed=False"

    # Simulate final AnalysisResult compilation as in JobManager
    assessed_scores = [
        v.authenticity_score
        for v in verdicts
        if v.is_authenticity_assessed and v.authenticity_score is not None
    ]
    assert len(assessed_scores) == 0

    analysis_result = AnalysisResult(
        job_id="job-test-guardrails",
        metadata=meta,
        overall_authenticity_score=None,
        is_manipulated=None,
        modality_verdicts=verdicts,
        execution_time_seconds=1.23,
        summary_explanation=result.explanation,
        fused_evidence=result,
    )

    assert analysis_result.overall_authenticity_score is None
    assert analysis_result.is_manipulated is None
    assert "Forensic Integrity Notice" in analysis_result.summary_explanation


def test_13_atypical_blink_rate_not_interpreted_as_manipulation():
    """Requirement: Unusual or atypical blink rate is preserved as biological variation and never flagged as manipulation."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    # Extremely low blink rate (e.g., 1.5 blinks/min during high visual concentration)
    verdict_low_blink = _make_facial_dynamics_verdict(blink_rate=1.5, confidence=0.85)

    result = engine.fuse(
        modality_verdicts=[verdict_low_blink],
        video_metadata=meta,
        requested_modalities=[ForensicModality.FACIAL_DYNAMICS],
    )

    # Must be INSUFFICIENT_EVIDENCE (not CONFLICTING_EVIDENCE or fake verdict)
    assert result.overall_state == FusedEvidenceState.INSUFFICIENT_EVIDENCE
    assert result.is_evidence_sufficient is False

    dyn_rec = result.modality_evidence["facial_dynamics"]
    assert dyn_rec.authenticity_score is None
    assert dyn_rec.is_authenticity_assessed is False

    dyn_summary = result.family_summaries["facial_dynamics"]
    assert any("atypical blink rate" in f.lower() for f in dyn_summary.family_findings)


def test_14_shared_video_confounders_documented():
    """Requirement: Shared video-level confounders are explicitly documented in family summaries."""
    engine = MultimodalEvidenceFusionEngine()
    meta = _dummy_metadata(duration_seconds=5.0)

    verdicts = [
        _make_rppg_verdict(hr_bpm=72.0, snr_db=6.0, confidence=0.90),
        _make_facial_dynamics_verdict(blink_rate=16.0, confidence=0.90),
    ]

    result = engine.fuse(
        modality_verdicts=verdicts,
        video_metadata=meta,
        requested_modalities=[ForensicModality.RPPG, ForensicModality.FACIAL_DYNAMICS],
    )

    opt_sum = result.family_summaries["optical_cardiovascular"]
    dyn_sum = result.family_summaries["facial_dynamics"]

    # Confounders documented across both families
    assert len(opt_sum.shared_video_confounders) >= 4
    assert "dermal_skin_reflectance" in opt_sum.shared_video_confounders
    assert "subject_rigid_motion" in opt_sum.shared_video_confounders
    assert "video_compression" in opt_sum.shared_video_confounders

    assert len(dyn_sum.shared_video_confounders) >= 4
    assert "landmark_tracking_stability" in dyn_sum.shared_video_confounders
    assert "head_pose_motion" in dyn_sum.shared_video_confounders
    assert "video_compression" in dyn_sum.shared_video_confounders
