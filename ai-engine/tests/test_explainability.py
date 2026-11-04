"""Comprehensive unit and integration test suite for Stage 7 Explainability & Forensic Reporting."""

import json
import os
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage, SignalType
from biotrace.domain.models import (
    AnalysisResult,
    ForensicSignal,
    JobCreationRequest,
    JobProgress,
    ModalityVerdict,
    SignalPoint,
    VideoMetadata,
)
from biotrace.evidence_fusion.domain import (
    EvidenceQualityStatus,
    FusedEvidenceResult,
    FusedEvidenceState,
)
from biotrace.explainability.domain import (
    ForensicReportData,
    GradCamResult,
    GradCamStatus,
    ModalityExplanation,
    TemporalTimeline,
)
from biotrace.explainability.evidence_explainer import ExplainabilityEngine
from biotrace.explainability.physiological_explainer import PhysiologicalExplainer
from biotrace.explainability.provenance import ProvenanceCollector
from biotrace.explainability.report_generator import ForensicReportGenerator
from biotrace.explainability.visual_explainer import GradCamGenerator, VisualExplainer
from biotrace.main import create_app
from biotrace.visual_detection.domain import (
    FaceDetectionStatus,
    FramePrediction,
    ModelAvailabilityStatus,
    VisualDetectionResult,
)
from biotrace.visual_detection.model import LightweightVisualBaseline


@pytest.fixture
def sample_metadata():
    return VideoMetadata(
        file_path="C:\\Users\\SensitiveUser\\Videos\\test_subject.mp4",
        file_name="test_subject.mp4",
        width=1280,
        height=720,
        fps=30.0,
        frame_count=150,
        duration_seconds=5.0,
        codec="h264",
        file_size_bytes=1048576,
    )


@pytest.fixture
def sample_analysis_result(sample_metadata):
    points = [
        SignalPoint(timestamp_seconds=i * 0.033, frame_index=i, value=0.28 + 0.05 * np.sin(i * 0.2))
        for i in range(90)
    ]
    rppg_signal = ForensicSignal(
        modality=ForensicModality.RPPG,
        signal_type=SignalType.RPPG_POS,
        sampling_rate_hz=30.0,
        points=points,
        metadata={
            "algorithm": "POS",
            "quality_grade": "HIGH_QUALITY",
            "heart_rate_bpm": "72.0",
            "snr_db": "4.2",
        },
    )
    rppg_verdict = ModalityVerdict(
        modality=ForensicModality.RPPG,
        authenticity_score=None,
        is_authenticity_assessed=False,
        anomaly_detected=False,
        confidence=0.88,
        raw_signal=rppg_signal,
        explanation="Extracted clean optical pulse waveform.",
    )

    ear_points = [
        SignalPoint(timestamp_seconds=i * 0.033, frame_index=i, value=0.32 if (i < 40 or i > 50) else 0.15)
        for i in range(90)
    ]
    dynamics_signal = ForensicSignal(
        modality=ForensicModality.FACIAL_DYNAMICS,
        signal_type=SignalType.EYE_BLINK_EAR,
        sampling_rate_hz=30.0,
        points=ear_points,
        metadata={
            "calibration_state": "CALIBRATED_STABLE",
            "valid_landmark_frames": 85,
            "total_frames": 90,
            "timing_source": "PTS",
        },
    )
    dynamics_verdict = ModalityVerdict(
        modality=ForensicModality.FACIAL_DYNAMICS,
        authenticity_score=None,
        is_authenticity_assessed=False,
        anomaly_detected=False,
        confidence=0.92,
        raw_signal=dynamics_signal,
        explanation="Facial dynamics tracking established stable baseline.",
    )

    fused = FusedEvidenceResult(
        overall_state=FusedEvidenceState.SUPPORTING_EVIDENCE,
        is_evidence_sufficient=True,
        agreements=["Optical pulse and facial kinematics show continuous physiological consistency."],
        disagreements=[],
        missing_or_insufficient_modalities=[],
        quality_limitations=[],
        explanation="Consolidated evidence exhibits high-quality biological plausibility.",
    )

    vis_preds = [
        FramePrediction(
            frame_index=i,
            timestamp_seconds=i * 0.033,
            raw_logit=0.15,
            raw_score=0.537,
            detection_status=FaceDetectionStatus.SUCCESS if i < 80 else FaceDetectionStatus.NO_FACE_DETECTED,
            face_detected=i < 80,
            is_valid=i < 80,
        )
        for i in range(90)
    ]
    vis_result = VisualDetectionResult(
        model_status=ModelAvailabilityStatus.AVAILABLE,
        model_name="biotrace_visual_baseline",
        model_version="1.0.0",
        backbone="lightweight",
        aggregation_method="TRIMMED_MEAN",
        total_frames_examined=90,
        valid_face_frames=80,
        frames_with_missing_face=10,
        aggregated_raw_score=0.537,
        aggregated_logit=0.15,
        is_calibrated=False,
        calibrated_probability=None,
        frame_predictions=vis_preds,
        temporal_variance=0.012,
        lag1_autocorrelation=0.85,
        effective_sample_size=14.2,
        descriptive_score_range=(0.48, 0.59),
        explanation="Baseline visual artifact evaluation completed with uncalibrated scores.",
    )

    return AnalysisResult(
        job_id="job-explain-unit-01",
        metadata=sample_metadata,
        overall_authenticity_score=None,
        is_manipulated=None,
        modality_verdicts=[rppg_verdict, dynamics_verdict],
        execution_time_seconds=1.45,
        summary_explanation="Multimodal evidence processed successfully.",
        fused_evidence=fused,
        visual_detection=vis_result,
    )


# ============================================================================
# 1. PROVENANCE & SANITIZATION TESTS
# ============================================================================

def test_provenance_sanitization_and_metadata(sample_metadata, sample_analysis_result):
    prov = ProvenanceCollector.collect(
        job_id="job-101",
        video_path=sample_metadata.file_path,
        metadata=sample_metadata,
        visual_result=sample_analysis_result.visual_detection,
        analysis_result=sample_analysis_result,
    )

    # Assure sensitive paths are redacted
    assert prov.sanitized_paths is True
    assert prov.video_identifier == "test_subject.mp4"
    assert "SensitiveUser" not in prov.video_identifier
    assert "C:\\Users" not in prov.video_identifier

    # Check software environment recorded
    assert "biotrace" in prov.software_versions
    assert "python" in prov.software_versions
    assert "torch" in prov.software_versions
    assert "opencv" in prov.software_versions

    # Check seeds & calibration
    assert "torch_initial_seed" in prov.random_seeds
    assert prov.calibration_status == "UNCALIBRATED_BASELINE"
    assert len(prov.unavailable_metadata) > 0


# ============================================================================
# 2. MODALITY EXPLANATIONS & CAVEATS TESTS
# ============================================================================

def test_missing_modality_explanations():
    exp_rppg = PhysiologicalExplainer.explain_rppg(None)
    assert exp_rppg.quality_confidence == 0.0
    assert len(exp_rppg.usable_observations) == 0
    assert "not requested" in exp_rppg.unreliable_or_unavailable_observations[0]

    exp_dyn = PhysiologicalExplainer.explain_facial_dynamics(None)
    assert exp_dyn.quality_confidence == 0.0
    assert "not requested" in exp_dyn.unreliable_or_unavailable_observations[0]

    exp_vis = VisualExplainer.explain(None)
    assert exp_vis.quality_confidence == 0.0
    assert "not requested" in exp_vis.unreliable_or_unavailable_observations[0]


def test_missing_model_weights_explanation():
    vis_unavail = VisualDetectionResult(
        model_status=ModelAvailabilityStatus.MODEL_UNAVAILABLE,
        model_name="biotrace_visual_baseline",
        model_version="1.0.0",
        backbone="lightweight",
        aggregation_method="TRIMMED_MEAN",
        explanation="Model weights missing.",
    )
    exp = VisualExplainer.explain(vis_unavail)
    assert exp.quality_confidence == 0.0
    assert "MODEL_UNAVAILABLE" in exp.unreliable_or_unavailable_observations[0]
    assert "synthetic" in exp.scientific_caveats[1].lower()


def test_uncalibrated_visual_score_caveats(sample_analysis_result):
    exp = VisualExplainer.explain(sample_analysis_result.visual_detection)
    assert exp.quality_confidence > 0.0
    assert any("uncalibrated heuristic" in c.lower() for c in exp.scientific_caveats)
    assert any("cannot assert video manipulation" in u.lower() for u in exp.unsupported_conclusions)


# ============================================================================
# 3. GRAD-CAM TESTS (RESPONSIBLE SENSITIVITY VISUALIZATION)
# ============================================================================

def test_gradcam_missing_weights():
    # If weights are missing, no fake heatmaps should be fabricated
    res = GradCamGenerator.generate(
        model=None,
        input_tensor=None,
        original_bgr=None,
        model_status=ModelAvailabilityStatus.MODEL_UNAVAILABLE,
    )
    assert res.status == GradCamStatus.UNAVAILABLE_WEIGHTS_MISSING
    assert res.raw_heatmap is None
    assert res.heatmap_png_base64 is None
    assert "strictly prohibited" in res.interpretation_notice


def test_gradcam_unsupported_architecture():
    class PureMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    model = PureMlp()
    dummy_tensor = torch.randn(1, 10)
    dummy_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    res = GradCamGenerator.generate(
        model=model,
        input_tensor=dummy_tensor,
        original_bgr=dummy_bgr,
        model_status=ModelAvailabilityStatus.AVAILABLE,
    )
    assert res.status == GradCamStatus.UNSUPPORTED_ARCHITECTURE
    assert "No suitable convolutional" in res.error_message


def test_gradcam_lightweight_baseline():
    model = LightweightVisualBaseline()
    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

    res = GradCamGenerator.generate(
        model=model,
        input_tensor=dummy_input,
        original_bgr=dummy_bgr,
        model_status=ModelAvailabilityStatus.AVAILABLE,
    )
    assert res.status == GradCamStatus.AVAILABLE
    assert res.target_layer is not None
    assert res.raw_heatmap is not None
    assert len(res.raw_heatmap) == 28
    assert res.heatmap_png_base64 is not None
    assert res.overlay_png_base64 is not None
    assert "sensitivity" in res.interpretation_notice.lower()


# ============================================================================
# 4. TEMPORAL EVIDENCE EXPLORER & GAPS
# ============================================================================

def test_temporal_timeline_with_tracking_gaps(sample_metadata, sample_analysis_result):
    timeline = ExplainabilityEngine.build_temporal_timeline(
        metadata=sample_metadata,
        modality_verdicts=sample_analysis_result.modality_verdicts,
        visual_result=sample_analysis_result.visual_detection,
    )
    assert isinstance(timeline, TemporalTimeline)
    assert len(timeline.points) > 0
    assert timeline.valid_frames_ratio > 0.0

    # Verification of gap tracking at frame indices >= 80 where face_status == NO_FACE_DETECTED
    assert len(timeline.gaps) > 0
    first_gap = timeline.gaps[0]
    assert first_gap.duration_seconds > 0.0
    assert "NO_FACE_DETECTED" in first_gap.reason


# ============================================================================
# 5. FORENSIC REPORT GENERATION (JSON & VECTOR PDF)
# ============================================================================

def test_forensic_report_data_assembly(sample_analysis_result):
    report_data = ExplainabilityEngine.generate_report_data(sample_analysis_result)
    assert isinstance(report_data, ForensicReportData)
    assert report_data.job_id == sample_analysis_result.job_id
    assert report_data.report_id.startswith("REP-")
    assert "SUPPORTING_EVIDENCE" in report_data.executive_summary
    assert len(report_data.scientific_limitations) >= 4


def test_json_report_serialization(sample_analysis_result):
    report_data = ExplainabilityEngine.generate_report_data(sample_analysis_result)
    json_str = ForensicReportGenerator.generate_json(report_data)

    parsed = json.loads(json_str)
    assert parsed["job_id"] == sample_analysis_result.job_id
    assert parsed["provenance"]["sanitized_paths"] is True
    assert "rppg" in parsed["modality_explanations"]


def test_pdf_report_generation(sample_analysis_result):
    report_data = ExplainabilityEngine.generate_report_data(sample_analysis_result)
    pdf_bytes = ForensicReportGenerator.generate_pdf(report_data)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 20000  # Multi-page vector PDF is non-trivial size
    assert pdf_bytes.startswith(b"%PDF-")


# ============================================================================
# 6. FASTAPI API ROUTE INTEGRATION TESTS
# ============================================================================

def test_api_explanation_and_report_endpoints(sample_analysis_result):
    app = create_app()
    client = TestClient(app)

    # Inject mock completed job directly into JobManager
    from datetime import datetime
    from biotrace.api.dependencies import get_job_manager
    jm = app.dependency_overrides.get(get_job_manager, None)
    if jm is None:
        from biotrace.core.config import get_settings
        from biotrace.services.video_inspector import VideoInspector
        from biotrace.services.job_manager import JobManager
        settings = get_settings()
        jm = JobManager(settings, VideoInspector(settings))
        app.dependency_overrides[get_job_manager] = lambda: jm

    job_id = sample_analysis_result.job_id
    progress = JobProgress(
        job_id=job_id,
        video_path=sample_analysis_result.metadata.file_path,
        status=JobStatus.COMPLETED,
        stage=ProcessingStage.COMPLETED,
        progress_percentage=100.0,
        created_at=datetime.utcnow(),
        result=sample_analysis_result,
    )
    with jm._lock:
        jm._jobs[job_id] = progress

    # 1. GET /explanation
    resp_exp = client.get(f"/api/v1/jobs/{job_id}/explanation")
    assert resp_exp.status_code == 200
    exp_json = resp_exp.json()
    assert exp_json["job_id"] == job_id
    assert "evidence_synthesis" in exp_json

    # 2. GET /report/json
    resp_json = client.get(f"/api/v1/jobs/{job_id}/report/json")
    assert resp_json.status_code == 200
    assert "application/json" in resp_json.headers["content-type"]
    assert "attachment" in resp_json.headers["content-disposition"]

    # 3. GET /report/pdf
    resp_pdf = client.get(f"/api/v1/jobs/{job_id}/report/pdf")
    assert resp_pdf.status_code == 200
    assert resp_pdf.headers["content-type"] == "application/pdf"
    assert resp_pdf.content.startswith(b"%PDF-")

    # 4. GET /visual-explanation
    resp_vis = client.get(f"/api/v1/jobs/{job_id}/visual-explanation")
    assert resp_vis.status_code == 200
    vis_data = resp_vis.json()
    assert "status" in vis_data
    assert "interpretation_notice" in vis_data

    # 5. Error handling for non-existent job
    resp_404 = client.get("/api/v1/jobs/job-unknown-999/explanation")
    assert resp_404.status_code == 404

    # 6. Explicit rejection of FAILED job
    failed_job_id = "job-failed-001"
    jm._jobs[failed_job_id] = JobProgress(
        job_id=failed_job_id,
        video_path=sample_analysis_result.metadata.file_path,
        status=JobStatus.FAILED,
        stage=ProcessingStage.QUEUED,
        progress_percentage=45.0,
        created_at=datetime.utcnow(),
        error_message="Video decoding corrupted at frame 45",
    )
    resp_failed_exp = client.get(f"/api/v1/jobs/{failed_job_id}/explanation")
    assert resp_failed_exp.status_code == 400
    assert "failed" in resp_failed_exp.json()["detail"].lower()

    resp_failed_json = client.get(f"/api/v1/jobs/{failed_job_id}/report/json")
    assert resp_failed_json.status_code == 400

    resp_failed_pdf = client.get(f"/api/v1/jobs/{failed_job_id}/report/pdf")
    assert resp_failed_pdf.status_code == 400

    # 7. Explicit rejection of CANCELLED job
    cancelled_job_id = "job-cancelled-001"
    jm._jobs[cancelled_job_id] = JobProgress(
        job_id=cancelled_job_id,
        video_path=sample_analysis_result.metadata.file_path,
        status=JobStatus.CANCELLED,
        stage=ProcessingStage.QUEUED,
        progress_percentage=20.0,
        created_at=datetime.utcnow(),
    )
    resp_canc_exp = client.get(f"/api/v1/jobs/{cancelled_job_id}/explanation")
    assert resp_canc_exp.status_code == 400
    assert "cancelled" in resp_canc_exp.json()["detail"].lower()

    # 8. Explicit rejection of IN-PROGRESS job
    running_job_id = "job-running-001"
    jm._jobs[running_job_id] = JobProgress(
        job_id=running_job_id,
        video_path=sample_analysis_result.metadata.file_path,
        status=JobStatus.RUNNING,
        stage=ProcessingStage.EXTRACTING_RPPG,
        progress_percentage=60.0,
        created_at=datetime.utcnow(),
    )
    resp_run_exp = client.get(f"/api/v1/jobs/{running_job_id}/explanation")
    assert resp_run_exp.status_code == 400
    assert "not yet completed" in resp_run_exp.json()["detail"].lower()


# ============================================================================
# 7. STAGE 7.1 AUDIT SPECIFIC TESTS (HEADLESS PDF, CONCURRENCY, EMPTY/FAILED)
# ============================================================================

def test_headless_pdf_generation_concurrency_and_cleanup(sample_analysis_result):
    """Verifies that PDF generation operates in headless mode and is thread-safe."""
    import concurrent.futures
    import matplotlib.pyplot as plt

    report_data = ExplainabilityEngine.generate_report_data(sample_analysis_result)

    def worker():
        return ForensicReportGenerator.generate_pdf(report_data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker) for _ in range(3)]
        results = [f.result() for f in futures]

    for pdf_bytes in results:
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 20000
        assert pdf_bytes.startswith(b"%PDF-")

    # Assert no dangling figure references remained in global pyplot state
    assert len(plt.get_fignums()) == 0


def test_empty_results_and_missing_modalities_integrity(sample_metadata):
    """Verifies that completely empty results or missing modalities do not invent measurements."""
    empty_result = AnalysisResult(
        job_id="job-empty-001",
        metadata=sample_metadata,
        overall_authenticity_score=None,
        is_manipulated=None,
        modality_verdicts=[],
        execution_time_seconds=0.10,
        summary_explanation="No modalities executed.",
        fused_evidence=None,
        visual_detection=None,
    )

    report_data = ExplainabilityEngine.generate_report_data(empty_result)
    assert report_data.job_id == "job-empty-001"
    assert report_data.evidence_synthesis.is_sufficient is False

    # Check all modalities report NOT_REQUESTED with 0.0 confidence
    for mod_name, exp in report_data.modality_explanations.items():
        assert exp.quality_confidence == 0.0
        assert len(exp.usable_observations) == 0

    # Ensure JSON and PDF generation work cleanly without crashing on empty signals
    json_str = ForensicReportGenerator.generate_json(report_data)
    assert "job-empty-001" in json_str

    pdf_bytes = ForensicReportGenerator.generate_pdf(report_data)
    assert pdf_bytes.startswith(b"%PDF-")


def test_gradcam_lifecycle_and_eval_mode_restoration():
    """Verifies that Grad-CAM restores model training mode and cleans hooks."""
    model = LightweightVisualBaseline()
    model.train(True)  # Start in training mode
    assert model.training is True

    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_bgr = np.zeros((224, 224, 3), dtype=np.uint8)

    res = GradCamGenerator.generate(
        model=model,
        input_tensor=dummy_input,
        original_bgr=dummy_bgr,
        model_status=ModelAvailabilityStatus.AVAILABLE,
    )

    assert res.status == GradCamStatus.AVAILABLE
    # Verify training mode was restored to True
    assert model.training is True
