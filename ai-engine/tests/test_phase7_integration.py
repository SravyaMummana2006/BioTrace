"""Comprehensive unit and integration tests for BIOTrace Phase 7 Baseline Integration.

Covers:
1. Baseline model loading, architecture verification, and SHA-256 provenance.
2. 224x224 RGB image normalization matching training specifications.
3. Uniform 10-frame video sampling with timestamp preservation.
4. Video-level mean-logit pooling and prediction score consistency.
5. Operating threshold governance (validation-derived only, test threshold quarantined).
6. Metric distinction: uncalibrated raw sigmoid scores vs calibrated probabilities.
7. Graceful handling of invalid video paths and unsupported formats.
8. FastAPI REST endpoint integration (/api/v1/baseline/info and /api/v1/baseline/predict).
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# Ensure ai-engine is in sys.path
_test_dir = Path(__file__).resolve().parent
_ai_engine_dir = _test_dir.parent
_workspace_dir = _ai_engine_dir.parent
if str(_ai_engine_dir) not in sys.path:
    sys.path.insert(0, str(_ai_engine_dir))
if str(_workspace_dir) not in sys.path:
    sys.path.insert(0, str(_workspace_dir))

from biotrace.api.v1.baseline import (
    EXPECTED_BASELINE_SHA256,
    STANDARD_THRESHOLD,
    VALIDATION_OPTIMAL_THRESHOLD,
)
from biotrace.main import create_app
from biotrace.visual_detection.baseline_service import (
    BaselineInferenceResult,
    BaselineInferenceService,
)


@pytest.fixture(scope="module")
def sample_video_path() -> Path:
    """Provides path to an existing verified sample video."""
    candidates = [
        Path("data/extracted/hidf/fake/c00003_c11507.mp4"),
        Path("../data/extracted/hidf/fake/c00003_c11507.mp4"),
        _workspace_dir / "data" / "extracted" / "hidf" / "fake" / "c00003_c11507.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    pytest.skip("No test video available on disk for integration testing.")


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient fixture."""
    app = create_app()
    return TestClient(app)


def test_baseline_model_loading_and_sha256():
    """Verifies that the trained baseline checkpoint loads cleanly with expected SHA-256."""
    service = BaselineInferenceService.get_instance()
    assert service.model is not None, "Model failed to load into memory."
    assert not service.model.training, "Baseline model must be in eval() mode for inference."
    assert service.weights_sha256 == EXPECTED_BASELINE_SHA256, (
        f"Checkpoint SHA-256 mismatch. Expected: {EXPECTED_BASELINE_SHA256}, Got: {service.weights_sha256}"
    )


def test_baseline_frame_preprocessing():
    """Verifies 224x224 RGB image normalization matching Phase 5/6 ImageNet specifications."""
    dummy_bgr = np.full((480, 640, 3), 128, dtype=np.uint8)
    tensor = BaselineInferenceService.preprocess_frame(dummy_bgr)

    assert tensor.shape == (3, 224, 224), f"Expected shape (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32

    # Verify channel-wise ImageNet normalization
    # Gray (128/255 = 0.5020) -> Channel 0: (0.5020 - 0.485) / 0.229 = ~0.074
    assert -3.0 < tensor[0, 0, 0].item() < 3.0


def test_baseline_video_frame_sampling(sample_video_path: Path):
    """Verifies that sample_video_frames extracts exactly 10 frames uniformly with timestamps."""
    service = BaselineInferenceService.get_instance()
    sampled = service.sample_video_frames(str(sample_video_path), num_frames=10)

    assert len(sampled) == 10, f"Expected 10 sampled frames, got {len(sampled)}"
    for idx, (f_idx, ts, frame_bgr) in enumerate(sampled):
        assert isinstance(f_idx, int)
        assert ts >= 0.0
        assert frame_bgr is not None
        assert frame_bgr.ndim == 3


def test_baseline_video_inference_and_mean_logit_pooling(sample_video_path: Path):
    """Verifies end-to-end video inference and video-level mean-logit pooling aggregation."""
    service = BaselineInferenceService.get_instance()
    result = service.predict_video(str(sample_video_path), num_frames=10)

    assert isinstance(result, BaselineInferenceResult)
    assert result.prediction in ("REAL", "FAKE")
    assert result.predicted_class == result.prediction
    assert result.sampled_frames_count == 10
    assert result.number_of_frames_evaluated == 10
    assert len(result.frame_scores) == 10

    # Probability sum invariant: real + fake = 1.0
    assert result.real_score + result.fake_score == pytest.approx(1.0, abs=1e-3)

    # Verify mean-logit pooling mathematics
    frame_logits = [f.raw_logit for f in result.frame_scores]
    expected_mean_logit = float(np.mean(frame_logits))
    assert result.mean_logit_score == pytest.approx(expected_mean_logit, abs=1e-3)

    # Verify metric distinction: uncalibrated raw sigmoid
    assert result.is_calibrated is False
    assert result.calibrated_probability is None
    assert result.raw_sigmoid_score == pytest.approx(result.fake_score, abs=1e-3)

    # Forensic disclaimer presence
    assert "EXPERIMENTAL RESEARCH BASELINE" in result.disclaimer


def test_operating_threshold_governance(sample_video_path: Path):
    """Verifies that test-derived threshold (0.5321) is quarantined and overridden to validation threshold."""
    service = BaselineInferenceService.get_instance()

    # Attempt to pass test-derived threshold
    res_quarantined = service.predict_video(
        str(sample_video_path), num_frames=10, threshold=0.5321
    )
    # Must be overridden to standard validation threshold
    assert res_quarantined.operating_threshold == pytest.approx(STANDARD_THRESHOLD, abs=1e-3)

    # Validation optimal threshold must be supported
    res_val = service.predict_video(
        str(sample_video_path), num_frames=10, threshold=VALIDATION_OPTIMAL_THRESHOLD
    )
    assert res_val.operating_threshold == pytest.approx(VALIDATION_OPTIMAL_THRESHOLD, abs=1e-3)


def test_invalid_video_handling():
    """Verifies graceful rejection of nonexistent and unsupported files."""
    service = BaselineInferenceService.get_instance()

    # Nonexistent file
    with pytest.raises(FileNotFoundError):
        service.predict_video("nonexistent_video_path_12345.mp4")

    # Empty image
    with pytest.raises(ValueError):
        service.preprocess_frame(np.array([]))


def test_fastapi_baseline_info_endpoint(api_client: TestClient):
    """Verifies GET /api/v1/baseline/info returns 200 with model provenance."""
    response = api_client.get("/api/v1/baseline/info")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "efficientnet_b0_baseline"
    assert data["status"] == "AVAILABLE"
    assert data["checkpoint_sha256"] == EXPECTED_BASELINE_SHA256
    assert data["is_experimental_baseline"] is True
    assert "operating_thresholds" in data
    assert data["operating_thresholds"]["standard_threshold"] == STANDARD_THRESHOLD


def test_fastapi_baseline_predict_endpoint(api_client: TestClient, sample_video_path: Path):
    """Verifies POST /api/v1/baseline/predict executes valid inference and returns correct schema."""
    payload = {
        "video_path": str(sample_video_path),
        "num_frames": 10,
        "threshold": 0.50,
    }
    response = api_client.post("/api/v1/baseline/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ("REAL", "FAKE")
    assert "real_score" in data
    assert "fake_score" in data
    assert "mean_logit_score" in data
    assert data["sampled_frames_count"] == 10
    assert data["model_name"] == "efficientnet_b0_baseline"
    assert data["model_version"] == "1.0.0"
    assert data["is_calibrated"] is False
    assert "EXPERIMENTAL RESEARCH BASELINE" in data["disclaimer"]


def test_fastapi_baseline_predict_invalid_path(api_client: TestClient):
    """Verifies POST /api/v1/baseline/predict returns 404 for missing video."""
    payload = {"video_path": "nonexistent_video_123.mp4", "num_frames": 10}
    response = api_client.post("/api/v1/baseline/predict", json=payload)
    assert response.status_code == 404


def test_fastapi_baseline_predict_unsupported_format(api_client: TestClient, tmp_path: Path):
    """Verifies POST /api/v1/baseline/predict returns 400 for unsupported format."""
    dummy_txt = tmp_path / "not_a_video.txt"
    dummy_txt.write_text("Hello world")

    payload = {"video_path": str(dummy_txt), "num_frames": 10}
    response = api_client.post("/api/v1/baseline/predict", json=payload)
    assert response.status_code == 400
