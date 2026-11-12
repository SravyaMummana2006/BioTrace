"""Tests for Phase 10 Multimodal Forensic Analysis API & Service.

Verifies:
  1. Multimodal system info endpoint (GET /api/v1/multimodal/info)
  2. Input validation (missing file 404, bad format 400)
  3. Real video multimodal verification (Phase 6 + Phase 9 + Phase 10)
  4. Fake video multimodal verification (Phase 6 + Phase 9 + Phase 10)
  5. Scientific integrity: frozen manifest SHA-256 and zero-score fabrication guardrail
"""

from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient
import pytest


def test_multimodal_system_info(client: TestClient):
    """Verifies architectural metadata, model weights, and provenance hash."""
    resp = client.get("/api/v1/multimodal/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "AVAILABLE"
    assert data["manifest_provenance"] == "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"
    assert "visual_model" in data["models"]
    assert "physiological_model" in data["models"]
    assert "fusion_model" in data["models"]
    assert "governance_notice" in data


def test_multimodal_analyze_missing_file(client: TestClient):
    """Verifies that non-existent media paths return HTTP 404."""
    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": "non_existent_video_path.mp4"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_multimodal_analyze_invalid_extension(client: TestClient, tmp_path: Path):
    """Verifies that non-video file formats return HTTP 400."""
    txt_file = tmp_path / "suspect.txt"
    txt_file.write_text("not a video")
    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": str(txt_file)})
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_portrait_video_is_assessed_not_aspect_gated(client: TestClient, tmp_path: Path):
    """A decodable portrait video must not be forced to UNASSESSED by layout alone."""
    path = tmp_path / "portrait.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (216, 480)
    )
    for i in range(60):
        frame = np.zeros((480, 216, 3), dtype=np.uint8)
        frame[:, :, 1] = (i * 4) % 255
        writer.write(frame)
    writer.release()

    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": str(path)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["visual_model"]["non_square_input"] is True
    assert data["visual_model"]["status"] == "COMPLETED"
    assert data["overall_verdict"] in ("REAL", "FAKE")
    assert data["verdict_basis"] == "visual_only"
    assert "aspect" not in data["verdict_reason"].lower()


def test_unreadable_video_returns_concrete_unassessed_reason(client: TestClient, tmp_path: Path):
    """Corrupt media must produce an explicit UNASSESSED reason, not fabricated scores."""
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"\x00\x01\x02\x03\x04")

    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": str(path)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_verdict"] == "UNASSESSED"
    assert data["verdict_basis"] == "unassessed"
    assert data["verdict_reason"]
    assert data["visual_model"]["status"] in ("FAILED", "UNAVAILABLE")
    assert data["visual_model"]["prediction"] == "UNAVAILABLE"
    assert data["multimodal_fusion"]["fusion_score"] is None


def test_multimodal_analyze_real_video(client: TestClient):
    """Verifies end-to-end multimodal verification on authentic HiDF video."""
    real_path = Path("data/extracted/hidf/real/c00024.mp4")
    if not real_path.exists():
        pytest.skip(f"Test video {real_path} not found")

    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": str(real_path)})
    assert resp.status_code == 200
    data = resp.json()

    assert data["overall_verdict"] == "REAL"
    assert data["overall_confidence"] > 0.5
    assert data["telemetry_state"] == "SIGNAL_ACQUIRED"

    # Model 1: Visual
    assert data["visual_model"]["prediction"] == "REAL"
    assert data["visual_model"]["fake_score"] < 0.50
    assert data["visual_model"]["mean_logit_score"] < 0.0

    # Model 2: Physiological
    assert data["physiological_model"]["prediction"] == "REAL"
    assert data["physiological_model"]["fake_score"] < 0.5128

    # Model 3: Multimodal Fusion
    assert data["multimodal_fusion"]["prediction"] == "REAL"
    assert data["multimodal_fusion"]["fusion_score"] < 0.499895

    # Telemetry and synthesis
    assert data["waveform_telemetry"] is not None
    assert len(data["waveform_telemetry"]["points"]) > 0
    assert len(data["evidence_synthesis"]) >= 3
    assert data["provenance"]["frozen_manifest_sha256"] == "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


def test_multimodal_analyze_fake_video(client: TestClient):
    """Verifies end-to-end multimodal verification on manipulated HiDF video."""
    fake_path = Path("data/extracted/hidf/fake/c00653_c13614.mp4")
    if not fake_path.exists():
        pytest.skip(f"Test video {fake_path} not found")

    resp = client.post("/api/v1/multimodal/analyze", json={"video_path": str(fake_path)})
    assert resp.status_code == 200
    data = resp.json()

    assert data["overall_verdict"] == "FAKE"
    assert data["overall_confidence"] > 0.5
    assert data["telemetry_state"] == "SIGNAL_ACQUIRED"

    # Model 1: Visual
    assert data["visual_model"]["prediction"] == "FAKE"
    assert data["visual_model"]["fake_score"] > 0.50
    assert data["visual_model"]["mean_logit_score"] > 0.0

    # Model 2: Physiological
    assert data["physiological_model"]["prediction"] == "FAKE"
    assert data["physiological_model"]["fake_score"] > 0.5128

    # Model 3: Multimodal Fusion
    assert data["multimodal_fusion"]["prediction"] == "FAKE"
    assert data["multimodal_fusion"]["fusion_score"] > 0.499895

    # Waveform telemetry
    assert data["waveform_telemetry"] is not None
    assert len(data["waveform_telemetry"]["points"]) > 0
