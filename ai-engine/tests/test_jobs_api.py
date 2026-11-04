"""Tests for REST job API endpoints."""

from pathlib import Path
import time
from fastapi.testclient import TestClient


def test_submit_job_api(client: TestClient, sample_video: Path):
    response = client.post(
        "/api/v1/jobs",
        json={
            "video_path": str(sample_video),
            "requested_modalities": ["rppg", "blink"],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ("PENDING", "RUNNING")

    # Poll status
    job_id = data["job_id"]
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id


def test_get_nonexistent_job(client: TestClient):
    response = client.get("/api/v1/jobs/non_existent_id")
    assert response.status_code == 404
    data = response.json()
    assert data["error_code"] == "JOB_NOT_FOUND"


def test_cancel_job_api(client: TestClient, sample_video: Path):
    # Submit job
    response = client.post(
        "/api/v1/jobs",
        json={"video_path": str(sample_video)},
    )
    job_id = response.json()["job_id"]

    # Cancel immediately
    cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "CANCELLED"

    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "CANCELLED"


# ==============================================================================
# Stage 3.5: End-to-End API Pipeline Integration Tests
# ==============================================================================


def test_e2e_api_full_pipeline_both_modalities(client: TestClient, sample_video: Path):
    """Verifies end-to-end execution of both modalities via REST API to terminal completion."""
    submit_resp = client.post(
        "/api/v1/jobs",
        json={
            "video_path": str(sample_video),
            "requested_modalities": ["rppg", "facial_dynamics"],
        },
    )
    assert submit_resp.status_code == 201
    job_id = submit_resp.json()["job_id"]

    # Poll until terminal
    for _ in range(80):
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.1)

    final_resp = client.get(f"/api/v1/jobs/{job_id}")
    final_data = final_resp.json()
    assert final_data["status"] == "COMPLETED"
    assert final_data["progress_percentage"] == 100.0
    assert final_data["stage"] == "COMPLETED"

    result = final_data["result"]
    assert result is not None
    # Crucial scientific check: physiological evidence is unassessed for authenticity
    assert result["overall_authenticity_score"] is None
    assert result["is_manipulated"] is None
    assert len(result["modality_verdicts"]) == 2

    rppg_v = next(v for v in result["modality_verdicts"] if v["modality"] == "rppg")
    assert rppg_v["authenticity_score"] is None
    assert rppg_v["is_authenticity_assessed"] is False

    dyn_v = next(v for v in result["modality_verdicts"] if v["modality"] == "facial_dynamics")
    assert dyn_v["authenticity_score"] is None
    assert dyn_v["is_authenticity_assessed"] is False


def test_e2e_api_modality_combinations(client: TestClient, sample_video: Path):
    """Verifies execution of isolated single modalities via REST API."""
    # 1. rPPG only
    resp1 = client.post(
        "/api/v1/jobs",
        json={"video_path": str(sample_video), "requested_modalities": ["rppg"]},
    )
    j1 = resp1.json()["job_id"]
    for _ in range(50):
        d = client.get(f"/api/v1/jobs/{j1}").json()
        if d["status"] == "COMPLETED":
            break
        time.sleep(0.1)
    res1 = client.get(f"/api/v1/jobs/{j1}").json()["result"]
    assert len(res1["modality_verdicts"]) == 1
    assert res1["modality_verdicts"][0]["modality"] == "rppg"

    # 2. Facial dynamics only
    resp2 = client.post(
        "/api/v1/jobs",
        json={"video_path": str(sample_video), "requested_modalities": ["facial_dynamics"]},
    )
    j2 = resp2.json()["job_id"]
    for _ in range(50):
        d = client.get(f"/api/v1/jobs/{j2}").json()
        if d["status"] == "COMPLETED":
            break
        time.sleep(0.1)
    res2 = client.get(f"/api/v1/jobs/{j2}").json()["result"]
    assert len(res2["modality_verdicts"]) == 1
    assert res2["modality_verdicts"][0]["modality"] == "facial_dynamics"


def test_e2e_api_unsupported_modality_handling(client: TestClient, sample_video: Path):
    """Verifies graceful handling when uncalibrated or unsupported modalities are requested."""
    resp = client.post(
        "/api/v1/jobs",
        json={"video_path": str(sample_video), "requested_modalities": ["respiratory"]},
    )
    job_id = resp.json()["job_id"]
    for _ in range(50):
        d = client.get(f"/api/v1/jobs/{job_id}").json()
        if d["status"] == "COMPLETED":
            break
        time.sleep(0.1)
    data = client.get(f"/api/v1/jobs/{job_id}").json()
    assert data["status"] == "COMPLETED"
    assert data["result"]["modality_verdicts"] == []
    assert "unsupported" in data["result"]["summary_explanation"].lower()


def test_e2e_api_invalid_video_error_reporting(client: TestClient, tmp_path: Path):
    """Verifies that an unreadable corrupt video fails gracefully with informative error."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"\x00\x01\x02\x03\x04")

    resp = client.post(
        "/api/v1/jobs",
        json={"video_path": str(corrupt_file), "requested_modalities": ["rppg"]},
    )
    job_id = resp.json()["job_id"]
    for _ in range(50):
        d = client.get(f"/api/v1/jobs/{job_id}").json()
        if d["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)
    data = client.get(f"/api/v1/jobs/{job_id}").json()
    assert data["status"] == "FAILED"
    assert data["error_message"] is not None
    assert len(data["error_message"]) > 0


def test_e2e_api_concurrent_jobs_execution(client: TestClient, sample_video: Path):
    """Verifies concurrent submission and processing of multiple jobs over the API."""
    resp1 = client.post("/api/v1/jobs", json={"video_path": str(sample_video), "requested_modalities": ["rppg"]})
    resp2 = client.post("/api/v1/jobs", json={"video_path": str(sample_video), "requested_modalities": ["facial_dynamics"]})

    j1 = resp1.json()["job_id"]
    j2 = resp2.json()["job_id"]

    for _ in range(80):
        d1 = client.get(f"/api/v1/jobs/{j1}").json()
        d2 = client.get(f"/api/v1/jobs/{j2}").json()
        if d1["status"] == "COMPLETED" and d2["status"] == "COMPLETED":
            break
        time.sleep(0.1)

    final1 = client.get(f"/api/v1/jobs/{j1}").json()
    final2 = client.get(f"/api/v1/jobs/{j2}").json()
    assert final1["status"] == "COMPLETED"
    assert final2["status"] == "COMPLETED"
    assert len(final1["result"]["modality_verdicts"]) == 1
    assert len(final2["result"]["modality_verdicts"]) == 1

