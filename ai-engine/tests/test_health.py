"""Tests for health and diagnostic endpoint."""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime_seconds" in data
    assert "python_version" in data
    assert "torch_version" in data
    assert "cuda_available" in data
    assert "supported_modalities" in data
    assert "rppg" in data["supported_modalities"]
    assert "blink" in data["supported_modalities"]
