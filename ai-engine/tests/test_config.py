"""Tests for configuration management."""

from biotrace.core.config import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.app_name == "BIOTrace AI Engine"
    assert settings.port == 8000
    assert ".mp4" in settings.allowed_extensions
    assert settings.worker_threads >= 1


def test_singleton_settings():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
