"""Application configuration management with environment variable overrides."""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Automatically load .env if present
load_dotenv()


class Settings(BaseModel):
    """BIOTrace AI Engine configuration settings."""

    app_name: str = "BIOTrace AI Engine"
    version: str = "0.1.0"
    debug: bool = False

    # Server Binding
    host: str = Field(default_factory=lambda: os.getenv("BIOTRACE_HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("BIOTRACE_PORT", "8000")))

    # Media Constraints
    max_video_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("BIOTRACE_MAX_VIDEO_SIZE_MB", "500"))
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ext.strip()
            for ext in os.getenv(
                "BIOTRACE_ALLOWED_EXTENSIONS", ".mp4,.avi,.mov,.mkv"
            ).split(",")
            if ext.strip()
        ]
    )
    max_analysis_seconds: float = Field(
        default_factory=lambda: float(os.getenv("BIOTRACE_MAX_ANALYSIS_SECONDS", "60.0"))
    )
    default_sampling_fps: float = Field(
        default_factory=lambda: float(os.getenv("BIOTRACE_DEFAULT_SAMPLING_FPS", "30.0"))
    )

    # Worker Pool & Concurrency
    worker_threads: int = Field(
        default_factory=lambda: int(os.getenv("BIOTRACE_WORKER_THREADS", "4"))
    )
    device: str = Field(default_factory=lambda: os.getenv("BIOTRACE_DEVICE", "auto"))

    # Storage Directories
    upload_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("BIOTRACE_UPLOAD_DIR", "data/uploads")
        ).resolve()
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("BIOTRACE_CACHE_DIR", "data/cache")
        ).resolve()
    )

    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("BIOTRACE_LOG_LEVEL", "INFO"))

    def ensure_directories(self) -> None:
        """Creates runtime data directories if they do not exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Returns singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.ensure_directories()
    return _settings_instance
