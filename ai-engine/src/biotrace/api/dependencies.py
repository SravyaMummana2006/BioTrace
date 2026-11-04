"""FastAPI dependency injection providers."""

from fastapi import Depends

from biotrace.core.config import Settings, get_settings
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector

_video_inspector_instance: VideoInspector = None
_job_manager_instance: JobManager = None


def get_video_inspector(
    settings: Settings = Depends(get_settings),
) -> VideoInspector:
    global _video_inspector_instance
    if _video_inspector_instance is None:
        _video_inspector_instance = VideoInspector(settings)
    return _video_inspector_instance


def get_job_manager(
    settings: Settings = Depends(get_settings),
    inspector: VideoInspector = Depends(get_video_inspector),
) -> JobManager:
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager(settings, inspector)
    return _job_manager_instance
