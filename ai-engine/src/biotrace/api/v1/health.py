"""System health and diagnostic status endpoints."""

import sys
import time
from fastapi import APIRouter, Depends
import torch

from biotrace.api.dependencies import get_job_manager
from biotrace.core.config import Settings, get_settings
from biotrace.domain.enums import ForensicModality
from biotrace.domain.models import HealthStatus
from biotrace.services.job_manager import JobManager

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()


@router.get("", response_model=HealthStatus)
def get_health_status(
    settings: Settings = Depends(get_settings),
    job_manager: JobManager = Depends(get_job_manager),
) -> HealthStatus:
    """Returns engine operational status, hardware capabilities, and version info."""
    uptime = round(time.time() - START_TIME, 2)
    cuda_available = torch.cuda.is_available()

    return HealthStatus(
        status="healthy",
        version=settings.version,
        uptime_seconds=uptime,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        active_jobs=job_manager.get_active_job_count(),
        worker_threads=settings.worker_threads,
        supported_modalities=[m.value for m in ForensicModality],
    )
