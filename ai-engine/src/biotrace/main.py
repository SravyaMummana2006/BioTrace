"""FastAPI Application Entry Point for BIOTrace AI Engine."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import biotrace  # Initialize environment configurations and MPLCONFIGDIR
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from biotrace.api.v1 import baseline, health, jobs, multimodal
from biotrace.core.config import get_settings
from biotrace.core.exceptions import (
    BioTraceBaseException,
    JobCancellationError,
    JobNotFoundError,
    VideoFormatError,
    VideoProcessingError,
)
from biotrace.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger("biotrace.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing %s v%s...", settings.app_name, settings.version)
    settings.ensure_directories()
    yield
    logger.info("Shutting down %s...", settings.app_name)


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="Biological Intelligence & Media Forensics Platform Backend",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for desktop client & local integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global Exception Handlers ---
    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(request: Request, exc: JobNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=exc.to_dict(),
        )

    @app.exception_handler(VideoFormatError)
    async def video_format_handler(request: Request, exc: VideoFormatError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    @app.exception_handler(VideoProcessingError)
    async def video_processing_handler(request: Request, exc: VideoProcessingError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=exc.to_dict(),
        )

    @app.exception_handler(JobCancellationError)
    async def job_cancellation_handler(request: Request, exc: JobCancellationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=exc.to_dict(),
        )

    @app.exception_handler(BioTraceBaseException)
    async def biotrace_base_handler(request: Request, exc: BioTraceBaseException) -> JSONResponse:
        logger.error("Internal BIOTrace exception: %s", exc.message)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=exc.to_dict(),
        )

    # --- Mount Version 1 Routers ---
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(jobs.router, prefix="/api/v1")
    app.include_router(baseline.router, prefix="/api/v1")
    app.include_router(multimodal.router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "biotrace.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
