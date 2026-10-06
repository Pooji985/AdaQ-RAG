"""FastAPI application factory and lifecycle management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from adaq_rag.api.routes.health import router as health_router
from adaq_rag.api.routes.rag import router as rag_router
from adaq_rag.core.config import Settings, get_settings
from adaq_rag.core.logging import setup_logging
from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown events."""
    settings = get_settings()
    logger = setup_logging(settings.log_level)
    logger.info(
        "Starting %s (v%s) in [%s] mode",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a FastAPI application instance."""
    app_settings = settings or get_settings()

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        debug=app_settings.debug,
        lifespan=lifespan,
    )

    # Register routers
    application.include_router(health_router)
    application.include_router(rag_router)

    # Register exception handlers
    @application.exception_handler(LLMConfigurationError)
    async def llm_config_handler(request, exc: LLMConfigurationError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
        )

    @application.exception_handler(LLMGenerationError)
    async def llm_generation_handler(request, exc: LLMGenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc)},
        )

    return application


app = create_app()
