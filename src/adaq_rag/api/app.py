"""FastAPI application factory and lifecycle management."""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse

from adaq_rag.api.routes.evaluation import router as evaluation_router
from adaq_rag.api.routes.health import router as health_router
from adaq_rag.api.routes.rag import query_rag_endpoint, router as rag_router
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

    # Register API routers
    application.include_router(health_router)
    application.include_router(rag_router)
    application.include_router(evaluation_router)

    # Convenient alias for top-level /api/v1/query and /api/v1/compare
    from adaq_rag.api.routes.rag import compare_rag_endpoint

    application.add_api_route(
        "/api/v1/query",
        query_rag_endpoint,
        methods=["POST"],
        summary="Alias for /api/v1/rag/query",
        include_in_schema=False,
    )
    application.add_api_route(
        "/api/v1/compare",
        compare_rag_endpoint,
        methods=["POST"],
        summary="Alias for /api/v1/rag/compare",
        include_in_schema=False,
    )

    # Register frontend static file routes
    frontend_dir = Path("frontend")

    @application.get("/", include_in_schema=False)
    async def serve_index() -> HTMLResponse:
        index_file = frontend_dir / "index.html"
        if index_file.is_file():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html><body>"
                "<h1>AdaQ-RAG API</h1>"
                "<p>API documentation is available at <a href='/docs'>/docs</a></p>"
                "</body></html>"
            )
        )

    @application.get("/style.css", include_in_schema=False)
    async def serve_css() -> Response:
        css_file = frontend_dir / "style.css"
        if css_file.is_file():
            return Response(content=css_file.read_text(encoding="utf-8"), media_type="text/css")
        return Response(content="", media_type="text/css", status_code=404)

    @application.get("/app.js", include_in_schema=False)
    async def serve_js() -> Response:
        js_file = frontend_dir / "app.js"
        if js_file.is_file():
            return Response(content=js_file.read_text(encoding="utf-8"), media_type="application/javascript")
        return Response(content="", media_type="application/javascript", status_code=404)

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
