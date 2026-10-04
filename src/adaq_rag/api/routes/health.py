"""Health check endpoint router."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from adaq_rag.core.config import Settings, get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Schema for service health status."""

    status: str = Field(default="ok", description="Operational status of the service")
    app_name: str = Field(description="Application name")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current deployment environment")


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Return application health status and basic runtime metadata."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
