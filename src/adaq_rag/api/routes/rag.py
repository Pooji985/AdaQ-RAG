"""FastAPI endpoint for the baseline RAG pipeline."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError
from adaq_rag.rag.models import RAGResponse
from adaq_rag.rag.pipeline import BasicRAGPipeline
from adaq_rag.rag.service import get_rag_pipeline

logger = logging.getLogger("adaq_rag.api.routes.rag")

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Baseline"])


class RAGQueryRequest(BaseModel):
    """Request payload for baseline RAG query endpoint."""

    query: str = Field(..., min_length=1, description="User question about scikit-learn")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Optional number of dense chunks to retrieve")


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the baseline RAG pipeline",
    description="Retrieves top-k dense chunks and generates a grounded response using the configured LLM.",
)
async def query_rag_endpoint(
    payload: RAGQueryRequest,
    pipeline: BasicRAGPipeline = Depends(get_rag_pipeline),
) -> RAGResponse:
    """Execute baseline RAG pipeline for the provided question."""
    try:
        return await pipeline.query_async(question=payload.query, top_k=payload.top_k)
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMGenerationError as exc:
        logger.error("LLM generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {exc}",
        ) from exc
