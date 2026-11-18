"""FastAPI endpoints for Baseline and Multi-System Adaptive RAG pipelines."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError
from adaq_rag.rag.models import RAGResponse, RoutingInfo
from adaq_rag.rag.pipeline import BasicRAGPipeline
from adaq_rag.rag.service import (
    SYSTEM_BASIC_RAG,
    MultiSystemRAGService,
    get_multi_system_service,
    get_rag_pipeline,
    normalize_system_name,
)

logger = logging.getLogger("adaq_rag.api.routes.rag")

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Systems"])


class RAGQueryRequest(BaseModel):
    """Request payload for RAG query endpoints."""

    query: str = Field(..., min_length=1, description="User question about scikit-learn")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Optional number of chunks to retrieve")
    system: str = Field(
        default=SYSTEM_BASIC_RAG,
        description="RAG system: 'basic_rag', 'hybrid_rag', 'adaptive_rag', or 'full_adaq_rag'",
    )


class QueryAnalysisRequest(BaseModel):
    """Request payload for query intent analysis and routing plan inspection."""

    query: str = Field(..., min_length=1, description="User question to analyze")


class QueryAnalysisResponse(BaseModel):
    """Routing and intent breakdown without running LLM generation."""

    query: str
    intent: str
    complexity: str
    strategy: str
    routing_reason: str
    candidate_top_k: int
    final_top_k: int
    use_reranking: bool


@router.post(
    "/query",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG pipeline",
    description="Executes selected RAG system (basic_rag, hybrid_rag, adaptive_rag, full_adaq_rag) and returns grounded response with provenance and routing info.",
)
async def query_rag_endpoint(
    payload: RAGQueryRequest,
    pipeline: BasicRAGPipeline = Depends(get_rag_pipeline),
    service: MultiSystemRAGService = Depends(get_multi_system_service),
) -> RAGResponse:
    """Execute selected RAG system for the provided question."""
    try:
        canonical_system = normalize_system_name(payload.system)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        # If basic_rag is requested, use pipeline dependency (ensuring override compatibility)
        if canonical_system == SYSTEM_BASIC_RAG:
            resp = await pipeline.query_async(question=payload.query, top_k=payload.top_k)
            resp.system = SYSTEM_BASIC_RAG
            if not resp.routing:
                resp.routing = RoutingInfo(
                    predicted_strategy="DENSE",
                    routing_reason="Fixed dense retrieval baseline.",
                    candidate_top_k=payload.top_k or pipeline.default_top_k,
                    final_top_k=payload.top_k or pipeline.default_top_k,
                )
            return resp

        # Multi-system execution (hybrid_rag, adaptive_rag, full_adaq_rag)
        return await service.query_async(
            question=payload.query,
            system=canonical_system,
            top_k=payload.top_k,
        )

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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {exc}",
        ) from exc


@router.post(
    "/analyze",
    response_model=QueryAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze query intent and routing strategy",
    description="Inspect how the query is classified (intent, complexity) and routed without running retrieval or generation.",
)
def analyze_query_endpoint(
    payload: QueryAnalysisRequest,
    service: MultiSystemRAGService = Depends(get_multi_system_service),
) -> QueryAnalysisResponse:
    """Analyze query and return routing plan."""
    try:
        res = service.analyze_query(payload.query)
        return QueryAnalysisResponse(**res)
    except Exception as exc:
        logger.exception("Failed to analyze query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query analysis failed: {exc}",
        ) from exc


class SystemComparisonItem(BaseModel):
    """Execution output for one RAG system in comparison mode."""

    system: str
    status: str
    result: RAGResponse | None = None
    error: str | None = None


class RAGCompareRequest(BaseModel):
    """Request payload for comparing all 4 RAG systems on the same query."""

    query: str = Field(..., min_length=1, description="User question to compare across systems")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Optional context chunks count")
    systems: list[str] | None = Field(default=None, description="Optional target system subset")


class RAGCompareResponse(BaseModel):
    """Comparative execution response across systems."""

    query: str
    systems: dict[str, SystemComparisonItem]


@router.post(
    "/compare",
    response_model=RAGCompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare all 4 RAG systems on the same query",
    description="Executes the exact same query across Basic, Hybrid, Adaptive, and Full AdaQ-RAG with independent failure isolation.",
)
async def compare_rag_endpoint(
    payload: RAGCompareRequest,
    pipeline: BasicRAGPipeline = Depends(get_rag_pipeline),
    service: MultiSystemRAGService = Depends(get_multi_system_service),
) -> RAGCompareResponse:
    """Execute all selected RAG systems on the same query."""
    from adaq_rag.rag.service import SUPPORTED_SYSTEMS

    target_systems = [
        normalize_system_name(s) for s in (payload.systems or SUPPORTED_SYSTEMS)
    ]
    results: dict[str, SystemComparisonItem] = {}

    for sys_name in target_systems:
        try:
            if sys_name == SYSTEM_BASIC_RAG:
                resp = await pipeline.query_async(question=payload.query, top_k=payload.top_k)
                resp.system = SYSTEM_BASIC_RAG
                resp.retrieval_calls = 1
                if not resp.routing:
                    resp.routing = RoutingInfo(
                        predicted_strategy="DENSE",
                        routing_reason="Fixed dense retrieval baseline.",
                        candidate_top_k=payload.top_k or pipeline.default_top_k,
                        final_top_k=payload.top_k or pipeline.default_top_k,
                    )
                results[sys_name] = SystemComparisonItem(system=sys_name, status="success", result=resp)
            else:
                resp = await service.query_async(question=payload.query, system=sys_name, top_k=payload.top_k)
                results[sys_name] = SystemComparisonItem(system=sys_name, status="success", result=resp)
        except Exception as exc:
            logger.error("Comparison execution failed for system %s: %s", sys_name, exc)
            results[sys_name] = SystemComparisonItem(
                system=sys_name,
                status="error",
                error=str(exc),
            )

    return RAGCompareResponse(query=payload.query, systems=results)


