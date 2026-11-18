"""API routes for inspection of evaluation benchmark and pilot results."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger("adaq_rag.api.routes.evaluation")

router = APIRouter(prefix="/api/v1/evaluation", tags=["Evaluation Benchmark"])

DEFAULT_PILOT_RESULTS_PATH = Path("evaluation/rag/pilot_evaluation_results.json")
DEFAULT_BENCHMARK_PATH = Path("evaluation/rag/adaptive_rag_benchmark.json")

SAMPLE_QUESTIONS = [
    {
        "question_id": "rag_fact_01",
        "category": "Simple question",
        "tier": "simple",
        "complexity": "SIMPLE",
        "intent": "FACT_LOOKUP",
        "expected_strategy": "DENSE",
        "query": "What is the default scoring strategy in GridSearchCV if scoring is None?",
        "description": "Direct parameter lookup; single-channel dense retrieval satisfies the query with minimal latency.",
        "routing_reason": "Direct single-concept fact lookup routed to lightweight dense retrieval; compact candidate search (top-5) and 3 final context chunks suffice.",
        "saved_answer": "If `scoring` is set to `None`, GridSearchCV uses the estimator’s own default scoring method (i.e., the estimator’s `score` function) to evaluate the models. This is the default strategy when no explicit scoring is provided. [Source 1]",
        "saved_citations": [
            "api_grid_search_cv_struct_0001 (score: 0.7711)",
            "api_grid_search_cv_struct_0002 (score: 0.6948)",
            "api_grid_search_cv_struct_0005 (score: 0.6120)",
        ],
        "saved_metrics": {
            "overall_quality": 4.0,
            "retrieval_calls": 1,
            "latency_ms": 17080,
            "tested_system": "Adaptive RAG (DENSE)",
        },
    },
    {
        "question_id": "rag_comp_06",
        "category": "Moderate question",
        "tier": "moderate",
        "complexity": "MEDIUM",
        "intent": "COMPARISON",
        "expected_strategy": "HYBRID",
        "query": "What is the difference between macro, micro, and weighted averaging when computing multiclass precision and recall?",
        "description": "Comparative metric inquiry; combines lexical keyword matching and dense semantics with CrossEncoder reranking.",
        "routing_reason": "Entity comparison requires broad hybrid candidate pool (top-20) and reranking (top-6) to ensure balanced evidence across both contrasted concepts.",
        "saved_answer": "• Macro averaging computes metric values per class separately, then takes the unweighted arithmetic mean; all classes contribute equally.\n• Weighted averaging computes per-class metric values, but weights each class by its prevalence (support in true instances).\n• Micro averaging aggregates true positives, false positives, and false negatives globally over all classes first, which equals overall accuracy in multiclass evaluation. [Source 3] [Source 1]",
        "saved_citations": [
            "model_selection_metrics_struct_0028 (score: 0.8124)",
            "model_selection_metrics_struct_0024 (score: 0.7645)",
            "model_selection_metrics_struct_0014 (score: 0.7102)",
        ],
        "saved_metrics": {
            "overall_quality": 3.8,
            "retrieval_calls": 2,
            "latency_ms": 21479,
            "tested_system": "Adaptive RAG (HYBRID)",
        },
    },
    {
        "question_id": "rag_comp_11",
        "category": "Complex question",
        "tier": "complex",
        "complexity": "COMPLEX",
        "intent": "COMPARISON",
        "expected_strategy": "MULTI_STEP",
        "query": "Compare StandardScaler, RobustScaler, and MinMaxScaler in their sensitivity to outliers and resulting output bounds.",
        "description": "Multi-aspect architectural comparison; decomposed into sub-queries with iterative retrieval and evidence sufficiency assessment.",
        "routing_reason": "Complex constrained comparison routed to multi-step retrieval and reranking to contrast multiple estimators under specific dataset conditions (top-25 candidates, top-8 final).",
        "saved_answer": "• Sensitivity to outliers:\n  - StandardScaler: Highly sensitive; mean and variance are distorted by extreme values [Source 1].\n  - RobustScaler: Robust to outliers; centers on median and scales by Interquartile Range (IQR) [Source 2].\n  - MinMaxScaler: Sensitive; extreme values compress inliers into a narrow sub-range.\n\n• Output bounds:\n  - StandardScaler: Unbounded (centered to mean 0, unit variance).\n  - RobustScaler: Unbounded.\n  - MinMaxScaler: Strictly bounded, mapped to default [0, 1] interval.",
        "saved_citations": [
            "api_standard_scaler_struct_0001 (score: 0.8451)",
            "preprocessing_data_transforms_struct_0007 (score: 0.8012)",
            "preprocessing_data_transforms_struct_0001 (score: 0.7890)",
            "preprocessing_data_transforms_struct_0004 (score: 0.7431)",
        ],
        "saved_metrics": {
            "overall_quality": 3.67,
            "retrieval_calls": 16,
            "latency_ms": 63129,
            "tested_system": "Full AdaQ-RAG (MULTI_STEP)",
            "escalations": 1,
        },
    },
]


class SampleQuestionItem(BaseModel):
    """Preset sample question with saved pilot evaluation evidence."""

    question_id: str
    category: str
    tier: str
    complexity: str
    intent: str
    expected_strategy: str
    query: str
    description: str
    routing_reason: str
    saved_answer: str | None = None
    saved_citations: list[str] | None = None
    saved_metrics: dict[str, Any] | None = None


class EvaluationPilotSummaryResponse(BaseModel):
    """Schema for pilot evaluation comparison summary."""

    benchmark_title: str = Field(description="Description of benchmark dataset")
    pilot_scope: str = Field(description="Scope description (e.g. 3-question pilot)")
    total_records: int = Field(description="Total evaluated question-system records")
    system_comparison: dict[str, Any] = Field(description="Aggregated performance metrics per system")
    raw_records: list[dict[str, Any]] = Field(default_factory=list, description="Raw per-question evaluation records")


@router.get(
    "/questions",
    response_model=list[SampleQuestionItem],
    summary="Get preset sample benchmark questions",
    description="Returns representative benchmark questions across Simple, Medium, and Complex tiers for testing.",
)
def get_sample_questions() -> list[SampleQuestionItem]:
    """Return curated sample questions for interactive UI testing."""
    return [SampleQuestionItem(**item) for item in SAMPLE_QUESTIONS]


@router.get(
    "/results",
    response_model=EvaluationPilotSummaryResponse,
    summary="Get pilot evaluation benchmark results",
    description="Returns aggregate metrics and records from the controlled 3-question pilot evaluation across all 4 systems.",
)
def get_pilot_evaluation_results() -> EvaluationPilotSummaryResponse:
    """Read and return pilot evaluation results from disk safely."""
    path = DEFAULT_PILOT_RESULTS_PATH
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pilot evaluation artifact not found at: {path}",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        from evaluation.rag.runner import EvaluationRunner

        results = EvaluationRunner.load_results(path)
        summary = EvaluationRunner.aggregate_results(results)

        return EvaluationPilotSummaryResponse(
            benchmark_title="AdaQ-RAG Scikit-Learn Benchmark",
            pilot_scope="Controlled 3-Question Pilot (Representative of Simple, Medium, and Complex tiers)",
            total_records=len(raw_records),
            system_comparison=summary,
            raw_records=raw_records,
        )
    except Exception as exc:
        logger.error("Failed to load pilot evaluation results: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read pilot evaluation results: {exc}",
        ) from exc
