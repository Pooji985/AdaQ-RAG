"""AdaQ-RAG RAG evaluation foundation package.

Exports:
- Typed evaluation models (AnswerQualityScores, RetrievalMetrics, EfficiencyMetrics,
  RoutingEvaluation, QuestionEvaluationResult)
- Ranking and routing metrics (recall_at_k, reciprocal_rank, mean_reciprocal_rank,
  ndcg_at_k, compute_routing_accuracy, compute_retrieval_metrics)
- Provider-independent LLM judging interface (BaseJudge, LLMJudge, MockJudge)
- Efficiency utilities (EfficiencyTracker, extract_efficiency_from_response,
  aggregate_efficiency_metrics)
- Benchmark runner and harness (EvaluationHarness, EvaluationRunner, run_evaluation,
  load_benchmark, filter_benchmark, adapt_evidence_to_retrieval_results, SUPPORTED_SYSTEMS)
"""

from evaluation.rag.efficiency import (
    EfficiencyTracker,
    aggregate_efficiency_metrics,
    extract_efficiency_from_response,
)
from evaluation.rag.judge import BaseJudge, LLMJudge, MockJudge, format_evidence
from evaluation.rag.metrics import (
    compute_retrieval_metrics,
    compute_routing_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.rag.models import (
    AnswerQualityScores,
    EfficiencyMetrics,
    QuestionEvaluationResult,
    RetrievalMetrics,
    RoutingEvaluation,
    interpret_score,
)
from evaluation.rag.runner import (
    SUPPORTED_SYSTEMS,
    SYSTEM_ADAPTIVE_RAG,
    SYSTEM_BASIC_RAG,
    SYSTEM_FULL_ADAQ_RAG,
    SYSTEM_HYBRID_RAG,
    EvaluationHarness,
    EvaluationRunner,
    adapt_evidence_to_retrieval_results,
    filter_benchmark,
    load_benchmark,
    normalize_system_name,
    run_evaluation,
)

__all__ = [
    # Models
    "AnswerQualityScores",
    "RetrievalMetrics",
    "EfficiencyMetrics",
    "RoutingEvaluation",
    "QuestionEvaluationResult",
    "interpret_score",
    # Metrics
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "compute_routing_accuracy",
    "compute_retrieval_metrics",
    # Judge
    "BaseJudge",
    "LLMJudge",
    "MockJudge",
    "format_evidence",
    # Efficiency
    "EfficiencyTracker",
    "extract_efficiency_from_response",
    "aggregate_efficiency_metrics",
    # Runner & Harness
    "EvaluationHarness",
    "EvaluationRunner",
    "run_evaluation",
    "load_benchmark",
    "filter_benchmark",
    "adapt_evidence_to_retrieval_results",
    "normalize_system_name",
    "SUPPORTED_SYSTEMS",
    "SYSTEM_BASIC_RAG",
    "SYSTEM_HYBRID_RAG",
    "SYSTEM_ADAPTIVE_RAG",
    "SYSTEM_FULL_ADAQ_RAG",
]
