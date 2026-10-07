"""Data models for AdaQ-RAG evaluation foundation.

Defines typed schemas for:
- Retrieval metrics (Recall@K, MRR, nDCG@K)
- Answer-quality scores (0-4 scale: correctness, faithfulness, relevance, overall)
- Efficiency metrics (latency, retrieval calls, chunks, escalation, token usage)
- Routing evaluation (strategy, intent, complexity fidelity)
- Unified per-question evaluation records
"""

from typing import Any
from pydantic import BaseModel, Field, model_validator

from adaq_rag.llm.models import LLMUsage


def interpret_score(score: float) -> str:
    """Map a 0-4 evaluation score to qualitative interpretation band.

    Interpretation bands:
        0: does not satisfy
        0–<1: very poor
        1–<2: partially satisfies
        2–<3: mostly satisfactory
        3–4: strongly satisfies

    Args:
        score: Numerical score between 0.0 and 4.0.

    Returns:
        Qualitative interpretation string.

    Raises:
        ValueError: If score is outside [0.0, 4.0].
    """
    if score < 0.0 or score > 4.0:
        raise ValueError(f"Score {score} is out of valid range [0.0, 4.0]")
    if score == 0.0:
        return "does not satisfy"
    if score < 1.0:
        return "very poor"
    if score < 2.0:
        return "partially satisfies"
    if score < 3.0:
        return "mostly satisfactory"
    return "strongly satisfies"


class AnswerQualityScores(BaseModel):
    """Answer-quality evaluation scores graded on a 0-4 scale.

    Attributes:
        correctness: Factual correctness against gold reference answer (0-4).
        faithfulness: Grounding in retrieved evidence without hallucinations (0-4).
        relevance: Direct relevance to user question (0-4).
        overall: Arithmetic mean of correctness, faithfulness, and relevance (0-4).
        reason: Short rationale/justification for scores.
    """

    correctness: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Factual correctness against reference answer (0 to 4)",
    )
    faithfulness: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Grounding in retrieved context without hallucination (0 to 4)",
    )
    relevance: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Relevance and directness in answering the user question (0 to 4)",
    )
    overall: float = Field(
        default=0.0,
        ge=0.0,
        le=4.0,
        description="Arithmetic mean of correctness, faithfulness, and relevance",
    )
    reason: str = Field(
        default="",
        description="Short rationale justifying the assigned scores",
    )

    @model_validator(mode="before")
    @classmethod
    def precompute_overall(cls, data: Any) -> Any:
        """Calculate overall arithmetic mean before validation if not explicitly provided."""
        if isinstance(data, dict):
            c = data.get("correctness")
            f = data.get("faithfulness")
            r = data.get("relevance")
            if c is not None and f is not None and r is not None:
                # If overall not in dict or None, compute arithmetic mean
                if "overall" not in data or data.get("overall") is None:
                    data["overall"] = round((float(c) + float(f) + float(r)) / 3.0, 4)
        return data

    @model_validator(mode="after")
    def validate_or_finalize_overall(self) -> "AnswerQualityScores":
        """Ensure overall is the arithmetic mean if left at default 0.0."""
        expected_mean = round((self.correctness + self.faithfulness + self.relevance) / 3.0, 4)
        if self.overall == 0.0 and (self.correctness > 0.0 or self.faithfulness > 0.0 or self.relevance > 0.0):
            self.overall = expected_mean
        return self

    @property
    def interpretation(self) -> str:
        """Qualitative interpretation of the overall score."""
        return interpret_score(self.overall)


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics computed against ground truth chunks."""

    recall_at_k: dict[int, float] = Field(
        default_factory=dict,
        description="Recall at cutoff K (e.g. {1: 0.5, 3: 1.0, 5: 1.0})",
    )
    mrr: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reciprocal rank of the first relevant chunk in retrieved list",
    )
    ndcg_at_k: dict[int, float] = Field(
        default_factory=dict,
        description="Normalized Discounted Cumulative Gain at cutoff K",
    )
    hit_at_k: dict[int, bool] = Field(
        default_factory=dict,
        description="Binary hit indicator at cutoff K",
    )
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of retrieved chunk IDs in retrieval order",
    )
    ground_truth_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ground truth relevant chunk IDs for this question",
    )


class EfficiencyMetrics(BaseModel):
    """System efficiency and resource consumption metrics."""

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total end-to-end processing latency in milliseconds",
    )
    retrieval_calls: int = Field(
        default=0,
        ge=0,
        description="Number of retrieval invocations",
    )
    chunks_processed: int = Field(
        default=0,
        ge=0,
        description="Total candidate or retrieved chunks processed",
    )
    escalation_count: int = Field(
        default=0,
        ge=0,
        description="Number of routing escalation or fallback steps triggered",
    )
    token_usage: LLMUsage | None = Field(
        default=None,
        description="Optional LLM token usage if available; None if unavailable",
    )


class RoutingEvaluation(BaseModel):
    """Routing strategy fidelity and query classification evaluation."""

    predicted_strategy: str | None = Field(
        default=None,
        description="Strategy selected by the router (e.g. DENSE, HYBRID, MULTI_STEP)",
    )
    expected_strategy: str | None = Field(
        default=None,
        description="Ground truth expected strategy from benchmark",
    )
    strategy_match: bool = Field(
        default=False,
        description="Whether predicted strategy matches expected strategy",
    )
    predicted_intent: str | None = Field(
        default=None,
        description="Intent classified by query analyzer",
    )
    expected_intent: str | None = Field(
        default=None,
        description="Ground truth expected intent from benchmark",
    )
    intent_match: bool = Field(
        default=False,
        description="Whether predicted intent matches expected intent",
    )
    predicted_complexity: str | None = Field(
        default=None,
        description="Complexity classified by query analyzer",
    )
    expected_complexity: str | None = Field(
        default=None,
        description="Ground truth expected complexity from benchmark",
    )
    complexity_match: bool = Field(
        default=False,
        description="Whether predicted complexity matches expected complexity",
    )
    routing_reason: str = Field(
        default="",
        description="Router decision rationale",
    )


class QuestionEvaluationResult(BaseModel):
    """Per-question evaluation record encapsulating all evaluation dimensions."""

    question_id: str = Field(description="Unique question identifier from benchmark")
    query: str = Field(description="User question text")
    reference_answer: str = Field(description="Gold reference answer")
    generated_answer: str = Field(description="System generated answer")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Retrieved chunk IDs used for generation",
    )
    ground_truth_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ground truth relevant chunk IDs",
    )
    retrieval_metrics: RetrievalMetrics = Field(
        description="Retrieval performance metrics (Recall, MRR, nDCG)",
    )
    answer_scores: AnswerQualityScores = Field(
        description="Answer quality scores judged on 0-4 scale",
    )
    efficiency: EfficiencyMetrics = Field(
        description="Execution efficiency and latency metrics",
    )
    routing: RoutingEvaluation | None = Field(
        default=None,
        description="Optional routing fidelity metrics for adaptive systems",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (system name, timestamp, etc.)",
    )
