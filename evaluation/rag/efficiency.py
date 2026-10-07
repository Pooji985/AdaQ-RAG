"""Efficiency tracking and profiling utilities for RAG evaluation.

Supports recording and aggregating:
- End-to-end processing latency (ms)
- Number of retrieval calls
- Number of chunks processed
- Number of routing escalations/fallbacks
- Token usage statistics (LLMUsage, strictly optional; never fabricated)
"""

import time
from typing import Any, Sequence

from adaq_rag.llm.models import LLMUsage
from evaluation.rag.models import EfficiencyMetrics


class EfficiencyTracker:
    """Context manager and recorder for tracking RAG pipeline execution efficiency."""

    def __init__(self) -> None:
        """Initialize empty efficiency recorder."""
        self.latency_ms: float = 0.0
        self.retrieval_calls: int = 0
        self.chunks_processed: int = 0
        self.escalation_count: int = 0
        self.token_usage: LLMUsage | None = None
        self._start_perf: float | None = None

    def start(self) -> "EfficiencyTracker":
        """Start latency timer."""
        self._start_perf = time.perf_counter()
        return self

    def stop(self) -> "EfficiencyTracker":
        """Stop latency timer and compute elapsed latency in milliseconds."""
        if self._start_perf is not None:
            elapsed_s = time.perf_counter() - self._start_perf
            self.latency_ms = round(elapsed_s * 1000.0, 2)
            self._start_perf = None
        return self

    def __enter__(self) -> "EfficiencyTracker":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def record_retrieval(self, chunks_count: int = 0) -> None:
        """Record an index retrieval call and candidate chunks count.

        Args:
            chunks_count: Number of chunks retrieved or processed in this step.
        """
        self.retrieval_calls += 1
        self.chunks_processed += max(0, chunks_count)

    def record_escalation(self, count: int = 1) -> None:
        """Record an escalation / fallback event.

        Args:
            count: Number of escalation steps taken (default: 1).
        """
        self.escalation_count += max(0, count)

    def record_tokens(self, usage: LLMUsage | None) -> None:
        """Record LLM token usage if provided.

        Args:
            usage: Optional LLMUsage instance. If None, token usage remains None.
        """
        if usage is None:
            return

        if self.token_usage is None:
            self.token_usage = LLMUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        else:
            self.token_usage.prompt_tokens += usage.prompt_tokens
            self.token_usage.completion_tokens += usage.completion_tokens
            self.token_usage.total_tokens += usage.total_tokens

    def get_metrics(self) -> EfficiencyMetrics:
        """Return populated EfficiencyMetrics instance."""
        return EfficiencyMetrics(
            latency_ms=self.latency_ms,
            retrieval_calls=self.retrieval_calls,
            chunks_processed=self.chunks_processed,
            escalation_count=self.escalation_count,
            token_usage=self.token_usage,
        )


def extract_efficiency_from_response(
    response: Any,
    default_retrieval_calls: int = 1,
) -> EfficiencyMetrics:
    """Extract efficiency metrics from a RAGResponse or similar pipeline output.

    Token usage is extracted ONLY if explicitly present in response metadata or attributes.
    Never invents token usage or counts.

    Args:
        response: RAGResponse instance or dict containing pipeline execution attributes.
        default_retrieval_calls: Default retrieval count if not specified in metadata.

    Returns:
        EfficiencyMetrics instance.
    """
    if isinstance(response, dict):
        latency_ms = float(response.get("latency_ms", 0.0))
        chunks = response.get("retrieved_chunk_ids") or []
        chunks_count = len(chunks)
        metadata = response.get("metadata") or {}
        retrieval_calls = int(metadata.get("retrieval_calls", default_retrieval_calls))
        escalations = int(metadata.get("escalation_count", 0))
        token_usage = metadata.get("token_usage")
    else:
        latency_ms = float(getattr(response, "latency_ms", 0.0))
        chunks = getattr(response, "retrieved_chunk_ids", []) or []
        chunks_count = len(chunks)
        metadata = getattr(response, "metadata", {}) or {}
        retrieval_calls = int(metadata.get("retrieval_calls", default_retrieval_calls))
        escalations = int(metadata.get("escalation_count", 0))
        token_usage = metadata.get("token_usage")

    parsed_token_usage: LLMUsage | None = None
    if isinstance(token_usage, LLMUsage):
        parsed_token_usage = token_usage
    elif isinstance(token_usage, dict):
        parsed_token_usage = LLMUsage(**token_usage)

    return EfficiencyMetrics(
        latency_ms=latency_ms,
        retrieval_calls=retrieval_calls,
        chunks_processed=chunks_count,
        escalation_count=escalations,
        token_usage=parsed_token_usage,
    )


def aggregate_efficiency_metrics(
    metrics_list: Sequence[EfficiencyMetrics],
) -> dict[str, Any]:
    """Compute aggregate summary statistics across multiple efficiency metric records.

    Args:
        metrics_list: Sequence of EfficiencyMetrics records.

    Returns:
        Summary dict containing mean latency, total retrieval calls, etc.
    """
    if not metrics_list:
        return {
            "count": 0,
            "mean_latency_ms": 0.0,
            "total_retrieval_calls": 0,
            "mean_retrieval_calls": 0.0,
            "total_chunks_processed": 0,
            "mean_chunks_processed": 0.0,
            "total_escalations": 0,
            "total_tokens": None,
        }

    n = len(metrics_list)
    total_latency = sum(m.latency_ms for m in metrics_list)
    total_calls = sum(m.retrieval_calls for m in metrics_list)
    total_chunks = sum(m.chunks_processed for m in metrics_list)
    total_escalations = sum(m.escalation_count for m in metrics_list)

    # Token usage aggregation: only if at least one record has token_usage
    has_tokens = any(m.token_usage is not None for m in metrics_list)
    total_tokens: dict[str, int] | None = None
    if has_tokens:
        total_prompt = sum(
            m.token_usage.prompt_tokens for m in metrics_list if m.token_usage is not None
        )
        total_comp = sum(
            m.token_usage.completion_tokens for m in metrics_list if m.token_usage is not None
        )
        total_all = sum(
            m.token_usage.total_tokens for m in metrics_list if m.token_usage is not None
        )
        total_tokens = {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "total_tokens": total_all,
        }

    return {
        "count": n,
        "mean_latency_ms": round(total_latency / n, 2),
        "total_retrieval_calls": total_calls,
        "mean_retrieval_calls": round(total_calls / n, 2),
        "total_chunks_processed": total_chunks,
        "mean_chunks_processed": round(total_chunks / n, 2),
        "total_escalations": total_escalations,
        "total_tokens": total_tokens,
    }
