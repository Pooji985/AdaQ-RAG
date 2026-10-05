"""Chunking evaluation package."""

from adaq_rag.evaluation.chunking.dataset import ChunkEvalQuestion, get_evaluation_questions
from adaq_rag.evaluation.chunking.evaluator import (
    compute_chunk_stats,
    evaluate_information_preservation,
    get_side_by_side_samples,
)
from adaq_rag.evaluation.chunking.report import generate_markdown_report

__all__ = [
    "ChunkEvalQuestion",
    "get_evaluation_questions",
    "compute_chunk_stats",
    "evaluate_information_preservation",
    "get_side_by_side_samples",
    "generate_markdown_report",
]
