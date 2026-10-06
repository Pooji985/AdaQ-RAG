"""Reciprocal Rank Fusion (RRF) for hybrid retrieval candidate merging."""

from copy import deepcopy
from typing import Any

from adaq_rag.retrieval.models import RetrievalResult

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Combine dense vector and BM25 lexical retrieval results using Reciprocal Rank Fusion (RRF).

    RRF formula for each chunk d across retriever rankings R:
        RRF(d) = sum_{r in R} (1 / (rrf_k + rank_r(d)))

    Where rank_r(d) is 1-based rank position. This rank-based method avoids raw score scale
    mismatches between FAISS cosine/L2 similarities and BM25 unbounded scores.

    Args:
        dense_results: Ordered list of RetrievalResult from DenseRetriever (FAISS).
        bm25_results: Ordered list of RetrievalResult from BM25Retriever.
        rrf_k: Smoothing constant to control relative weight of high vs low ranks (default: 60).
        top_k: Optional maximum number of fused candidates to return.

    Returns:
        list[RetrievalResult]: De-duplicated candidates sorted descending by RRF score.

    Raises:
        ValueError: If rrf_k <= 0 or top_k <= 0.
    """
    if rrf_k <= 0:
        raise ValueError(f"RRF constant rrf_k must be positive, got {rrf_k}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    rrf_scores: dict[str, float] = {}
    chunk_dense_info: dict[str, tuple[int, float, RetrievalResult]] = {}
    chunk_bm25_info: dict[str, tuple[int, float, RetrievalResult]] = {}

    # 1. Process Dense retriever rankings
    for rank, res in enumerate(dense_results, start=1):
        chunk_id = res.chunk_id
        chunk_dense_info[chunk_id] = (rank, res.score, res)
        contribution = 1.0 / (rrf_k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

    # 2. Process BM25 retriever rankings
    for rank, res in enumerate(bm25_results, start=1):
        chunk_id = res.chunk_id
        chunk_bm25_info[chunk_id] = (rank, res.score, res)
        contribution = 1.0 / (rrf_k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

    # 3. Build fused results with score attribution and de-duplication
    fused_results: list[RetrievalResult] = []
    for chunk_id, fused_score in rrf_scores.items():
        dense_entry = chunk_dense_info.get(chunk_id)
        bm25_entry = chunk_bm25_info.get(chunk_id)

        # Base item template from whichever retriever saw it first
        base_item = dense_entry[2] if dense_entry is not None else bm25_entry[2]  # type: ignore[index]
        merged_meta: dict[str, Any] = deepcopy(base_item.metadata)

        dense_rank = dense_entry[0] if dense_entry is not None else None
        dense_raw = dense_entry[1] if dense_entry is not None else None
        bm25_rank = bm25_entry[0] if bm25_entry is not None else None
        bm25_raw = bm25_entry[1] if bm25_entry is not None else None

        merged_meta["rrf_dense_rank"] = dense_rank
        merged_meta["rrf_dense_score"] = dense_raw
        merged_meta["rrf_bm25_rank"] = bm25_rank
        merged_meta["rrf_bm25_score"] = bm25_raw
        merged_meta["rrf_score"] = fused_score
        merged_meta["fused_sources"] = [
            src for src, entry in [("dense", dense_entry), ("bm25", bm25_entry)] if entry is not None
        ]

        fused_item = RetrievalResult(
            chunk_id=chunk_id,
            score=fused_score,
            retrieval_method="hybrid_rrf",
            content=base_item.content,
            doc_id=base_item.doc_id,
            doc_title=base_item.doc_title,
            section_title=base_item.section_title,
            section_level=base_item.section_level,
            dense_score=dense_raw,
            bm25_score=bm25_raw,
            rrf_score=fused_score,
            metadata=merged_meta,
        )
        fused_results.append(fused_item)

    # 4. Sort descending by RRF score, breaking ties deterministically by chunk_id
    fused_results.sort(key=lambda item: (-item.score, item.chunk_id))

    if top_k is not None:
        return fused_results[:top_k]

    return fused_results
