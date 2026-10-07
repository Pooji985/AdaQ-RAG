"""Information retrieval and routing metrics for RAG evaluation.

Implements standard ranking and classification metrics:
- Recall@K
- Mean Reciprocal Rank (MRR) / Reciprocal Rank (RR)
- Normalized Discounted Cumulative Gain at K (nDCG@K)
- Routing accuracy

All metrics strictly treat ground_truth_chunk_ids as relevance ground truth
without fabricating non-binary relevance grades. Edge cases (empty sets,
k <= 0, duplicates) are explicitly handled.
"""

import math
from typing import Sequence

from evaluation.rag.models import RetrievalMetrics


def recall_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str],
    k: int,
) -> float:
    """Compute Recall@K against ground truth relevant chunk IDs.

    Recall@K measures the proportion of relevant chunks retrieved in top K.
    Formula: |retrieved[:k] ∩ ground_truth| / |ground_truth|

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        ground_truth: Set/list of gold relevant chunk IDs.
        k: Retrieval rank cutoff.

    Returns:
        Recall@K score in range [0.0, 1.0]. Returns 0.0 for empty ground truth
        or invalid k (<= 0).
    """
    if k <= 0:
        return 0.0
    if not retrieved or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    top_k = retrieved[:k]
    # Unique hits in top k to prevent duplicate chunks from artificially inflating recall
    hits = set(top_k) & gt_set
    return len(hits) / len(gt_set)


def reciprocal_rank(
    retrieved: Sequence[str],
    ground_truth: Sequence[str],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR is 1 / rank of the first relevant chunk retrieved.
    If no relevant chunk is retrieved (or none within rank k), RR is 0.0.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        ground_truth: Set/list of gold relevant chunk IDs.
        k: Optional cutoff. If set and > 0, chunks beyond rank k are ignored.

    Returns:
        Reciprocal rank in range [0.0, 1.0]. Returns 0.0 if no hit.
    """
    if k is not None and k <= 0:
        return 0.0
    if not retrieved or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    candidates = retrieved[:k] if k is not None and k > 0 else retrieved
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in gt_set:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    queries_retrieved: Sequence[Sequence[str]],
    queries_ground_truth: Sequence[Sequence[str]],
    k: int | None = None,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries.

    Formula: (1 / |Q|) * sum_{q in Q} RR(q)

    Args:
        queries_retrieved: List of retrieved chunk lists for each query.
        queries_ground_truth: List of ground truth chunk lists for each query.
        k: Optional cutoff rank.

    Returns:
        MRR score in range [0.0, 1.0]. Returns 0.0 if empty queries list.

    Raises:
        ValueError: If queries_retrieved and queries_ground_truth have different lengths.
    """
    if len(queries_retrieved) != len(queries_ground_truth):
        raise ValueError(
            f"Length mismatch: {len(queries_retrieved)} retrieved vs "
            f"{len(queries_ground_truth)} ground truth entries"
        )
    if not queries_retrieved:
        return 0.0

    total_rr = sum(
        reciprocal_rank(retrieved, gt, k=k)
        for retrieved, gt in zip(queries_retrieved, queries_ground_truth)
    )
    return total_rr / len(queries_retrieved)


def ndcg_at_k(
    retrieved: Sequence[str],
    ground_truth: Sequence[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff K (nDCG@K).

    Uses standard binary relevance (rel_i = 1 if retrieved[i-1] in ground_truth else 0).
    DCG@K = sum_{i=1}^k (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(k, |ground_truth|)} (1 / log2(i + 1))
    nDCG@K = DCG@K / IDCG@K

    Duplicate chunk IDs in retrieved results are deduplicated so duplicate
    occurrences cannot inflate DCG.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        ground_truth: Gold relevant chunk IDs.
        k: Cutoff rank.

    Returns:
        nDCG@K score in range [0.0, 1.0]. Returns 0.0 for empty inputs or k <= 0.
    """
    if k <= 0:
        return 0.0
    if not retrieved or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0

    # Ideal DCG: perfect ranking places min(k, |gt|) relevant items in top positions
    ideal_hits = min(k, len(gt_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0

    # Discounted Cumulative Gain
    dcg = 0.0
    seen_relevant: set[str] = set()
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in gt_set and doc_id not in seen_relevant:
            seen_relevant.add(doc_id)
            dcg += 1.0 / math.log2(rank + 1)

    return dcg / idcg


def compute_routing_accuracy(
    predicted: Sequence[str | None],
    expected: Sequence[str | None],
) -> float:
    """Compute classification accuracy for router strategy decisions.

    Accuracy = (number of exact strategy matches) / (total predictions)

    Comparison is case-insensitive and whitespace-stripped.

    Args:
        predicted: Sequence of predicted routing strategies.
        expected: Sequence of ground-truth expected strategies.

    Returns:
        Accuracy score in range [0.0, 1.0]. Returns 0.0 if input is empty.

    Raises:
        ValueError: If predicted and expected sequences have different lengths.
    """
    if len(predicted) != len(expected):
        raise ValueError(
            f"Length mismatch: {len(predicted)} predicted vs {len(expected)} expected"
        )
    if not predicted:
        return 0.0

    matches = sum(
        1
        for p, e in zip(predicted, expected)
        if p is not None
        and e is not None
        and str(p).strip().upper() == str(e).strip().upper()
    )
    return matches / len(predicted)


def compute_retrieval_metrics(
    retrieved: Sequence[str],
    ground_truth: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Compute complete retrieval metric bundle for a question.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        ground_truth: Ground truth relevant chunk IDs.
        k_values: Cutoff values for Recall, Hit, and nDCG (default: 1, 3, 5, 10).

    Returns:
        RetrievalMetrics model instance populated with computed metrics.
    """
    recalls: dict[int, float] = {}
    hits: dict[int, bool] = {}
    ndcgs: dict[int, float] = {}

    for k in k_values:
        rec = recall_at_k(retrieved, ground_truth, k)
        recalls[k] = round(rec, 4)
        hits[k] = rec > 0.0
        ndcgs[k] = round(ndcg_at_k(retrieved, ground_truth, k), 4)

    mrr = round(reciprocal_rank(retrieved, ground_truth), 4)

    return RetrievalMetrics(
        recall_at_k=recalls,
        mrr=mrr,
        ndcg_at_k=ndcgs,
        hit_at_k=hits,
        retrieved_chunk_ids=list(retrieved),
        ground_truth_chunk_ids=list(ground_truth),
    )
