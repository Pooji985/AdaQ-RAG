"""Hybrid retrieval component combining FAISS dense vector search and BM25 lexical search with RRF and reranking."""

import logging
from pathlib import Path
from typing import Any

from adaq_rag.core.config import get_settings
from adaq_rag.retrieval.embeddings import EmbeddingModel
from adaq_rag.retrieval.fusion import reciprocal_rank_fusion
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from adaq_rag.retrieval.retriever import BaseRetriever, BM25Retriever, DenseRetriever
from adaq_rag.routing.models import RetrievalPlan, RetrievalStrategy

logger = logging.getLogger("adaq_rag.retrieval.hybrid")


class HybridRetriever(BaseRetriever):
    """Hybrid Retriever: Dense FAISS + Sparse BM25 -> RRF Fusion -> Cross-Encoder Reranking.

    Implements the retrieval execution tier requested by Phase 7 RetrievalPlan for MEDIUM queries.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        reranker: BaseReranker | None = None,
        rrf_k: int | None = None,
        default_candidate_top_k: int | None = None,
        default_final_top_k: int | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            dense_retriever: Dense vector retriever instance.
            bm25_retriever: Lexical BM25 retriever instance.
            reranker: Optional cross-encoder reranker instance.
            rrf_k: Smoothing constant for Reciprocal Rank Fusion (defaults to settings.hybrid_rrf_k).
            default_candidate_top_k: Default candidate pool size (defaults to settings.hybrid_candidate_top_k).
            default_final_top_k: Default final top-k returned (defaults to settings.hybrid_final_top_k).
        """
        settings = get_settings()
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k
        self.default_candidate_top_k = (
            default_candidate_top_k if default_candidate_top_k is not None else settings.hybrid_candidate_top_k
        )
        self.default_final_top_k = (
            default_final_top_k if default_final_top_k is not None else settings.hybrid_final_top_k
        )

    @classmethod
    def load(
        cls,
        indexes_dir: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
        reranker: BaseReranker | None = None,
    ) -> "HybridRetriever":
        """Load HybridRetriever from persisted FAISS and BM25 index artifacts.

        Args:
            indexes_dir: Path to directory containing indexes. If None, uses settings.indexes_dir.
            embedding_model: Optional pre-loaded EmbeddingModel.
            reranker: Optional custom or pre-loaded reranker (defaults to CrossEncoderReranker).

        Returns:
            HybridRetriever: Initialized hybrid retriever with both indexes and cross-encoder.
        """
        settings = get_settings()
        base_dir = Path(indexes_dir) if indexes_dir else Path(settings.indexes_dir)

        dense = DenseRetriever.load(
            index_path=base_dir / settings.vector_index_file,
            metadata_path=base_dir / settings.vector_metadata_file,
            embedding_model=embedding_model,
        )
        bm25 = BM25Retriever.load(
            index_path=base_dir / settings.bm25_index_file,
            metadata_path=base_dir / settings.bm25_metadata_file,
        )
        active_reranker = reranker or CrossEncoderReranker()
        return cls(dense_retriever=dense, bm25_retriever=bm25, reranker=active_reranker)

    def retrieve(
        self,
        query: str,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
        use_reranking: bool = True,
    ) -> list[RetrievalResult]:
        """Execute hybrid search: dense + BM25 -> RRF -> optional cross-encoder reranking.

        Args:
            query: Query text.
            candidate_top_k: Number of candidates to retrieve from each individual retriever.
            final_top_k: Number of top chunks to return after fusion and reranking.
            use_reranking: Whether to apply cross-encoder reranking.

        Returns:
            list[RetrievalResult]: Final ranked evidence chunks.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        c_top_k = candidate_top_k or self.default_candidate_top_k
        f_top_k = final_top_k or self.default_final_top_k

        if c_top_k <= 0 or f_top_k <= 0:
            raise ValueError(f"Candidate and final top_k must be positive, got {c_top_k}, {f_top_k}")

        # 1. Retrieve candidates from Dense FAISS index
        dense_results = self.dense_retriever.retrieve(query, top_k=c_top_k)

        # 2. Retrieve candidates from Sparse BM25 index
        bm25_results = self.bm25_retriever.retrieve(query, top_k=c_top_k)

        # 3. Fuse candidate sets using Reciprocal Rank Fusion (RRF)
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            rrf_k=self.rrf_k,
            top_k=c_top_k,
        )

        if not fused_candidates:
            return []

        # 4. Optional Cross-Encoder Reranking
        if use_reranking:
            if self.reranker is None:
                raise RuntimeError("Reranker is requested (use_reranking=True) but not configured in HybridRetriever.")
            return self.reranker.rerank(query, fused_candidates, top_k=f_top_k)

        return fused_candidates[:f_top_k]

    def retrieve_from_plan(self, plan: RetrievalPlan) -> list[RetrievalResult]:
        """Execute retrieval as specified by a Phase 7 RetrievalPlan.

        Args:
            plan: RetrievalPlan containing strategy, candidate_top_k, final_top_k, and use_reranking.

        Returns:
            list[RetrievalResult]: Final retrieved and ranked results.

        Raises:
            ValueError: If the plan specifies an unsupported strategy for this retriever.
        """
        if not isinstance(plan, RetrievalPlan):
            raise ValueError(f"Expected RetrievalPlan instance, got {type(plan).__name__}")

        if plan.strategy == RetrievalStrategy.HYBRID:
            return self.retrieve(
                query=plan.query,
                candidate_top_k=plan.candidate_top_k,
                final_top_k=plan.final_top_k,
                use_reranking=plan.use_reranking,
            )
        elif plan.strategy == RetrievalStrategy.DENSE:
            # Fallback to pure dense retrieval if plan specifies DENSE
            return self.dense_retriever.retrieve(plan.query, top_k=plan.final_top_k)
        else:
            raise ValueError(
                f"HybridRetriever currently executes HYBRID (and DENSE fallback) plans; "
                f"strategy '{plan.strategy.value}' requires dedicated orchestrator."
            )
