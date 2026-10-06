"""Cross-encoder reranking module for hybrid retrieval evidence."""

from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from typing import Any

from adaq_rag.core.config import get_settings
from adaq_rag.retrieval.models import RetrievalResult

logger = logging.getLogger("adaq_rag.retrieval.reranker")


class BaseReranker(ABC):
    """Abstract base class for retrieval candidate reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates based on cross-attention relevance with the query.

        Args:
            query: User query string.
            candidates: Initial pool of retrieved candidate chunks.
            top_k: Optional maximum number of top reranked chunks to return.

        Returns:
            list[RetrievalResult]: Reranked candidates ordered descending by relevance.
        """


class CrossEncoderReranker(BaseReranker):
    """Production cross-encoder reranker using sentence-transformers CrossEncoder.

    Uses a pre-trained cross-attention transformer (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
    to compute fine-grained (query, passage) relevance logits.
    """

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Hugging Face model identifier (defaults to settings.reranker_model).
            model: Optional pre-instantiated CrossEncoder model (e.g. for testing or custom caching).
            batch_size: Inference batch size (defaults to settings.reranker_batch_size).
            device: Computation device ('cpu', 'cuda', etc.).
        """
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size or settings.reranker_batch_size
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        """Lazy-load the sentence-transformers CrossEncoder model on first inference."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model '{self.model_name}': {e}")
                raise RuntimeError(
                    f"Could not load CrossEncoder model '{self.model_name}': {e}"
                ) from e
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Score (query, chunk_content) pairs and return candidates sorted by relevance score."""
        if not query.strip() or not candidates:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        # Construct (query, passage) text pairs for cross-encoder inference
        pairs = [(query, c.content) for c in candidates]

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"CrossEncoder inference failed on query '{query}': {e}")
            raise RuntimeError(f"CrossEncoder reranking inference failed: {e}") from e

        reranked_results: list[RetrievalResult] = []
        for candidate, raw_score in zip(candidates, scores):
            score_float = float(raw_score)
            updated_meta = deepcopy(candidate.metadata)
            updated_meta["rerank_score"] = score_float

            reranked_item = RetrievalResult(
                chunk_id=candidate.chunk_id,
                score=score_float,
                retrieval_method="hybrid_reranked",
                content=candidate.content,
                doc_id=candidate.doc_id,
                doc_title=candidate.doc_title,
                section_title=candidate.section_title,
                section_level=candidate.section_level,
                dense_score=candidate.dense_score,
                bm25_score=candidate.bm25_score,
                rrf_score=candidate.rrf_score,
                rerank_score=score_float,
                metadata=updated_meta,
            )
            reranked_results.append(reranked_item)

        # Sort descending by cross-encoder score; break ties deterministically by chunk_id
        reranked_results.sort(key=lambda item: (-item.score, item.chunk_id))

        if top_k is not None:
            return reranked_results[:top_k]

        return reranked_results
