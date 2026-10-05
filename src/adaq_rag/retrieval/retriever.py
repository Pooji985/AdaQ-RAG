"""Unified retrieval interfaces for dense and sparse retrieval."""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

from adaq_rag.core.config import get_settings
from adaq_rag.retrieval.bm25_index import BM25Index
from adaq_rag.retrieval.embeddings import EmbeddingModel
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.vector_index import FAISSVectorIndex

logger = logging.getLogger("adaq_rag.retrieval.retriever")


class BaseRetriever(ABC):
    """Abstract base class for chunk retrieval methods."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top_k chunks matching the query.

        Args:
            query: User query text.
            top_k: Number of chunks to retrieve.

        Returns:
            list[RetrievalResult]: Ranked retrieval results.
        """


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using embedding model and FAISS index."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: FAISSVectorIndex,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_index = vector_index

    @classmethod
    def load(
        cls,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> "DenseRetriever":
        """Load DenseRetriever from persisted index and metadata files.

        Args:
            index_path: Path to FAISS binary index. If None, uses configuration default.
            metadata_path: Path to JSON metadata. If None, uses configuration default.
            embedding_model: Optional pre-initialized EmbeddingModel.

        Returns:
            DenseRetriever: Configured dense retriever.
        """
        settings = get_settings()
        idx_dir = Path(settings.indexes_dir)
        idx_p = Path(index_path) if index_path else idx_dir / settings.vector_index_file
        meta_p = Path(metadata_path) if metadata_path else idx_dir / settings.vector_metadata_file

        v_index = FAISSVectorIndex.load(idx_p, meta_p)
        model = embedding_model or EmbeddingModel()
        return cls(embedding_model=model, vector_index=v_index)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Perform dense embedding retrieval for a query."""
        if not query.strip():
            return []

        q_vec = self.embedding_model.encode_text(query)
        matches = self.vector_index.search(q_vec, top_k=top_k)

        results: list[RetrievalResult] = []
        for chunk_id, score, meta in matches:
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    retrieval_method="dense",
                    content=meta.get("content", ""),
                    doc_id=meta.get("doc_id", ""),
                    doc_title=meta.get("doc_title", ""),
                    section_title=meta.get("section_title", ""),
                    section_level=meta.get("section_level", 1),
                    metadata=meta,
                )
            )
        return results


class BM25Retriever(BaseRetriever):
    """Sparse lexical retriever using BM25Okapi."""

    def __init__(self, bm25_index: BM25Index) -> None:
        self.bm25_index = bm25_index

    @classmethod
    def load(
        cls,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> "BM25Retriever":
        """Load BM25Retriever from persisted files.

        Args:
            index_path: Path to pickled BM25 index. If None, uses configuration default.
            metadata_path: Path to JSON metadata. If None, uses configuration default.

        Returns:
            BM25Retriever: Configured BM25 retriever.
        """
        settings = get_settings()
        idx_dir = Path(settings.indexes_dir)
        idx_p = Path(index_path) if index_path else idx_dir / settings.bm25_index_file
        meta_p = Path(metadata_path) if metadata_path else idx_dir / settings.bm25_metadata_file

        bm25_idx = BM25Index.load(idx_p, meta_p)
        return cls(bm25_index=bm25_idx)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Perform BM25 lexical retrieval for a query."""
        if not query.strip():
            return []

        matches = self.bm25_index.search(query, top_k=top_k)

        results: list[RetrievalResult] = []
        for chunk_id, score, meta in matches:
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    retrieval_method="bm25",
                    content=meta.get("content", ""),
                    doc_id=meta.get("doc_id", ""),
                    doc_title=meta.get("doc_title", ""),
                    section_title=meta.get("section_title", ""),
                    section_level=meta.get("section_level", 1),
                    metadata=meta,
                )
            )
        return results


class UnifiedRetriever:
    """Unified access interface for dense vector and BM25 sparse retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    @classmethod
    def load(
        cls,
        indexes_dir: str | Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> "UnifiedRetriever":
        """Load both dense and BM25 retrievers from an indexes directory.

        Args:
            indexes_dir: Directory containing index artifacts. If None, uses config.
            embedding_model: Optional pre-loaded EmbeddingModel.

        Returns:
            UnifiedRetriever: Initialized retriever with both index strategies.
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
        return cls(dense_retriever=dense, bm25_retriever=bm25)

    def retrieve(
        self,
        query: str,
        method: str = "dense",
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Dispatch query retrieval to specified strategy ('dense' or 'bm25').

        Args:
            query: Query string.
            method: Retrieval method ('dense' or 'bm25').
            top_k: Maximum number of chunks to return.

        Returns:
            list[RetrievalResult]: Ranked results matching the query.
        """
        norm_method = method.strip().lower()
        if norm_method == "dense":
            return self.retrieve_dense(query, top_k=top_k)
        elif norm_method == "bm25":
            return self.retrieve_bm25(query, top_k=top_k)
        else:
            raise ValueError(f"Unsupported retrieval method: '{method}'. Choose 'dense' or 'bm25'.")

    def retrieve_dense(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve using dense vector search."""
        if self.dense_retriever is None:
            raise RuntimeError("DenseRetriever is not configured in UnifiedRetriever.")
        return self.dense_retriever.retrieve(query, top_k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve using BM25 lexical search."""
        if self.bm25_retriever is None:
            raise RuntimeError("BM25Retriever is not configured in UnifiedRetriever.")
        return self.bm25_retriever.retrieve(query, top_k=top_k)
