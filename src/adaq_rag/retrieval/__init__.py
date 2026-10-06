"""Retrieval package for AdaQ-RAG: embeddings, vector indexes, and lexical retrieval."""

from adaq_rag.retrieval.bm25_index import BM25Index, tokenize_text
from adaq_rag.retrieval.embeddings import EmbeddingModel
from adaq_rag.retrieval.fusion import reciprocal_rank_fusion
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from adaq_rag.retrieval.retriever import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    UnifiedRetriever,
)
from adaq_rag.retrieval.vector_index import FAISSVectorIndex

__all__ = [
    "RetrievalResult",
    "EmbeddingModel",
    "FAISSVectorIndex",
    "BM25Index",
    "tokenize_text",
    "BaseRetriever",
    "DenseRetriever",
    "BM25Retriever",
    "UnifiedRetriever",
    "reciprocal_rank_fusion",
    "BaseReranker",
    "CrossEncoderReranker",
    "HybridRetriever",
]
