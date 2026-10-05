"""Unit tests for Phase 4 retrieval components: embeddings, FAISS vector index, and BM25 index."""

from pathlib import Path
import numpy as np
import pytest

from adaq_rag.chunking.models import Chunk
from adaq_rag.retrieval.bm25_index import BM25Index, tokenize_text
from adaq_rag.retrieval.embeddings import EmbeddingModel
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    UnifiedRetriever,
)
from adaq_rag.retrieval.vector_index import FAISSVectorIndex


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Provide a small collection of fixture chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_001",
            strategy="structure_aware",
            doc_id="linear_models",
            source_url="https://example.com/linear",
            doc_title="Linear Models",
            section_title="Ridge Regression",
            section_level=2,
            content="Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of the coefficients with parameter alpha.",
            token_count=28,
            char_count=155,
            metadata={"category": "supervised"},
        ),
        Chunk(
            chunk_id="chunk_002",
            strategy="structure_aware",
            doc_id="linear_models",
            source_url="https://example.com/linear",
            doc_title="Linear Models",
            section_title="Lasso Regression",
            section_level=2,
            content="The Lasso is a linear model that estimates sparse coefficients. It is useful due to its tendency to prefer solutions with fewer non-zero coefficients.",
            token_count=27,
            char_count=151,
            metadata={"category": "supervised"},
        ),
        Chunk(
            chunk_id="chunk_003",
            strategy="structure_aware",
            doc_id="model_selection",
            source_url="https://example.com/cv",
            doc_title="Cross Validation",
            section_title="KFold Splitting",
            section_level=2,
            content="KFold divides all the samples in k groups of samples, called folds, of equal sizes. The prediction function is evaluated on k-1 folds.",
            token_count=28,
            char_count=138,
            metadata={"category": "model_selection"},
        ),
        Chunk(
            chunk_id="chunk_004",
            strategy="structure_aware",
            doc_id="pipeline_guide",
            source_url="https://example.com/pipeline",
            doc_title="Pipeline Guide",
            section_title="Preventing Data Leakage",
            section_level=2,
            content="Pipeline can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, normalization and classification, avoiding data leakage.",
            token_count=38,
            char_count=233,
            metadata={"category": "best_practices"},
        ),
    ]


def test_tokenize_text() -> None:
    """Verify lexical tokenization and normalization."""
    tokens = tokenize_text("LogisticRegression(solver='lbfgs', max_iter=100)")
    assert "logisticregression" in tokens
    assert "solver" in tokens
    assert "lbfgs" in tokens
    assert "max_iter" in tokens
    assert "100" in tokens


def test_embedding_model_shape_and_consistency() -> None:
    """Verify EmbeddingModel produces consistent normalized embeddings."""
    model = EmbeddingModel()
    assert model.dimension > 0

    text = "Hyperparameter optimization with GridSearchCV"
    vec = model.encode_text(text)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == model.dimension
    # Verify unit L2 norm
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    texts = [text, "Cross validation folds"]
    matrix = model.encode_texts(texts)
    assert matrix.shape == (2, model.dimension)
    for i in range(2):
        assert np.isclose(np.linalg.norm(matrix[i]), 1.0, atol=1e-4)


def test_faiss_vector_index_build_and_search(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Verify building, persisting, loading, and searching FAISSVectorIndex."""
    embedder = EmbeddingModel()
    embeddings, chunk_ids, metadata = embedder.encode_chunks(sample_chunks, show_progress=False)

    index_path = tmp_path / "vector_index.faiss"
    metadata_path = tmp_path / "vector_metadata.json"

    # Build and persist
    index = FAISSVectorIndex.build(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        metadata=metadata,
        index_path=index_path,
        metadata_path=metadata_path,
    )
    assert index.total_vectors == len(sample_chunks)
    assert index.dimension == embedder.dimension

    # Reload from disk
    loaded_index = FAISSVectorIndex.load(index_path, metadata_path)
    assert loaded_index.total_vectors == len(sample_chunks)

    # Search with query vector
    q_vec = embedder.encode_text("preventing test data leakage with pipeline")
    results = loaded_index.search(q_vec, top_k=2)

    assert len(results) == 2
    top_chunk_id, top_score, top_meta = results[0]
    # Data leakage query should rank chunk_004 first
    assert top_chunk_id == "chunk_004"
    assert top_score > 0.3
    assert top_meta["section_title"] == "Preventing Data Leakage"


def test_bm25_index_build_and_search(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Verify BM25 lexical index construction, disk persistence, and exact term retrieval."""
    index_path = tmp_path / "bm25_index.pkl"
    metadata_path = tmp_path / "bm25_metadata.json"

    # Build
    bm25 = BM25Index.build(sample_chunks, index_path, metadata_path)
    assert bm25.total_documents == len(sample_chunks)

    # Load from disk
    loaded_bm25 = BM25Index.load(index_path, metadata_path)
    assert loaded_bm25.total_documents == len(sample_chunks)

    # Keyword search for unique term "alpha"
    results = loaded_bm25.search("alpha penalty coefficients", top_k=2)
    assert len(results) == 2
    top_chunk_id, top_score, top_meta = results[0]
    assert top_chunk_id == "chunk_001"
    assert top_score > 0.0
    assert top_meta["section_title"] == "Ridge Regression"

    # Test top_k limiting
    top1 = loaded_bm25.search("linear model", top_k=1)
    assert len(top1) == 1


def test_unified_retriever_interface(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Verify UnifiedRetriever provides consistent result format for both dense and sparse retrieval."""
    embedder = EmbeddingModel()
    embeddings, chunk_ids, metadata = embedder.encode_chunks(sample_chunks, show_progress=False)

    v_idx_path = tmp_path / "vector_index.faiss"
    v_meta_path = tmp_path / "vector_metadata.json"
    b_idx_path = tmp_path / "bm25_index.pkl"
    b_meta_path = tmp_path / "bm25_metadata.json"

    FAISSVectorIndex.build(embeddings, chunk_ids, metadata, v_idx_path, v_meta_path)
    BM25Index.build(sample_chunks, b_idx_path, b_meta_path)

    dense_retriever = DenseRetriever.load(v_idx_path, v_meta_path, embedding_model=embedder)
    bm25_retriever = BM25Retriever.load(b_idx_path, b_meta_path)
    unified = UnifiedRetriever(dense_retriever=dense_retriever, bm25_retriever=bm25_retriever)

    # Test Dense retrieval result structure
    dense_results = unified.retrieve("KFold groups of samples", method="dense", top_k=2)
    assert len(dense_results) == 2
    for r in dense_results:
        assert isinstance(r, RetrievalResult)
        assert r.retrieval_method == "dense"
        assert r.chunk_id in [c.chunk_id for c in sample_chunks]
        assert isinstance(r.score, float)
        assert r.content != ""
        assert r.doc_id != ""

    # Test BM25 retrieval result structure
    bm25_results = unified.retrieve("KFold groups of samples", method="bm25", top_k=2)
    assert len(bm25_results) == 2
    for r in bm25_results:
        assert isinstance(r, RetrievalResult)
        assert r.retrieval_method == "bm25"
        assert r.chunk_id in [c.chunk_id for c in sample_chunks]
        assert isinstance(r.score, float)

    # Top hit for KFold query should be chunk_003
    assert bm25_results[0].chunk_id == "chunk_003"

    # Test invalid method handling
    with pytest.raises(ValueError, match="Unsupported retrieval method"):
        unified.retrieve("test", method="invalid_method")

    # Test empty query handling
    assert unified.retrieve("", method="dense") == []
    assert unified.retrieve("   ", method="bm25") == []
