"""Unit tests for Phase 5: Basic RAG Baseline pipeline, context builder, and API routes."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from adaq_rag.api.app import create_app
from adaq_rag.core.config import Settings
from adaq_rag.llm.exceptions import LLMConfigurationError
from adaq_rag.llm.factory import get_llm_provider
from adaq_rag.llm.mock import MockLLMProvider
from adaq_rag.rag.context_builder import ContextBuilder
from adaq_rag.rag.models import RAGResponse, SourceReference
from adaq_rag.rag.pipeline import BasicRAGPipeline
from adaq_rag.rag.service import get_rag_pipeline
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.retriever import DenseRetriever


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    """Provide dummy retrieval results for pipeline and context builder testing."""
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            score=0.85,
            retrieval_method="dense",
            content="Ridge regression solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.2. Ridge regression and classification",
            section_level=3,
            metadata={"source_url": "https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression"},
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            score=0.72,
            retrieval_method="dense",
            content="Lasso is a linear model that estimates sparse coefficients. It consists of a linear model with an added l1 regularization term.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.3. Lasso",
            section_level=3,
            metadata={"source_url": "https://scikit-learn.org/stable/modules/linear_model.html#lasso"},
        ),
        RetrievalResult(
            chunk_id="chunk_003",
            score=0.61,
            retrieval_method="dense",
            content="Pipeline can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in processing the data.",
            doc_id="pipelines",
            doc_title="Pipelines and composite estimators",
            section_title="8.1. Pipeline: chaining estimators",
            section_level=2,
            metadata={"source_url": "https://scikit-learn.org/stable/modules/compose.html#pipeline"},
        ),
    ]


@pytest.fixture
def mock_dense_retriever(sample_retrieval_results: list[RetrievalResult]) -> DenseRetriever:
    """Create a mock DenseRetriever returning fixture results."""
    retriever = MagicMock(spec=DenseRetriever)

    def _retrieve(query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            return []
        return sample_retrieval_results[:top_k]

    retriever.retrieve.side_effect = _retrieve
    return retriever


def test_context_builder_formatting(sample_retrieval_results: list[RetrievalResult]) -> None:
    """Verify ContextBuilder correctly formats chunks with headers, sources, and metadata."""
    builder = ContextBuilder(snippet_max_chars=80)
    context_str, sources = builder.build_context(sample_retrieval_results)

    assert len(sources) == 3
    assert "[Source 1] (Document: Linear Models | Section: 1.1.2. Ridge regression and classification)" in context_str
    assert "[Source 2] (Document: Linear Models | Section: 1.1.3. Lasso)" in context_str
    assert "[Source 3] (Document: Pipelines and composite estimators | Section: 8.1. Pipeline: chaining estimators)" in context_str

    # Verify source reference fields
    s1: SourceReference = sources[0]
    assert s1.source_index == 1
    assert s1.chunk_id == "chunk_001"
    assert s1.score == 0.85
    assert s1.doc_title == "Linear Models"
    assert s1.section_title == "1.1.2. Ridge regression and classification"
    assert len(s1.content_snippet) > 0


def test_context_builder_empty() -> None:
    """Verify ContextBuilder handles empty retrieval gracefully."""
    builder = ContextBuilder()
    context_str, sources = builder.build_context([])
    assert "No relevant documentation excerpts were found." in context_str
    assert sources == []


def test_mock_llm_provider() -> None:
    """Verify MockLLMProvider operates deterministically and logs call history."""
    provider = MockLLMProvider(default_response="Custom mock answer [Source 1].")
    resp = provider.generate("What is Ridge?", system_prompt="System instructions")

    assert resp.content == "Custom mock answer [Source 1]."
    assert resp.model == "mock-llm-v1"
    assert resp.usage is not None
    assert resp.usage.total_tokens > 0
    assert len(provider.call_history) == 1
    assert provider.call_history[0]["prompt"] == "What is Ridge?"


def test_missing_llm_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that unconfigured LLM providers raise LLMConfigurationError clearly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings_openai = Settings(llm_provider="openai", llm_api_key=None)
    with pytest.raises(LLMConfigurationError, match="OpenAI API key is missing"):
        get_llm_provider(settings_openai)

    settings_gemini = Settings(llm_provider="gemini", llm_api_key=None)
    with pytest.raises(LLMConfigurationError, match="Gemini API key is missing"):
        get_llm_provider(settings_gemini)

    settings_invalid = Settings(llm_provider="invalid_provider")
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        get_llm_provider(settings_invalid)


def test_rag_pipeline_flow(mock_dense_retriever: DenseRetriever) -> None:
    """Verify BasicRAGPipeline executes end-to-end with dense retrieval and mocked LLM."""
    mock_llm = MockLLMProvider(default_response="Ridge regression minimizes penalized sum of squares [Source 1].")
    pipeline = BasicRAGPipeline(
        dense_retriever=mock_dense_retriever,
        llm_provider=mock_llm,
        default_top_k=2,
    )

    query = "How does Ridge regression work?"
    response: RAGResponse = pipeline.query(query)

    assert response.query == query
    assert response.answer == "Ridge regression minimizes penalized sum of squares [Source 1]."
    assert response.retrieval_method == "dense"
    assert response.top_k == 2
    assert len(response.sources) == 2
    assert response.retrieved_chunk_ids == ["chunk_001", "chunk_002"]
    assert response.retrieval_scores == [0.85, 0.72]
    assert response.latency_ms >= 0.0
    assert response.metadata["model"] == "mock-llm-v1"
    assert response.metadata["pipeline_stage"] == "phase_5_baseline"


def test_rag_pipeline_top_k_override(mock_dense_retriever: DenseRetriever) -> None:
    """Verify custom top_k argument overrides pipeline default."""
    mock_llm = MockLLMProvider()
    pipeline = BasicRAGPipeline(
        dense_retriever=mock_dense_retriever,
        llm_provider=mock_llm,
        default_top_k=2,
    )

    response = pipeline.query("Question", top_k=3)
    assert response.top_k == 3
    assert len(response.sources) == 3
    assert response.retrieved_chunk_ids == ["chunk_001", "chunk_002", "chunk_003"]


@pytest.mark.anyio
async def test_rag_pipeline_async(mock_dense_retriever: DenseRetriever) -> None:
    """Verify asynchronous query execution."""
    mock_llm = MockLLMProvider(default_response="Async answer [Source 1].")
    pipeline = BasicRAGPipeline(
        dense_retriever=mock_dense_retriever,
        llm_provider=mock_llm,
        default_top_k=2,
    )

    response = await pipeline.query_async("What is Lasso?")
    assert response.answer == "Async answer [Source 1]."
    assert len(response.sources) == 2


def test_rag_api_endpoint(mock_dense_retriever: DenseRetriever) -> None:
    """Verify POST /api/v1/rag/query endpoint returns expected structure."""
    mock_llm = MockLLMProvider(default_response="API answer from mock [Source 1].")
    test_pipeline = BasicRAGPipeline(
        dense_retriever=mock_dense_retriever,
        llm_provider=mock_llm,
        default_top_k=2,
    )

    app = create_app()
    app.dependency_overrides[get_rag_pipeline] = lambda: test_pipeline

    client = TestClient(app)
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "Explain Ridge and Lasso differences", "top_k": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Explain Ridge and Lasso differences"
    assert data["answer"] == "API answer from mock [Source 1]."
    assert data["retrieval_method"] == "dense"
    assert data["top_k"] == 2
    assert len(data["sources"]) == 2
    assert data["sources"][0]["chunk_id"] == "chunk_001"
    assert data["sources"][0]["score"] == 0.85
    assert "latency_ms" in data


def test_rag_api_validation(mock_dense_retriever: DenseRetriever) -> None:
    """Verify validation on empty query or invalid top_k."""
    mock_llm = MockLLMProvider()
    test_pipeline = BasicRAGPipeline(
        dense_retriever=mock_dense_retriever,
        llm_provider=mock_llm,
        default_top_k=2,
    )

    app = create_app()
    app.dependency_overrides[get_rag_pipeline] = lambda: test_pipeline
    client = TestClient(app)

    # Empty query string
    r1 = client.post("/api/v1/rag/query", json={"query": "", "top_k": 3})
    assert r1.status_code == 422

    # Negative or zero top_k
    r2 = client.post("/api/v1/rag/query", json={"query": "valid query", "top_k": 0})
    assert r2.status_code == 422

    # Excessive top_k (> 20)
    r3 = client.post("/api/v1/rag/query", json={"query": "valid query", "top_k": 50})
    assert r3.status_code == 422


def test_rag_api_missing_llm_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify API returns 503 when LLM credentials are not configured."""
    from adaq_rag.rag.service import reset_rag_pipeline
    reset_rag_pipeline()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    app = create_app()
    client = TestClient(app)

    response = client.post("/api/v1/rag/query", json={"query": "What is Ridge?", "top_k": 2})
    assert response.status_code == 503
    assert "API key is missing" in response.json()["detail"]
