"""Tests for multi-system RAG API endpoints, routing inspection, and evaluation endpoints."""

import pytest
from fastapi.testclient import TestClient

from adaq_rag.api.app import create_app
from adaq_rag.rag.models import RAGResponse, RoutingInfo
from adaq_rag.rag.service import (
    SYSTEM_ADAPTIVE_RAG,
    SYSTEM_BASIC_RAG,
    SYSTEM_FULL_ADAQ_RAG,
    SYSTEM_HYBRID_RAG,
    MultiSystemRAGService,
    get_multi_system_service,
)


@pytest.fixture
def mock_multi_service() -> MultiSystemRAGService:
    """Fixture providing an offline mock MultiSystemRAGService."""
    return MultiSystemRAGService.create_mock(
        default_response="Mock answer for multi-system API test [Source 1]."
    )


@pytest.fixture
def client(mock_multi_service: MultiSystemRAGService) -> TestClient:
    """Fixture providing FastAPI test client with injected mock service."""
    app = create_app()
    app.dependency_overrides[get_multi_system_service] = lambda: mock_multi_service
    return TestClient(app)


class TestMultiSystemEndpoints:
    """Tests executing and verifying all 4 RAG systems via the API."""

    def test_query_basic_rag(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "What is Ridge regression?", "system": "basic_rag", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is Ridge regression?"
        assert data["system"] == SYSTEM_BASIC_RAG
        assert "answer" in data
        assert len(data["sources"]) > 0
        assert data["routing"] is not None
        assert data["routing"]["predicted_strategy"] == "DENSE"

    def test_query_hybrid_rag(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "Compare Ridge and Lasso", "system": "hybrid_rag", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == SYSTEM_HYBRID_RAG
        assert data["retrieval_method"] == "hybrid"
        assert len(data["sources"]) > 0
        assert data["routing"]["predicted_strategy"] == "HYBRID"

    def test_query_adaptive_rag(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "Explain how cross-validation works in GridSearchCV", "system": "adaptive_rag"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == SYSTEM_ADAPTIVE_RAG
        assert data["routing"] is not None
        assert "predicted_intent" in data["routing"]
        assert "predicted_complexity" in data["routing"]
        assert "predicted_strategy" in data["routing"]
        assert data["routing"]["routing_reason"] is not None

    def test_query_full_adaq_rag(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/query",
            json={
                "query": "Compare soft voting and stacking classifier with logistic regression",
                "system": "full_adaq_rag",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == SYSTEM_FULL_ADAQ_RAG
        assert data["routing"] is not None
        assert data["routing"]["predicted_strategy"] in ("DENSE", "HYBRID", "MULTI_STEP")

    def test_query_alias_endpoint(self, client: TestClient) -> None:
        """Verify /api/v1/query functions identically to /api/v1/rag/query."""
        resp = client.post(
            "/api/v1/query",
            json={"query": "What is ElasticNet?", "system": "adaptive_rag"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == SYSTEM_ADAPTIVE_RAG

    def test_query_invalid_system_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/query",
            json={"query": "Valid query", "system": "non_existent_rag"},
        )
        assert resp.status_code == 400
        assert "Unknown system name" in resp.json()["detail"]

    def test_compare_all_systems(self, client: TestClient) -> None:
        """Verify comparing all 4 systems returns isolated results for each system."""
        resp = client.post(
            "/api/v1/rag/compare",
            json={"query": "Compare StandardScaler and MinMaxScaler", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Compare StandardScaler and MinMaxScaler"
        assert len(data["systems"]) == 4
        systems_dict = data["systems"]
        for expected_sys in [SYSTEM_BASIC_RAG, SYSTEM_HYBRID_RAG, SYSTEM_ADAPTIVE_RAG, SYSTEM_FULL_ADAQ_RAG]:
            assert expected_sys in systems_dict
            item = systems_dict[expected_sys]
            assert item["status"] == "success"
            assert item["result"] is not None
            assert len(item["result"]["sources"]) > 0

    def test_compare_alias_endpoint(self, client: TestClient) -> None:
        """Verify /api/v1/compare functions identically to /api/v1/rag/compare."""
        resp = client.post(
            "/api/v1/compare",
            json={"query": "What is Ridge?", "systems": [SYSTEM_BASIC_RAG, SYSTEM_ADAPTIVE_RAG]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["systems"]) == 2


class TestAnalysisAndEvaluationEndpoints:
    """Tests for routing analysis and pilot benchmark endpoints."""

    def test_analyze_query_endpoint(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/rag/analyze",
            json={"query": "What is the penalty parameter in Ridge?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is the penalty parameter in Ridge?"
        assert data["intent"] in ("FACT_LOOKUP", "CONCEPT_EXPLANATION", "COMPARISON", "CODE_IMPLEMENTATION", "PARAMETER_TUNING", "COMPLEX_ANALYSIS")
        assert data["complexity"] in ("SIMPLE", "MEDIUM", "COMPLEX")
        assert data["strategy"] in ("DENSE", "HYBRID", "MULTI_STEP")
        assert len(data["routing_reason"]) > 0

    def test_get_sample_questions_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/evaluation/questions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        # Ensure simple, moderate, complex are present
        tiers = [item["tier"] for item in data]
        assert "simple" in tiers
        assert "moderate" in tiers
        assert "complex" in tiers
        for sample in data:
            assert "question_id" in sample
            assert "query" in sample
            assert "expected_strategy" in sample
            assert sample["saved_answer"] is not None
            assert len(sample["saved_citations"]) > 0
            assert sample["saved_metrics"] is not None
            assert "overall_quality" in sample["saved_metrics"]
            assert "latency_ms" in sample["saved_metrics"]

    def test_get_pilot_evaluation_results_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/evaluation/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 12
        assert "system_comparison" in data
        assert SYSTEM_BASIC_RAG in data["system_comparison"]
        assert SYSTEM_FULL_ADAQ_RAG in data["system_comparison"]


class TestFrontendStaticRoutes:
    """Tests for serving frontend HTML, CSS, and JS assets."""

    def test_serve_index_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        text = resp.text
        assert "AdaQ-RAG" in text
        assert "The right retrieval strategy for each question." in text
        assert "Explore example questions." in text
        assert "Ask your own question." in text
        assert "How do the systems compare?" in text
        assert "Controlled 3-Question Pilot Evaluation" in text
        assert "Final Year Project Demo" not in text
        assert "API (development)" not in text

    def test_serve_style_css(self, client: TestClient) -> None:
        resp = client.get("/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert "--bg-page" in resp.text
        assert "--primary" in resp.text

    def test_serve_app_js(self, client: TestClient) -> None:
        resp = client.get("/app.js")
        assert resp.status_code == 200
        assert "application/javascript" in resp.headers["content-type"]
        assert "updatePipelineTrace" in resp.text
        assert "executeSingleSystem" in resp.text
        assert "executeComparison" in resp.text


