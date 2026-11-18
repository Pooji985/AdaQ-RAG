"""Service-level dependency providers for Baseline and Multi-System RAG Pipelines."""

import logging
from pathlib import Path
import time
from typing import Any

from fastapi import Depends

from adaq_rag.core.config import Settings, get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.factory import get_llm_provider
from adaq_rag.llm.mock import MockLLMProvider
from adaq_rag.query_analysis import QueryAnalyzer
from adaq_rag.rag.context_builder import ContextBuilder
from adaq_rag.rag.models import RAGResponse, RoutingInfo, SourceReference
from adaq_rag.rag.pipeline import BasicRAGPipeline
from adaq_rag.rag.prompts import DEFAULT_RAG_SYSTEM_PROMPT
from adaq_rag.retrieval.evidence import EvidenceItem
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.orchestrator import ComplexRetrievalOrchestrator
from adaq_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from adaq_rag.retrieval.retriever import BaseRetriever, BM25Retriever, DenseRetriever
from adaq_rag.routing import AdaptiveRetrievalRouter
from adaq_rag.routing.models import RetrievalStrategy

logger = logging.getLogger("adaq_rag.rag.service")

SYSTEM_BASIC_RAG = "basic_rag"
SYSTEM_HYBRID_RAG = "hybrid_rag"
SYSTEM_ADAPTIVE_RAG = "adaptive_rag"
SYSTEM_FULL_ADAQ_RAG = "full_adaq_rag"

SUPPORTED_SYSTEMS = [
    SYSTEM_BASIC_RAG,
    SYSTEM_HYBRID_RAG,
    SYSTEM_ADAPTIVE_RAG,
    SYSTEM_FULL_ADAQ_RAG,
]


def normalize_system_name(name: str) -> str:
    """Normalize system name string to canonical identifier."""
    clean = name.strip().lower().replace("-", "_").replace(" ", "_")
    if clean in ("basic", "basic_rag", "baseline", "baseline_rag"):
        return SYSTEM_BASIC_RAG
    if clean in ("hybrid", "hybrid_rag"):
        return SYSTEM_HYBRID_RAG
    if clean in ("adaptive", "adaptive_rag"):
        return SYSTEM_ADAPTIVE_RAG
    if clean in ("full", "full_adaq", "full_adaq_rag", "adaq", "adaq_rag"):
        return SYSTEM_FULL_ADAQ_RAG

    raise ValueError(
        f"Unknown system name '{name}'. Supported systems: {SUPPORTED_SYSTEMS}"
    )


def adapt_evidence_to_retrieval_results(
    evidence_items: list[Any],
) -> list[RetrievalResult]:
    """Convert EvidenceItem list from ComplexRetrievalOrchestrator into RetrievalResult list."""
    adapted: list[RetrievalResult] = []
    for item in evidence_items:
        if isinstance(item, RetrievalResult):
            adapted.append(item)
        elif isinstance(item, EvidenceItem):
            effective_score = (
                item.final_rerank_score
                if item.final_rerank_score is not None
                else item.initial_score
            )
            adapted.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    score=effective_score,
                    retrieval_method="multi_step",
                    content=item.content,
                    doc_id=item.doc_id,
                    doc_title=item.doc_title,
                    section_title=item.section_title,
                    section_level=item.section_level,
                    rerank_score=item.final_rerank_score,
                    metadata=dict(item.metadata),
                )
            )
        else:
            score = getattr(item, "score", getattr(item, "final_rerank_score", 0.0))
            adapted.append(
                RetrievalResult(
                    chunk_id=getattr(item, "chunk_id", ""),
                    score=float(score or 0.0),
                    retrieval_method="adapted",
                    content=getattr(item, "content", ""),
                    doc_id=getattr(item, "doc_id", ""),
                    doc_title=getattr(item, "doc_title", ""),
                    section_title=getattr(item, "section_title", ""),
                    section_level=getattr(item, "section_level", 1),
                )
            )
    return adapted


class MultiSystemRAGService:
    """Unified service container managing all 4 RAG systems for live API querying."""

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        reranker: BaseReranker,
        hybrid_retriever: HybridRetriever,
        query_analyzer: QueryAnalyzer,
        router: AdaptiveRetrievalRouter,
        orchestrator: ComplexRetrievalOrchestrator,
        context_builder: ContextBuilder,
        llm_provider: BaseLLMProvider,
        basic_pipeline: BasicRAGPipeline,
        settings: Settings | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.hybrid_retriever = hybrid_retriever
        self.query_analyzer = query_analyzer
        self.router = router
        self.orchestrator = orchestrator
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.basic_pipeline = basic_pipeline
        self.settings = settings or get_settings()

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        indexes_dir: str | Path | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> "MultiSystemRAGService":
        """Factory assembling service with loaded indexes and models."""
        cfg = settings or get_settings()
        idx_dir = Path(indexes_dir) if indexes_dir else Path(cfg.indexes_dir)

        logger.info("Initializing MultiSystemRAGService from indexes at %s", idx_dir)

        dense = DenseRetriever.load(
            index_path=idx_dir / cfg.vector_index_file,
            metadata_path=idx_dir / cfg.vector_metadata_file,
        )
        bm25 = BM25Retriever.load(
            index_path=idx_dir / cfg.bm25_index_file,
            metadata_path=idx_dir / cfg.bm25_metadata_file,
        )
        reranker = CrossEncoderReranker(model_name=cfg.reranker_model)

        hybrid = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            reranker=reranker,
            default_candidate_top_k=cfg.hybrid_candidate_top_k,
            default_final_top_k=cfg.hybrid_final_top_k,
            rrf_k=cfg.hybrid_rrf_k,
        )

        analyzer = QueryAnalyzer()
        router = AdaptiveRetrievalRouter()
        orchestrator = ComplexRetrievalOrchestrator(
            hybrid_retriever=hybrid,
            reranker=reranker,
            default_candidate_top_k=cfg.complex_candidate_top_k,
            default_final_top_k=cfg.complex_final_top_k,
            max_retrieval_attempts=cfg.complex_max_retrieval_attempts,
        )

        context_builder = ContextBuilder()
        provider = llm_provider or get_llm_provider(cfg)

        basic_pipeline = BasicRAGPipeline(
            dense_retriever=dense,
            llm_provider=provider,
            context_builder=context_builder,
            default_top_k=cfg.rag_top_k,
        )

        return cls(
            dense_retriever=dense,
            bm25_retriever=bm25,
            reranker=reranker,
            hybrid_retriever=hybrid,
            query_analyzer=analyzer,
            router=router,
            orchestrator=orchestrator,
            context_builder=context_builder,
            llm_provider=provider,
            basic_pipeline=basic_pipeline,
            settings=cfg,
        )

    @classmethod
    def create_mock(
        cls,
        default_response: str = "Mock answer from MultiSystemRAGService [Source 1].",
        settings: Settings | None = None,
    ) -> "MultiSystemRAGService":
        """Factory for testing without loading heavy FAISS or cross-encoders."""
        cfg = settings or get_settings()
        mock_provider = MockLLMProvider(default_response=default_response)

        class DummyDenseRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(
                        chunk_id=f"chunk_{i:03d}",
                        score=1.0 - (i * 0.1),
                        retrieval_method="dense",
                        content=f"Dense mock content for chunk {i}",
                        doc_id="doc_sklearn",
                        doc_title="Scikit-Learn Guide",
                        section_title="Model Overview",
                    )
                    for i in range(1, top_k + 1)
                ]

        class DummyBM25Retriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(
                        chunk_id=f"chunk_{i:03d}",
                        score=0.9 - (i * 0.1),
                        retrieval_method="bm25",
                        content=f"BM25 mock content for chunk {i}",
                        doc_id="doc_sklearn",
                        doc_title="Scikit-Learn Guide",
                        section_title="Model Overview",
                    )
                    for i in range(1, top_k + 1)
                ]

        class DummyReranker(BaseReranker):
            def rerank(
                self,
                query: str,
                candidates: list[RetrievalResult],
                top_k: int | None = None,
            ) -> list[RetrievalResult]:
                k = top_k or len(candidates)
                for res in candidates:
                    res.rerank_score = res.score
                return candidates[:k]

        dummy_dense = DummyDenseRetriever()
        dummy_bm25 = DummyBM25Retriever()
        dummy_reranker = DummyReranker()

        hybrid = HybridRetriever(
            dense_retriever=dummy_dense,
            bm25_retriever=dummy_bm25,
            reranker=dummy_reranker,
        )
        analyzer = QueryAnalyzer()
        router = AdaptiveRetrievalRouter()
        orchestrator = ComplexRetrievalOrchestrator(
            hybrid_retriever=hybrid,
            reranker=dummy_reranker,
        )
        context_builder = ContextBuilder()
        basic_pipeline = BasicRAGPipeline(
            dense_retriever=dummy_dense,
            llm_provider=mock_provider,
            context_builder=context_builder,
        )

        return cls(
            dense_retriever=dummy_dense,
            bm25_retriever=dummy_bm25,
            reranker=dummy_reranker,
            hybrid_retriever=hybrid,
            query_analyzer=analyzer,
            router=router,
            orchestrator=orchestrator,
            context_builder=context_builder,
            llm_provider=mock_provider,
            basic_pipeline=basic_pipeline,
            settings=cfg,
        )

    def analyze_query(self, question: str) -> dict[str, Any]:
        """Analyze question intent, complexity, and recommended retrieval strategy."""
        clean_q = question.strip()
        analysis = self.query_analyzer.analyze(clean_q)
        plan = self.router.route(analysis)
        return {
            "query": clean_q,
            "intent": analysis.intent.value,
            "complexity": analysis.complexity.value,
            "strategy": plan.strategy.value,
            "routing_reason": plan.routing_reason,
            "candidate_top_k": plan.candidate_top_k,
            "final_top_k": plan.final_top_k,
            "use_reranking": plan.use_reranking,
        }

    async def query_async(
        self,
        question: str,
        system: str = SYSTEM_BASIC_RAG,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Execute selected RAG system asynchronously."""
        canonical = normalize_system_name(system)
        clean_q = question.strip()
        start = time.perf_counter()

        # -------------------------------------------------------------
        # System 1: Basic RAG (Phase 5 Baseline)
        # -------------------------------------------------------------
        if canonical == SYSTEM_BASIC_RAG:
            k = top_k or self.settings.rag_top_k
            resp = await self.basic_pipeline.query_async(clean_q, top_k=k)
            resp.system = SYSTEM_BASIC_RAG
            resp.retrieval_calls = 1
            resp.routing = RoutingInfo(
                predicted_strategy="DENSE",
                routing_reason="Fixed dense retrieval baseline.",
                candidate_top_k=k,
                final_top_k=k,
            )
            return resp

        # -------------------------------------------------------------
        # System 2: Hybrid RAG (Phase 8 Hybrid + CrossEncoder)
        # -------------------------------------------------------------
        if canonical == SYSTEM_HYBRID_RAG:
            c_k = self.settings.hybrid_candidate_top_k
            f_k = top_k or self.settings.hybrid_final_top_k
            retrieved = self.hybrid_retriever.retrieve(
                query=clean_q,
                candidate_top_k=c_k,
                final_top_k=f_k,
                use_reranking=True,
            )
            user_prompt, sources = self.context_builder.format_prompt(clean_q, retrieved)
            llm_resp = await self.llm_provider.generate_async(
                prompt=user_prompt,
                system_prompt=DEFAULT_RAG_SYSTEM_PROMPT,
            )
            latency = (time.perf_counter() - start) * 1000.0
            return RAGResponse(
                query=clean_q,
                answer=llm_resp.content.strip(),
                system=SYSTEM_HYBRID_RAG,
                retrieval_method="hybrid",
                top_k=f_k,
                sources=sources,
                retrieved_chunk_ids=[c.chunk_id for c in retrieved],
                retrieval_scores=[c.score for c in retrieved],
                routing=RoutingInfo(
                    predicted_strategy="HYBRID",
                    routing_reason="Fixed hybrid retrieval with cross-encoder reranking.",
                    candidate_top_k=c_k,
                    final_top_k=f_k,
                ),
                latency_ms=round(latency, 2),
                retrieval_calls=2,
                metadata={
                    "model": llm_resp.model,
                    "usage": llm_resp.usage.model_dump() if llm_resp.usage else None,
                },
            )

        # -------------------------------------------------------------
        # Systems 3 & 4: Adaptive RAG & Full AdaQ-RAG
        # -------------------------------------------------------------
        analysis = self.query_analyzer.analyze(clean_q)
        plan = self.router.route(analysis)
        f_k = top_k or plan.final_top_k
        sub_queries: list[str] = []
        was_escalated = False
        retrieval_calls = 1

        if canonical == SYSTEM_ADAPTIVE_RAG:
            if plan.strategy == RetrievalStrategy.DENSE:
                retrieved = self.dense_retriever.retrieve(clean_q, top_k=f_k)
                method = "dense"
                retrieval_calls = 1
            else:
                retrieved = self.hybrid_retriever.retrieve(
                    query=clean_q,
                    candidate_top_k=plan.candidate_top_k,
                    final_top_k=f_k,
                    use_reranking=plan.use_reranking,
                )
                method = "hybrid"
                retrieval_calls = 2
        else:
            # SYSTEM_FULL_ADAQ_RAG
            if plan.strategy == RetrievalStrategy.DENSE:
                retrieved = self.dense_retriever.retrieve(clean_q, top_k=f_k)
                method = "dense"
                retrieval_calls = 1
            elif plan.strategy == RetrievalStrategy.HYBRID:
                retrieved = self.hybrid_retriever.retrieve(
                    query=clean_q,
                    candidate_top_k=plan.candidate_top_k,
                    final_top_k=f_k,
                    use_reranking=plan.use_reranking,
                )
                method = "hybrid"
                retrieval_calls = 2
            elif plan.strategy == RetrievalStrategy.MULTI_STEP:
                complex_res = self.orchestrator.retrieve_from_plan(plan)
                was_escalated = complex_res.was_escalated
                sub_queries = [sq.query_text for sq in complex_res.decomposed_query.sub_queries]
                retrieved = adapt_evidence_to_retrieval_results(complex_res.final_evidence)
                method = "multi_step"
                retrieval_calls = len(sub_queries) * 2 * complex_res.retrieval_attempts
            else:
                retrieved = self.dense_retriever.retrieve(clean_q, top_k=f_k)
                method = "dense"
                retrieval_calls = 1

        user_prompt, sources = self.context_builder.format_prompt(clean_q, retrieved)
        llm_resp = await self.llm_provider.generate_async(
            prompt=user_prompt,
            system_prompt=DEFAULT_RAG_SYSTEM_PROMPT,
        )
        latency = (time.perf_counter() - start) * 1000.0

        return RAGResponse(
            query=clean_q,
            answer=llm_resp.content.strip(),
            system=canonical,
            retrieval_method=method,
            top_k=f_k,
            sources=sources,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
            retrieval_scores=[c.score for c in retrieved],
            routing=RoutingInfo(
                predicted_intent=analysis.intent.value,
                predicted_complexity=analysis.complexity.value,
                predicted_strategy=plan.strategy.value,
                routing_reason=plan.routing_reason,
                candidate_top_k=plan.candidate_top_k,
                final_top_k=f_k,
                sub_queries=sub_queries,
                was_escalated=was_escalated,
            ),
            latency_ms=round(latency, 2),
            retrieval_calls=retrieval_calls,
            metadata={
                "model": llm_resp.model,
                "usage": llm_resp.usage.model_dump() if llm_resp.usage else None,
            },
        )


_pipeline_instance: BasicRAGPipeline | None = None
_multi_service_instance: MultiSystemRAGService | None = None


def get_rag_pipeline(settings: Settings = Depends(get_settings)) -> BasicRAGPipeline:
    """Return singleton BasicRAGPipeline instance, initializing lazily on first access."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing singleton BasicRAGPipeline instance")
        _pipeline_instance = BasicRAGPipeline.create(settings=settings)
    return _pipeline_instance


def reset_rag_pipeline() -> None:
    """Reset singleton BasicRAGPipeline instance (useful during testing)."""
    global _pipeline_instance
    _pipeline_instance = None


def get_multi_system_service(
    settings: Settings = Depends(get_settings),
) -> MultiSystemRAGService:
    """Return singleton MultiSystemRAGService instance, initializing lazily on first access."""
    global _multi_service_instance
    if _multi_service_instance is None:
        logger.info("Initializing singleton MultiSystemRAGService instance")
        _multi_service_instance = MultiSystemRAGService.create(settings=settings)
    return _multi_service_instance


def reset_multi_system_service() -> None:
    """Reset singleton MultiSystemRAGService instance."""
    global _multi_service_instance
    _multi_service_instance = None

