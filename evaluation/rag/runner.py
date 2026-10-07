"""AdaQ-RAG Evaluation Runner.

Connects existing AdaQ-RAG components to the 90-question benchmark across:
1. Basic RAG (Phase 5: BasicRAGPipeline)
2. Hybrid RAG (Phase 8: HybridRetriever + CrossEncoder reranker + generation)
3. Adaptive RAG (Phase 6 & 7: QueryAnalyzer + AdaptiveRetrievalRouter + strategy retrieval + generation)
4. Full AdaQ-RAG (Phase 9: Full adaptive routing + multi-step decomposition + evidence sufficiency + generation)

Features:
- Benchmark loading and flexible question ID subset selection
- Shared component harness to avoid reloading vector/lexical indexes
- Safe adapter for ComplexRetrievalOrchestrator EvidenceItems
- Execution of selected systems with latency, retrieval calls, and token tracking
- Integration with BaseJudge (LLMJudge or MockJudge)
- Per-question result serialization and system-level metrics aggregation
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Sequence

from adaq_rag.core.config import Settings, get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.factory import get_llm_provider
from adaq_rag.llm.mock import MockLLMProvider
from adaq_rag.llm.models import LLMUsage
from adaq_rag.query_analysis import QueryAnalyzer
from adaq_rag.rag.context_builder import ContextBuilder
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
from evaluation.rag.efficiency import EfficiencyTracker
from evaluation.rag.judge import BaseJudge, MockJudge
from evaluation.rag.metrics import compute_retrieval_metrics
from evaluation.rag.models import (
    AnswerQualityScores,
    EfficiencyMetrics,
    QuestionEvaluationResult,
    RoutingEvaluation,
)

logger = logging.getLogger("adaq_rag.evaluation.runner")

DEFAULT_BENCHMARK_PATH = Path("evaluation/rag/adaptive_rag_benchmark.json")

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
    """Normalize system name string to canonical identifier.

    Args:
        name: User-provided system identifier.

    Returns:
        Canonical system name string.

    Raises:
        ValueError: If name is not a supported system.
    """
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


def load_benchmark(
    benchmark_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load benchmark questions from JSON dataset file.

    Args:
        benchmark_path: Optional path to JSON benchmark file. Defaults to
            'evaluation/rag/adaptive_rag_benchmark.json'.

    Returns:
        List of benchmark question dictionary items.

    Raises:
        FileNotFoundError: If benchmark file does not exist.
        ValueError: If benchmark data is empty or invalid JSON.
    """
    path = Path(benchmark_path) if benchmark_path else DEFAULT_BENCHMARK_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Benchmark file {path} must contain a non-empty list of question items.")

    return data


def filter_benchmark(
    questions: list[dict[str, Any]],
    question_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter benchmark questions by question IDs and/or limit count.

    Args:
        questions: Full list of benchmark question items.
        question_ids: Optional sequence of target question_ids to include.
        limit: Optional maximum number of items to return.

    Returns:
        Filtered list of question items.
    """
    selected = questions
    if question_ids is not None:
        id_set = set(question_ids)
        selected = [q for q in selected if q.get("question_id") in id_set]

    if limit is not None and limit > 0:
        selected = selected[:limit]

    return selected


def adapt_evidence_to_retrieval_results(
    evidence_items: Sequence[Any],
) -> list[RetrievalResult]:
    """Safe adapter mapping EvidenceItem to RetrievalResult for ContextBuilder.

    Resolves interface mismatch: ContextBuilder expects objects with `.score`,
    whereas Phase 9 EvidenceItem defines `.final_rerank_score` and `.initial_score`.

    Args:
        evidence_items: Sequence of EvidenceItem or RetrievalResult objects.

    Returns:
        List of RetrievalResult instances compatible with ContextBuilder.
    """
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
            # Generic duck-typed object fallback
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


class EvaluationHarness:
    """Shared component container and executor for the 4 evaluation systems.

    Maintains singletons of FAISS, BM25, and CrossEncoder to avoid reloading
    heavy model artifacts during evaluation runs.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        hybrid_retriever: HybridRetriever,
        query_analyzer: QueryAnalyzer,
        router: AdaptiveRetrievalRouter,
        orchestrator: ComplexRetrievalOrchestrator,
        basic_pipeline: BasicRAGPipeline,
        context_builder: ContextBuilder,
        llm_provider: BaseLLMProvider,
        judge: BaseJudge | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.query_analyzer = query_analyzer
        self.router = router
        self.orchestrator = orchestrator
        self.basic_pipeline = basic_pipeline
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.judge = judge
        self.settings = settings or get_settings()

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        indexes_dir: str | Path | None = None,
        llm_provider: BaseLLMProvider | None = None,
        judge: BaseJudge | None = None,
    ) -> "EvaluationHarness":
        """Factory assembling harness with loaded indexes and models.

        Args:
            settings: Optional Settings instance.
            indexes_dir: Optional path to index directory.
            llm_provider: Optional BaseLLMProvider instance.
            judge: Optional BaseJudge instance.

        Returns:
            EvaluationHarness instance ready for benchmarking.
        """
        cfg = settings or get_settings()
        idx_dir = Path(indexes_dir) if indexes_dir else Path(cfg.indexes_dir)

        logger.info("Initializing EvaluationHarness from indexes at %s", idx_dir)

        # 1. Retrievers & Reranker
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

        # 2. Query Analysis, Router & Orchestrator
        analyzer = QueryAnalyzer()
        router = AdaptiveRetrievalRouter()
        orchestrator = ComplexRetrievalOrchestrator(
            hybrid_retriever=hybrid,
            reranker=reranker,
            default_candidate_top_k=cfg.complex_candidate_top_k,
            default_final_top_k=cfg.complex_final_top_k,
            max_retrieval_attempts=cfg.complex_max_retrieval_attempts,
        )

        # 3. Context Builder & LLM Provider
        context_builder = ContextBuilder()
        provider = llm_provider or get_llm_provider(cfg)

        # 4. Basic RAG Pipeline
        basic_pipeline = BasicRAGPipeline(
            dense_retriever=dense,
            llm_provider=provider,
            context_builder=context_builder,
            default_top_k=cfg.rag_top_k,
        )

        return cls(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            query_analyzer=analyzer,
            router=router,
            orchestrator=orchestrator,
            basic_pipeline=basic_pipeline,
            context_builder=context_builder,
            llm_provider=provider,
            judge=judge,
            settings=cfg,
        )

    @classmethod
    def create_mock(
        cls,
        judge: BaseJudge | None = None,
        default_response: str | None = None,
    ) -> "EvaluationHarness":
        """Factory for lightweight testing without loading heavy FAISS or cross-encoders."""
        mock_provider = MockLLMProvider(
            default_response=default_response or "Mock answer for evaluation."
        )
        mock_judge = judge or MockJudge()

        # Build dummy retriever
        class DummyDenseRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(
                        chunk_id=f"mock_chunk_{i}",
                        score=1.0 - (i * 0.1),
                        retrieval_method="dense",
                        content=f"Mock content for chunk {i}",
                        doc_id="mock_doc",
                        doc_title="Mock Doc",
                        section_title="Mock Section",
                    )
                    for i in range(1, top_k + 1)
                ]

        class DummyBM25Retriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(
                        chunk_id=f"mock_chunk_{i}",
                        score=0.9 - (i * 0.1),
                        retrieval_method="bm25",
                        content=f"Mock content for chunk {i}",
                        doc_id="mock_doc",
                        doc_title="Mock Doc",
                        section_title="Mock Section",
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
            hybrid_retriever=hybrid,
            query_analyzer=analyzer,
            router=router,
            orchestrator=orchestrator,
            basic_pipeline=basic_pipeline,
            context_builder=context_builder,
            llm_provider=mock_provider,
            judge=mock_judge,
        )

    def _judge_answer(
        self,
        question: str,
        reference_answer: str,
        retrieved_results: list[RetrievalResult],
        generated_answer: str,
    ) -> AnswerQualityScores:
        """Evaluate generated answer with injected judge.

        CRITICAL: Does NOT pass system name to judge.
        """
        if self.judge is None:
            return AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="No judge configured.",
            )

        evidence_snippets = [c.content for c in retrieved_results]
        return self.judge.evaluate(
            question=question,
            reference_answer=reference_answer,
            retrieved_evidence=evidence_snippets,
            generated_answer=generated_answer,
        )

    def run_basic_rag(self, item: dict[str, Any]) -> QuestionEvaluationResult:
        """Execute System 1: Basic RAG (Phase 5)."""
        query = item["query"]
        qid = item["question_id"]
        ref_ans = item.get("reference_answer", "")
        gt_chunks = item.get("ground_truth_chunk_ids", [])
        exp_intent = item.get("expected_intent")
        exp_complexity = item.get("expected_complexity")
        exp_strategy = item.get("expected_strategy")

        start = time.perf_counter()
        rag_resp = self.basic_pipeline.query(query)
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_chunk_ids = list(rag_resp.retrieved_chunk_ids)
        retrieval_metrics = compute_retrieval_metrics(retrieved_chunk_ids, gt_chunks)

        token_usage: LLMUsage | None = None
        if rag_resp.metadata and rag_resp.metadata.get("usage"):
            try:
                token_usage = LLMUsage(**rag_resp.metadata["usage"])
            except Exception:
                token_usage = None

        eff = EfficiencyMetrics(
            latency_ms=round(latency_ms, 2),
            retrieval_calls=1,
            chunks_processed=len(retrieved_chunk_ids),
            escalation_count=0,
            token_usage=token_usage,
        )

        # Basic RAG is non-adaptive, fixed to DENSE
        actual_strategy = "DENSE"
        routing_eval = RoutingEvaluation(
            predicted_strategy=actual_strategy,
            expected_strategy=exp_strategy,
            strategy_match=(actual_strategy == str(exp_strategy).upper() if exp_strategy else False),
            predicted_intent=None,
            expected_intent=exp_intent,
            intent_match=False,
            predicted_complexity=None,
            expected_complexity=exp_complexity,
            complexity_match=False,
            routing_reason="Fixed dense retrieval baseline.",
        )

        # Convert SourceReferences to RetrievalResult for judging
        judging_results = [
            RetrievalResult(
                chunk_id=s.chunk_id,
                score=s.score,
                retrieval_method="dense",
                content=s.content_snippet,
                doc_id=s.doc_id,
                doc_title=s.doc_title,
                section_title=s.section_title,
            )
            for s in rag_resp.sources
        ]
        scores = self._judge_answer(query, ref_ans, judging_results, rag_resp.answer)

        return QuestionEvaluationResult(
            question_id=qid,
            query=query,
            reference_answer=ref_ans,
            generated_answer=rag_resp.answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            ground_truth_chunk_ids=gt_chunks,
            retrieval_metrics=retrieval_metrics,
            answer_scores=scores,
            efficiency=eff,
            routing=routing_eval,
            metadata={
                "system_name": SYSTEM_BASIC_RAG,
                "expected_strategy": exp_strategy,
                "actual_strategy": actual_strategy,
                "expected_intent": exp_intent,
                "actual_intent": None,
                "expected_complexity": exp_complexity,
                "actual_complexity": None,
            },
        )

    def run_hybrid_rag(self, item: dict[str, Any]) -> QuestionEvaluationResult:
        """Execute System 2: Hybrid RAG (Phase 8)."""
        query = item["query"]
        qid = item["question_id"]
        ref_ans = item.get("reference_answer", "")
        gt_chunks = item.get("ground_truth_chunk_ids", [])
        exp_intent = item.get("expected_intent")
        exp_complexity = item.get("expected_complexity")
        exp_strategy = item.get("expected_strategy")

        start = time.perf_counter()
        c_k = self.settings.hybrid_candidate_top_k
        f_k = self.settings.hybrid_final_top_k

        retrieved_results = self.hybrid_retriever.retrieve(
            query=query,
            candidate_top_k=c_k,
            final_top_k=f_k,
            use_reranking=True,
        )

        user_prompt, sources = self.context_builder.format_prompt(query, retrieved_results)
        llm_resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=DEFAULT_RAG_SYSTEM_PROMPT,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_chunk_ids = [c.chunk_id for c in retrieved_results]
        retrieval_metrics = compute_retrieval_metrics(retrieved_chunk_ids, gt_chunks)

        eff = EfficiencyMetrics(
            latency_ms=round(latency_ms, 2),
            retrieval_calls=2,  # Dense + BM25
            chunks_processed=c_k * 2,
            escalation_count=0,
            token_usage=llm_resp.usage,
        )

        actual_strategy = "HYBRID"
        routing_eval = RoutingEvaluation(
            predicted_strategy=actual_strategy,
            expected_strategy=exp_strategy,
            strategy_match=(actual_strategy == str(exp_strategy).upper() if exp_strategy else False),
            predicted_intent=None,
            expected_intent=exp_intent,
            intent_match=False,
            predicted_complexity=None,
            expected_complexity=exp_complexity,
            complexity_match=False,
            routing_reason="Fixed hybrid retrieval with cross-encoder reranking.",
        )

        scores = self._judge_answer(query, ref_ans, retrieved_results, llm_resp.content.strip())

        return QuestionEvaluationResult(
            question_id=qid,
            query=query,
            reference_answer=ref_ans,
            generated_answer=llm_resp.content.strip(),
            retrieved_chunk_ids=retrieved_chunk_ids,
            ground_truth_chunk_ids=gt_chunks,
            retrieval_metrics=retrieval_metrics,
            answer_scores=scores,
            efficiency=eff,
            routing=routing_eval,
            metadata={
                "system_name": SYSTEM_HYBRID_RAG,
                "expected_strategy": exp_strategy,
                "actual_strategy": actual_strategy,
                "expected_intent": exp_intent,
                "actual_intent": None,
                "expected_complexity": exp_complexity,
                "actual_complexity": None,
            },
        )

    def run_adaptive_rag(self, item: dict[str, Any]) -> QuestionEvaluationResult:
        """Execute System 3: Adaptive RAG (Phase 6 Query Analysis + Phase 7 Router)."""
        query = item["query"]
        qid = item["question_id"]
        ref_ans = item.get("reference_answer", "")
        gt_chunks = item.get("ground_truth_chunk_ids", [])
        exp_intent = item.get("expected_intent")
        exp_complexity = item.get("expected_complexity")
        exp_strategy = item.get("expected_strategy")

        start = time.perf_counter()

        # 1. Query Analysis & Routing
        analysis = self.query_analyzer.analyze(query)
        plan = self.router.route(analysis)

        # 2. Strategy-based Retrieval
        if plan.strategy == RetrievalStrategy.DENSE:
            retrieved_results = self.dense_retriever.retrieve(query, top_k=plan.final_top_k)
            retrieval_calls = 1
            chunks_processed = len(retrieved_results)
        else:
            # HYBRID or MULTI_STEP executed via HybridRetriever
            retrieved_results = self.hybrid_retriever.retrieve(
                query=query,
                candidate_top_k=plan.candidate_top_k,
                final_top_k=plan.final_top_k,
                use_reranking=plan.use_reranking,
            )
            retrieval_calls = 2
            chunks_processed = plan.candidate_top_k * 2

        # 3. Generation
        user_prompt, sources = self.context_builder.format_prompt(query, retrieved_results)
        llm_resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=DEFAULT_RAG_SYSTEM_PROMPT,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_chunk_ids = [c.chunk_id for c in retrieved_results]
        retrieval_metrics = compute_retrieval_metrics(retrieved_chunk_ids, gt_chunks)

        eff = EfficiencyMetrics(
            latency_ms=round(latency_ms, 2),
            retrieval_calls=retrieval_calls,
            chunks_processed=chunks_processed,
            escalation_count=0,
            token_usage=llm_resp.usage,
        )

        actual_strategy = plan.strategy.value
        actual_intent = analysis.intent.value
        actual_complexity = analysis.complexity.value

        routing_eval = RoutingEvaluation(
            predicted_strategy=actual_strategy,
            expected_strategy=exp_strategy,
            strategy_match=(actual_strategy == str(exp_strategy).upper() if exp_strategy else False),
            predicted_intent=actual_intent,
            expected_intent=exp_intent,
            intent_match=(actual_intent == str(exp_intent).upper() if exp_intent else False),
            predicted_complexity=actual_complexity,
            expected_complexity=exp_complexity,
            complexity_match=(
                actual_complexity == str(exp_complexity).upper() if exp_complexity else False
            ),
            routing_reason=plan.routing_reason,
        )

        scores = self._judge_answer(query, ref_ans, retrieved_results, llm_resp.content.strip())

        return QuestionEvaluationResult(
            question_id=qid,
            query=query,
            reference_answer=ref_ans,
            generated_answer=llm_resp.content.strip(),
            retrieved_chunk_ids=retrieved_chunk_ids,
            ground_truth_chunk_ids=gt_chunks,
            retrieval_metrics=retrieval_metrics,
            answer_scores=scores,
            efficiency=eff,
            routing=routing_eval,
            metadata={
                "system_name": SYSTEM_ADAPTIVE_RAG,
                "expected_strategy": exp_strategy,
                "actual_strategy": actual_strategy,
                "expected_intent": exp_intent,
                "actual_intent": actual_intent,
                "expected_complexity": exp_complexity,
                "actual_complexity": actual_complexity,
            },
        )

    def run_full_adaq_rag(self, item: dict[str, Any]) -> QuestionEvaluationResult:
        """Execute System 4: Full AdaQ-RAG (Phases 6, 7, 8 & 9)."""
        query = item["query"]
        qid = item["question_id"]
        ref_ans = item.get("reference_answer", "")
        gt_chunks = item.get("ground_truth_chunk_ids", [])
        exp_intent = item.get("expected_intent")
        exp_complexity = item.get("expected_complexity")
        exp_strategy = item.get("expected_strategy")

        start = time.perf_counter()

        # 1. Query Analysis & Routing
        analysis = self.query_analyzer.analyze(query)
        plan = self.router.route(analysis)

        escalation_count = 0

        # 2. Retrieval Execution
        if plan.strategy == RetrievalStrategy.DENSE:
            retrieved_results = self.dense_retriever.retrieve(query, top_k=plan.final_top_k)
            retrieval_calls = 1
            chunks_processed = len(retrieved_results)

        elif plan.strategy == RetrievalStrategy.HYBRID:
            retrieved_results = self.hybrid_retriever.retrieve(
                query=query,
                candidate_top_k=plan.candidate_top_k,
                final_top_k=plan.final_top_k,
                use_reranking=plan.use_reranking,
            )
            retrieval_calls = 2
            chunks_processed = plan.candidate_top_k * 2

        elif plan.strategy == RetrievalStrategy.MULTI_STEP:
            # Phase 9: Complex Orchestrator with query decomposition & sufficiency evaluation
            complex_res = self.orchestrator.retrieve_from_plan(plan)
            escalation_count = 1 if complex_res.was_escalated else 0
            sub_count = len(complex_res.decomposed_query.sub_queries)
            retrieval_calls = sub_count * 2 * complex_res.retrieval_attempts
            chunks_processed = complex_res.evidence_pool.total_candidates_examined

            # Small safe adapter converting EvidenceItem to RetrievalResult
            retrieved_results = adapt_evidence_to_retrieval_results(complex_res.final_evidence)

        else:
            # Fallback
            retrieved_results = self.dense_retriever.retrieve(query, top_k=plan.final_top_k)
            retrieval_calls = 1
            chunks_processed = len(retrieved_results)

        # 3. Grounded Generation
        user_prompt, sources = self.context_builder.format_prompt(query, retrieved_results)
        llm_resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=DEFAULT_RAG_SYSTEM_PROMPT,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_chunk_ids = [c.chunk_id for c in retrieved_results]
        retrieval_metrics = compute_retrieval_metrics(retrieved_chunk_ids, gt_chunks)

        eff = EfficiencyMetrics(
            latency_ms=round(latency_ms, 2),
            retrieval_calls=retrieval_calls,
            chunks_processed=chunks_processed,
            escalation_count=escalation_count,
            token_usage=llm_resp.usage,
        )

        actual_strategy = plan.strategy.value
        actual_intent = analysis.intent.value
        actual_complexity = analysis.complexity.value

        routing_eval = RoutingEvaluation(
            predicted_strategy=actual_strategy,
            expected_strategy=exp_strategy,
            strategy_match=(actual_strategy == str(exp_strategy).upper() if exp_strategy else False),
            predicted_intent=actual_intent,
            expected_intent=exp_intent,
            intent_match=(actual_intent == str(exp_intent).upper() if exp_intent else False),
            predicted_complexity=actual_complexity,
            expected_complexity=exp_complexity,
            complexity_match=(
                actual_complexity == str(exp_complexity).upper() if exp_complexity else False
            ),
            routing_reason=plan.routing_reason,
        )

        scores = self._judge_answer(query, ref_ans, retrieved_results, llm_resp.content.strip())

        return QuestionEvaluationResult(
            question_id=qid,
            query=query,
            reference_answer=ref_ans,
            generated_answer=llm_resp.content.strip(),
            retrieved_chunk_ids=retrieved_chunk_ids,
            ground_truth_chunk_ids=gt_chunks,
            retrieval_metrics=retrieval_metrics,
            answer_scores=scores,
            efficiency=eff,
            routing=routing_eval,
            metadata={
                "system_name": SYSTEM_FULL_ADAQ_RAG,
                "expected_strategy": exp_strategy,
                "actual_strategy": actual_strategy,
                "expected_intent": exp_intent,
                "actual_intent": actual_intent,
                "expected_complexity": exp_complexity,
                "actual_complexity": actual_complexity,
            },
        )

    def run_system(
        self,
        system_name: str,
        question_item: dict[str, Any],
    ) -> QuestionEvaluationResult:
        """Dispatch evaluation for a specific system and question."""
        canonical = normalize_system_name(system_name)
        if canonical == SYSTEM_BASIC_RAG:
            return self.run_basic_rag(question_item)
        if canonical == SYSTEM_HYBRID_RAG:
            return self.run_hybrid_rag(question_item)
        if canonical == SYSTEM_ADAPTIVE_RAG:
            return self.run_adaptive_rag(question_item)
        if canonical == SYSTEM_FULL_ADAQ_RAG:
            return self.run_full_adaq_rag(question_item)

        raise ValueError(f"Unsupported system '{system_name}'")


class EvaluationRunner:
    """High-level runner executing benchmark questions across selected RAG systems."""

    def __init__(
        self,
        harness: EvaluationHarness | None = None,
        judge: BaseJudge | None = None,
    ) -> None:
        self.harness = harness
        self.judge = judge
        if self.harness and self.judge:
            self.harness.judge = self.judge

    def get_harness(self) -> EvaluationHarness:
        """Lazily initialize harness if not provided."""
        if self.harness is None:
            self.harness = EvaluationHarness.create(judge=self.judge)
        elif self.judge is not None:
            self.harness.judge = self.judge
        return self.harness

    def evaluate_questions(
        self,
        questions: list[dict[str, Any]],
        systems: Sequence[str] | None = None,
    ) -> list[QuestionEvaluationResult]:
        """Evaluate a collection of benchmark questions across specified systems.

        Args:
            questions: List of benchmark question dicts.
            systems: Optional list of system names to run. Defaults to all 4 systems.

        Returns:
            List of QuestionEvaluationResult instances.
        """
        harness = self.get_harness()
        target_systems = [
            normalize_system_name(s) for s in (systems or SUPPORTED_SYSTEMS)
        ]

        results: list[QuestionEvaluationResult] = []

        for q_idx, item in enumerate(questions, start=1):
            qid = item.get("question_id", f"q_{q_idx}")
            logger.info("Evaluating [%d/%d] %s across %d systems", q_idx, len(questions), qid, len(target_systems))

            for sys_name in target_systems:
                try:
                    res = harness.run_system(sys_name, item)
                    results.append(res)
                except Exception as err:
                    logger.error("Failed evaluation of %s on %s: %s", qid, sys_name, err, exc_info=True)
                    # Create partial/failure evaluation record
                    dummy_res = QuestionEvaluationResult(
                        question_id=qid,
                        query=item.get("query", ""),
                        reference_answer=item.get("reference_answer", ""),
                        generated_answer="",
                        retrieval_metrics=compute_retrieval_metrics([], item.get("ground_truth_chunk_ids", [])),
                        answer_scores=AnswerQualityScores(
                            correctness=0.0,
                            faithfulness=0.0,
                            relevance=0.0,
                            overall=0.0,
                            reason=f"Evaluation execution failure: {err}",
                        ),
                        efficiency=EfficiencyMetrics(latency_ms=0.0),
                        routing=None,
                        metadata={
                            "system_name": sys_name,
                            "error": str(err),
                        },
                    )
                    results.append(dummy_res)

        return results

    @staticmethod
    def aggregate_results(
        results: Sequence[QuestionEvaluationResult],
    ) -> dict[str, Any]:
        """Produce system-level aggregate performance metrics from completed results.

        Args:
            results: List of QuestionEvaluationResult instances.

        Returns:
            Dictionary mapping system names to aggregate metrics summaries.
        """
        by_system: dict[str, list[QuestionEvaluationResult]] = {}
        for r in results:
            sys_name = r.metadata.get("system_name", "unknown")
            by_system.setdefault(sys_name, []).append(r)

        summary: dict[str, Any] = {}

        for sys_name, records in by_system.items():
            n = len(records)
            if n == 0:
                continue

            # 1. Retrieval averages
            r_at_1 = sum(r.retrieval_metrics.recall_at_k.get(1, 0.0) for r in records) / n
            r_at_3 = sum(r.retrieval_metrics.recall_at_k.get(3, 0.0) for r in records) / n
            r_at_5 = sum(r.retrieval_metrics.recall_at_k.get(5, 0.0) for r in records) / n
            r_at_10 = sum(r.retrieval_metrics.recall_at_k.get(10, 0.0) for r in records) / n
            mrr = sum(r.retrieval_metrics.mrr for r in records) / n

            ndcg_1 = sum(r.retrieval_metrics.ndcg_at_k.get(1, 0.0) for r in records) / n
            ndcg_3 = sum(r.retrieval_metrics.ndcg_at_k.get(3, 0.0) for r in records) / n
            ndcg_5 = sum(r.retrieval_metrics.ndcg_at_k.get(5, 0.0) for r in records) / n
            ndcg_10 = sum(r.retrieval_metrics.ndcg_at_k.get(10, 0.0) for r in records) / n

            # 2. Routing fidelity
            routing_evals = [r.routing for r in records if r.routing is not None]
            routing_acc = (
                sum(1 for rt in routing_evals if rt.strategy_match) / len(routing_evals)
                if routing_evals
                else 0.0
            )
            intent_acc = (
                sum(1 for rt in routing_evals if rt.intent_match) / len(routing_evals)
                if routing_evals
                else 0.0
            )
            complexity_acc = (
                sum(1 for rt in routing_evals if rt.complexity_match) / len(routing_evals)
                if routing_evals
                else 0.0
            )

            # 3. Efficiency averages
            mean_latency = sum(r.efficiency.latency_ms for r in records) / n
            total_calls = sum(r.efficiency.retrieval_calls for r in records)
            mean_calls = total_calls / n
            total_chunks = sum(r.efficiency.chunks_processed for r in records)
            mean_chunks = total_chunks / n
            total_escalations = sum(r.efficiency.escalation_count for r in records)

            # Token usage
            tokens_available = any(r.efficiency.token_usage is not None for r in records)
            token_stats: dict[str, int] | None = None
            if tokens_available:
                p_tokens = sum(
                    r.efficiency.token_usage.prompt_tokens
                    for r in records
                    if r.efficiency.token_usage is not None
                )
                c_tokens = sum(
                    r.efficiency.token_usage.completion_tokens
                    for r in records
                    if r.efficiency.token_usage is not None
                )
                t_tokens = sum(
                    r.efficiency.token_usage.total_tokens
                    for r in records
                    if r.efficiency.token_usage is not None
                )
                token_stats = {
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": t_tokens,
                }

            # 4. Answer Quality Scores
            mean_c = sum(r.answer_scores.correctness for r in records) / n
            mean_f = sum(r.answer_scores.faithfulness for r in records) / n
            mean_rel = sum(r.answer_scores.relevance for r in records) / n
            mean_overall = sum(r.answer_scores.overall for r in records) / n

            summary[sys_name] = {
                "question_count": n,
                "retrieval": {
                    "recall_at_1": round(r_at_1, 4),
                    "recall_at_3": round(r_at_3, 4),
                    "recall_at_5": round(r_at_5, 4),
                    "recall_at_10": round(r_at_10, 4),
                    "mrr": round(mrr, 4),
                    "ndcg_at_1": round(ndcg_1, 4),
                    "ndcg_at_3": round(ndcg_3, 4),
                    "ndcg_at_5": round(ndcg_5, 4),
                    "ndcg_at_10": round(ndcg_10, 4),
                },
                "routing": {
                    "strategy_accuracy": round(routing_acc, 4),
                    "intent_accuracy": round(intent_acc, 4),
                    "complexity_accuracy": round(complexity_acc, 4),
                },
                "efficiency": {
                    "mean_latency_ms": round(mean_latency, 2),
                    "total_retrieval_calls": total_calls,
                    "mean_retrieval_calls": round(mean_calls, 2),
                    "total_chunks_processed": total_chunks,
                    "mean_chunks_processed": round(mean_chunks, 2),
                    "total_escalations": total_escalations,
                    "token_usage": token_stats,
                },
                "quality": {
                    "mean_correctness": round(mean_c, 4),
                    "mean_faithfulness": round(mean_f, 4),
                    "mean_relevance": round(mean_rel, 4),
                    "mean_overall": round(mean_overall, 4),
                },
            }

        return summary

    @staticmethod
    def save_results(
        results: Sequence[QuestionEvaluationResult],
        output_file: str | Path,
    ) -> None:
        """Save evaluation results to JSON file.

        Args:
            results: Sequence of QuestionEvaluationResult instances.
            output_file: Destination file path.
        """
        out_p = Path(output_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        data = [r.model_dump() for r in results]
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Saved %d evaluation results to %s", len(results), out_p)


def run_evaluation(
    benchmark_path: str | Path | None = None,
    systems: Sequence[str] | None = None,
    question_ids: Sequence[str] | None = None,
    limit: int | None = None,
    judge: BaseJudge | None = None,
    output_path: str | Path | None = None,
    harness: EvaluationHarness | None = None,
) -> tuple[list[QuestionEvaluationResult], dict[str, Any]]:
    """Convenience entry point to execute evaluation run.

    Args:
        benchmark_path: Path to benchmark JSON file.
        systems: List of system names to evaluate.
        question_ids: Optional question ID subset.
        limit: Optional maximum question count.
        judge: Optional Judge instance (MockJudge or LLMJudge).
        output_path: Optional destination path to persist JSON results.
        harness: Optional pre-configured EvaluationHarness.

    Returns:
        tuple[list[QuestionEvaluationResult], dict[str, Any]]:
            List of per-question results and aggregate metrics dictionary.
    """
    raw_benchmark = load_benchmark(benchmark_path)
    questions = filter_benchmark(raw_benchmark, question_ids=question_ids, limit=limit)

    runner = EvaluationRunner(harness=harness, judge=judge)
    results = runner.evaluate_questions(questions=questions, systems=systems)
    summary = runner.aggregate_results(results)

    if output_path:
        runner.save_results(results, output_path)

    return results, summary
