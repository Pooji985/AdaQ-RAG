"""Focused test suite for AdaQ-RAG Evaluation Runner (Part 2).

Tests cover:
- Benchmark loading and validation
- Question filtering and subset selection
- System name normalization and selection
- EvidenceItem to RetrievalResult adapter
- Harness execution across all 4 systems (mock mode)
- Result construction and serialization
- MockJudge integration and system isolation
- Error/failure handling during evaluation
- Aggregate metric computation
- Saving and reloading evaluation results
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest

from adaq_rag.llm.mock import MockLLMProvider
from evaluation.rag.judge import MockJudge
from evaluation.rag.models import (
    AnswerQualityScores,
    EfficiencyMetrics,
    LLMUsage,
    QuestionEvaluationResult,
)
from evaluation.rag.runner import (
    DEFAULT_BENCHMARK_PATH,
    SUPPORTED_SYSTEMS,
    SYSTEM_ADAPTIVE_RAG,
    SYSTEM_BASIC_RAG,
    SYSTEM_FULL_ADAQ_RAG,
    SYSTEM_HYBRID_RAG,
    EvaluationHarness,
    EvaluationRunner,
    JUDGE_PARSE_FAILURE_PREFIX,
    adapt_evidence_to_retrieval_results,
    compute_benchmark_fingerprint,
    compute_question_fingerprint,
    compute_retrieval_metrics,
    filter_benchmark,
    load_benchmark,
    normalize_system_name,
)
from adaq_rag.retrieval.evidence import EvidenceItem


# =============================================================================
# 1. BENCHMARK LOADING & FILTERING TESTS
# =============================================================================


class TestBenchmarkLoadingAndFiltering:
    """Tests for loading and filtering the 90-question benchmark dataset."""

    def test_load_benchmark_default(self) -> None:
        data = load_benchmark()
        assert isinstance(data, list)
        assert len(data) == 90

        sample = data[0]
        assert "question_id" in sample
        assert "query" in sample
        assert "expected_intent" in sample
        assert "expected_complexity" in sample
        assert "expected_strategy" in sample
        assert "ground_truth_chunk_ids" in sample
        assert "reference_answer" in sample

    def test_load_benchmark_custom_path(self, tmp_path: Path) -> None:
        p = tmp_path / "custom_bench.json"
        items = [{"question_id": "test_01", "query": "Test query"}]
        p.write_text(json.dumps(items), encoding="utf-8")

        loaded = load_benchmark(p)
        assert len(loaded) == 1
        assert loaded[0]["question_id"] == "test_01"

    def test_load_benchmark_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_benchmark(tmp_path / "non_existent.json")

    def test_filter_benchmark_by_ids(self) -> None:
        questions = [
            {"question_id": "q1", "query": "one"},
            {"question_id": "q2", "query": "two"},
            {"question_id": "q3", "query": "three"},
        ]
        filtered = filter_benchmark(questions, question_ids=["q1", "q3"])
        assert len(filtered) == 2
        assert [q["question_id"] for q in filtered] == ["q1", "q3"]

    def test_filter_benchmark_by_limit(self) -> None:
        questions = [
            {"question_id": "q1"},
            {"question_id": "q2"},
            {"question_id": "q3"},
        ]
        filtered = filter_benchmark(questions, limit=2)
        assert len(filtered) == 2
        assert [q["question_id"] for q in filtered] == ["q1", "q2"]


# =============================================================================
# 2. SYSTEM SELECTION & NORMALIZATION TESTS
# =============================================================================


class TestSystemSelection:
    """Tests for system name normalization and selection."""

    def test_normalize_system_name(self) -> None:
        assert normalize_system_name("basic") == SYSTEM_BASIC_RAG
        assert normalize_system_name("Basic RAG") == SYSTEM_BASIC_RAG
        assert normalize_system_name("baseline_rag") == SYSTEM_BASIC_RAG

        assert normalize_system_name("hybrid") == SYSTEM_HYBRID_RAG
        assert normalize_system_name("Hybrid RAG") == SYSTEM_HYBRID_RAG

        assert normalize_system_name("adaptive") == SYSTEM_ADAPTIVE_RAG
        assert normalize_system_name("Adaptive RAG") == SYSTEM_ADAPTIVE_RAG

        assert normalize_system_name("full") == SYSTEM_FULL_ADAQ_RAG
        assert normalize_system_name("Full AdaQ-RAG") == SYSTEM_FULL_ADAQ_RAG
        assert normalize_system_name("adaq_rag") == SYSTEM_FULL_ADAQ_RAG

    def test_normalize_system_name_invalid(self) -> None:
        with pytest.raises(ValueError):
            normalize_system_name("unknown_pipeline")


# =============================================================================
# 3. EVIDENCE ADAPTER TESTS
# =============================================================================


class TestEvidenceAdapter:
    """Tests for adapt_evidence_to_retrieval_results adapter."""

    def test_adapt_evidence_item_to_retrieval_result(self) -> None:
        ev = EvidenceItem(
            chunk_id="chunk_101",
            content="Evidence content text",
            doc_id="doc_linear",
            doc_title="Linear Models",
            section_title="Ridge",
            section_level=2,
            initial_score=0.75,
            final_rerank_score=0.92,
            metadata={"source_url": "https://example.com"},
        )

        results = adapt_evidence_to_retrieval_results([ev])
        assert len(results) == 1
        res = results[0]

        assert res.chunk_id == "chunk_101"
        assert res.score == 0.92  # final_rerank_score mapped to score
        assert res.content == "Evidence content text"
        assert res.doc_id == "doc_linear"
        assert res.doc_title == "Linear Models"
        assert res.section_title == "Ridge"
        assert res.section_level == 2
        assert res.rerank_score == 0.92

    def test_adapt_evidence_fallback_initial_score(self) -> None:
        ev = EvidenceItem(
            chunk_id="chunk_102",
            content="Content",
            initial_score=0.65,
            final_rerank_score=None,
        )
        results = adapt_evidence_to_retrieval_results([ev])
        assert results[0].score == 0.65


# =============================================================================
# 4. HARNESS & ALL 4 SYSTEMS EXECUTION (MOCK MODE)
# =============================================================================


class TestHarnessExecution:
    """Tests executing all 4 systems using Mock Harness."""

    @pytest.fixture
    def mock_harness(self) -> EvaluationHarness:
        judge = MockJudge(
            default_scores=AnswerQualityScores(
                correctness=3.5,
                faithfulness=3.5,
                relevance=3.5,
                overall=3.5,
                reason="Mock judge score",
            )
        )
        return EvaluationHarness.create_mock(judge=judge)

    @pytest.fixture
    def sample_question(self) -> dict:
        return {
            "question_id": "test_fact_01",
            "query": "What is the default scoring strategy in GridSearchCV if scoring is None?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_strategy": "DENSE",
            "ground_truth_chunk_ids": ["mock_chunk_1"],
            "reference_answer": "It uses the estimator default score method.",
        }

    def test_run_basic_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_basic_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_BASIC_RAG
        assert res.efficiency.retrieval_calls == 1
        assert res.routing.predicted_strategy == "DENSE"
        assert res.routing.strategy_match is True
        assert res.retrieval_metrics.mrr > 0.0
        assert res.answer_scores.overall == 3.5

    def test_run_hybrid_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_hybrid_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_HYBRID_RAG
        assert res.efficiency.retrieval_calls == 2
        assert res.routing.predicted_strategy == "HYBRID"
        assert res.answer_scores.overall == 3.5

    def test_run_adaptive_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_adaptive_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_ADAPTIVE_RAG
        assert res.routing.predicted_strategy in ("DENSE", "HYBRID", "MULTI_STEP")
        assert res.answer_scores.overall == 3.5

    def test_run_full_adaq_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_full_adaq_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_FULL_ADAQ_RAG
        assert res.routing.predicted_strategy in ("DENSE", "HYBRID", "MULTI_STEP")
        assert res.answer_scores.overall == 3.5

    def test_dispatch_run_system(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        for sys_name in SUPPORTED_SYSTEMS:
            res = mock_harness.run_system(sys_name, sample_question)
            assert res.metadata["system_name"] == sys_name


# =============================================================================
# 5. RUNNER SUBSET EXECUTION & AGGREGATION TESTS
# =============================================================================


class TestRunnerExecutionAndAggregation:
    """Tests EvaluationRunner executing subsets and computing aggregate metrics."""

    def test_runner_evaluates_subset(self) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {
                "question_id": "q1",
                "query": "What is Ridge?",
                "expected_strategy": "DENSE",
                "ground_truth_chunk_ids": ["mock_chunk_1"],
                "reference_answer": "Ridge uses L2 norm.",
            },
            {
                "question_id": "q2",
                "query": "Compare Ridge and Lasso.",
                "expected_strategy": "HYBRID",
                "ground_truth_chunk_ids": ["mock_chunk_2"],
                "reference_answer": "Ridge is L2, Lasso is L1.",
            },
        ]

        # Run 2 questions on 2 systems (basic_rag, adaptive_rag)
        results = runner.evaluate_questions(
            questions=questions,
            systems=["basic_rag", "adaptive_rag"],
        )

        assert len(results) == 4  # 2 questions * 2 systems

        # Compute summary
        summary = runner.aggregate_results(results)
        assert SYSTEM_BASIC_RAG in summary
        assert SYSTEM_ADAPTIVE_RAG in summary

        basic_summary = summary[SYSTEM_BASIC_RAG]
        assert basic_summary["question_count"] == 2
        assert "retrieval" in basic_summary
        assert "routing" in basic_summary
        assert "efficiency" in basic_summary
        assert "quality" in basic_summary
        assert basic_summary["quality"]["mean_overall"] > 0.0

    def test_runner_failure_handling(self) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        # Mock run_system to fail for one system
        original_run = harness.run_system

        def flappy_run(sys_name: str, item: dict):
            if sys_name == SYSTEM_BASIC_RAG:
                raise RuntimeError("Simulated pipeline failure")
            return original_run(sys_name, item)

        harness.run_system = flappy_run

        runner = EvaluationRunner(harness=harness, judge=judge)
        questions = [{"question_id": "q_fail", "query": "Test query"}]

        results = runner.evaluate_questions(questions, systems=["basic_rag", "hybrid_rag"])
        assert len(results) == 2

        # Basic RAG should be caught with error metadata
        basic_res = next(r for r in results if r.metadata.get("system_name") == SYSTEM_BASIC_RAG)
        assert "error" in basic_res.metadata
        assert "Simulated pipeline failure" in basic_res.metadata["error"]
        assert basic_res.answer_scores.overall == 0.0

        # Hybrid RAG should have succeeded normally
        hybrid_res = next(r for r in results if r.metadata.get("system_name") == SYSTEM_HYBRID_RAG)
        assert "error" not in hybrid_res.metadata
        assert hybrid_res.answer_scores.overall > 0.0

    def test_save_and_reload_results(self, tmp_path: Path) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {
                "question_id": "q_save",
                "query": "Sample query",
                "ground_truth_chunk_ids": ["mock_chunk_1"],
                "reference_answer": "Ref ans",
            }
        ]
        results = runner.evaluate_questions(questions, systems=["basic_rag"])

        out_file = tmp_path / "eval_results.json"
        runner.save_results(results, out_file)

        assert out_file.is_file()
        loaded = runner.load_results(out_file)
        assert len(loaded) == 1
        assert loaded[0].question_id == "q_save"
        assert loaded[0].metadata["system_name"] == SYSTEM_BASIC_RAG

        # Verify atomic write left no temporary files behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_results_uses_atomic_os_replace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that save_results writes to a .tmp file first and calls os.replace."""
        import os
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r = QuestionEvaluationResult(
            question_id="q_atom",
            query="Query",
            reference_answer="Ref",
            generated_answer="Ans",
            retrieval_metrics=compute_retrieval_metrics([], []),
            answer_scores=AnswerQualityScores(
                correctness=4.0,
                faithfulness=4.0,
                relevance=4.0,
                overall=4.0,
                reason="Ok",
            ),
            efficiency=EfficiencyMetrics(latency_ms=10.0),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        out_file = tmp_path / "atomic_out.json"
        replace_called: list[tuple[str, str]] = []
        original_replace = os.replace

        def tracked_replace(src, dst):
            replace_called.append((str(src), str(dst)))
            original_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracked_replace)

        EvaluationRunner.save_results([r], out_file)
        assert len(replace_called) == 1
        src, dst = replace_called[0]
        assert src.endswith(".tmp")
        assert dst == str(out_file)
        assert out_file.is_file()

    def test_load_legacy_result_schema(self, tmp_path: Path) -> None:
        """Verify that older result formats lacking new metadata fields load cleanly."""
        legacy_data = [
            {
                "question_id": "legacy_q1",
                "query": "Old query",
                "reference_answer": "Old ref",
                "generated_answer": "Old ans",
                "retrieval_metrics": {
                    "recall_at_k": {"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
                    "mrr": 1.0,
                    "ndcg_at_k": {"1": 1.0, "3": 1.0, "5": 1.0, "10": 1.0},
                },
                "answer_scores": {
                    "correctness": 4.0,
                    "faithfulness": 4.0,
                    "relevance": 4.0,
                    "overall": 4.0,
                    "reason": "Fine",
                },
                "efficiency": {
                    "latency_ms": 120.0,
                    "retrieval_calls": 1,
                    "chunks_processed": 5,
                    "escalation_count": 0,
                    "token_usage": None,
                },
                "routing": None,
                "metadata": {
                    "system_name": "basic_rag"
                },
            }
        ]
        legacy_file = tmp_path / "legacy_eval.json"
        legacy_file.write_text(json.dumps(legacy_data), encoding="utf-8")

        loaded = EvaluationRunner.load_results(legacy_file)
        assert len(loaded) == 1
        assert loaded[0].question_id == "legacy_q1"
        assert loaded[0].metadata.get("system_name") == "basic_rag"
        assert loaded[0].metadata.get("error") is None

        summary = EvaluationRunner.aggregate_results(loaded)
        assert summary["basic_rag"]["successful_count"] == 1
        assert summary["basic_rag"]["failure_count"] == 0
        assert summary["basic_rag"]["error_rate"] == 0.0

    def test_aggregate_results_failure_separation(self) -> None:
        """Verify successful evaluations are averaged separately from failures."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r_success = QuestionEvaluationResult(
            question_id="q1",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="Answer 1",
            retrieval_metrics=compute_retrieval_metrics(["c1"], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=4.0,
                faithfulness=4.0,
                relevance=4.0,
                overall=4.0,
                reason="Good answer",
            ),
            efficiency=EfficiencyMetrics(latency_ms=1000.0, retrieval_calls=1, chunks_processed=5),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        r_failed = QuestionEvaluationResult(
            question_id="q2",
            query="Query 2",
            reference_answer="Ref 2",
            generated_answer="",
            retrieval_metrics=compute_retrieval_metrics([], ["c2"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="Evaluation execution failure: Connection timed out",
            ),
            efficiency=EfficiencyMetrics(latency_ms=0.0, retrieval_calls=0, chunks_processed=0),
            metadata={"system_name": SYSTEM_BASIC_RAG, "error": "Connection timed out"},
        )

        summary = EvaluationRunner.aggregate_results([r_success, r_failed])
        basic_stats = summary[SYSTEM_BASIC_RAG]

        assert basic_stats["question_count"] == 2
        assert basic_stats["successful_count"] == 1
        assert basic_stats["failure_count"] == 1
        assert basic_stats["error_rate"] == 0.5
        assert basic_stats["errors"] == ["Connection timed out"]

        # Quality metrics must reflect the successful evaluation (4.0), not diluted by the failure to 2.0
        assert basic_stats["quality"]["mean_overall"] == 4.0
        assert basic_stats["quality"]["mean_correctness"] == 4.0

        # Efficiency metrics must reflect successful evaluation
        assert basic_stats["efficiency"]["mean_latency_ms"] == 1000.0
        assert basic_stats["efficiency"]["mean_retrieval_calls"] == 1.0

    def test_aggregate_results_all_failed_handling(self) -> None:
        """Verify all-failed edge case returns zeros safely without ZeroDivisionError."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r_failed = QuestionEvaluationResult(
            question_id="q1",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="",
            retrieval_metrics=compute_retrieval_metrics([], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="Evaluation execution failure: Rate limited",
            ),
            efficiency=EfficiencyMetrics(latency_ms=0.0),
            metadata={"system_name": SYSTEM_BASIC_RAG, "error": "Rate limited"},
        )

        summary = EvaluationRunner.aggregate_results([r_failed])
        basic_stats = summary[SYSTEM_BASIC_RAG]

        assert basic_stats["question_count"] == 1
        assert basic_stats["successful_count"] == 0
        assert basic_stats["failure_count"] == 1
        assert basic_stats["error_rate"] == 1.0
        assert basic_stats["quality"]["mean_overall"] == 0.0
        assert basic_stats["retrieval"]["recall_at_1"] == 0.0
        assert basic_stats["efficiency"]["mean_latency_ms"] == 0.0

    def test_checkpoint_incremental_saving_and_deduplication(self, tmp_path: Path) -> None:
        """Verify incremental checkpointing saves progress and deduplicates upon resume."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {"question_id": "q1", "query": "Q1 query", "reference_answer": "A1"},
            {"question_id": "q2", "query": "Q2 query", "reference_answer": "A2"},
        ]
        chk_file = tmp_path / "checkpoint.json"

        # 1. Run only question 1
        res1 = runner.evaluate_questions([questions[0]], systems=["basic_rag"], checkpoint_path=chk_file)
        assert len(res1) == 1
        assert chk_file.is_file()

        # 2. Track harness execution count on second run
        call_count = 0
        original_run = harness.run_system

        def count_run(sys_name, item):
            nonlocal call_count
            call_count += 1
            return original_run(sys_name, item)

        harness.run_system = count_run

        # 3. Resume running with both questions 1 and 2
        res2 = runner.evaluate_questions(questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)

        # Total results must be 2 (q1 and q2), exactly deduplicated
        assert len(res2) == 2
        assert {r.question_id for r in res2} == {"q1", "q2"}

        # Only question 2 should have been executed by harness; question 1 was skipped
        assert call_count == 1

        # File on disk must match deduplicated 2 results
        on_disk = runner.load_results(chk_file)
        assert len(on_disk) == 2

    def test_checkpoint_incompatible_question_scope_raises(self, tmp_path: Path) -> None:
        """Verify attempting to resume with an incompatible question set raises ValueError."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        # Save checkpoint with question q_foreign
        foreign_questions = [{"question_id": "q_foreign", "query": "Foreign query"}]
        chk_file = tmp_path / "chk_foreign.json"
        runner.evaluate_questions(foreign_questions, systems=["basic_rag"], checkpoint_path=chk_file)

        # Target questions do not include q_foreign
        target_questions = [{"question_id": "q_target", "query": "Target query"}]
        with pytest.raises(ValueError, match="outside target benchmark questions"):
            runner.evaluate_questions(target_questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)

    def test_checkpoint_incompatible_system_scope_raises(self, tmp_path: Path) -> None:
        """Verify attempting to resume with incompatible systems raises ValueError."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [{"question_id": "q1", "query": "Query"}]
        chk_file = tmp_path / "chk_sys.json"
        # Save checkpoint with hybrid_rag
        runner.evaluate_questions(questions, systems=["hybrid_rag"], checkpoint_path=chk_file)

        # Resume with only basic_rag
        with pytest.raises(ValueError, match="outside selected target systems"):
            runner.evaluate_questions(questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)

    def test_checkpoint_resume_retries_failed_records_and_preserves_errors(self, tmp_path: Path) -> None:
        """Verify that failed records in checkpoint are retried on resume, and prior errors are preserved."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        chk_file = tmp_path / "chk_fail.json"
        questions = [{"question_id": "q1", "query": "Q1 query"}]

        # 1. First run fails
        harness.run_system = MagicMock(side_effect=RuntimeError("Network drop 500"))
        res_fail = runner.evaluate_questions(questions, systems=["basic_rag"], checkpoint_path=chk_file)
        assert len(res_fail) == 1
        assert "error" in res_fail[0].metadata

        # 2. Second run succeeds: harness recovers
        recovered_mock = EvaluationHarness.create_mock(judge=judge)
        runner.harness = recovered_mock

        res_retry = runner.evaluate_questions(questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)
        assert len(res_retry) == 1
        recovered_res = res_retry[0]

        # Prior error must NOT be silently discarded; preserved in previous_errors
        assert "error" not in recovered_res.metadata
        assert "previous_errors" in recovered_res.metadata
        assert "Network drop 500" in recovered_res.metadata["previous_errors"][0]
        assert recovered_res.answer_scores.overall > 0.0


# =============================================================================
# 6. JUDGE PARSING FALLBACK & FAILURE-AWARE AGGREGATION TESTS
# =============================================================================


class TestJudgeFailureAwareAggregation:
    """Tests isolating judge parsing fallbacks from genuine scores and RAG execution errors."""

    def test_aggregate_valid_judge_scores(self) -> None:
        """Verify normal, valid judge scores are included in quality averages."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r1 = QuestionEvaluationResult(
            question_id="q1",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="Answer 1",
            retrieval_metrics=compute_retrieval_metrics(["c1"], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=3.0,
                faithfulness=3.0,
                relevance=3.0,
                overall=3.0,
                reason="Evaluated by LLM Judge.",
            ),
            efficiency=EfficiencyMetrics(latency_ms=100.0, retrieval_calls=1, chunks_processed=3),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )
        r2 = QuestionEvaluationResult(
            question_id="q2",
            query="Query 2",
            reference_answer="Ref 2",
            generated_answer="Answer 2",
            retrieval_metrics=compute_retrieval_metrics(["c2"], ["c2"]),
            answer_scores=AnswerQualityScores(
                correctness=4.0,
                faithfulness=4.0,
                relevance=4.0,
                overall=4.0,
                reason="Evaluated by LLM Judge.",
            ),
            efficiency=EfficiencyMetrics(latency_ms=200.0, retrieval_calls=1, chunks_processed=5),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        summary = EvaluationRunner.aggregate_results([r1, r2])
        stats = summary[SYSTEM_BASIC_RAG]

        assert stats["question_count"] == 2
        assert stats["successful_count"] == 2
        assert stats["failure_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["errors"] == []

        assert stats["judge_successful_count"] == 2
        assert stats["judge_failure_count"] == 0
        assert stats["judge_error_rate"] == 0.0
        assert stats["judge_errors"] == []

        assert stats["quality"]["mean_overall"] == 3.5
        assert stats["quality"]["mean_correctness"] == 3.5
        assert stats["quality"]["scored_count"] == 2

        assert stats["efficiency"]["mean_latency_ms"] == 150.0

    def test_aggregate_judge_parsing_fallback(self) -> None:
        """Verify judge parsing fallback records preserve RAG efficiency/retrieval but are excluded from quality averages."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        parse_err_reason = f"{JUDGE_PARSE_FAILURE_PREFIX} Expecting value: line 1 column 1 (char 0)"
        r_judge_fallback = QuestionEvaluationResult(
            question_id="q1",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="Answer 1",
            retrieval_metrics=compute_retrieval_metrics(["c1"], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason=parse_err_reason,
            ),
            efficiency=EfficiencyMetrics(latency_ms=250.0, retrieval_calls=1, chunks_processed=4),
            metadata={"system_name": SYSTEM_BASIC_RAG},  # No metadata["error"], RAG succeeded
        )

        summary = EvaluationRunner.aggregate_results([r_judge_fallback])
        stats = summary[SYSTEM_BASIC_RAG]

        # RAG pipeline execution is marked successful
        assert stats["question_count"] == 1
        assert stats["successful_count"] == 1
        assert stats["failure_count"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["errors"] == []

        # Retrieval & efficiency metrics are completely preserved
        assert stats["retrieval"]["recall_at_1"] == 1.0
        assert stats["retrieval"]["mrr"] == 1.0
        assert stats["efficiency"]["mean_latency_ms"] == 250.0
        assert stats["efficiency"]["total_retrieval_calls"] == 1

        # Judge parsing failure is explicitly recorded
        assert stats["judge_successful_count"] == 0
        assert stats["judge_failure_count"] == 1
        assert stats["judge_error_rate"] == 1.0
        assert stats["judge_errors"] == [parse_err_reason]

        # Quality scores exclude the invalid 0.0 fallback
        assert stats["quality"]["mean_overall"] == 0.0
        assert stats["quality"]["scored_count"] == 0

    def test_aggregate_rag_execution_failure(self) -> None:
        """Verify RAG execution failures report pipeline errors and zero judge counts."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r_rag_fail = QuestionEvaluationResult(
            question_id="q_fail",
            query="Query Fail",
            reference_answer="Ref",
            generated_answer="",
            retrieval_metrics=compute_retrieval_metrics([], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="Evaluation execution failure: ConnectionRefusedError: Connection refused",
            ),
            efficiency=EfficiencyMetrics(latency_ms=0.0),
            metadata={"system_name": SYSTEM_BASIC_RAG, "error": "ConnectionRefusedError: Connection refused"},
        )

        summary = EvaluationRunner.aggregate_results([r_rag_fail])
        stats = summary[SYSTEM_BASIC_RAG]

        assert stats["question_count"] == 1
        assert stats["successful_count"] == 0
        assert stats["failure_count"] == 1
        assert stats["error_rate"] == 1.0
        assert stats["errors"] == ["ConnectionRefusedError: Connection refused"]

        # No judge evaluations were attempted or completed
        assert stats["judge_successful_count"] == 0
        assert stats["judge_failure_count"] == 0
        assert stats["judge_error_rate"] == 0.0
        assert stats["judge_errors"] == []

        assert stats["quality"]["mean_overall"] == 0.0
        assert stats["quality"]["scored_count"] == 0

    def test_aggregate_mixture_of_all_three(self) -> None:
        """Verify mixture of valid judge scores, judge parsing fallback, and RAG execution failure."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        # 1. Valid judge evaluation
        r_valid = QuestionEvaluationResult(
            question_id="q_valid",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="Answer 1",
            retrieval_metrics=compute_retrieval_metrics(["c1"], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=4.0,
                faithfulness=4.0,
                relevance=4.0,
                overall=4.0,
                reason="Evaluated by LLM Judge.",
            ),
            efficiency=EfficiencyMetrics(latency_ms=100.0, retrieval_calls=1, chunks_processed=2),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        # 2. Judge parsing fallback (RAG succeeded, judge output unparseable)
        parse_err = f"{JUDGE_PARSE_FAILURE_PREFIX} Unterminated string starting at line 1"
        r_judge_fail = QuestionEvaluationResult(
            question_id="q_judge_fail",
            query="Query 2",
            reference_answer="Ref 2",
            generated_answer="Answer 2",
            retrieval_metrics=compute_retrieval_metrics(["c2"], ["c2"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason=parse_err,
            ),
            efficiency=EfficiencyMetrics(latency_ms=200.0, retrieval_calls=1, chunks_processed=4),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        # 3. RAG execution failure
        r_rag_fail = QuestionEvaluationResult(
            question_id="q_rag_fail",
            query="Query 3",
            reference_answer="Ref 3",
            generated_answer="",
            retrieval_metrics=compute_retrieval_metrics([], ["c3"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason="Evaluation execution failure: Groq 429 Rate Limit",
            ),
            efficiency=EfficiencyMetrics(latency_ms=0.0),
            metadata={"system_name": SYSTEM_BASIC_RAG, "error": "Groq 429 Rate Limit"},
        )

        summary = EvaluationRunner.aggregate_results([r_valid, r_judge_fail, r_rag_fail])
        stats = summary[SYSTEM_BASIC_RAG]

        # Overall question and RAG execution counts
        assert stats["question_count"] == 3
        assert stats["successful_count"] == 2
        assert stats["failure_count"] == 1
        assert stats["error_rate"] == round(1 / 3, 4)
        assert stats["errors"] == ["Groq 429 Rate Limit"]

        # Retrieval and efficiency averages reflect the 2 successful RAG runs
        assert stats["retrieval"]["recall_at_1"] == 1.0
        assert stats["efficiency"]["mean_latency_ms"] == 150.0  # (100 + 200) / 2
        assert stats["efficiency"]["total_retrieval_calls"] == 2

        # Judge parsing stats reflect 1 valid and 1 parse failure out of 2 attempted evaluations
        assert stats["judge_successful_count"] == 1
        assert stats["judge_failure_count"] == 1
        assert stats["judge_error_rate"] == 0.5  # 1 fail / 2 attempted judge calls
        assert stats["judge_errors"] == [parse_err]

        # Quality averages are computed SOLELY from the 1 valid judge score (4.0), not diluted by the 0.0 fallback!
        assert stats["quality"]["mean_overall"] == 4.0
        assert stats["quality"]["mean_correctness"] == 4.0
        assert stats["quality"]["mean_faithfulness"] == 4.0
        assert stats["quality"]["mean_relevance"] == 4.0
        assert stats["quality"]["scored_count"] == 1

    def test_aggregate_no_valid_judge_scores(self) -> None:
        """Verify safe handling when all successful RAG executions have judge parsing fallbacks."""
        from evaluation.rag.metrics import compute_retrieval_metrics
        from evaluation.rag.models import EfficiencyMetrics

        r1 = QuestionEvaluationResult(
            question_id="q1",
            query="Query 1",
            reference_answer="Ref 1",
            generated_answer="Answer 1",
            retrieval_metrics=compute_retrieval_metrics(["c1"], ["c1"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason=f"{JUDGE_PARSE_FAILURE_PREFIX} Unexpected EOF",
            ),
            efficiency=EfficiencyMetrics(latency_ms=120.0, retrieval_calls=1, chunks_processed=2),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )
        r2 = QuestionEvaluationResult(
            question_id="q2",
            query="Query 2",
            reference_answer="Ref 2",
            generated_answer="Answer 2",
            retrieval_metrics=compute_retrieval_metrics(["c2"], ["c2"]),
            answer_scores=AnswerQualityScores(
                correctness=0.0,
                faithfulness=0.0,
                relevance=0.0,
                overall=0.0,
                reason=f"{JUDGE_PARSE_FAILURE_PREFIX} Unmatched brace",
            ),
            efficiency=EfficiencyMetrics(latency_ms=180.0, retrieval_calls=1, chunks_processed=4),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        summary = EvaluationRunner.aggregate_results([r1, r2])
        stats = summary[SYSTEM_BASIC_RAG]

        assert stats["question_count"] == 2
        assert stats["successful_count"] == 2
        assert stats["failure_count"] == 0
        assert stats["error_rate"] == 0.0

        # Retrieval and efficiency are preserved
        assert stats["retrieval"]["recall_at_1"] == 1.0
        assert stats["efficiency"]["mean_latency_ms"] == 150.0

        # Judge stats show 100% judge parse failure
        assert stats["judge_successful_count"] == 0
        assert stats["judge_failure_count"] == 2
        assert stats["judge_error_rate"] == 1.0
        assert len(stats["judge_errors"]) == 2

        # Quality metrics safely default to 0.0 without ZeroDivisionError
        assert stats["quality"]["mean_overall"] == 0.0
        assert stats["quality"]["mean_correctness"] == 0.0
        assert stats["quality"]["mean_faithfulness"] == 0.0
        assert stats["quality"]["mean_relevance"] == 0.0
        assert stats["quality"]["scored_count"] == 0


# =============================================================================
# 7. BENCHMARK CONTENT FINGERPRINTING & RESUME VALIDATION TESTS
# =============================================================================


class TestBenchmarkContentFingerprinting:
    """Tests deterministic benchmark content fingerprinting and checkpoint integrity validation."""

    def test_identical_content_produces_identical_fingerprint(self) -> None:
        """Verify identical benchmark content produces matching SHA-256 fingerprints."""
        q1 = {
            "question_id": "q1",
            "query": "What is the penalty in Ridge regression?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_strategy": "DENSE",
            "ground_truth_chunk_ids": ["chunk_ridge_1"],
            "reference_answer": "Ridge uses L2 regularization.",
        }
        q2 = {
            "question_id": "q1",
            "query": "What is the penalty in Ridge regression?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_strategy": "DENSE",
            "ground_truth_chunk_ids": ["chunk_ridge_1"],
            "reference_answer": "Ridge uses L2 regularization.",
        }

        fp1 = compute_benchmark_fingerprint([q1])
        fp2 = compute_benchmark_fingerprint([q2])
        assert fp1 == fp2
        assert len(fp1) == 64  # Valid SHA-256 hex string

        # Single question fingerprint matches
        assert compute_question_fingerprint(q1) == compute_question_fingerprint(q2)
        assert EvaluationRunner.compute_benchmark_fingerprint([q1]) == fp1

    def test_changed_question_text_causes_mismatch(self) -> None:
        """Verify modifying query text changes the fingerprint."""
        base_q = {
            "question_id": "q1",
            "query": "What is Ridge regression?",
            "reference_answer": "Ridge uses L2 regularization.",
        }
        changed_q = {
            "question_id": "q1",
            "query": "What is Lasso regression?",  # Changed text
            "reference_answer": "Ridge uses L2 regularization.",
        }

        fp_base = compute_benchmark_fingerprint([base_q])
        fp_changed = compute_benchmark_fingerprint([changed_q])
        assert fp_base != fp_changed
        assert compute_question_fingerprint(base_q) != compute_question_fingerprint(changed_q)

    def test_changed_reference_answer_causes_mismatch(self) -> None:
        """Verify modifying reference answer changes the fingerprint."""
        base_q = {
            "question_id": "q1",
            "query": "What is Ridge regression?",
            "reference_answer": "Ridge uses L2 regularization.",
        }
        changed_q = {
            "question_id": "q1",
            "query": "What is Ridge regression?",
            "reference_answer": "Ridge uses L1 regularization.",  # Changed reference answer
        }

        fp_base = compute_benchmark_fingerprint([base_q])
        fp_changed = compute_benchmark_fingerprint([changed_q])
        assert fp_base != fp_changed
        assert compute_question_fingerprint(base_q) != compute_question_fingerprint(changed_q)

    def test_json_key_ordering_does_not_affect_fingerprint(self) -> None:
        """Verify key ordering in dict does not affect the deterministic fingerprint."""
        q_order_a = {
            "question_id": "q1",
            "query": "Explain ElasticNet.",
            "reference_answer": "Combines L1 and L2.",
            "expected_strategy": "HYBRID",
            "expected_intent": "CONCEPT_EXPLANATION",
        }
        # Inverted key insertion order
        q_order_b = {
            "expected_intent": "CONCEPT_EXPLANATION",
            "expected_strategy": "HYBRID",
            "reference_answer": "Combines L1 and L2.",
            "query": "Explain ElasticNet.",
            "question_id": "q1",
        }

        fp_a = compute_benchmark_fingerprint([q_order_a])
        fp_b = compute_benchmark_fingerprint([q_order_b])
        assert fp_a == fp_b
        assert compute_question_fingerprint(q_order_a) == compute_question_fingerprint(q_order_b)

    def test_checkpoint_resume_rejects_changed_question_content(self, tmp_path: Path) -> None:
        """Verify attempting to resume from a checkpoint with altered question text raises ValueError."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        original_questions = [
            {"question_id": "q1", "query": "Original query text", "reference_answer": "Ref answer"}
        ]
        chk_file = tmp_path / "chk_content_mismatch.json"

        # 1. Run with original question content
        runner.evaluate_questions(original_questions, systems=["basic_rag"], checkpoint_path=chk_file)
        assert chk_file.is_file()

        # 2. Attempt to resume with modified query text
        modified_questions = [
            {"question_id": "q1", "query": "Altered query text", "reference_answer": "Ref answer"}
        ]
        with pytest.raises(ValueError, match="Checkpoint benchmark content mismatch for question 'q1'"):
            runner.evaluate_questions(modified_questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)

    def test_checkpoint_resume_rejects_changed_reference_answer(self, tmp_path: Path) -> None:
        """Verify attempting to resume from a checkpoint with altered reference answer raises ValueError."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        original_questions = [
            {"question_id": "q1", "query": "Query text", "reference_answer": "Original reference answer"}
        ]
        chk_file = tmp_path / "chk_ref_mismatch.json"

        # 1. Run with original reference answer
        runner.evaluate_questions(original_questions, systems=["basic_rag"], checkpoint_path=chk_file)
        assert chk_file.is_file()

        # 2. Attempt to resume with altered reference answer
        modified_questions = [
            {"question_id": "q1", "query": "Query text", "reference_answer": "Altered reference answer"}
        ]
        with pytest.raises(ValueError, match="Checkpoint benchmark content mismatch for question 'q1'"):
            runner.evaluate_questions(modified_questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)

    def test_checkpoint_resume_rejects_legacy_checkpoint_without_fingerprint(self, tmp_path: Path) -> None:
        """Verify unverified legacy checkpoints lacking fingerprints are rejected on resume."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        legacy_data = [
            {
                "question_id": "q_legacy",
                "query": "Old query",
                "reference_answer": "Old ref",
                "generated_answer": "Old ans",
                "retrieval_metrics": {
                    "recall_at_k": {"1": 1.0},
                    "mrr": 1.0,
                    "ndcg_at_k": {"1": 1.0},
                },
                "answer_scores": {
                    "correctness": 3.0,
                    "faithfulness": 3.0,
                    "relevance": 3.0,
                    "overall": 3.0,
                    "reason": "Fine",
                },
                "efficiency": {
                    "latency_ms": 100.0,
                    "retrieval_calls": 1,
                    "chunks_processed": 2,
                    "escalation_count": 0,
                    "token_usage": None,
                },
                "routing": None,
                "metadata": {
                    "system_name": "basic_rag"
                    # Intentionally omitting question_fingerprint and benchmark_fingerprint
                },
            }
        ]
        legacy_chk = tmp_path / "legacy_checkpoint.json"
        legacy_chk.write_text(json.dumps(legacy_data), encoding="utf-8")

        target_questions = [{"question_id": "q_legacy", "query": "Old query", "reference_answer": "Old ref"}]
        with pytest.raises(ValueError, match="missing benchmark fingerprint"):
            runner.evaluate_questions(target_questions, systems=["basic_rag"], checkpoint_path=legacy_chk, resume=True)

    def test_checkpoint_resume_preserves_valid_resume_behavior(self, tmp_path: Path) -> None:
        """Verify normal resume with matching fingerprint skips already completed questions."""
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {"question_id": "q1", "query": "Q1 query", "reference_answer": "A1"},
            {"question_id": "q2", "query": "Q2 query", "reference_answer": "A2"},
        ]
        chk_file = tmp_path / "valid_resume_chk.json"

        # 1. Run first question
        runner.evaluate_questions([questions[0]], systems=["basic_rag"], checkpoint_path=chk_file)
        assert chk_file.is_file()

        # 2. Track calls on resume
        harness_calls = 0
        original_run = harness.run_system

        def tracked_run(sys_name, item):
            nonlocal harness_calls
            harness_calls += 1
            return original_run(sys_name, item)

        harness.run_system = tracked_run

        # 3. Resume with both questions
        results = runner.evaluate_questions(questions, systems=["basic_rag"], checkpoint_path=chk_file, resume=True)
        assert len(results) == 2
        # Only question 2 should be executed; question 1 was safely resumed with verified fingerprint
        assert harness_calls == 1
        assert {r.question_id for r in results} == {"q1", "q2"}


# =============================================================================
# 6. EVALUATION RELIABILITY & JUDGE TOKEN TESTS
# =============================================================================


class TestEvaluationReliabilityAndJudgeTokens:
    """Tests for empty-response validation, skipped judging, resume recovery, and judge token tracking."""

    def test_empty_generated_answer_skips_judge_and_marks_failure(self) -> None:
        """Verify an empty string response from the generator skips judging and records an error."""
        judge = MockJudge()
        judge.evaluate = MagicMock(wraps=judge.evaluate)  # type: ignore[method-assign]
        harness = EvaluationHarness.create_mock(judge=judge, default_response="")
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {"question_id": "q_empty", "query": "What is Ridge?", "reference_answer": "Ref"}
        ]
        results = runner.evaluate_questions(questions, systems=["basic_rag"])
        assert len(results) == 1
        res = results[0]

        # Judge evaluate must NOT be called for empty response
        judge.evaluate.assert_not_called()
        assert res.generated_answer == ""
        assert "empty or whitespace-only answer" in res.metadata.get("error", "").lower()
        assert "empty or whitespace-only answer" in res.metadata.get("generation_error", "").lower()
        assert res.answer_scores.overall == 0.0
        assert "skipped" in res.answer_scores.reason.lower()
        assert res.efficiency.judge_token_usage is None

    def test_whitespace_generated_answer_skips_judge_across_systems(self) -> None:
        """Verify whitespace-only answers across systems skip judging and record failures."""
        judge = MockJudge()
        judge.evaluate = MagicMock(wraps=judge.evaluate)  # type: ignore[method-assign]
        harness = EvaluationHarness.create_mock(judge=judge, default_response="   \n\t  ")

        item = {"question_id": "q_ws", "query": "Test query", "reference_answer": "Ref"}

        for sys_name in [SYSTEM_BASIC_RAG, SYSTEM_HYBRID_RAG, SYSTEM_ADAPTIVE_RAG, SYSTEM_FULL_ADAQ_RAG]:
            res = harness.run_system(sys_name, item)
            assert res.metadata.get("error") is not None
            assert "empty or whitespace" in res.metadata["error"].lower()
            assert res.answer_scores.overall == 0.0
            assert res.efficiency.judge_token_usage is None

        # Verify judge was never called for any system
        judge.evaluate.assert_not_called()

    def test_resume_retries_empty_and_failed_records(self, tmp_path: Path) -> None:
        """Verify resume mechanism retries both explicit errors and empty-response records without corrupting valid ones."""
        questions = [
            {"question_id": "q1", "query": "Q1 query", "reference_answer": "A1"},
            {"question_id": "q2", "query": "Q2 query", "reference_answer": "A2"},
            {"question_id": "q3", "query": "Q3 query", "reference_answer": "A3"},
            {"question_id": "q4", "query": "Q4 query", "reference_answer": "A4"},
        ]
        bench_fp = compute_benchmark_fingerprint(questions)

        # Build checkpoint with:
        # q1: valid success
        # q2: explicit error (e.g. rate limit)
        # q3: empty answer with error=None (simulating old buggy checkpoint)
        # q4: whitespace answer with error=None
        from evaluation.rag.metrics import compute_retrieval_metrics

        checkpoint_records = [
            QuestionEvaluationResult(
                question_id="q1",
                query="Q1 query",
                reference_answer="A1",
                generated_answer="Valid answer for Q1.",
                retrieval_metrics=compute_retrieval_metrics([], []),
                answer_scores=AnswerQualityScores(
                    correctness=4.0, faithfulness=4.0, relevance=4.0, overall=4.0, reason="Good"
                ),
                efficiency=EfficiencyMetrics(latency_ms=100.0),
                metadata={
                    "system_name": SYSTEM_BASIC_RAG,
                    "benchmark_fingerprint": bench_fp,
                    "question_fingerprint": compute_question_fingerprint(questions[0]),
                },
            ),
            QuestionEvaluationResult(
                question_id="q2",
                query="Q2 query",
                reference_answer="A2",
                generated_answer="",
                retrieval_metrics=compute_retrieval_metrics([], []),
                answer_scores=AnswerQualityScores(
                    correctness=0.0, faithfulness=0.0, relevance=0.0, overall=0.0, reason="Failed"
                ),
                efficiency=EfficiencyMetrics(latency_ms=10.0),
                metadata={
                    "system_name": SYSTEM_BASIC_RAG,
                    "benchmark_fingerprint": bench_fp,
                    "question_fingerprint": compute_question_fingerprint(questions[1]),
                    "error": "HTTP 429: Rate limit exceeded",
                },
            ),
            QuestionEvaluationResult(
                question_id="q3",
                query="Q3 query",
                reference_answer="A3",
                generated_answer="",  # empty string, no error key (legacy edge case)
                retrieval_metrics=compute_retrieval_metrics([], []),
                answer_scores=AnswerQualityScores(
                    correctness=0.0, faithfulness=0.0, relevance=0.0, overall=0.0, reason="No answer"
                ),
                efficiency=EfficiencyMetrics(latency_ms=50.0),
                metadata={
                    "system_name": SYSTEM_BASIC_RAG,
                    "benchmark_fingerprint": bench_fp,
                    "question_fingerprint": compute_question_fingerprint(questions[2]),
                },
            ),
            QuestionEvaluationResult(
                question_id="q4",
                query="Q4 query",
                reference_answer="A4",
                generated_answer="   \n   ",  # whitespace string
                retrieval_metrics=compute_retrieval_metrics([], []),
                answer_scores=AnswerQualityScores(
                    correctness=0.0, faithfulness=0.0, relevance=0.0, overall=0.0, reason="Whitespace"
                ),
                efficiency=EfficiencyMetrics(latency_ms=50.0),
                metadata={
                    "system_name": SYSTEM_BASIC_RAG,
                    "benchmark_fingerprint": bench_fp,
                    "question_fingerprint": compute_question_fingerprint(questions[3]),
                },
            ),
        ]

        chk_file = tmp_path / "resume_recovery_chk.json"
        EvaluationRunner.save_results(checkpoint_records, chk_file)

        judge = MockJudge(default_scores=AnswerQualityScores(correctness=3.5, faithfulness=3.5, relevance=3.5, overall=3.5, reason="Good"))
        harness = EvaluationHarness.create_mock(judge=judge, default_response="Retried valid answer.")
        runner = EvaluationRunner(harness=harness, judge=judge)

        executed_qids: list[str] = []
        original_run = harness.run_system

        def tracked_run(sys_name: str, item: dict[str, Any]):
            executed_qids.append(item["question_id"])
            return original_run(sys_name, item)

        harness.run_system = tracked_run

        results = runner.evaluate_questions(
            questions=questions,
            systems=["basic_rag"],
            checkpoint_path=chk_file,
            resume=True,
        )

        assert len(results) == 4
        # q1 was valid, should NOT be executed
        # q2 (error), q3 (empty answer), and q4 (whitespace answer) MUST be executed
        assert "q1" not in executed_qids
        assert "q2" in executed_qids
        assert "q3" in executed_qids
        assert "q4" in executed_qids

        # All results should now be valid and successful
        for r in results:
            assert r.metadata.get("error") is None
            assert len(r.generated_answer.strip()) > 0
            assert r.answer_scores.overall > 0.0

    def test_judge_token_usage_captured_in_efficiency(self) -> None:
        """Verify judge token usage is stored separately in EfficiencyMetrics."""
        expected_judge_usage = LLMUsage(
            prompt_tokens=150,
            completion_tokens=45,
            total_tokens=195,
        )
        judge = MockJudge(
            default_scores=AnswerQualityScores(correctness=3.8, faithfulness=3.8, relevance=3.8, overall=3.8, reason="Good"),
            default_token_usage=expected_judge_usage,
        )
        harness = EvaluationHarness.create_mock(judge=judge, default_response="A well-formed answer.")
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {"question_id": "q_tokens", "query": "Test query", "reference_answer": "Ref"}
        ]
        results = runner.evaluate_questions(questions, systems=["basic_rag"])
        assert len(results) == 1
        res = results[0]

        assert res.efficiency.judge_token_usage is not None
        assert res.efficiency.judge_token_usage.prompt_tokens == 150
        assert res.efficiency.judge_token_usage.completion_tokens == 45
        assert res.efficiency.judge_token_usage.total_tokens == 195
        # Generation token_usage must remain distinct from judge_token_usage
        assert res.efficiency.token_usage != res.efficiency.judge_token_usage

    def test_judge_token_usage_none_when_unsupported(self) -> None:
        """Verify judge token usage is None when judge provider does not return tokens."""
        judge = MockJudge(
            default_scores=AnswerQualityScores(correctness=3.8, faithfulness=3.8, relevance=3.8, overall=3.8, reason="Good"),
            default_token_usage=None,
        )
        harness = EvaluationHarness.create_mock(judge=judge, default_response="A valid answer.")
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {"question_id": "q_no_tokens", "query": "Test query", "reference_answer": "Ref"}
        ]
        results = runner.evaluate_questions(questions, systems=["basic_rag"])
        assert len(results) == 1
        res = results[0]

        assert res.efficiency.judge_token_usage is None
        # Does not fabricate fake numbers
        assert res.answer_scores.overall == 3.8

    def test_aggregate_results_aggregates_judge_token_usage(self) -> None:
        """Verify aggregate_results computes separate totals for judge tokens."""
        from evaluation.rag.metrics import compute_retrieval_metrics

        r1 = QuestionEvaluationResult(
            question_id="q1",
            query="Q1",
            reference_answer="Ref1",
            generated_answer="Ans1",
            retrieval_metrics=compute_retrieval_metrics([], []),
            answer_scores=AnswerQualityScores(correctness=4.0, faithfulness=4.0, relevance=4.0, overall=4.0, reason="Good"),
            efficiency=EfficiencyMetrics(
                latency_ms=100.0,
                token_usage=LLMUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
                judge_token_usage=LLMUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
            ),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )
        r2 = QuestionEvaluationResult(
            question_id="q2",
            query="Q2",
            reference_answer="Ref2",
            generated_answer="Ans2",
            retrieval_metrics=compute_retrieval_metrics([], []),
            answer_scores=AnswerQualityScores(correctness=3.0, faithfulness=3.0, relevance=3.0, overall=3.0, reason="Good"),
            efficiency=EfficiencyMetrics(
                latency_ms=120.0,
                token_usage=LLMUsage(prompt_tokens=60, completion_tokens=30, total_tokens=90),
                judge_token_usage=LLMUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
            ),
            metadata={"system_name": SYSTEM_BASIC_RAG},
        )

        summary = EvaluationRunner.aggregate_results([r1, r2])
        basic_eff = summary[SYSTEM_BASIC_RAG]["efficiency"]

        # Generation token_usage
        assert basic_eff["token_usage"]["prompt_tokens"] == 110
        assert basic_eff["token_usage"]["completion_tokens"] == 50
        assert basic_eff["token_usage"]["total_tokens"] == 160

        # Judge token_usage
        assert basic_eff["judge_token_usage"]["prompt_tokens"] == 300
        assert basic_eff["judge_token_usage"]["completion_tokens"] == 80
        assert basic_eff["judge_token_usage"]["total_tokens"] == 380

    def test_legacy_checkpoint_compatibility_without_judge_tokens(self, tmp_path: Path) -> None:
        """Verify legacy checkpoints without judge_token_usage deserialize and aggregate smoothly."""
        legacy_data = [
            {
                "question_id": "legacy_q",
                "query": "Old query",
                "reference_answer": "Old ref",
                "generated_answer": "Old ans",
                "retrieval_metrics": {
                    "recall_at_k": {"1": 1.0},
                    "mrr": 1.0,
                    "ndcg_at_k": {"1": 1.0},
                },
                "answer_scores": {
                    "correctness": 3.5,
                    "faithfulness": 3.5,
                    "relevance": 3.5,
                    "overall": 3.5,
                    "reason": "Ok",
                },
                "efficiency": {
                    "latency_ms": 100.0,
                    "retrieval_calls": 1,
                    "chunks_processed": 2,
                    "escalation_count": 0,
                    "token_usage": None,
                    # judge_token_usage completely omitted
                },
                "routing": None,
                "metadata": {
                    "system_name": "basic_rag",
                },
            }
        ]
        p = tmp_path / "legacy_eval_chk.json"
        p.write_text(json.dumps(legacy_data), encoding="utf-8")

        loaded = EvaluationRunner.load_results(p)
        assert len(loaded) == 1
        assert loaded[0].efficiency.judge_token_usage is None

        summary = EvaluationRunner.aggregate_results(loaded)
        assert summary["basic_rag"]["efficiency"]["judge_token_usage"] is None


    def test_evaluate_questions_tpd_exhaustion_early_abort(self, tmp_path: Path) -> None:
        """Verify runner aborts evaluation loop early when TPD quota exhaustion occurs."""
        mock_harness = MagicMock()
        mock_harness.run_system.side_effect = RuntimeError(
            "Rate limit reached for model openai/gpt-oss-20b on tokens per day (TPD): Limit 200000, Used 198770."
        )

        runner = EvaluationRunner(harness=mock_harness)
        questions = [
            {"question_id": "q1", "query": "query 1", "reference_answer": "ref 1"},
            {"question_id": "q2", "query": "query 2", "reference_answer": "ref 2"},
        ]
        chk_path = tmp_path / "tpd_abort_chk.json"

        results = runner.evaluate_questions(
            questions=questions,
            systems=["basic_rag", "hybrid_rag"],
            checkpoint_path=chk_path,
        )

        # Should abort on very first system of q1, so only 1 result recorded instead of 4
        assert len(results) == 1
        assert results[0].question_id == "q1"
        assert "tokens per day (TPD)" in results[0].metadata["error"]
        # Checkpoint should have been saved
        assert chk_path.is_file()

    def test_evaluate_questions_max_eval_questions_limit(self, tmp_path: Path) -> None:
        """Verify runner respects max_eval_questions limit on newly executed questions."""
        def mock_run_system(sys_name: str, item: dict[str, Any]) -> QuestionEvaluationResult:
            return QuestionEvaluationResult(
                question_id=item["question_id"],
                query=item.get("query", ""),
                reference_answer=item.get("reference_answer", ""),
                generated_answer="dummy ans",
                retrieval_metrics=compute_retrieval_metrics([], []),
                answer_scores=AnswerQualityScores(correctness=3.0, faithfulness=3.0, relevance=3.0, overall=3.0, reason="ok"),
                efficiency=EfficiencyMetrics(latency_ms=10.0),
                routing=None,
                metadata={"system_name": sys_name},
            )

        mock_harness = MagicMock()
        mock_harness.run_system.side_effect = mock_run_system

        runner = EvaluationRunner(harness=mock_harness)
        questions = [
            {"question_id": "q1", "query": "q1", "reference_answer": "r1"},
            {"question_id": "q2", "query": "q2", "reference_answer": "r2"},
            {"question_id": "q3", "query": "q3", "reference_answer": "r3"},
        ]
        chk_path = tmp_path / "limit_chk.json"

        results = runner.evaluate_questions(
            questions=questions,
            systems=["basic_rag"],
            checkpoint_path=chk_path,
            max_eval_questions=1,
        )

        # Should only evaluate 1 question and stop
        assert len(results) == 1
        assert results[0].question_id == "q1"
        assert chk_path.is_file()





