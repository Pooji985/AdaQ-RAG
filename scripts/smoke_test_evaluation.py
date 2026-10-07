"""Smoke test script for Part 2 Evaluation Runner.

Runs 4 representative questions across all 4 evaluation systems:
1. Basic RAG (Phase 5)
2. Hybrid RAG (Phase 8)
3. Adaptive RAG (Phases 6 & 7)
4. Full AdaQ-RAG (Phases 6, 7, 8 & 9)

Representative questions:
- SIMPLE: rag_fact_01 (FACT_LOOKUP, SIMPLE, DENSE)
- MEDIUM: rag_fact_19 (FACT_LOOKUP, MEDIUM, HYBRID)
- COMPLEX: rag_expl_14 (EXPLANATION, COMPLEX, MULTI_STEP)
- MULTI_CONCEPT: rag_multi_01 (MULTI_CONCEPT, MEDIUM, HYBRID)
"""

import json
from pathlib import Path
import sys

# Ensure src and root are in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaq_rag.core.logging import setup_logging
from adaq_rag.llm.mock import MockLLMProvider
from evaluation.rag.judge import MockJudge
from evaluation.rag.runner import (
    SUPPORTED_SYSTEMS,
    EvaluationHarness,
    EvaluationRunner,
    filter_benchmark,
    load_benchmark,
)

SMOKE_TEST_QUESTION_IDS = [
    "rag_fact_01",   # SIMPLE
    "rag_fact_19",   # MEDIUM
    "rag_expl_14",   # COMPLEX
    "rag_multi_01",  # MULTI_CONCEPT
]


def run_smoke_test() -> None:
    setup_logging("INFO")
    print("=" * 80)
    print("AdaQ-RAG Part 2: Evaluation Runner Smoke Test (4 Questions x 4 Systems)")
    print("=" * 80)

    # 1. Load benchmark & filter target questions
    all_questions = load_benchmark()
    smoke_questions = filter_benchmark(all_questions, question_ids=SMOKE_TEST_QUESTION_IDS)

    print(f"\nLoaded {len(smoke_questions)} representative questions:")
    for q in smoke_questions:
        print(f"  - [{q['question_id']}] ({q.get('expected_complexity')}/{q.get('expected_intent')} -> {q.get('expected_strategy')}): {q['query'][:60]}...")

    # 2. Initialize Harness with real FAISS/BM25 indexes, MockLLMProvider, and MockJudge
    print("\nInitializing EvaluationHarness with real indexes, MockLLMProvider, and MockJudge...")
    mock_llm = MockLLMProvider(
        default_response="Based on the official scikit-learn documentation, the configuration operates as specified."
    )
    mock_judge = MockJudge()
    harness = EvaluationHarness.create(llm_provider=mock_llm, judge=mock_judge)
    print("EvaluationHarness loaded successfully.")

    # 3. Execute runner
    runner = EvaluationRunner(harness=harness, judge=mock_judge)
    print(f"\nExecuting evaluation on systems: {SUPPORTED_SYSTEMS}...")
    results = runner.evaluate_questions(smoke_questions, systems=SUPPORTED_SYSTEMS)

    print(f"\nCompleted {len(results)} question evaluation runs.")

    # 4. Print Per-System / Per-Question Breakdown
    print("\n" + "=" * 80)
    print("DETAILED SMOKE TEST RESULTS")
    print("=" * 80)

    for res in results:
        sys_name = res.metadata.get("system_name", "unknown")
        rt = res.routing
        pred_strat = rt.predicted_strategy if rt else "N/A"
        strat_match = rt.strategy_match if rt else "N/A"
        mrr = res.retrieval_metrics.mrr
        recall_5 = res.retrieval_metrics.recall_at_k.get(5, 0.0)
        ndcg_5 = res.retrieval_metrics.ndcg_at_k.get(5, 0.0)
        overall_score = res.answer_scores.overall

        print(
            f"[{sys_name.upper():<14}] QID={res.question_id:<12} "
            f"Strat={pred_strat:<10} Match={str(strat_match):<5} "
            f"MRR={mrr:.2f} R@5={recall_5:.2f} nDCG@5={ndcg_5:.2f} "
            f"Calls={res.efficiency.retrieval_calls:<2} "
            f"Chunks={res.efficiency.chunks_processed:<2} "
            f"Latency={res.efficiency.latency_ms:>6.1f}ms "
            f"JudgeScore={overall_score:.1f}"
        )

    # 5. Aggregate Summary
    summary = runner.aggregate_results(results)
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    # Verification Assertions
    assert len(results) == 16, f"Expected 16 results (4x4), got {len(results)}"
    for sys_name in SUPPORTED_SYSTEMS:
        assert sys_name in summary, f"Missing system {sys_name} in summary"
        assert summary[sys_name]["question_count"] == 4

    print("\nAll 4 systems invoked successfully with valid evaluation results!")


if __name__ == "__main__":
    run_smoke_test()
