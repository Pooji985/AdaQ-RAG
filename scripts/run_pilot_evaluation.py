"""Controlled Pilot Evaluation Script for AdaQ-RAG.

Executes a small, controlled pilot across all 4 systems (Basic, Hybrid, Adaptive, Full AdaQ)
using the configured real LLM provider and real LLM judge on representative benchmark questions.

Validates:
1. All 4 systems execute correctly
2. Retrieval metrics computation
3. Real LLM Judge evaluation and parsing
4. Latency and token usage tracking
5. Checkpoint saving and benchmark-content fingerprint integrity
6. Resume deduplication (skips completed questions)
7. Failure-aware aggregation
"""

import json
import logging
from pathlib import Path
import sys
import time

# Ensure src and root are in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaq_rag.core.config import get_settings
from adaq_rag.core.logging import setup_logging
from adaq_rag.llm.factory import get_llm_provider
from evaluation.rag.judge import LLMJudge
from evaluation.rag.runner import (
    SUPPORTED_SYSTEMS,
    EvaluationHarness,
    EvaluationRunner,
    compute_benchmark_fingerprint,
    compute_question_fingerprint,
    filter_benchmark,
    load_benchmark,
)

logger = logging.getLogger("adaq_rag.evaluation.pilot")

PILOT_OUTPUT_PATH = Path("evaluation/rag/pilot_evaluation_results.json")

# Representative question subset covering all 3 complexity tiers & strategies
PILOT_QUESTION_IDS = [
    "rag_fact_01",   # SIMPLE | FACT_LOOKUP -> DENSE
    "rag_comp_06",   # MEDIUM | COMPARISON -> HYBRID
    "rag_comp_11",   # COMPLEX | COMPARISON -> MULTI_STEP
]


def run_pilot() -> None:
    setup_logging("INFO")
    print("=" * 80)
    print("AdaQ-RAG: Controlled Evaluation Pilot (Real LLM + Real LLMJudge)")
    print("=" * 80)

    # 0. Clean prior pilot artifact if present
    if PILOT_OUTPUT_PATH.is_file():
        PILOT_OUTPUT_PATH.unlink()
        print(f"Removed previous pilot checkpoint at: {PILOT_OUTPUT_PATH}")

    # 1. Load benchmark & select pilot questions
    all_questions = load_benchmark()
    pilot_questions = filter_benchmark(all_questions, question_ids=PILOT_QUESTION_IDS)
    print(f"\nSelected {len(pilot_questions)} representative pilot questions:")
    for q in pilot_questions:
        print(f"  - [{q['question_id']}] ({q.get('expected_complexity')} / {q.get('expected_intent')} -> {q.get('expected_strategy')})")
        print(f"    Query: {q['query']}")
        print(f"    Ref:   {q['reference_answer'][:80]}...")

    # 2. Initialize Harness with real FAISS/BM25/Reranker, real LLMProvider, and real LLMJudge
    cfg = get_settings()
    print(f"\nInitializing EvaluationHarness with provider={cfg.llm_provider}, model={cfg.llm_model}...")
    llm_provider = get_llm_provider(cfg)
    judge = LLMJudge(provider=llm_provider)
    harness = EvaluationHarness.create(settings=cfg, llm_provider=llm_provider, judge=judge)
    runner = EvaluationRunner(harness=harness, judge=judge)
    print("EvaluationHarness and LLMJudge initialized successfully.")

    # -------------------------------------------------------------------------
    # STAGE 1: Run Question 1 to establish initial checkpoint & verify fingerprinting
    # -------------------------------------------------------------------------
    q1_subset = [pilot_questions[0]]
    q1_id = q1_subset[0]["question_id"]
    print(f"\n--- STAGE 1: Initial Checkpoint Run with [{q1_id}] across 4 systems ---")
    start_stage1 = time.perf_counter()
    stage1_results = runner.evaluate_questions(
        questions=q1_subset,
        systems=SUPPORTED_SYSTEMS,
        checkpoint_path=PILOT_OUTPUT_PATH,
        resume=False,
    )
    stage1_elapsed = time.perf_counter() - start_stage1
    print(f"Stage 1 completed in {stage1_elapsed:.1f}s. Results count: {len(stage1_results)}")

    # Verify Stage 1 checkpoint on disk
    assert PILOT_OUTPUT_PATH.is_file(), f"Expected checkpoint file at {PILOT_OUTPUT_PATH}"
    checkpoint_records = runner.load_results(PILOT_OUTPUT_PATH)
    assert len(checkpoint_records) == 4, f"Expected 4 checkpoint records, got {len(checkpoint_records)}"
    for r in checkpoint_records:
        assert "question_fingerprint" in r.metadata, f"Missing question_fingerprint in record {r.question_id}"
        assert "benchmark_fingerprint" in r.metadata, f"Missing benchmark_fingerprint in record {r.question_id}"
    print(f"Verified Stage 1 checkpoint: {len(checkpoint_records)} records saved with valid fingerprints.")

    # -------------------------------------------------------------------------
    # STAGE 2: Resume with all 3 questions to verify resume skipping & completion
    # -------------------------------------------------------------------------
    print(f"\n--- STAGE 2: Resume Run with all {len(pilot_questions)} questions (should skip {q1_id}) ---")
    start_stage2 = time.perf_counter()
    final_results = runner.evaluate_questions(
        questions=pilot_questions,
        systems=SUPPORTED_SYSTEMS,
        checkpoint_path=PILOT_OUTPUT_PATH,
        resume=True,
    )
    stage2_elapsed = time.perf_counter() - start_stage2
    print(f"Stage 2 completed in {stage2_elapsed:.1f}s. Total final results: {len(final_results)}")

    # Verify deduplication and record count
    assert len(final_results) == 12, f"Expected 12 records (3 questions x 4 systems), got {len(final_results)}"
    res_keys = {(r.question_id, r.metadata.get("system_name")) for r in final_results}
    assert len(res_keys) == 12, f"Duplicate (question_id, system_name) records detected: {len(res_keys)}"

    # 3. Print Per-Question / Per-System Details
    print("\n" + "=" * 80)
    print("PILOT EVALUATION PER-RECORD BREAKDOWN")
    print("=" * 80)
    for r in final_results:
        sys_name = r.metadata.get("system_name", "unknown")
        err = r.metadata.get("error")
        judge_score = f"{r.answer_scores.overall:.2f}" if not err else "N/A (Error)"
        c_score = f"{r.answer_scores.correctness:.1f}"
        f_score = f"{r.answer_scores.faithfulness:.1f}"
        r_score = f"{r.answer_scores.relevance:.1f}"
        mrr = f"{r.retrieval_metrics.mrr:.2f}"
        r_at_5 = f"{r.retrieval_metrics.recall_at_k.get(5, 0.0):.2f}"
        latency = f"{r.efficiency.latency_ms:.0f}ms"
        tokens = r.efficiency.token_usage.total_tokens if r.efficiency.token_usage else 0

        status_str = f"ERROR: {err[:30]}" if err else "SUCCESS"
        print(
            f"[{sys_name:<14}] {r.question_id:<12} | "
            f"Judge={judge_score} (C={c_score}, F={f_score}, R={r_score}) | "
            f"MRR={mrr} R@5={r_at_5} | "
            f"Lat={latency:>6} Tokens={tokens:<5} | {status_str}"
        )

    # 4. Aggregate Metrics Summary
    summary = runner.aggregate_results(final_results)
    print("\n" + "=" * 80)
    print("PILOT AGGREGATE METRICS SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print("PILOT EXECUTION COMPLETE")
    print(f"Artifact persisted to: {PILOT_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    run_pilot()
