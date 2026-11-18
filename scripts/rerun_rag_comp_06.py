"""Rerun single benchmark question (rag_comp_06) across all 4 systems.

Executes only the affected question using the real LLM provider and real LLMJudge,
validates the updated retrieval metrics and fingerprints, and safely updates the pilot results.
"""

import json
import logging
from pathlib import Path
import shutil
import sys
import time

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

logger = logging.getLogger("adaq_rag.evaluation.rerun")

PILOT_OUTPUT_PATH = Path("evaluation/rag/pilot_evaluation_results.json")
BACKUP_PILOT_PATH = Path("evaluation/rag/pilot_evaluation_results_pre_correction.json")
RERUN_CHECKPOINT_PATH = Path("evaluation/rag/pilot_rag_comp_06_updated.json")


def main() -> None:
    setup_logging("INFO")
    print("=" * 80)
    print("AdaQ-RAG: Rerun rag_comp_06 (Corrected Benchmark Annotation)")
    print("=" * 80)

    # 1. Load benchmark & select rag_comp_06
    bench = load_benchmark()
    q6_list = filter_benchmark(bench, question_ids=["rag_comp_06"])
    assert len(q6_list) == 1, "rag_comp_06 not found in benchmark"
    q6 = q6_list[0]

    q6_fp = compute_question_fingerprint(q6)
    bench_fp = compute_benchmark_fingerprint(bench)
    print(f"\nQuestion: {q6['question_id']}")
    print(f"Query: {q6['query']}")
    print(f"Ground Truth Chunk IDs: {q6['ground_truth_chunk_ids']}")
    print(f"New Question Fingerprint:  {q6_fp}")
    print(f"New Benchmark Fingerprint: {bench_fp}")

    # 2. Verify Checkpoint Compatibility Behavior (Task 7)
    print("\n--- Verifying Checkpoint Compatibility Protection ---")
    if PILOT_OUTPUT_PATH.is_file():
        existing_records = EvaluationRunner.load_results(PILOT_OUTPUT_PATH)
        # Attempting to resume with the old multi-question checkpoint for just q6 must fail
        try:
            EvaluationRunner._validate_checkpoint_compatibility(
                checkpoint_records=existing_records,
                target_questions=[q6],
                target_systems=SUPPORTED_SYSTEMS,
            )
            print("WARNING: Checkpoint validation unexpectedly passed!")
        except ValueError as ex:
            print(f"Protection Verified: Incompatible checkpoint correctly rejected:\n  -> {ex}")

        # Also verify that if we isolate the old q6 record from the checkpoint, its fingerprint mismatch is caught
        old_q6_records = [r for r in existing_records if r.question_id == "rag_comp_06"]
        if old_q6_records:
            try:
                EvaluationRunner._validate_checkpoint_compatibility(
                    checkpoint_records=old_q6_records,
                    target_questions=[q6],
                    target_systems=SUPPORTED_SYSTEMS,
                )
                print("WARNING: Stale question fingerprint validation unexpectedly passed!")
            except ValueError as ex:
                print(f"Protection Verified: Stale fingerprint correctly rejected:\n  -> {ex}")

    # 3. Clean prior rerun checkpoint if present
    if RERUN_CHECKPOINT_PATH.is_file():
        RERUN_CHECKPOINT_PATH.unlink()

    # 4. Initialize Harness with real FAISS/BM25/Reranker, real LLMProvider, and real LLMJudge
    cfg = get_settings()
    print(f"\nInitializing EvaluationHarness with provider={cfg.llm_provider}, model={cfg.llm_model}...")
    llm_provider = get_llm_provider(cfg)
    judge = LLMJudge(provider=llm_provider)
    harness = EvaluationHarness.create(settings=cfg, llm_provider=llm_provider, judge=judge)
    runner = EvaluationRunner(harness=harness, judge=judge)

    # 5. Evaluate rag_comp_06 across all 4 systems
    print("\n--- Executing rag_comp_06 across all 4 systems ---")
    start = time.perf_counter()
    new_records = runner.evaluate_questions(
        questions=[q6],
        systems=SUPPORTED_SYSTEMS,
        checkpoint_path=RERUN_CHECKPOINT_PATH,
        resume=False,
    )
    elapsed = time.perf_counter() - start
    print(f"Completed 4 evaluations in {elapsed:.1f}s.")

    # 6. Verify and display results
    print("\n" + "=" * 80)
    print("UPDATED RESULTS FOR rag_comp_06")
    print("=" * 80)
    for r in new_records:
        sys_name = r.metadata.get("system_name", "unknown")
        err = r.metadata.get("error")
        judge_score = f"{r.answer_scores.overall:.2f}" if not err else "N/A (Error)"
        c_score = f"{r.answer_scores.correctness:.1f}"
        f_score = f"{r.answer_scores.faithfulness:.1f}"
        r_score = f"{r.answer_scores.relevance:.1f}"
        mrr = f"{r.retrieval_metrics.mrr:.2f}"
        r_at_1 = f"{r.retrieval_metrics.recall_at_k.get(1, 0.0):.2f}"
        r_at_5 = f"{r.retrieval_metrics.recall_at_k.get(5, 0.0):.2f}"
        ndcg_5 = f"{r.retrieval_metrics.ndcg_at_k.get(5, 0.0):.2f}"
        latency = f"{r.efficiency.latency_ms:.0f}ms"
        tokens = r.efficiency.token_usage.total_tokens if r.efficiency.token_usage else 0

        print(
            f"[{sys_name:<14}] {r.question_id:<12} | "
            f"R@1={r_at_1} R@5={r_at_5} MRR={mrr} nDCG@5={ndcg_5} | "
            f"Judge={judge_score} (C={c_score}, F={f_score}, R={r_score}) | "
            f"Lat={latency:>6} Tokens={tokens:<5}"
        )
        print(f"  Retrieved Chunks: {r.retrieved_chunk_ids[:3]} ...")
        print(f"  Ground Truth:    {r.ground_truth_chunk_ids}")

    # 7. Backup existing pilot results and update with new rag_comp_06 records
    if PILOT_OUTPUT_PATH.is_file():
        if not BACKUP_PILOT_PATH.is_file():
            shutil.copy2(PILOT_OUTPUT_PATH, BACKUP_PILOT_PATH)
            print(f"\nBacked up original pilot results to: {BACKUP_PILOT_PATH}")

        existing = EvaluationRunner.load_results(PILOT_OUTPUT_PATH)
        # Keep non-rag_comp_06 records
        merged = [r for r in existing if r.question_id != "rag_comp_06"]
        merged.extend(new_records)
        # Sort by (question_id, system_name) for deterministic ordering
        merged.sort(key=lambda r: (r.question_id, r.metadata.get("system_name", "")))

        EvaluationRunner.save_results(merged, PILOT_OUTPUT_PATH)
        print(f"Merged updated records into {PILOT_OUTPUT_PATH}. Total records: {len(merged)}")

        # Print overall updated pilot summary
        summary = EvaluationRunner.aggregate_results(merged)
        print("\n" + "=" * 80)
        print("UPDATED PILOT AGGREGATE SUMMARY (All 3 Questions x 4 Systems)")
        print("=" * 80)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
