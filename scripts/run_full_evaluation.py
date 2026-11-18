"""Full 90-Question Benchmark Evaluation Script for AdaQ-RAG.

Executes the formal benchmark across all 4 RAG systems:
1. Basic RAG (Phase 5)
2. Hybrid RAG (Phase 8)
3. Adaptive RAG (Phases 6 & 7)
4. Full AdaQ-RAG (Phases 6, 7, 8 & 9)

Reuses the core evaluation infrastructure in `evaluation/rag/runner.py`.
Provides question-level progress reporting, elapsed and estimated remaining time,
checkpoint resumption, rate-limit mitigation, and atomic result persistence.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from adaq_rag.core.config import get_settings
from adaq_rag.core.logging import setup_logging
from adaq_rag.llm.factory import get_llm_provider
from evaluation.rag.judge import LLMJudge
from evaluation.rag.runner import (
    DEFAULT_BENCHMARK_PATH,
    SUPPORTED_SYSTEMS,
    EvaluationHarness,
    EvaluationRunner,
    compute_benchmark_fingerprint,
    filter_benchmark,
    load_benchmark,
    normalize_system_name,
)

logger = logging.getLogger("adaq_rag.evaluation.full")

DEFAULT_OUTPUT_PATH = Path("evaluation/rag/full_evaluation_results.json")
DEFAULT_SUMMARY_PATH = Path("evaluation/rag/full_evaluation_summary.json")

# Protected files that must NEVER be overwritten by evaluation runs
PROTECTED_PATHS = {
    (PROJECT_ROOT / "evaluation/rag/pilot_evaluation_results.json").resolve(),
    (PROJECT_ROOT / "evaluation/rag/adaptive_rag_benchmark.json").resolve(),
}


def validate_paths(
    output_path: str | Path,
    summary_path: str | Path,
    benchmark_path: str | Path | None = None,
) -> None:
    """Validate that output and summary paths do not overwrite protected assets.

    Args:
        output_path: Path where raw evaluation results will be saved.
        summary_path: Path where aggregate summary JSON will be saved.
        benchmark_path: Optional benchmark input path to verify exists.

    Raises:
        ValueError: If output or summary path targets a protected file, or if paths collide.
        FileNotFoundError: If benchmark_path is specified but does not exist.
    """
    out_resolved = Path(output_path).resolve()
    sum_resolved = Path(summary_path).resolve()

    for p in PROTECTED_PATHS:
        if out_resolved == p:
            raise ValueError(
                f"Output path '{output_path}' targets protected file '{p.name}'. "
                "The pilot evaluation results and benchmark source must not be overwritten."
            )
        if sum_resolved == p:
            raise ValueError(
                f"Summary path '{summary_path}' targets protected file '{p.name}'. "
                "The pilot evaluation results and benchmark source must not be overwritten."
            )

    if out_resolved == sum_resolved:
        raise ValueError(
            f"Output path and summary path cannot be identical: '{output_path}'"
        )

    if benchmark_path:
        bench_p = Path(benchmark_path)
        if not bench_p.is_file():
            raise FileNotFoundError(f"Benchmark file not found at: {bench_p}")


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds into human-readable string (e.g. '12m 34s' or '45s')."""
    if seconds is None or seconds < 0:
        return "--"
    secs = int(round(seconds))
    if secs < 60:
        return f"{secs}s"
    mins, rem_secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {rem_secs:02d}s"
    hours, rem_mins = divmod(mins, 60)
    return f"{hours}h {rem_mins:02d}m {rem_secs:02d}s"


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the full benchmark evaluation."""
    parser = argparse.ArgumentParser(
        description="AdaQ-RAG Full 90-Question Benchmark Evaluation Runner."
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing checkpoint if present (default: --resume).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to first N questions (default: None, evaluate all).",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            f"Target RAG systems to evaluate. Choices: {SUPPORTED_SYSTEMS}. "
            "Defaults to all four systems."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Inter-system pacing delay in seconds to mitigate API rate limits (default: 0.5s).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help=f"Path to benchmark JSON dataset (default: {DEFAULT_BENCHMARK_PATH}).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write raw evaluation results checkpoint (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Path to write aggregate evaluation summary JSON (default: {DEFAULT_SUMMARY_PATH}).",
    )
    parser.add_argument(
        "--max-eval-questions",
        type=int,
        default=None,
        help="Maximum number of previously uncompleted questions to evaluate before stopping gracefully (default: None, evaluate all).",
    )
    return parser.parse_args(args)


def save_summary_atomic(summary: dict[str, Any], summary_path: Path) -> None:
    """Save aggregate summary atomically to prevent corruption on interruption."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = summary_path.with_suffix(f"{summary_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_path, summary_path)
    logger.info("Saved evaluation summary to %s", summary_path)


def make_progress_reporter():
    """Create a progress reporter callback for the evaluation loop."""
    def progress_callback(
        q_idx: int,
        total_questions: int,
        qid: str,
        elapsed_s: float,
        remaining_s: float | None,
        was_resumed: bool,
    ) -> None:
        pct = (q_idx / total_questions) * 100.0 if total_questions > 0 else 0.0
        elapsed_str = format_duration(elapsed_s)
        rem_str = format_duration(remaining_s)
        status_tag = "[RESUMED]" if was_resumed else "[EVALUATED]"

        print(
            f"  -> [{q_idx:02d}/{total_questions:02d}] ({pct:5.1f}%) {qid:<14} {status_tag} | "
            f"Elapsed: {elapsed_str} | Est. Remaining: {rem_str}"
        )
        sys.stdout.flush()

    return progress_callback


def run_full_evaluation(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    """Execute the full benchmark evaluation workflow with the provided arguments."""
    setup_logging("INFO")

    validate_paths(
        output_path=args.output_path,
        summary_path=args.summary_path,
        benchmark_path=args.benchmark_path,
    )

    # Resolve target systems
    target_systems = [
        normalize_system_name(s) for s in (args.systems or SUPPORTED_SYSTEMS)
    ]

    print("=" * 80)
    print("AdaQ-RAG: Formal Benchmark Evaluation Runner")
    print("=" * 80)
    print(f"Benchmark Path:  {args.benchmark_path}")
    print(f"Output Path:     {args.output_path}")
    print(f"Summary Path:    {args.summary_path}")
    print(f"Target Systems:  {target_systems}")
    print(f"Resume Mode:     {args.resume}")
    print(f"Pacing Delay:    {args.delay}s")
    if args.limit:
        print(f"Question Limit:  {args.limit}")
    print("=" * 80)

    # 1. Load questions
    raw_benchmark = load_benchmark(args.benchmark_path)
    questions = filter_benchmark(raw_benchmark, limit=args.limit)
    print(f"\nLoaded {len(questions)} benchmark questions for evaluation.")

    # 2. Initialize EvaluationHarness and LLMJudge
    cfg = get_settings()
    print(f"Initializing EvaluationHarness with provider={cfg.llm_provider}, model={cfg.llm_model}...")
    llm_provider = get_llm_provider(cfg)
    judge = LLMJudge(provider=llm_provider)
    harness = EvaluationHarness.create(
        settings=cfg,
        llm_provider=llm_provider,
        judge=judge,
    )
    runner = EvaluationRunner(harness=harness, judge=judge)
    print("Harness and LLM Judge ready. Starting evaluation loop...\n")

    # 3. Execute evaluation loop with progress tracking
    progress_reporter = make_progress_reporter()
    total_start = time.perf_counter()

    results = runner.evaluate_questions(
        questions=questions,
        systems=target_systems,
        checkpoint_path=args.output_path,
        resume=args.resume,
        progress_callback=progress_reporter,
        delay_between_systems=args.delay,
        max_eval_questions=args.max_eval_questions,
    )
    total_elapsed = time.perf_counter() - total_start

    print(f"\nEvaluation loop completed in {format_duration(total_elapsed)} ({total_elapsed:.1f}s).")
    print(f"Total question-system records: {len(results)}")

    # 4. Aggregate results
    bench_fp = compute_benchmark_fingerprint(questions)
    summary = runner.aggregate_results(results)

    # 5. Persist final raw results and summary
    runner.save_results(results, args.output_path, benchmark_fingerprint=bench_fp)
    save_summary_atomic(summary, args.summary_path)

    # 6. Print summary metrics table
    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"{'System':<16} | {'Quality':<7} | {'Recall@5':<8} | {'MRR':<6} | "
        f"{'RouteAcc':<8} | {'Calls':<5} | {'Latency':<8} | {'Errors':<6}"
    )
    print("-" * 80)

    for sys_name in target_systems:
        s = summary.get(sys_name)
        if not s:
            continue
        qual = f"{s['quality']['mean_overall']:.2f}"
        rec = f"{s['retrieval']['recall_at_5']:.4f}"
        mrr = f"{s['retrieval']['mrr']:.4f}"
        rt_acc = f"{s['routing']['strategy_accuracy']*100:.1f}%" if "adaptive" in sys_name or "full" in sys_name else "N/A"
        calls = f"{s['efficiency']['mean_retrieval_calls']:.2f}"
        lat = f"{s['efficiency']['mean_latency_ms']/1000:.1f}s"
        errs = f"{s['failure_count']}/{s['question_count']}"

        print(
            f"{sys_name:<16} | {qual:<7} | {rec:<8} | {mrr:<6} | "
            f"{rt_acc:<8} | {calls:<5} | {lat:<8} | {errs:<6}"
        )

    print("=" * 80)
    print(f"Raw results saved to:   {args.output_path}")
    print(f"Summary metrics to:     {args.summary_path}")

    return results, summary


def main() -> None:
    args = parse_args()
    try:
        run_full_evaluation(args)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Intermediate progress saved to checkpoint.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nEvaluation failed with error: {exc}", file=sys.stderr)
        logger.exception("Full evaluation encountered an unhandled exception: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
