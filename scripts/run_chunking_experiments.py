"""CLI script to execute and evaluate chunking strategies on scikit-learn documentation."""

import json
import sys
from pathlib import Path

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaq_rag.chunking.pipeline import ChunkingPipeline
from adaq_rag.core.logging import setup_logging
from adaq_rag.evaluation.chunking.evaluator import (
    compute_chunk_stats,
    evaluate_information_preservation,
    get_side_by_side_samples,
)
from adaq_rag.evaluation.chunking.report import generate_markdown_report


def main() -> int:
    """Run all chunking strategies and generate evaluation reports."""
    setup_logging("INFO")

    processed_dir = Path("data/processed")
    chunks_output_dir = Path("data/processed/chunks")
    eval_output_dir = Path("evaluation/chunking")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("        AdaQ-RAG: PHASE 3 CHUNKING STRATEGIES EXPERIMENT")
    print("=" * 70)

    # 1. Run Chunking Pipeline
    pipeline = ChunkingPipeline(processed_dir=processed_dir, output_dir=chunks_output_dir)
    print(f"\n[1/3] Executing chunking strategies on {processed_dir}...")
    chunk_map = pipeline.run_all()

    # 2. Compute Statistics and Evaluation Metrics
    print("\n[2/3] Computing statistical distributions and context preservation...")
    stats_map = {}
    preservation_map = {}

    for strat_name, chunks in chunk_map.items():
        stats = compute_chunk_stats(chunks, strat_name)
        stats_map[strat_name] = stats
        preservation = evaluate_information_preservation(chunks)
        preservation_map[strat_name] = preservation

    # Extract sample inspections
    samples = [
        get_side_by_side_samples(chunk_map, "trouble_01"),  # Data leakage in feature selection
        get_side_by_side_samples(chunk_map, "proc_01"),     # Pipeline chaining
    ]

    # 3. Save Experiment Artifacts
    print("\n[3/3] Saving experiment reports...")
    json_report_path = eval_output_dir / "chunking_experiment_report.json"
    md_report_path = eval_output_dir / "chunking_experiment_report.md"

    json_data = {
        "statistics": {k: v.model_dump() for k, v in stats_map.items()},
        "information_preservation": preservation_map,
        "sample_inspections": samples,
    }
    json_report_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    md_report = generate_markdown_report(
        stats_map=stats_map,
        preservation_map=preservation_map,
        sample_inspections=samples,
        output_path=md_report_path,
    )

    # Terminal summary display
    print("\n" + "=" * 70)
    print("                  CHUNKING STRATEGIES COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<32} | {'Fixed (~500t)':<12} | {'Overlap (~500t)':<15} | {'Structure-Aware':<15}"
    print(header)
    print("-" * 80)

    f_s = stats_map["fixed"]
    o_s = stats_map["fixed_overlap"]
    s_s = stats_map["structure_aware"]

    print(f"{'Total Chunks':<32} | {f_s.total_chunks:<12} | {o_s.total_chunks:<15} | {s_s.total_chunks:<15}")
    print(f"{'Mean Tokens / Chunk':<32} | {f_s.avg_tokens:<12.1f} | {o_s.avg_tokens:<15.1f} | {s_s.avg_tokens:<15.1f}")
    print(f"{'Token Range (Min - Max)':<32} | {f_s.min_tokens}-{f_s.max_tokens:<8} | {o_s.min_tokens}-{o_s.max_tokens:<11} | {s_s.min_tokens}-{s_s.max_tokens:<11}")
    print(f"{'Undersized Chunks (<50t)':<32} | {f_s.undersized_pct:<11.1f}% | {o_s.undersized_pct:<14.1f}% | {s_s.undersized_pct:<14.1f}%")
    print(f"{'Oversized Chunks (>550t)':<32} | {f_s.oversized_pct:<11.1f}% | {o_s.oversized_pct:<14.1f}% | {s_s.oversized_pct:<14.1f}%")
    print(f"{'Metadata Completeness':<32} | {f_s.metadata_completeness_pct:<11.1f}% | {o_s.metadata_completeness_pct:<14.1f}% | {s_s.metadata_completeness_pct:<14.1f}%")

    f_p = preservation_map["fixed"]["preservation_rate"]
    o_p = preservation_map["fixed_overlap"]["preservation_rate"]
    s_p = preservation_map["structure_aware"]["preservation_rate"]
    print(f"{'Info Preservation Rate':<32} | {f_p:<11.1f}% | {o_p:<14.1f}% | {s_p:<14.1f}%")
    print("=" * 80)

    # Category / Archetype Breakdown
    print("\n" + "=" * 70)
    print("           PRESERVATION BY QUESTION ARCHETYPE (10 QUESTIONS EACH)")
    print("=" * 70)
    cat_header = f"{'Archetype':<20} | {'Fixed':<15} | {'Overlap':<15} | {'Structure-Aware':<15}"
    print(cat_header)
    print("-" * 70)
    f_cats = preservation_map["fixed"].get("category_breakdown", {})
    o_cats = preservation_map["fixed_overlap"].get("category_breakdown", {})
    s_cats = preservation_map["structure_aware"].get("category_breakdown", {})

    for cat in sorted(s_cats.keys()):
        tot = s_cats[cat]["total"]
        f_str = f"{f_cats.get(cat, {}).get('preserved', 0)}/{tot} ({f_cats.get(cat, {}).get('rate', 0.0):.0f}%)"
        o_str = f"{o_cats.get(cat, {}).get('preserved', 0)}/{tot} ({o_cats.get(cat, {}).get('rate', 0.0):.0f}%)"
        s_str = f"{s_cats.get(cat, {}).get('preserved', 0)}/{tot} ({s_cats.get(cat, {}).get('rate', 0.0):.0f}%)"
        print(f"{cat:<20} | {f_str:<15} | {o_str:<15} | {s_str:<15}")
    print("=" * 70)

    print(f"\nDetailed report written to: {md_report_path}")
    print(f"JSON metrics written to:     {json_report_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
