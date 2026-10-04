"""CLI script to run scikit-learn documentation ingestion pipeline."""

import argparse
import sys
from pathlib import Path

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaq_rag.core.logging import setup_logging
from adaq_rag.ingestion.pipeline import IngestionPipeline


def print_report(report) -> None:
    """Print a formatted terminal summary report of ingestion results."""
    separator = "=" * 65
    print("\n" + separator)
    print("           AdaQ-RAG KNOWLEDGE INGESTION REPORT")
    print(separator)
    print(f"Total Sources Targeted:      {report.total_sources}")
    print(f"Sources Collected:           {report.collected_count}")
    print(f"Documents Processed:         {report.processed_count}")
    print(f"Failed Sources:              {report.failed_count}")
    print(f"Total Corpus Characters:     {report.total_characters:,}")
    print(f"Total Corpus Words:          {report.total_words:,}")
    approx_mb = report.total_characters / (1024 * 1024)
    print(f"Approximate Text Size:       {approx_mb:.2f} MB")
    print("-" * 65)
    print("Category Breakdown:")
    for cat, count in sorted(report.categories_breakdown.items()):
        print(f"  - {cat:30s}: {count} document(s)")
    if report.failed_sources:
        print("-" * 65)
        print("Failed Sources:")
        for failure in report.failed_sources:
            print(f"  [!] {failure}")
    print(separator + "\n")


def main() -> int:
    """Parse CLI arguments and execute the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest and structure scikit-learn documentation for AdaQ-RAG."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetching remote documentation even if cached locally in data/raw/",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to store raw HTML pages (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to store cleaned JSON/Markdown documents (default: data/processed)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    pipeline = IngestionPipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    report = pipeline.run(force_download=args.force)
    print_report(report)

    return 1 if report.failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
