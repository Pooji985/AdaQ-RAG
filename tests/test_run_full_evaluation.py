"""Focused unit tests for scripts/run_full_evaluation.py.

Validates:
1. Argument parsing (defaults, custom flags, systems selection, limits)
2. Output path safety (protecting pilot results and benchmark dataset)
3. Duration formatting and progress reporting
"""

from pathlib import Path
import pytest

from scripts.run_full_evaluation import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUMMARY_PATH,
    format_duration,
    make_progress_reporter,
    parse_args,
    validate_paths,
)


class TestArgumentParsing:
    """Test CLI argument parsing and defaults."""

    def test_default_arguments(self) -> None:
        args = parse_args([])
        assert args.resume is True
        assert args.limit is None
        assert args.systems is None
        assert args.delay == 0.5
        assert args.benchmark_path == DEFAULT_BENCHMARK_PATH
        assert args.output_path == DEFAULT_OUTPUT_PATH
        assert args.summary_path == DEFAULT_SUMMARY_PATH

    def test_custom_arguments(self) -> None:
        args = parse_args([
            "--no-resume",
            "--limit", "10",
            "--systems", "basic_rag", "hybrid_rag",
            "--delay", "1.5",
            "--output-path", "evaluation/rag/custom_results.json",
            "--summary-path", "evaluation/rag/custom_summary.json",
        ])
        assert args.resume is False
        assert args.limit == 10
        assert args.systems == ["basic_rag", "hybrid_rag"]
        assert args.delay == 1.5
        assert args.output_path == Path("evaluation/rag/custom_results.json")
        assert args.summary_path == Path("evaluation/rag/custom_summary.json")

    def test_resume_flag_variants(self) -> None:
        args_resume = parse_args(["--resume"])
        assert args_resume.resume is True

        args_no_resume = parse_args(["--no-resume"])
        assert args_no_resume.resume is False


class TestPathSafety:
    """Verify that pilot results and benchmark datasets cannot be overwritten."""

    def test_reject_overwriting_pilot_results_as_output(self) -> None:
        pilot_path = Path("evaluation/rag/pilot_evaluation_results.json")
        with pytest.raises(ValueError, match="targets protected file 'pilot_evaluation_results.json'"):
            validate_paths(
                output_path=pilot_path,
                summary_path=Path("evaluation/rag/some_summary.json"),
            )

    def test_reject_overwriting_pilot_results_as_summary(self) -> None:
        pilot_path = Path("evaluation/rag/pilot_evaluation_results.json")
        with pytest.raises(ValueError, match="targets protected file 'pilot_evaluation_results.json'"):
            validate_paths(
                output_path=Path("evaluation/rag/some_results.json"),
                summary_path=pilot_path,
            )

    def test_reject_overwriting_benchmark_file(self) -> None:
        bench_path = Path("evaluation/rag/adaptive_rag_benchmark.json")
        with pytest.raises(ValueError, match="targets protected file 'adaptive_rag_benchmark.json'"):
            validate_paths(
                output_path=bench_path,
                summary_path=Path("evaluation/rag/some_summary.json"),
            )

    def test_reject_colliding_output_and_summary_paths(self) -> None:
        same_path = Path("evaluation/rag/same_file.json")
        with pytest.raises(ValueError, match="cannot be identical"):
            validate_paths(
                output_path=same_path,
                summary_path=same_path,
            )

    def test_accept_valid_non_protected_paths(self, tmp_path: Path) -> None:
        out_p = tmp_path / "valid_results.json"
        sum_p = tmp_path / "valid_summary.json"
        # Should not raise
        validate_paths(output_path=out_p, summary_path=sum_p)

    def test_reject_missing_benchmark_path(self, tmp_path: Path) -> None:
        missing_bench = tmp_path / "non_existent_benchmark.json"
        with pytest.raises(FileNotFoundError, match="Benchmark file not found"):
            validate_paths(
                output_path=tmp_path / "out.json",
                summary_path=tmp_path / "sum.json",
                benchmark_path=missing_bench,
            )


class TestFormattingAndProgress:
    """Test duration and progress calculation utilities."""

    def test_format_duration_seconds(self) -> None:
        assert format_duration(0.0) == "0s"
        assert format_duration(45.2) == "45s"
        assert format_duration(59.2) == "59s"
        assert format_duration(59.9) == "1m 00s"

    def test_format_duration_minutes(self) -> None:
        assert format_duration(65.0) == "1m 05s"
        assert format_duration(125.0) == "2m 05s"
        assert format_duration(3599.0) == "59m 59s"

    def test_format_duration_hours(self) -> None:
        assert format_duration(3600.0) == "1h 00m 00s"
        assert format_duration(3665.0) == "1h 01m 05s"

    def test_format_duration_none_or_negative(self) -> None:
        assert format_duration(None) == "--"
        assert format_duration(-10.0) == "--"

    def test_progress_reporter_callback(self, capsys: pytest.CaptureFixture) -> None:
        reporter = make_progress_reporter()
        reporter(
            q_idx=5,
            total_questions=90,
            qid="rag_fact_05",
            elapsed_s=120.0,
            remaining_s=2040.0,
            was_resumed=False,
        )
        captured = capsys.readouterr().out
        assert "[05/90]" in captured
        assert "5.6%" in captured
        assert "rag_fact_05" in captured
        assert "[EVALUATED]" in captured
        assert "Elapsed: 2m 00s" in captured
        assert "Est. Remaining: 34m 00s" in captured

    def test_progress_reporter_callback_resumed(self, capsys: pytest.CaptureFixture) -> None:
        reporter = make_progress_reporter()
        reporter(
            q_idx=10,
            total_questions=90,
            qid="rag_fact_10",
            elapsed_s=0.05,
            remaining_s=None,
            was_resumed=True,
        )
        captured = capsys.readouterr().out
        assert "[10/90]" in captured
        assert "rag_fact_10" in captured
        assert "[RESUMED]" in captured
