"""Unit tests for chunking strategies, metadata preservation, and evaluation."""

from adaq_rag.chunking.models import Chunk, ChunkEvalQuestion
from adaq_rag.chunking.strategies import (
    FixedChunker,
    FixedOverlapChunker,
    StructureAwareChunker,
)
from adaq_rag.chunking.tokenizer import count_tokens
from adaq_rag.evaluation.chunking.dataset import get_evaluation_questions
from adaq_rag.evaluation.chunking.evaluator import (
    compute_chunk_stats,
    evaluate_information_preservation,
)
from adaq_rag.ingestion.models import DocumentSection, ProcessedDocument


def create_sample_doc(num_words: int = 1200) -> ProcessedDocument:
    """Helper creating a synthetic ProcessedDocument with sections."""
    part1 = "Linear models make predictions using linear functions. " * 30  # ~300 words
    part2 = "Ridge regression adds an L2 penalty to the loss function. " * 40  # ~440 words
    part3 = "Lasso regression produces sparse weights using L1 regularization. " * 30  # ~300 words

    sections = [
        DocumentSection(title="1.1. Introduction to Linear Models", level=1, content=part1),
        DocumentSection(title="1.1.2. Ridge Regression", level=2, content=part2),
        DocumentSection(title="1.1.3. Lasso Sparsity", level=2, content=part3),
    ]

    full_content = (
        f"# 1.1. Introduction to Linear Models\n\n{part1}\n\n"
        f"## 1.1.2. Ridge Regression\n\n{part2}\n\n"
        f"## 1.1.3. Lasso Sparsity\n\n{part3}"
    )

    return ProcessedDocument(
        doc_id="test_linear_models",
        source_url="https://scikit-learn.org/stable/modules/linear_model.html",
        title="1.1. Linear Models",
        category="supervised_learning",
        version="1.9",
        headings=["1.1. Introduction", "1.1.2. Ridge", "1.1.3. Lasso"],
        sections=sections,
        content=full_content,
        char_count=len(full_content),
        word_count=len(full_content.split()),
    )


def test_fixed_chunker_metadata_and_boundaries() -> None:
    """Verify FixedChunker preserves metadata and adheres to target token budget."""
    doc = create_sample_doc()
    chunker = FixedChunker(target_tokens=400, min_tokens=40)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strategy == "fixed"
        assert chunk.doc_id == doc.doc_id
        assert chunk.source_url == doc.source_url
        assert chunk.doc_title == doc.title
        assert chunk.section_title
        assert chunk.token_count > 0
        assert chunk.char_count > 0
        assert chunk.chunk_id.startswith("test_linear_models_fixed_")


def test_fixed_overlap_chunker_boundary_sharing() -> None:
    """Verify FixedOverlapChunker generates overlapping content between adjacent chunks."""
    doc = create_sample_doc()
    chunker = FixedOverlapChunker(target_tokens=300, overlap_tokens=60, min_tokens=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    # Verify that consecutive chunks share text
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        assert c1.strategy == "fixed_overlap"
        assert c2.strategy == "fixed_overlap"
        # Check that consecutive chunks share content from the sliding window
        c1_words = set(c1.content.split())
        c2_words = set(c2.content.split())
        common = c1_words.intersection(c2_words)
        assert len(common) >= 10, "Adjacent overlapping chunks must share text from the overlap window"


def test_structure_aware_chunker_section_preservation() -> None:
    """Verify StructureAwareChunker partitions according to sections and sub-splits large sections."""
    doc = create_sample_doc()
    chunker = StructureAwareChunker(max_tokens=350, min_tokens=30)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    section_titles = [c.section_title for c in chunks]
    assert "1.1. Introduction to Linear Models" in section_titles
    assert "1.1.2. Ridge Regression" in section_titles
    assert "1.1.3. Lasso Sparsity" in section_titles

    for c in chunks:
        assert c.strategy == "structure_aware"
        assert c.doc_id == doc.doc_id
        assert c.chunk_id.startswith("test_linear_models_struct_")
        assert c.token_count <= 450  # Section with prefix header fits comfortably


def test_compute_chunk_stats_calculation() -> None:
    """Verify statistical metrics calculation on chunk collections."""
    chunks = [
        Chunk(
            chunk_id="c1",
            strategy="fixed",
            doc_id="d1",
            source_url="http://example.com",
            doc_title="Doc 1",
            section_title="Sec 1",
            content="Small text",
            token_count=20,
            char_count=10,
        ),
        Chunk(
            chunk_id="c2",
            strategy="fixed",
            doc_id="d1",
            source_url="http://example.com",
            doc_title="Doc 1",
            section_title="Sec 2",
            content="Standard text " * 50,
            token_count=150,
            char_count=700,
        ),
    ]

    stats = compute_chunk_stats(chunks, "fixed", undersized_thresh=50, oversized_thresh=200)
    assert stats.total_chunks == 2
    assert stats.total_tokens == 170
    assert stats.min_tokens == 20
    assert stats.max_tokens == 150
    assert stats.avg_tokens == 85.0
    assert stats.undersized_count == 1
    assert stats.undersized_pct == 50.0
    assert stats.oversized_count == 0
    assert stats.metadata_completeness_pct == 100.0


def test_information_preservation_evaluation() -> None:
    """Verify context preservation auditor detects intact co-occurrence vs fragmentation."""
    chunks = [
        Chunk(
            chunk_id="c1",
            strategy="test",
            doc_id="best_practices_common_pitfalls",
            source_url="http://example.com",
            doc_title="Common Pitfalls",
            section_title="12.2.2. Data leakage",
            content="When using SelectKBest you must call fit_transform only on train data from train_test_split to prevent data leakage.",
            token_count=50,
            char_count=120,
        )
    ]

    question = ChunkEvalQuestion(
        question_id="test_q1",
        query="How to avoid leakage in SelectKBest?",
        category="pitfalls",
        target_doc_id="best_practices_common_pitfalls",
        expected_section="12.2.2. Data leakage",
        key_phrases=["SelectKBest", "fit_transform", "train_test_split", "leakage"],
    )

    res = evaluate_information_preservation(chunks, [question])
    assert res["total_questions"] == 1
    assert res["preserved_count"] == 1
    assert res["preservation_rate"] == 100.0
    assert res["question_results"][0]["is_preserved"] is True


def test_evaluation_dataset_structure_and_balance() -> None:
    """Verify that the evaluation dataset has exactly 60 questions balanced across 6 archetypes."""
    questions = get_evaluation_questions()
    assert len(questions) == 60

    expected_categories = {
        "FACT_LOOKUP": 10,
        "EXPLANATION": 10,
        "PROCEDURE": 10,
        "PARAMETER_API": 10,
        "COMPARISON": 10,
        "TROUBLESHOOTING": 10,
    }

    counts: dict[str, int] = {}
    q_ids = set()
    queries = set()

    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
        assert q.question_id not in q_ids, f"Duplicate question_id: {q.question_id}"
        q_ids.add(q.question_id)

        assert q.query not in queries, f"Duplicate query text: {q.query}"
        queries.add(q.query)

        assert len(q.key_phrases) >= 2, f"Question {q.question_id} must have >= 2 key phrases"
        assert q.target_doc_id, f"Question {q.question_id} missing target_doc_id"
        assert q.expected_section, f"Question {q.question_id} missing expected_section"

    assert counts == expected_categories
