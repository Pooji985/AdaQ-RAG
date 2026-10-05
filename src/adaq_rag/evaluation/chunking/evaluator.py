"""Statistical and context preservation evaluator for chunking strategies."""

import statistics
from typing import Any

from adaq_rag.chunking.models import Chunk, ChunkStats
from adaq_rag.evaluation.chunking.dataset import ChunkEvalQuestion, get_evaluation_questions


def compute_chunk_stats(
    chunks: list[Chunk],
    strategy_name: str,
    undersized_thresh: int = 50,
    oversized_thresh: int = 550,
) -> ChunkStats:
    """Compute statistical distribution and metadata completeness for a chunk dataset."""
    if not chunks:
        return ChunkStats(
            strategy=strategy_name,
            total_chunks=0,
            total_tokens=0,
            total_chars=0,
            avg_tokens=0.0,
            min_tokens=0,
            max_tokens=0,
            avg_chars=0.0,
            min_chars=0,
            max_chars=0,
            undersized_count=0,
            undersized_pct=0.0,
            oversized_count=0,
            oversized_pct=0.0,
            metadata_completeness_pct=0.0,
        )

    tokens = [c.token_count for c in chunks]
    chars = [c.char_count for c in chunks]

    undersized = sum(1 for t in tokens if t < undersized_thresh)
    oversized = sum(1 for t in tokens if t > oversized_thresh)

    complete_metadata_count = 0
    for c in chunks:
        if (
            c.chunk_id
            and c.strategy
            and c.doc_id
            and c.source_url
            and c.doc_title
            and c.section_title
            and c.token_count > 0
        ):
            complete_metadata_count += 1

    total = len(chunks)
    return ChunkStats(
        strategy=strategy_name,
        total_chunks=total,
        total_tokens=sum(tokens),
        total_chars=sum(chars),
        avg_tokens=round(statistics.mean(tokens), 2),
        min_tokens=min(tokens),
        max_tokens=max(tokens),
        avg_chars=round(statistics.mean(chars), 2),
        min_chars=min(chars),
        max_chars=max(chars),
        undersized_count=undersized,
        undersized_pct=round((undersized / total) * 100, 2),
        oversized_count=oversized,
        oversized_pct=round((oversized / total) * 100, 2),
        metadata_completeness_pct=round((complete_metadata_count / total) * 100, 2),
    )


def evaluate_information_preservation(
    chunks: list[Chunk],
    questions: list[ChunkEvalQuestion] | None = None,
) -> dict[str, Any]:
    """Audit whether target concepts and their key contextual phrases are preserved intact within chunks."""
    eval_qs = questions or get_evaluation_questions()
    results: list[dict[str, Any]] = []
    preserved_count = 0

    for q in eval_qs:
        matching_chunks: list[Chunk] = []

        # Target doc candidates first, fallback to all if needed
        doc_chunks = [c for c in chunks if c.doc_id == q.target_doc_id] or chunks

        for chunk in doc_chunks:
            chunk_lower = chunk.content.lower()
            # Check if all key phrases co-occur in the same chunk
            if all(phrase.lower() in chunk_lower for phrase in q.key_phrases):
                matching_chunks.append(chunk)

        is_preserved = len(matching_chunks) > 0
        if is_preserved:
            preserved_count += 1

        best_chunk = matching_chunks[0] if matching_chunks else None

        results.append(
            {
                "question_id": q.question_id,
                "query": q.query,
                "category": q.category,
                "target_doc_id": q.target_doc_id,
                "expected_section": q.expected_section,
                "key_phrases": q.key_phrases,
                "is_preserved": is_preserved,
                "matching_chunks_count": len(matching_chunks),
                "best_chunk_id": best_chunk.chunk_id if best_chunk else None,
                "best_chunk_section": best_chunk.section_title if best_chunk else None,
                "best_chunk_tokens": best_chunk.token_count if best_chunk else 0,
                "sample_excerpt": best_chunk.content[:280] + "..." if best_chunk else None,
            }
        )

    total_qs = len(eval_qs)
    preservation_rate = round((preserved_count / total_qs) * 100, 2) if total_qs else 0.0

    category_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "preserved": 0, "rate": 0.0}
        category_stats[cat]["total"] += 1
        if r["is_preserved"]:
            category_stats[cat]["preserved"] += 1

    for cat, c_data in category_stats.items():
        c_data["rate"] = (
            round((c_data["preserved"] / c_data["total"]) * 100, 2) if c_data["total"] else 0.0
        )

    return {
        "total_questions": total_qs,
        "preserved_count": preserved_count,
        "preservation_rate": preservation_rate,
        "category_breakdown": category_stats,
        "question_results": results,
    }


def get_side_by_side_samples(
    all_chunks: dict[str, list[Chunk]],
    question_id: str = "eval_q01",
) -> dict[str, Any]:
    """Retrieve sample matching chunks across all strategies for side-by-side inspection."""
    questions = {q.question_id: q for q in get_evaluation_questions()}
    q = questions.get(question_id)
    if not q:
        return {}

    samples: dict[str, Any] = {"question": q.query, "key_phrases": q.key_phrases, "strategies": {}}

    for strategy, chunks in all_chunks.items():
        doc_chunks = [c for c in chunks if c.doc_id == q.target_doc_id]
        matching = [
            c
            for c in doc_chunks
            if all(phrase.lower() in c.content.lower() for phrase in q.key_phrases)
        ]
        chosen = matching[0] if matching else (doc_chunks[0] if doc_chunks else None)
        if chosen:
            samples["strategies"][strategy] = {
                "chunk_id": chosen.chunk_id,
                "section_title": chosen.section_title,
                "token_count": chosen.token_count,
                "content_preview": chosen.content[:350] + ("..." if len(chosen.content) > 350 else ""),
            }

    return samples
