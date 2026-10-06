"""Real retrieval sanity check for Phase 9 Complex Retrieval Orchestrator."""

from adaq_rag.query_analysis import QueryAnalyzer
from adaq_rag.retrieval.orchestrator import ComplexRetrievalOrchestrator
from adaq_rag.routing import AdaptiveRetrievalRouter

SANITY_COMPLEX_QUERIES = [
    (
        "MULTI_CONCEPT",
        "How do preprocessing, pipelines, and model selection work together in scikit-learn?",
    ),
    (
        "TROUBLESHOOTING",
        "Why does pipeline cross-validation produce data leakage when scaling features with StandardScaler?",
    ),
    (
        "COMPARISON",
        "Compare StandardScaler, RobustScaler, and MinMaxScaler when datasets have severe outliers and sparse features",
    ),
]


def run_complex_sanity_check() -> None:
    print("Initializing Analyzer, Router, and ComplexRetrievalOrchestrator with real indexes and CrossEncoder...")
    analyzer = QueryAnalyzer()
    router = AdaptiveRetrievalRouter()
    orchestrator = ComplexRetrievalOrchestrator.load()
    print("Orchestrator loaded successfully.\n")

    for category, query in SANITY_COMPLEX_QUERIES:
        print("================================================================================")
        print(f"Category: {category}")
        print(f"Query   : '{query}'")

        # 1. Analyze & Route
        analysis = analyzer.analyze(query)
        plan = router.route(analysis)
        print(f"Phase 6 Intent: {analysis.intent.value} | Complexity: {analysis.complexity.value}")
        print(f"Phase 7 Plan  : Strategy={plan.strategy.value} | Effort={plan.effort.value} | k=({plan.candidate_top_k}->{plan.final_top_k})")

        # 2. Execute complex retrieval from plan
        result = orchestrator.retrieve_from_plan(plan)

        # 3. Verify results
        assert result.original_query == query
        assert result.decomposed_query.was_decomposed is True, "Expected decomposition for complex query"
        print(f"\nDecomposition ({len(result.decomposed_query.sub_queries)} sub-queries, bounded <= 4):")
        for sq in result.decomposed_query.sub_queries:
            print(f"  [{sq.sub_query_id}] ({sq.sub_query_type}) -> '{sq.query_text}'")

        print(f"\nEvidence Pool: {result.evidence_pool.unique_chunks_count} unique chunks from {result.evidence_pool.total_candidates_examined} examined candidates.")
        coverage = result.evidence_pool.get_sub_query_coverage()
        print(f"Sub-query Evidence Coverage: {coverage}")

        print(f"\nSufficiency Assessment:")
        print(f"  Sufficient?      : {result.sufficiency.is_sufficient}")
        print(f"  Score            : {result.sufficiency.sufficiency_score:.3f}")
        print(f"  Coverage Fraction: {result.sufficiency.concept_coverage:.0%}")
        print(f"  Top Relevance    : {result.sufficiency.top_relevance_score:.3f}")
        print(f"  Recommendation   : {result.sufficiency.recommendation}")
        print(f"  Attempts Used    : {result.retrieval_attempts} (escalated={result.was_escalated})")
        print(f"  Reason           : {result.sufficiency.reason}")

        print(f"\nFinal Ranked Evidence (Top {len(result.final_evidence)}):")
        for i, item in enumerate(result.final_evidence[:3], start=1):
            assert item.chunk_id, "Missing chunk_id"
            assert item.final_rerank_score is not None, "Missing final_rerank_score"
            print(f"  {i}. Chunk ID: {item.chunk_id}")
            print(f"     Title   : {item.doc_title} -> {item.section_title}")
            print(f"     Rerank  : {item.final_rerank_score:.4f} | Sub-queries: {item.retrieved_by_sub_queries} (overlap={item.overlap_count})")
            print(f"     Snippet : {item.content[:120]}...\n")

    print("All Phase 9 sanity checks passed successfully!")


if __name__ == "__main__":
    run_complex_sanity_check()
