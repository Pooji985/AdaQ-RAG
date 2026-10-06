"""Real retrieval sanity check for Phase 8 Hybrid Retrieval and Reranking."""

from adaq_rag.retrieval.hybrid import HybridRetriever

SANITY_QUERIES = [
    ("FACT_LOOKUP", "What is the default scoring strategy used by GridSearchCV if None is specified?"),
    ("EXPLANATION", "Why does Lasso regression produce sparse models with exact zero coefficients compared to Ridge?"),
    ("PROCEDURE", "How to configure cross-validation with StratifiedKFold"),
    ("COMPARISON", "What is the difference between LinearSVC and SVC with a linear kernel?"),
    ("TROUBLESHOOTING", "Why does fit_transform on train data produce data leakage when scaling?"),
]


def run_sanity_check() -> None:
    print("Loading HybridRetriever with real FAISS, BM25, and CrossEncoder...")
    retriever = HybridRetriever.load()
    print("HybridRetriever loaded successfully.\n")

    for category, query in SANITY_QUERIES:
        print(f"============================================================")
        print(f"Category: {category}")
        print(f"Query   : {query}")

        results = retriever.retrieve(
            query=query,
            candidate_top_k=15,
            final_top_k=3,
            use_reranking=True,
        )

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        print(f"Returned {len(results)} reranked results:")

        for i, res in enumerate(results, start=1):
            assert res.chunk_id, "Missing chunk_id"
            assert res.rerank_score is not None, "Missing rerank_score"
            print(f"  {i}. Chunk ID: {res.chunk_id}")
            print(f"     Title   : {res.doc_title} -> {res.section_title}")
            print(f"     Method  : {res.retrieval_method}")
            print(f"     Scores  : rerank={res.rerank_score:.4f}, rrf={res.rrf_score:.5f}, dense={res.dense_score}, bm25={res.bm25_score}")
            print(f"     Snippet : {res.content[:120]}...\n")

    print("All sanity checks passed successfully!")


if __name__ == "__main__":
    run_sanity_check()
