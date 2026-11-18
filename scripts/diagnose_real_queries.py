"""Diagnostic script to inspect query analysis, routing, retrieval, and orchestrator
for standard real-world queries.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaq_rag.query_analysis import QueryAnalyzer
from adaq_rag.routing import AdaptiveRetrievalRouter
from adaq_rag.retrieval.retriever import DenseRetriever, BM25Retriever
from adaq_rag.retrieval.reranker import CrossEncoderReranker
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.orchestrator import ComplexRetrievalOrchestrator
from adaq_rag.rag.context_builder import ContextBuilder

print("Loading indexes and models...", flush=True)
dense = DenseRetriever.load("indexes/vector_index.faiss", "indexes/vector_metadata.json")
print("Loaded Dense Retriever", flush=True)
bm25 = BM25Retriever.load("indexes/bm25_index.pkl", "indexes/bm25_metadata.json")
print("Loaded BM25 Retriever", flush=True)

reranker = CrossEncoderReranker()
print("Loaded Reranker", flush=True)

hybrid = HybridRetriever(dense, bm25, reranker)
analyzer = QueryAnalyzer()
router = AdaptiveRetrievalRouter()
orchestrator = ComplexRetrievalOrchestrator(hybrid, reranker)
builder = ContextBuilder()

queries = [
    "What is Scikit-learn?",
    "What is Random Forest?",
    "Explain decision trees",
    "What is the purpose of train_test_split?",
    "Compare Random Forest and Decision Tree",
]

for q in queries:
    print("\n" + "=" * 60, flush=True)
    print(f"QUERY: {q}", flush=True)
    analysis = analyzer.analyze(q)
    plan = router.route(analysis)
    print(f"Analysis: Intent={analysis.intent.value}, Complexity={analysis.complexity.value}", flush=True)
    print(f"Plan: Strategy={plan.strategy.value}, Reason={plan.routing_reason}", flush=True)
    
    # 1. Dense retrieval
    dense_res = dense.retrieve(q, top_k=3)
    print(f"Dense top 3 chunks: {[c.chunk_id for c in dense_res]}", flush=True)
    for c in dense_res:
        print(f"   [{c.chunk_id}] score={c.score:.3f} | {c.doc_title} -> {c.section_title}", flush=True)
        print(f"      Excerpt: {c.content[:120].strip()}...", flush=True)

    # 2. BM25 retrieval
    bm25_res = bm25.retrieve(q, top_k=3)
    print(f"BM25 top 3 chunks: {[c.chunk_id for c in bm25_res]}", flush=True)
    for c in bm25_res:
        print(f"   [{c.chunk_id}] score={c.score:.3f} | {c.doc_title} -> {c.section_title}", flush=True)

    # 3. Hybrid retrieval
    hybrid_res = hybrid.retrieve(q, candidate_top_k=10, final_top_k=3, use_reranking=True)
    print(f"Hybrid (reranked) top 3 chunks: {[c.chunk_id for c in hybrid_res]}", flush=True)
    for c in hybrid_res:
        print(f"   [{c.chunk_id}] rerank_score={getattr(c, 'rerank_score', 0):.3f} | {c.doc_title} -> {c.section_title}", flush=True)

    # 4. If MULTI_STEP, test orchestrator
    if plan.strategy.value == "MULTI_STEP":
        try:
            comp_res = orchestrator.retrieve_from_plan(plan)
            print(f"Orchestrator: Sub-queries={len(comp_res.decomposed_query.sub_queries)}, Escalated={comp_res.was_escalated}, Evidence={len(comp_res.final_evidence)}", flush=True)
            for ev in comp_res.final_evidence[:3]:
                print(f"   Evidence [{ev.chunk_id}]: {ev.doc_title} -> {ev.section_title}", flush=True)
        except Exception as e:
            print(f"Orchestrator error: {e}", flush=True)

print("\nDone diagnosis!", flush=True)

