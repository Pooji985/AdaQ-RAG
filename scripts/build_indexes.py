"""CLI script to build dense vector (FAISS) and sparse lexical (BM25) indexes from Phase 3 chunks."""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Ensure src is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaq_rag.chunking.models import Chunk
from adaq_rag.core.config import get_settings
from adaq_rag.core.logging import setup_logging
from adaq_rag.retrieval.bm25_index import BM25Index
from adaq_rag.retrieval.embeddings import EmbeddingModel
from adaq_rag.retrieval.retriever import UnifiedRetriever
from adaq_rag.retrieval.vector_index import FAISSVectorIndex

logger = logging.getLogger("adaq_rag.build_indexes")


def load_chunks(chunks_file: Path) -> list[Chunk]:
    """Load structure-aware chunks from JSON file."""
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
    raw_data = json.loads(chunks_file.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw_data]


def build_all_indexes(
    chunks_path: Path = Path("data/processed/chunks/structure_aware.json"),
    output_dir: Path | None = None,
    embedding_model_name: str | None = None,
) -> dict:
    """Orchestrate embedding generation, FAISS vector index, and BM25 index creation."""
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir else Path(settings.indexes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("           AdaQ-RAG: PHASE 4 OFFLINE INDEXING PIPELINE")
    print("=" * 70)

    # 1. Load Chunks
    print(f"\n[1/4] Loading Phase 3 structure-aware chunks from: {chunks_path}")
    start_time = time.perf_counter()
    chunks = load_chunks(chunks_path)
    print(f"      Loaded {len(chunks)} chunks.")

    # 2. Dense Embeddings & FAISS Vector Index
    model_name = embedding_model_name or settings.embedding_model
    print(f"\n[2/4] Generating dense embeddings using model: {model_name}")
    embedder = EmbeddingModel(model_name=model_name)
    embed_start = time.perf_counter()
    embeddings, chunk_ids, metadata = embedder.encode_chunks(chunks, show_progress=True)
    embed_duration = time.perf_counter() - embed_start
    print(f"      Encoded {embeddings.shape[0]} chunks into {embeddings.shape[1]}-dim vectors in {embed_duration:.2f}s.")

    vector_index_path = out_dir / settings.vector_index_file
    vector_meta_path = out_dir / settings.vector_metadata_file
    print(f"      Building FAISS IndexFlatIP (cosine similarity) -> {vector_index_path}")
    FAISSVectorIndex.build(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        metadata=metadata,
        index_path=vector_index_path,
        metadata_path=vector_meta_path,
    )

    # 3. BM25 Lexical Index
    print(f"\n[3/4] Building BM25 lexical index from tokenized chunks")
    bm25_start = time.perf_counter()
    bm25_index_path = out_dir / settings.bm25_index_file
    bm25_meta_path = out_dir / settings.bm25_metadata_file
    BM25Index.build(
        chunks=chunks,
        index_path=bm25_index_path,
        metadata_path=bm25_meta_path,
    )
    bm25_duration = time.perf_counter() - bm25_start
    print(f"      Indexed {len(chunks)} documents in {bm25_duration:.2f}s -> {bm25_index_path}")

    # 4. Manifest & Verification
    total_duration = time.perf_counter() - start_time
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(chunks),
        "source_chunks_file": str(chunks_path),
        "embedding_model": model_name,
        "embedding_dimension": embedder.dimension,
        "vector_index_type": "FAISS_IndexFlatIP",
        "vector_index_file": str(vector_index_path),
        "vector_metadata_file": str(vector_meta_path),
        "bm25_index_file": str(bm25_index_path),
        "bm25_metadata_file": str(bm25_meta_path),
        "build_duration_seconds": round(total_duration, 2),
    }

    manifest_path = out_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Sanity check retrieval
    print(f"\n[4/4] Verifying index retrieval with sample query...")
    retriever = UnifiedRetriever.load(indexes_dir=out_dir, embedding_model=embedder)
    test_query = "How to avoid data leakage with Pipeline and cross-validation?"
    dense_hits = retriever.retrieve_dense(test_query, top_k=3)
    bm25_hits = retriever.retrieve_bm25(test_query, top_k=3)

    print(f"\nQuery: '{test_query}'")
    print(f"  Dense top hit: {dense_hits[0].chunk_id} (score={dense_hits[0].score:.4f}, section={dense_hits[0].section_title})")
    print(f"  BM25  top hit: {bm25_hits[0].chunk_id} (score={bm25_hits[0].score:.4f}, section={bm25_hits[0].section_title})")

    print("\n" + "=" * 70)
    print("                    BUILD SUMMARY")
    print("=" * 70)
    print(f"Chunks Indexed:         {len(chunks)}")
    print(f"Embedding Model:        {model_name}")
    print(f"Embedding Dimension:    {embedder.dimension}")
    print(f"Vector Index Type:      FAISS IndexFlatIP (Cosine Similarity)")
    print(f"BM25 Index Type:        BM25Okapi (Normalized lexical)")
    print(f"Artifacts Directory:    {out_dir}")
    print(f"Total Build Time:       {total_duration:.2f}s")
    print("=" * 70)

    return manifest


def main() -> int:
    """CLI entrypoint."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Build dense vector and BM25 sparse indexes.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/chunks/structure_aware.json"),
        help="Path to structure-aware chunks JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for index files (defaults to settings.indexes_dir).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Sentence Transformer model name (defaults to settings.embedding_model).",
    )

    args = parser.parse_args()
    build_all_indexes(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        embedding_model_name=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
