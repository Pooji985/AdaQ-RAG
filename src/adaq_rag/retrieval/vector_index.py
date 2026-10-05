"""FAISS-based dense vector index construction and search."""

import json
import logging
from pathlib import Path
from typing import Any
import faiss
import numpy as np

logger = logging.getLogger("adaq_rag.retrieval.vector_index")


class FAISSVectorIndex:
    """Local vector index backed by FAISS IndexFlatIP (cosine similarity)."""

    def __init__(
        self,
        index: faiss.Index,
        id_mapping: list[str],
        chunk_metadata: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize the in-memory FAISS index wrapper.

        Args:
            index: FAISS index object.
            id_mapping: List mapping sequential row position to chunk_id.
            chunk_metadata: Dictionary mapping chunk_id to chunk metadata attributes.
        """
        self.index = index
        self.id_mapping = id_mapping
        self.chunk_metadata = chunk_metadata

    @property
    def total_vectors(self) -> int:
        """Total number of vectors in index."""
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self.index.d

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        metadata: list[dict[str, Any]],
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "FAISSVectorIndex":
        """Build and persist a new FAISS vector index from embeddings and metadata.

        Args:
            embeddings: (N, D) float32 embedding matrix.
            chunk_ids: Ordered list of chunk identifiers matching embeddings.
            metadata: Ordered list of chunk metadata dictionaries.
            index_path: Output file path for FAISS binary index.
            metadata_path: Output file path for JSON metadata mapping.

        Returns:
            FAISSVectorIndex: Instantiated index wrapper.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        n_vectors, dimension = embeddings.shape
        if n_vectors != len(chunk_ids) or n_vectors != len(metadata):
            raise ValueError(
                f"Mismatch: {n_vectors} embeddings, {len(chunk_ids)} chunk_ids, {len(metadata)} metadata items"
            )

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # IndexFlatIP calculates inner product; with normalized vectors, this is cosine similarity
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(vectors)

        # Save FAISS binary index
        faiss.write_index(faiss_index, str(index_path))

        # Save mapping and chunk metadata
        meta_dict = {
            "dimension": dimension,
            "total_vectors": n_vectors,
            "id_mapping": chunk_ids,
            "chunk_metadata": {m["chunk_id"]: m for m in metadata},
        }
        metadata_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

        logger.info(
            "Built FAISS vector index: %d vectors (dim=%d) -> %s, metadata -> %s",
            n_vectors,
            dimension,
            index_path,
            metadata_path,
        )
        return cls(index=faiss_index, id_mapping=chunk_ids, chunk_metadata=meta_dict["chunk_metadata"])

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "FAISSVectorIndex":
        """Load an existing FAISS index and its metadata from disk.

        Args:
            index_path: Path to FAISS binary index file.
            metadata_path: Path to JSON metadata file.

        Returns:
            FAISSVectorIndex: Loaded index ready for querying.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"FAISS metadata file not found: {metadata_path}")

        faiss_index = faiss.read_index(str(index_path))
        meta_dict = json.loads(metadata_path.read_text(encoding="utf-8"))

        id_mapping = meta_dict["id_mapping"]
        chunk_metadata = meta_dict["chunk_metadata"]

        logger.info("Loaded FAISS index with %d vectors (dim=%d)", faiss_index.ntotal, faiss_index.d)
        return cls(index=faiss_index, id_mapping=id_mapping, chunk_metadata=chunk_metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search nearest neighbor chunks for a query embedding vector.

        Args:
            query_vector: 1D or 2D query embedding vector.
            top_k: Number of nearest matches to return.

        Returns:
            list[tuple[str, float, dict[str, Any]]]: List of (chunk_id, similarity_score, metadata).
        """
        if self.index.ntotal == 0:
            return []

        # Format query to 2D contiguous float32
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # L2-normalize if not already unit length
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        if norm[0, 0] > 0:
            q = q / norm

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)

        results: list[tuple[str, float, dict[str, Any]]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.id_mapping):
                continue
            chunk_id = self.id_mapping[idx]
            meta = self.chunk_metadata.get(chunk_id, {})
            results.append((chunk_id, float(score), meta))

        return results
