"""BM25 lexical retrieval index construction and search."""

import json
import logging
from pathlib import Path
import pickle
import re
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi

from adaq_rag.chunking.models import Chunk

logger = logging.getLogger("adaq_rag.retrieval.bm25_index")

# Alphanumeric words, underscores for python identifiers (e.g. cv_results_, OneHotEncoder)
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text for BM25 lexical indexing.

    Args:
        text: Input string.

    Returns:
        list[str]: Lowercased word/identifier tokens.
    """
    if not text:
        return []
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class BM25Index:
    """Lexical search index using BM25Okapi."""

    def __init__(
        self,
        bm25: BM25Okapi,
        id_mapping: list[str],
        chunk_metadata: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize in-memory BM25 index.

        Args:
            bm25: Instantiated BM25Okapi model.
            id_mapping: Ordered list mapping document position to chunk_id.
            chunk_metadata: Mapping of chunk_id to metadata attributes.
        """
        self.bm25 = bm25
        self.id_mapping = id_mapping
        self.chunk_metadata = chunk_metadata

    @property
    def total_documents(self) -> int:
        """Number of indexed chunks."""
        return len(self.id_mapping)

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "BM25Index":
        """Build and persist BM25 index from text chunks.

        Args:
            chunks: List of Chunk objects to index.
            index_path: Path to write pickled BM25 model.
            metadata_path: Path to write JSON metadata.

        Returns:
            BM25Index: Instantiated BM25 index.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Building BM25 index for %d chunks...", len(chunks))
        tokenized_corpus = [tokenize_text(c.content) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        id_mapping = [c.chunk_id for c in chunks]
        chunk_metadata = {
            c.chunk_id: {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "section_title": c.section_title,
                "section_level": c.section_level,
                "source_url": c.source_url,
                "token_count": c.token_count,
                "char_count": c.char_count,
                "content": c.content,
            }
            for c in chunks
        }

        # Save binary model
        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)

        # Save metadata JSON
        meta_dict = {
            "total_documents": len(chunks),
            "id_mapping": id_mapping,
            "chunk_metadata": chunk_metadata,
        }
        metadata_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

        logger.info(
            "Built BM25 index: %d documents -> %s, metadata -> %s",
            len(chunks),
            index_path,
            metadata_path,
        )
        return cls(bm25=bm25, id_mapping=id_mapping, chunk_metadata=chunk_metadata)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        metadata_path: str | Path,
    ) -> "BM25Index":
        """Load an existing BM25 index and metadata from disk.

        Args:
            index_path: Path to pickled BM25 model file.
            metadata_path: Path to JSON metadata file.

        Returns:
            BM25Index: Loaded index ready for searching.
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"BM25 metadata file not found: {metadata_path}")

        with open(index_path, "rb") as f:
            bm25: BM25Okapi = pickle.load(f)

        meta_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        id_mapping = meta_dict["id_mapping"]
        chunk_metadata = meta_dict["chunk_metadata"]

        logger.info("Loaded BM25 index with %d documents", len(id_mapping))
        return cls(bm25=bm25, id_mapping=id_mapping, chunk_metadata=chunk_metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search top scoring chunks for a query string.

        Args:
            query: Query string.
            top_k: Number of highest ranking chunks to return.

        Returns:
            list[tuple[str, float, dict[str, Any]]]: List of (chunk_id, bm25_score, metadata).
        """
        if not self.id_mapping:
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k == 0:
            return []

        # Sort indices descending by score
        top_indices = np.argsort(-scores)[:k]

        results: list[tuple[str, float, dict[str, Any]]] = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk_id = self.id_mapping[idx]
            meta = self.chunk_metadata.get(chunk_id, {})
            results.append((chunk_id, score, meta))

        return results
