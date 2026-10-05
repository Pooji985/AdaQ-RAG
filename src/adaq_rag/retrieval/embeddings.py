"""Dense text embedding module using Sentence Transformers."""

import logging
from typing import Any
import numpy as np

from adaq_rag.chunking.models import Chunk
from adaq_rag.core.config import get_settings

logger = logging.getLogger("adaq_rag.retrieval.embeddings")


class EmbeddingModel:
    """Configurable local embedding generator using Sentence Transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: Compute device ('cpu', 'cuda', etc.). If None, auto-detected.
            batch_size: Batch size for inference encoding.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or "cpu"
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None
        self._dimension: int | None = None

    @property
    def model(self):
        """Lazily load SentenceTransformer model instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch

            # Ensure deterministic CPU inference
            torch.set_grad_enabled(False)
            logger.info("Loading sentence transformer model: %s on device: %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimension of the embedding output."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        return self._dimension

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string into a normalized 1D embedding vector.

        Args:
            text: Input string.

        Returns:
            np.ndarray: 1D float32 normalized vector.
        """
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_texts(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode multiple text strings into a normalized 2D embedding matrix.

        Args:
            texts: List of input strings.
            show_progress: Whether to show progress bar during encoding.

        Returns:
            np.ndarray: 2D float32 normalized matrix of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)

    def encode_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
        """Generate embeddings and mapping metadata for a list of Chunks.

        Args:
            chunks: Chunks to encode.
            show_progress: Display progress during encoding.

        Returns:
            tuple containing:
                - embeddings: (N, D) float32 matrix
                - chunk_ids: List of chunk ID strings in matching row order
                - metadata: List of chunk metadata dictionaries in matching row order
        """
        texts = [c.content for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        metadata = [
            {
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
        ]

        logger.info("Encoding %d chunks with model %s...", len(chunks), self.model_name)
        embeddings = self.encode_texts(texts, show_progress=show_progress)
        return embeddings, chunk_ids, metadata
