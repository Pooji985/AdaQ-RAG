"""Abstract base class for chunking strategies."""

from abc import ABC, abstractmethod

from adaq_rag.chunking.models import Chunk
from adaq_rag.ingestion.models import ProcessedDocument


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the unique identifier for the chunking strategy."""
        pass

    @abstractmethod
    def chunk_document(self, doc: ProcessedDocument) -> list[Chunk]:
        """Split a processed document into chunks.

        Args:
            doc: ProcessedDocument instance.

        Returns:
            list[Chunk]: List of chunks with preserved metadata.
        """
        pass

    def chunk_documents(self, docs: list[ProcessedDocument]) -> list[Chunk]:
        """Split multiple documents into chunks.

        Args:
            docs: Collection of ProcessedDocument objects.

        Returns:
            list[Chunk]: Aggregated list of all chunks.
        """
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
