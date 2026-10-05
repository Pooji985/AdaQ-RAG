"""Pipeline for running chunking strategies across processed documents."""

import json
import logging
from pathlib import Path

from adaq_rag.chunking.models import Chunk
from adaq_rag.chunking.strategies import (
    BaseChunker,
    FixedChunker,
    FixedOverlapChunker,
    StructureAwareChunker,
)
from adaq_rag.ingestion.models import ProcessedDocument

logger = logging.getLogger("adaq_rag.chunking")


def load_processed_documents(processed_dir: str | Path = "data/processed") -> list[ProcessedDocument]:
    """Load all processed document JSON files from data/processed/.

    Args:
        processed_dir: Path to directory containing processed JSON files.

    Returns:
        list[ProcessedDocument]: Parsed documents.
    """
    p_path = Path(processed_dir)
    docs: list[ProcessedDocument] = []

    for file_path in sorted(p_path.glob("*.json")):
        if file_path.name == "corpus_manifest.json":
            continue
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            docs.append(ProcessedDocument(**data))
        except Exception as exc:
            logger.warning("Could not parse processed doc %s: %exc", file_path.name, exc)

    logger.info("Loaded %d processed documents from %s", len(docs), p_path)
    return docs


class ChunkingPipeline:
    """Orchestrates document chunking and chunk dataset persistence."""

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        output_dir: str | Path = "data/processed/chunks",
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.strategies: dict[str, BaseChunker] = {
            "fixed": FixedChunker(target_tokens=500, min_tokens=50),
            "fixed_overlap": FixedOverlapChunker(target_tokens=500, overlap_tokens=100, min_tokens=50),
            "structure_aware": StructureAwareChunker(max_tokens=550, min_tokens=50),
        }

    def run_strategy(self, strategy_name: str, docs: list[ProcessedDocument]) -> list[Chunk]:
        """Execute a single named chunking strategy.

        Args:
            strategy_name: Key in self.strategies.
            docs: List of processed documents.

        Returns:
            list[Chunk]: Generated chunks.
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")

        chunker = self.strategies[strategy_name]
        chunks = chunker.chunk_documents(docs)

        # Save to output directory
        out_file = self.output_dir / f"{strategy_name}.json"
        out_file.write_text(
            json.dumps([c.model_dump() for c in chunks], indent=2),
            encoding="utf-8",
        )
        logger.info("Strategy '%s': generated %d chunks -> %s", strategy_name, len(chunks), out_file)
        return chunks

    def run_all(self, docs: list[ProcessedDocument] | None = None) -> dict[str, list[Chunk]]:
        """Execute all configured chunking strategies.

        Args:
            docs: Optional pre-loaded document list; if None, loads from processed_dir.

        Returns:
            dict[str, list[Chunk]]: Mapping of strategy name to generated chunks.
        """
        documents = docs if docs is not None else load_processed_documents(self.processed_dir)
        results: dict[str, list[Chunk]] = {}

        for name in self.strategies:
            results[name] = self.run_strategy(name, documents)

        return results
