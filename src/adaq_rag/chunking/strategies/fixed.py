"""Fixed-size chunking strategy without overlap."""

import re

from adaq_rag.chunking.models import Chunk
from adaq_rag.chunking.strategies.base import BaseChunker
from adaq_rag.chunking.tokenizer import count_tokens, split_into_paragraphs, split_into_sentences
from adaq_rag.ingestion.models import ProcessedDocument


class FixedChunker(BaseChunker):
    """Splits documents into fixed-size chunks of approximately target_tokens."""

    def __init__(self, target_tokens: int = 500, min_tokens: int = 50) -> None:
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk_document(self, doc: ProcessedDocument) -> list[Chunk]:
        paragraphs = split_into_paragraphs(doc.content)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        current_blocks: list[str] = []
        current_tokens = 0
        current_section = doc.title
        current_level = 1

        def flush_chunk() -> None:
            nonlocal current_blocks, current_tokens
            if not current_blocks:
                return
            chunk_text = "\n\n".join(current_blocks).strip()
            t_count = count_tokens(chunk_text)
            if t_count == 0:
                current_blocks = []
                current_tokens = 0
                return

            chunk_idx = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_fixed_{chunk_idx:04d}",
                    strategy=self.strategy_name,
                    doc_id=doc.doc_id,
                    source_url=doc.source_url,
                    doc_title=doc.title,
                    section_title=current_section,
                    section_level=current_level,
                    content=chunk_text,
                    token_count=t_count,
                    char_count=len(chunk_text),
                    metadata={"target_tokens": self.target_tokens},
                )
            )
            current_blocks = []
            current_tokens = 0

        for para in paragraphs:
            # Check if paragraph is a heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", para)
            if heading_match:
                current_level = len(heading_match.group(1))
                current_section = heading_match.group(2).strip()

            para_tokens = count_tokens(para)

            # If a single paragraph is oversized, split by sentences
            if para_tokens > self.target_tokens:
                sentences = split_into_sentences(para)
                for sentence in sentences:
                    s_tokens = count_tokens(sentence)
                    if current_tokens + s_tokens > self.target_tokens and current_tokens >= self.min_tokens:
                        flush_chunk()
                    current_blocks.append(sentence)
                    current_tokens += s_tokens
            else:
                if current_tokens + para_tokens > self.target_tokens and current_tokens >= self.min_tokens:
                    flush_chunk()
                current_blocks.append(para)
                current_tokens += para_tokens

        # Flush trailing block
        if current_blocks:
            trailing_tokens = count_tokens("\n\n".join(current_blocks))
            # Merge with previous chunk if too small and previous exists
            if trailing_tokens < self.min_tokens and chunks:
                prev = chunks[-1]
                merged_text = prev.content + "\n\n" + "\n\n".join(current_blocks).strip()
                chunks[-1] = Chunk(
                    chunk_id=prev.chunk_id,
                    strategy=prev.strategy,
                    doc_id=prev.doc_id,
                    source_url=prev.source_url,
                    doc_title=prev.doc_title,
                    section_title=prev.section_title,
                    section_level=prev.section_level,
                    content=merged_text,
                    token_count=count_tokens(merged_text),
                    char_count=len(merged_text),
                    metadata=prev.metadata,
                )
            else:
                flush_chunk()

        return chunks
