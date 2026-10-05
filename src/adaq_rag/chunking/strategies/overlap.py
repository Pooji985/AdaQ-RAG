"""Fixed-size chunking strategy with sliding window overlap."""

import re

from adaq_rag.chunking.models import Chunk
from adaq_rag.chunking.strategies.base import BaseChunker
from adaq_rag.chunking.tokenizer import count_tokens, split_into_paragraphs, split_into_sentences
from adaq_rag.ingestion.models import ProcessedDocument


class FixedOverlapChunker(BaseChunker):
    """Splits documents into fixed-size chunks with sliding window token overlap."""

    def __init__(
        self,
        target_tokens: int = 500,
        overlap_tokens: int = 100,
        min_tokens: int = 50,
    ) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    @property
    def strategy_name(self) -> str:
        return "fixed_overlap"

    def chunk_document(self, doc: ProcessedDocument) -> list[Chunk]:
        raw_paragraphs = split_into_paragraphs(doc.content)
        if not raw_paragraphs:
            return []

        # Decompose into atomic units (sentences if paragraph is large, else paragraph)
        units: list[tuple[str, int, str, int]] = []  # (text, tokens, section_title, section_level)
        current_section = doc.title
        current_level = 1

        for para in raw_paragraphs:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", para)
            if heading_match:
                current_level = len(heading_match.group(1))
                current_section = heading_match.group(2).strip()

            p_tokens = count_tokens(para)
            if p_tokens > self.overlap_tokens:
                for sent in split_into_sentences(para):
                    s_tokens = count_tokens(sent)
                    if s_tokens > 0:
                        units.append((sent, s_tokens, current_section, current_level))
            else:
                if p_tokens > 0:
                    units.append((para, p_tokens, current_section, current_level))

        if not units:
            return []

        chunks: list[Chunk] = []
        i = 0
        n = len(units)

        while i < n:
            chunk_units: list[str] = []
            chunk_tokens = 0
            start_idx = i
            active_section = units[i][2]
            active_level = units[i][3]

            while i < n:
                text, tokens, sec_title, sec_lvl = units[i]
                if chunk_tokens + tokens > self.target_tokens and chunk_tokens >= self.min_tokens:
                    break
                chunk_units.append(text)
                chunk_tokens += tokens
                i += 1

            if not chunk_units:
                # Fallback in case a single unit exceeds target_tokens
                text, tokens, sec_title, sec_lvl = units[i]
                chunk_units.append(text)
                chunk_tokens += tokens
                i += 1

            chunk_text = "\n\n".join(chunk_units).strip()
            chunk_idx = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_overlap_{chunk_idx:04d}",
                    strategy=self.strategy_name,
                    doc_id=doc.doc_id,
                    source_url=doc.source_url,
                    doc_title=doc.title,
                    section_title=active_section,
                    section_level=active_level,
                    content=chunk_text,
                    token_count=chunk_tokens,
                    char_count=len(chunk_text),
                    metadata={
                        "target_tokens": self.target_tokens,
                        "overlap_tokens": self.overlap_tokens,
                    },
                )
            )

            if i >= n:
                break

            # Calculate step-back for sliding overlap
            overlap_accum = 0
            step_back = 0
            # Look backwards from current position i to find overlap window
            for back_idx in range(i - 1, start_idx, -1):
                overlap_accum += units[back_idx][1]
                step_back += 1
                if overlap_accum >= self.overlap_tokens:
                    break

            # Advance by at least 1 unit to guarantee forward progress
            next_start = i - step_back
            if next_start <= start_idx:
                next_start = start_idx + 1

            i = next_start

        return chunks
