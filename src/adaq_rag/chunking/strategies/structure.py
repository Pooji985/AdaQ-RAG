"""Structure-aware chunking strategy using document section hierarchy."""

from adaq_rag.chunking.models import Chunk
from adaq_rag.chunking.strategies.base import BaseChunker
from adaq_rag.chunking.tokenizer import count_tokens, split_into_paragraphs, split_into_sentences
from adaq_rag.ingestion.models import ProcessedDocument


class StructureAwareChunker(BaseChunker):
    """Splits documents according to section boundaries with fallback sub-chunking."""

    def __init__(self, max_tokens: int = 550, min_tokens: int = 50) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    @property
    def strategy_name(self) -> str:
        return "structure_aware"

    def _split_large_section(
        self,
        title: str,
        level: int,
        content: str,
    ) -> list[tuple[str, int]]:
        """Split an oversized section into paragraph-aligned chunks."""
        prefix = "#" * max(1, min(level, 4))
        header_budget = count_tokens(f"{prefix} {title} (Part 1/1)\n\n")
        body_max = max(self.min_tokens, self.max_tokens - header_budget)

        paragraphs = split_into_paragraphs(content)
        sub_chunks: list[str] = []
        current_blocks: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current_blocks, current_tokens
            if current_blocks:
                sub_chunks.append("\n\n".join(current_blocks).strip())
                current_blocks = []
                current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)
            if para_tokens > body_max:
                sentences = split_into_sentences(para)
                for sentence in sentences:
                    s_tokens = count_tokens(sentence)
                    if current_tokens + s_tokens > body_max and current_tokens >= self.min_tokens:
                        flush()
                    current_blocks.append(sentence)
                    current_tokens += s_tokens
            else:
                if current_tokens + para_tokens > body_max and current_tokens >= self.min_tokens:
                    flush()
                current_blocks.append(para)
                current_tokens += para_tokens

        flush()
        return [(sc, count_tokens(sc)) for sc in sub_chunks if sc]

    def chunk_document(self, doc: ProcessedDocument) -> list[Chunk]:
        if not doc.sections:
            # Fallback if no sections extracted
            return []

        chunks: list[Chunk] = []

        # First pass: Coalesce trivial stub sections (< min_tokens) into logical units
        coalesced_sections: list[tuple[str, int, str]] = []  # (title, level, content)
        pending_title: str | None = None
        pending_level: int = 1
        pending_content: list[str] = []

        for section in doc.sections:
            sec_tokens = count_tokens(section.content)
            if sec_tokens < self.min_tokens:
                if pending_title is None:
                    pending_title = section.title
                    pending_level = section.level
                pending_content.append(f"### {section.title}\n{section.content}")
            else:
                if pending_content:
                    pending_content.append(f"### {section.title}\n{section.content}")
                    merged = "\n\n".join(pending_content)
                    coalesced_sections.append((pending_title or section.title, pending_level, merged))
                    pending_title = None
                    pending_level = 1
                    pending_content = []
                else:
                    coalesced_sections.append((section.title, section.level, section.content))

        if pending_content:
            merged = "\n\n".join(pending_content)
            coalesced_sections.append((pending_title or doc.title, pending_level, merged))

        # Second pass: Process sections (keep intact or sub-split if > max_tokens)
        for title, level, content in coalesced_sections:
            tokens = count_tokens(content)

            if tokens <= self.max_tokens:
                prefix = "#" * max(1, min(level, 4))
                formatted_text = f"{prefix} {title}\n\n{content}".strip()
                t_count = count_tokens(formatted_text)
                chunk_idx = len(chunks) + 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_struct_{chunk_idx:04d}",
                        strategy=self.strategy_name,
                        doc_id=doc.doc_id,
                        source_url=doc.source_url,
                        doc_title=doc.title,
                        section_title=title,
                        section_level=level,
                        content=formatted_text,
                        token_count=t_count,
                        char_count=len(formatted_text),
                        metadata={"is_split": False, "part": 1, "total_parts": 1},
                    )
                )
            else:
                # Oversized section: split further while preserving section metadata
                parts = self._split_large_section(title, level, content)
                total_parts = len(parts)
                for part_idx, (part_content, _) in enumerate(parts, start=1):
                    prefix = "#" * max(1, min(level, 4))
                    part_header = f"{prefix} {title} (Part {part_idx}/{total_parts})"
                    formatted_text = f"{part_header}\n\n{part_content}".strip()
                    t_count = count_tokens(formatted_text)
                    chunk_idx = len(chunks) + 1
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}_struct_{chunk_idx:04d}",
                            strategy=self.strategy_name,
                            doc_id=doc.doc_id,
                            source_url=doc.source_url,
                            doc_title=doc.title,
                            section_title=title,
                            section_level=level,
                            content=formatted_text,
                            token_count=t_count,
                            char_count=len(formatted_text),
                            metadata={
                                "is_split": True,
                                "part": part_idx,
                                "total_parts": total_parts,
                            },
                        )
                    )

        return chunks
