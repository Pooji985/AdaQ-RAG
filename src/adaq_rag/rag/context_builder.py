"""Context construction from retrieved documentation chunks."""

import logging
from adaq_rag.rag.models import SourceReference
from adaq_rag.rag.prompts import USER_PROMPT_TEMPLATE
from adaq_rag.retrieval.models import RetrievalResult

logger = logging.getLogger("adaq_rag.rag.context_builder")


class ContextBuilder:
    """Formats ranked retrieval chunks into structured LLM prompt context."""

    def __init__(self, snippet_max_chars: int = 150) -> None:
        self.snippet_max_chars = snippet_max_chars

    def build_context(
        self,
        retrieved_chunks: list[RetrievalResult],
    ) -> tuple[str, list[SourceReference]]:
        """Construct structured context string and matching source citations.

        Args:
            retrieved_chunks: Ranked list of retrieval results from the retriever.

        Returns:
            tuple[str, list[SourceReference]]: Formatted context block and source references.
        """
        if not retrieved_chunks:
            return "No relevant documentation excerpts were found.", []

        formatted_blocks: list[str] = []
        sources: list[SourceReference] = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            # Create snippet for citation reference
            snippet = chunk.content[: self.snippet_max_chars].replace("\n", " ").strip()
            if len(chunk.content) > self.snippet_max_chars:
                snippet += "..."

            source_ref = SourceReference(
                source_index=idx,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section_title=chunk.section_title,
                section_level=chunk.section_level,
                source_url=chunk.metadata.get("source_url", ""),
                score=chunk.score,
                content_snippet=snippet,
            )
            sources.append(source_ref)

            # Block formatting with clear demarcation and metadata
            header = f"[Source {idx}] (Document: {chunk.doc_title} | Section: {chunk.section_title})"
            block = f"{header}\n{chunk.content}"
            formatted_blocks.append(block)

        full_context = "\n\n---\n\n".join(formatted_blocks)
        return full_context, sources

    def format_prompt(
        self,
        question: str,
        retrieved_chunks: list[RetrievalResult],
    ) -> tuple[str, list[SourceReference]]:
        """Build full user prompt containing context and question.

        Args:
            question: User inquiry.
            retrieved_chunks: Ranked list of retrieval chunks.

        Returns:
            tuple[str, list[SourceReference]]: Final user prompt and source references.
        """
        context_str, sources = self.build_context(retrieved_chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, question=question.strip())
        return user_prompt, sources
