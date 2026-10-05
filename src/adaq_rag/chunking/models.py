"""Data models for text chunking and chunking evaluation."""

from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a discrete text chunk with associated document metadata."""

    chunk_id: str = Field(description="Unique chunk identifier")
    strategy: str = Field(description="Chunking strategy name (fixed, fixed_overlap, structure_aware)")
    doc_id: str = Field(description="Source document identifier")
    source_url: str = Field(description="Original documentation URL")
    doc_title: str = Field(description="Parent document title")
    section_title: str = Field(description="Active section or subsection title")
    section_level: int = Field(default=1, description="Depth level of the section heading")
    content: str = Field(description="The chunk text content")
    token_count: int = Field(description="Estimated or exact token count")
    char_count: int = Field(description="Character count of the content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")


class ChunkStats(BaseModel):
    """Statistical summary of a chunk dataset."""

    strategy: str
    total_chunks: int
    total_tokens: int
    total_chars: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    avg_chars: float
    min_chars: int
    max_chars: int
    undersized_count: int
    undersized_pct: float
    oversized_count: int
    oversized_pct: float
    metadata_completeness_pct: float


class ChunkEvalQuestion(BaseModel):
    """Representative probe question for measuring chunk information preservation."""

    question_id: str
    query: str
    category: str
    question_type: str = Field(default="", description="Question archetype (FACT_LOOKUP, EXPLANATION, etc.)")
    target_doc_id: str
    expected_section: str
    key_phrases: list[str] = Field(
        description="Key terms/phrases that must co-occur together in a single chunk to avoid context fragmentation"
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.question_type:
            self.question_type = self.category
