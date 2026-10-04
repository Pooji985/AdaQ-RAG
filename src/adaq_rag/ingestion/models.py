"""Data models for knowledge ingestion and processing."""

from pydantic import BaseModel, Field


class SourceDefinition(BaseModel):
    """Metadata specification for a target documentation source."""

    doc_id: str = Field(description="Unique snake_case identifier for the document")
    url: str = Field(description="Official documentation URL")
    category: str = Field(description="Functional area category")
    description: str = Field(description="Brief summary of document coverage")


class DocumentSection(BaseModel):
    """A structured subsection within a processed document."""

    title: str = Field(description="Section heading title")
    level: int = Field(description="Heading depth (1=H1, 2=H2, 3=H3, etc.)")
    content: str = Field(description="Clean markdown text content belonging to this section")


class ProcessedDocument(BaseModel):
    """Structured, cleaned representation of an ingested document."""

    doc_id: str
    source_url: str
    title: str
    category: str
    version: str
    headings: list[str]
    sections: list[DocumentSection]
    content: str
    char_count: int
    word_count: int


class IngestionReport(BaseModel):
    """Execution summary and validation metrics for the ingestion pipeline."""

    total_sources: int
    collected_count: int
    processed_count: int
    failed_count: int
    skipped_count: int
    total_characters: int
    total_words: int
    failed_sources: list[str] = Field(default_factory=list)
    categories_breakdown: dict[str, int] = Field(default_factory=dict)
