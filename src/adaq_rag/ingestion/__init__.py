"""Knowledge ingestion package for AdaQ-RAG."""

from adaq_rag.ingestion.cleaner import clean_html_document
from adaq_rag.ingestion.models import (
    DocumentSection,
    IngestionReport,
    ProcessedDocument,
    SourceDefinition,
)
from adaq_rag.ingestion.pipeline import IngestionPipeline
from adaq_rag.ingestion.sources import CURATED_SOURCES, get_curated_sources

__all__ = [
    "clean_html_document",
    "DocumentSection",
    "IngestionReport",
    "ProcessedDocument",
    "SourceDefinition",
    "IngestionPipeline",
    "CURATED_SOURCES",
    "get_curated_sources",
]
