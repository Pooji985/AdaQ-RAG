"""Knowledge ingestion pipeline orchestrating fetch, clean, and validation."""

import json
import logging
from pathlib import Path

import httpx

from adaq_rag.ingestion.cleaner import clean_html_document
from adaq_rag.ingestion.models import IngestionReport, ProcessedDocument, SourceDefinition
from adaq_rag.ingestion.sources import get_curated_sources

logger = logging.getLogger("adaq_rag.ingestion")


class IngestionPipeline:
    """Orchestrates scikit-learn documentation ingestion, transformation, and validation."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        sources: list[SourceDefinition] | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.sources = sources or get_curated_sources()

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_source_html(
        self,
        source: SourceDefinition,
        client: httpx.Client,
        force_download: bool = False,
    ) -> str:
        """Retrieve raw HTML from local cache or fetch from official URL.

        Args:
            source: Target source definition.
            client: HTTP client instance.
            force_download: If True, bypasses local file cache.

        Returns:
            str: Raw HTML content.
        """
        raw_path = self.raw_dir / f"{source.doc_id}.html"

        if raw_path.exists() and not force_download:
            logger.info("Loading cached raw HTML: %s", raw_path.name)
            return raw_path.read_text(encoding="utf-8")

        logger.info("Fetching documentation from URL: %s", source.url)
        response = client.get(source.url)
        response.raise_for_status()

        html_content = response.text
        raw_path.write_text(html_content, encoding="utf-8")
        logger.info("Cached raw HTML to: %s", raw_path.name)
        return html_content

    def run(self, force_download: bool = False) -> IngestionReport:
        """Execute the ingestion pipeline across all curated sources.

        Args:
            force_download: If True, forces re-fetching of all remote sources.

        Returns:
            IngestionReport: Detailed pipeline execution and validation summary.
        """
        collected = 0
        processed_docs: list[ProcessedDocument] = []
        failed_sources: list[str] = []
        category_counts: dict[str, int] = {}
        total_chars = 0
        total_words = 0

        timeout = httpx.Timeout(30.0, connect=15.0)

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            for source in self.sources:
                try:
                    html_content = self.fetch_source_html(
                        source=source,
                        client=client,
                        force_download=force_download,
                    )
                    collected += 1

                    doc = clean_html_document(html_content, source)
                    processed_docs.append(doc)

                    # Persist structured document JSON
                    json_path = self.processed_dir / f"{doc.doc_id}.json"
                    json_path.write_text(
                        doc.model_dump_json(indent=2),
                        encoding="utf-8",
                    )

                    # Persist clean Markdown representation
                    md_path = self.processed_dir / f"{doc.doc_id}.md"
                    md_path.write_text(doc.content, encoding="utf-8")

                    total_chars += doc.char_count
                    total_words += doc.word_count
                    category_counts[doc.category] = category_counts.get(doc.category, 0) + 1

                    logger.info(
                        "Processed '%s': %d sections, %d words",
                        doc.title,
                        len(doc.sections),
                        doc.word_count,
                    )

                except Exception as exc:
                    logger.error("Failed to ingest source %s: %s", source.doc_id, exc)
                    failed_sources.append(f"{source.doc_id} ({source.url}): {exc}")

        # Generate and save corpus manifest
        manifest_data = {
            "total_documents": len(processed_docs),
            "total_characters": total_chars,
            "total_words": total_words,
            "categories": category_counts,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "category": d.category,
                    "version": d.version,
                    "source_url": d.source_url,
                    "headings_count": len(d.headings),
                    "sections_count": len(d.sections),
                    "char_count": d.char_count,
                    "word_count": d.word_count,
                }
                for d in processed_docs
            ],
        }

        manifest_path = self.processed_dir / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        logger.info("Saved corpus manifest to: %s", manifest_path)

        report = IngestionReport(
            total_sources=len(self.sources),
            collected_count=collected,
            processed_count=len(processed_docs),
            failed_count=len(failed_sources),
            skipped_count=0,
            total_characters=total_chars,
            total_words=total_words,
            failed_sources=failed_sources,
            categories_breakdown=category_counts,
        )

        return report
