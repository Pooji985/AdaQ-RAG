"""HTML cleaning and structured document extraction for scikit-learn documentation."""

import re
from bs4 import BeautifulSoup, NavigableString, Tag

from adaq_rag.ingestion.models import DocumentSection, ProcessedDocument, SourceDefinition

NOISE_TAGS = {"nav", "aside", "header", "footer", "script", "style", "noscript", "svg", "iframe"}
NOISE_CLASSES = re.compile(
    r"headerlink|viewcode-link|copybutton|prev-next|sphx-glr-footer|sphx-glr-download|toctree-wrapper",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Normalize internal whitespaces and remove Sphinx permalink anchors."""
    cleaned = re.sub(r"[\s#¶]+$", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _render_node_to_markdown(node: Tag | NavigableString) -> str:
    """Recursively render a BeautifulSoup node into structured Markdown."""
    if isinstance(node, NavigableString):
        return str(node)

    tag_name = node.name

    # Ignore noise tags
    if tag_name in NOISE_TAGS:
        return ""

    # Headings
    if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(tag_name[1])
        prefix = "#" * level
        heading_text = _clean_text(node.get_text())
        return f"\n\n{prefix} {heading_text}\n\n"

    # Paragraphs
    if tag_name == "p":
        text = "".join(_render_node_to_markdown(c) for c in node.children).strip()
        return f"\n\n{text}\n\n" if text else ""

    # Code blocks
    if tag_name in ["pre"] or (tag_name == "div" and "highlight" in " ".join(node.get("class", []))):
        pre = node.find("pre") if tag_name == "div" else node
        code_text = pre.get_text() if pre else node.get_text()
        return f"\n\n```python\n{code_text.strip()}\n```\n\n"

    # Inline code
    if tag_name == "code":
        return f"`{node.get_text()}`"

    # List items
    if tag_name == "li":
        text = "".join(_render_node_to_markdown(c) for c in node.children).strip()
        return f"- {text}\n"

    # Unordered / Ordered lists
    if tag_name in ["ul", "ol"]:
        items = "".join(
            _render_node_to_markdown(c) for c in node.children if isinstance(c, Tag) and c.name == "li"
        )
        return f"\n\n{items}\n"

    # Definition lists (Parameters, Attributes, Methods)
    if tag_name == "dt":
        dt_text = _clean_text(" ".join(node.stripped_strings))
        return f"\n\n**{dt_text}**\n"

    if tag_name == "dd":
        dd_text = "".join(_render_node_to_markdown(c) for c in node.children).strip()
        return f": {dd_text}\n\n"

    # Tables
    if tag_name == "table":
        rows: list[str] = []
        all_trs = node.find_all("tr")
        if not all_trs:
            return ""
        for tr in all_trs:
            cells = [_clean_text(c.get_text()) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            num_cols = len(all_trs[0].find_all(["th", "td"]))
            sep = "| " + " | ".join(["---"] * num_cols) + " |"
            table_md = "\n".join([rows[0], sep] + rows[1:]) if len(rows) > 1 else rows[0]
            return f"\n\n{table_md}\n\n"

    # Admonitions / Callouts
    if tag_name in ["div", "aside"] and "admonition" in " ".join(node.get("class", [])):
        title_tag = node.find(class_=re.compile(r"admonition-title"))
        title = _clean_text(title_tag.get_text()) if title_tag else "Note"
        body_text = "".join(
            _render_node_to_markdown(c) for c in node.children if c != title_tag
        ).strip()
        return f"\n\n> **{title}:** {body_text}\n\n"

    # Blockquotes
    if tag_name == "blockquote":
        inner = "".join(_render_node_to_markdown(c) for c in node.children).strip()
        return f"\n\n> {inner}\n\n"

    # General container fallback: recurse into children
    return "".join(_render_node_to_markdown(c) for c in node.children)


def _extract_sections(
    article: Tag,
) -> tuple[list[str], list[DocumentSection]]:
    """Extract flat headings list and hierarchical section contents from article DOM."""
    headings: list[str] = []
    sections: list[DocumentSection] = []

    current_title = "Introduction"
    current_level = 1
    current_lines: list[str] = []

    # Find heading elements and associated content
    for elem in article.descendants:
        if isinstance(elem, Tag) and elem.name in ["h1", "h2", "h3", "h4"]:
            heading_title = _clean_text(elem.get_text())
            if not heading_title:
                continue

            headings.append(heading_title)

            # Flush previous section
            if current_lines:
                sec_content = _clean_text("\n".join(current_lines))
                if sec_content:
                    sections.append(
                        DocumentSection(
                            title=current_title,
                            level=current_level,
                            content=sec_content,
                        )
                    )
                current_lines = []

            current_title = heading_title
            current_level = int(elem.name[1])

        elif isinstance(elem, Tag) and elem.name in ["p", "pre", "table", "ul", "ol", "dt", "dd"]:
            # Only process if direct or top-level element to avoid duplicate text from nested tags
            parent_block = elem.find_parent(["p", "pre", "table", "ul", "ol", "dt", "dd"])
            if parent_block is None:
                rendered = _render_node_to_markdown(elem).strip()
                if rendered:
                    current_lines.append(rendered)

    # Flush final section
    if current_lines:
        sec_content = "\n\n".join(current_lines).strip()
        if sec_content:
            sections.append(
                DocumentSection(
                    title=current_title,
                    level=current_level,
                    content=sec_content,
                )
            )

    return headings, sections


def clean_html_document(
    html_content: str,
    source: SourceDefinition,
) -> ProcessedDocument:
    """Clean raw scikit-learn HTML document and return structured representation.

    Args:
        html_content: Raw HTML text of the page.
        source: Source metadata definition.

    Returns:
        ProcessedDocument: Cleaned, structured document.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract version
    version_meta = soup.find("meta", attrs={"name": re.compile(r"docsearch:version|version", re.I)})
    version = version_meta["content"] if version_meta and version_meta.get("content") else "stable"

    # Identify main article container
    article = soup.find("article", class_="bd-article") or soup.find("main") or soup.find("body") or soup

    # Decompose noise tags and classes
    for noise_tag in article.find_all(list(NOISE_TAGS)):
        noise_tag.decompose()

    for noise_elem in article.find_all(class_=NOISE_CLASSES):
        noise_elem.decompose()

    # Extract Document Title from primary H1
    h1 = article.find("h1")
    title = _clean_text(h1.get_text()) if h1 else source.doc_id.replace("_", " ").title()

    # Extract flat headings and structured sections
    headings, sections = _extract_sections(article)

    # Render full article to clean Markdown
    markdown_content = _render_node_to_markdown(article)
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

    words = markdown_content.split()

    return ProcessedDocument(
        doc_id=source.doc_id,
        source_url=source.url,
        title=title,
        category=source.category,
        version=version,
        headings=headings,
        sections=sections,
        content=markdown_content,
        char_count=len(markdown_content),
        word_count=len(words),
    )
