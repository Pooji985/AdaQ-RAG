"""Unit and integration tests for documentation ingestion and cleaning."""

from pathlib import Path
from unittest.mock import MagicMock

from adaq_rag.ingestion.cleaner import clean_html_document
from adaq_rag.ingestion.models import SourceDefinition
from adaq_rag.ingestion.pipeline import IngestionPipeline
from adaq_rag.ingestion.sources import get_curated_sources

SAMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Sample - scikit-learn 1.4.0 documentation</title>
    <meta name="docsearch:version" content="1.4.0">
</head>
<body>
    <nav class="bd-header">Navigation Bar to remove</nav>
    <aside class="bd-sidebar">Sidebar menu to remove</aside>
    <article class="bd-article">
        <a class="headerlink" href="#title">#</a>
        <h1>1.1. Supervised Linear Models<a class="headerlink" href="#">¶</a></h1>
        <p>Linear models are a set of methods for regression and classification.</p>
        
        <h2>1.1.1. Ordinary Least Squares<a class="headerlink" href="#">#</a></h2>
        <p>Ordinary least squares fits a linear model with coefficients <code>w</code>.</p>
        
        <div class="highlight-default notranslate">
            <div class="highlight">
                <pre><span></span><span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">linear_model</span>
<span class="n">reg</span> <span class="o">=</span> <span class="n">linear_model</span><span class="o">.</span><span class="n">LinearRegression</span><span class="p">()</span>
<span class="n">reg</span><span class="o">.</span><span class="n">fit</span><span class="p">([[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">]],</span> <span class="p">[</span><span class="mi">0</span><span class="p">])</span></pre>
            </div>
        </div>
        
        <button class="copybutton">Copy</button>
        
        <h3>1.1.1.1. Complexity</h3>
        <p>Complexity details here.</p>
        <ul>
            <li>Feature scaling helps.</li>
            <li>Matrix inversion takes O(p^3) time.</li>
        </ul>
        
        <div class="prev-next-area">Prev / Next links</div>
        <div class="sphx-glr-footer">Download script</div>
    </article>
    <footer>Footer boilerplate to remove</footer>
    <script>var x = 1;</script>
</body>
</html>
"""


def test_clean_html_removes_noise_and_extracts_metadata() -> None:
    """Verify boilerplate, navigation, scripts, and permalinks are stripped."""
    source = SourceDefinition(
        doc_id="test_linear_models",
        url="https://scikit-learn.org/stable/modules/linear_model.html",
        category="supervised_learning",
        description="Linear models test",
    )

    doc = clean_html_document(SAMPLE_HTML, source)

    # Metadata assertions
    assert doc.doc_id == "test_linear_models"
    assert doc.title == "1.1. Supervised Linear Models"
    assert doc.version == "1.4.0"
    assert doc.category == "supervised_learning"
    assert doc.source_url == source.url

    # Noise removal assertions
    assert "Navigation Bar" not in doc.content
    assert "Sidebar menu" not in doc.content
    assert "Footer boilerplate" not in doc.content
    assert "Prev / Next" not in doc.content
    assert "Download script" not in doc.content
    assert "var x = 1;" not in doc.content
    assert "¶" not in doc.content

    # Content structure assertions
    assert "# 1.1. Supervised Linear Models" in doc.content
    assert "## 1.1.1. Ordinary Least Squares" in doc.content
    assert "### 1.1.1.1. Complexity" in doc.content
    assert "```python" in doc.content
    assert "from sklearn import linear_model" in doc.content
    assert "`w`" in doc.content
    assert "- Feature scaling helps." in doc.content

    # Headings and sections assertions
    assert len(doc.headings) == 3
    assert doc.headings[0] == "1.1. Supervised Linear Models"
    assert doc.headings[1] == "1.1.1. Ordinary Least Squares"
    assert doc.headings[2] == "1.1.1.1. Complexity"

    assert len(doc.sections) >= 3
    assert doc.char_count > 0
    assert doc.word_count > 0


def test_table_conversion() -> None:
    """Verify HTML table is formatted as Markdown table."""
    html = """
    <article class="bd-article">
        <h1>Scoring Table</h1>
        <table>
            <tr><th>Metric</th><th>Function</th></tr>
            <tr><td>Accuracy</td><td>accuracy_score</td></tr>
            <tr><td>F1 Score</td><td>f1_score</td></tr>
        </table>
    </article>
    """
    source = SourceDefinition(
        doc_id="test_table",
        url="https://scikit-learn.org/stable/table.html",
        category="model_selection",
        description="Table test",
    )
    doc = clean_html_document(html, source)
    assert "| Metric | Function |" in doc.content
    assert "| Accuracy | accuracy_score |" in doc.content


def test_curated_sources_validity() -> None:
    """Verify curated sources registry is non-empty and has unique, valid definitions."""
    sources = get_curated_sources()
    assert len(sources) >= 15

    doc_ids = [s.doc_id for s in sources]
    assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_id found in curated sources!"

    categories = {s.category for s in sources}
    assert "supervised_learning" in categories
    assert "model_selection_evaluation" in categories
    assert "pipelines_and_composites" in categories
    assert "preprocessing_and_transforms" in categories
    assert "common_pitfalls" in categories
    assert "api_reference" in categories

    for s in sources:
        assert s.url.startswith("https://scikit-learn.org/stable/")
        assert len(s.description) > 10


def test_pipeline_local_cache_idempotence(tmp_path: Path) -> None:
    """Verify pipeline reads from data/raw cache without making network calls."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)

    source = SourceDefinition(
        doc_id="cached_doc",
        url="https://scikit-learn.org/stable/cached.html",
        category="test",
        description="Cached doc test",
    )

    # Pre-populate raw file
    cached_file = raw_dir / "cached_doc.html"
    cached_file.write_text(SAMPLE_HTML, encoding="utf-8")

    mock_client = MagicMock()
    pipeline = IngestionPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        sources=[source],
    )

    # Run fetch - should read cache and NEVER call client.get
    content = pipeline.fetch_source_html(source=source, client=mock_client, force_download=False)
    assert content == SAMPLE_HTML
    mock_client.get.assert_not_called()

    # Run full pipeline
    report = pipeline.run(force_download=False)
    assert report.total_sources == 1
    assert report.collected_count == 1
    assert report.processed_count == 1
    assert report.failed_count == 0

    assert (processed_dir / "cached_doc.json").exists()
    assert (processed_dir / "cached_doc.md").exists()
    assert (processed_dir / "corpus_manifest.json").exists()
