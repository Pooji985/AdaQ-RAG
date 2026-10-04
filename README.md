# AdaQ-RAG: Intent-Aware Adaptive Retrieval for Efficient Augmented Generation

AdaQ-RAG is a research-oriented, production-grade Retrieval-Augmented Generation (RAG) framework designed to dynamically modulate retrieval depth and routing strategies based on real-time query complexity and intent analysis. By avoiding one-size-fits-all retrieval pipelines, AdaQ-RAG optimizes retrieval quality, latency, and computational cost over technical domain corpora (such as scikit-learn documentation).

---

## Project Structure

```text
AdaQ-RAG/
├── data/                  # Corpus storage (raw and processed documentation)
│   ├── raw/
│   └── processed/
├── evaluation/            # Evaluation benchmarks, datasets, and harness runners
├── frontend/              # Frontend client placeholder
├── indexes/               # Persistent lexical (BM25) and dense vector indexes
├── scripts/               # Utility scripts for ingestion, indexing, and offline evaluation
├── src/                   # Core application source code
│   └── adaq_rag/
│       ├── api/           # FastAPI application layer, routes, and schemas
│       │   └── routes/
│       │       └── health.py
│       └── core/          # Configuration (Pydantic Settings) and centralized logging
├── tests/                 # Unit and integration test suites
├── .env.example           # Example environment variable configuration
├── .gitignore             # Standard Python and artifact ignore rules
├── pyproject.toml         # Project metadata and test configuration
├── requirements.txt       # Core runtime dependencies
└── requirements-dev.txt   # Development and testing dependencies
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Virtual environment tool (`venv`)

### 2. Environment Setup

Create and activate a virtual environment:

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```powershell
# Core dependencies
pip install -r requirements.txt

# Development and testing dependencies (includes core)
pip install -r requirements-dev.txt

# Install project in editable mode
pip install -e .
```

### 3. Configuration

Copy the example environment configuration:

```powershell
# Windows (PowerShell)
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Modify `.env` as needed. Default settings configure the service on `0.0.0.0:8000` with `INFO` logging.

---

## Running the API

Start the FastAPI development server with Uvicorn:

```powershell
uvicorn adaq_rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify Endpoints

- **Health Check**:
  ```bash
  curl http://127.0.0.1:8000/health
  ```
  Expected Response:
  ```json
  {
    "status": "ok",
    "app_name": "AdaQ-RAG API",
    "version": "0.1.0",
    "environment": "development"
  }
  ```

- **Interactive API Documentation (Swagger UI)**:
  Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

- **ReDoc API Documentation**:
  Navigate to [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

---

## Running Tests

Execute the test suite with `pytest`:

```powershell
pytest -v
```
