"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    app_name: str = "AdaQ-RAG API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    host: str = "0.0.0.0"
    port: int = 8000

    log_level: str = "INFO"

    # Retrieval and Indexing Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    indexes_dir: str = "indexes"
    vector_index_file: str = "vector_index.faiss"
    vector_metadata_file: str = "vector_metadata.json"
    bm25_index_file: str = "bm25_index.pkl"
    bm25_metadata_file: str = "bm25_metadata.json"

    # Hybrid Retrieval and Reranking Configuration
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32
    hybrid_rrf_k: int = 60
    hybrid_candidate_top_k: int = 20
    hybrid_final_top_k: int = 5

    # Complex Retrieval and Orchestration Configuration
    complex_max_sub_queries: int = 4
    complex_candidate_top_k: int = 15
    complex_final_top_k: int = 8
    complex_max_retrieval_attempts: int = 2
    sufficiency_min_top_score: float = 0.0
    sufficiency_min_coverage: float = 0.50
    sufficiency_min_chunks: int = 2

    # LLM and Baseline RAG Configuration
    llm_provider: str = "openai"  # "openai", "gemini", "mock"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    rag_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
