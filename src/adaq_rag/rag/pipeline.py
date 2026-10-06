"""Standard non-adaptive Baseline RAG pipeline using Dense FAISS retrieval."""

import logging
import time
from typing import Any

from adaq_rag.core.config import Settings, get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.factory import get_llm_provider
from adaq_rag.rag.context_builder import ContextBuilder
from adaq_rag.rag.models import RAGResponse
from adaq_rag.rag.prompts import DEFAULT_RAG_SYSTEM_PROMPT
from adaq_rag.retrieval.retriever import DenseRetriever

logger = logging.getLogger("adaq_rag.rag.pipeline")


class BasicRAGPipeline:
    """Baseline RAG Pipeline: Dense FAISS retrieval -> Context construction -> LLM generation."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        llm_provider: BaseLLMProvider,
        context_builder: ContextBuilder | None = None,
        default_top_k: int | None = None,
        system_prompt: str = DEFAULT_RAG_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the baseline RAG pipeline.

        Args:
            dense_retriever: FAISS dense vector retriever.
            llm_provider: Configured LLM generation provider.
            context_builder: Helper for formatting retrieved chunks into prompts.
            default_top_k: Default number of chunks to retrieve (defaults to settings.rag_top_k).
            system_prompt: Base system prompt for grounded generation.
        """
        settings = get_settings()
        self.dense_retriever = dense_retriever
        self.llm_provider = llm_provider
        self.context_builder = context_builder or ContextBuilder()
        self.default_top_k = default_top_k if default_top_k is not None else settings.rag_top_k
        self.system_prompt = system_prompt

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        dense_retriever: DenseRetriever | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> "BasicRAGPipeline":
        """Factory method to assemble BasicRAGPipeline using application configuration.

        Args:
            settings: Optional Settings instance.
            dense_retriever: Optional pre-loaded DenseRetriever.
            llm_provider: Optional pre-initialized BaseLLMProvider.

        Returns:
            BasicRAGPipeline: Ready-to-use baseline pipeline.
        """
        cfg = settings or get_settings()
        retriever = dense_retriever or DenseRetriever.load()
        provider = llm_provider or get_llm_provider(cfg)
        return cls(
            dense_retriever=retriever,
            llm_provider=provider,
            default_top_k=cfg.rag_top_k,
        )

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Execute baseline RAG pipeline synchronously.

        Args:
            question: User inquiry.
            top_k: Number of dense chunks to retrieve.

        Returns:
            RAGResponse: Grounded answer with source references and evaluation metrics.
        """
        start_time = time.perf_counter()
        k = top_k if top_k is not None else self.default_top_k
        clean_q = question.strip()

        logger.info("Executing Basic RAG for query: '%s' (top_k=%d)", clean_q, k)

        # 1. Dense FAISS retrieval (Strictly dense only for Phase 5 baseline)
        retrieved_chunks = self.dense_retriever.retrieve(clean_q, top_k=k)

        # 2. Context Construction
        user_prompt, sources = self.context_builder.format_prompt(clean_q, retrieved_chunks)

        # 3. LLM Generation
        llm_resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Evaluation Hooks & Response Assembly
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieval_scores = [c.score for c in retrieved_chunks]

        return RAGResponse(
            query=clean_q,
            answer=llm_resp.content.strip(),
            retrieval_method="dense",
            top_k=k,
            sources=sources,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieval_scores=retrieval_scores,
            latency_ms=round(latency_ms, 2),
            metadata={
                "model": llm_resp.model,
                "usage": llm_resp.usage.model_dump() if llm_resp.usage else None,
                "pipeline_stage": "phase_5_baseline",
            },
        )

    async def query_async(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Execute baseline RAG pipeline asynchronously.

        Args:
            question: User inquiry.
            top_k: Number of dense chunks to retrieve.

        Returns:
            RAGResponse: Grounded answer with source references and evaluation metrics.
        """
        start_time = time.perf_counter()
        k = top_k if top_k is not None else self.default_top_k
        clean_q = question.strip()

        logger.info("Executing async Basic RAG for query: '%s' (top_k=%d)", clean_q, k)

        # 1. Dense FAISS retrieval
        retrieved_chunks = self.dense_retriever.retrieve(clean_q, top_k=k)

        # 2. Context Construction
        user_prompt, sources = self.context_builder.format_prompt(clean_q, retrieved_chunks)

        # 3. LLM Generation
        llm_resp = await self.llm_provider.generate_async(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieval_scores = [c.score for c in retrieved_chunks]

        return RAGResponse(
            query=clean_q,
            answer=llm_resp.content.strip(),
            retrieval_method="dense",
            top_k=k,
            sources=sources,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieval_scores=retrieval_scores,
            latency_ms=round(latency_ms, 2),
            metadata={
                "model": llm_resp.model,
                "usage": llm_resp.usage.model_dump() if llm_resp.usage else None,
                "pipeline_stage": "phase_5_baseline",
            },
        )
