"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod

from adaq_rag.llm.models import LLMResponse


class BaseLLMProvider(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Synchronously generate text response from the LLM.

        Args:
            prompt: User / task prompt.
            system_prompt: Optional system role instructions.
            **kwargs: Provider-specific inference parameters.

        Returns:
            LLMResponse: Structured response containing generated text.
        """

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Asynchronously generate text response from the LLM.

        Args:
            prompt: User / task prompt.
            system_prompt: Optional system role instructions.
            **kwargs: Provider-specific inference parameters.

        Returns:
            LLMResponse: Structured response containing generated text.
        """
