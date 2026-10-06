"""Mock LLM provider for unit tests and local simulation."""

from typing import Any, Callable

from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.models import LLMResponse, LLMUsage


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for offline testing and evaluation without API calls."""

    def __init__(
        self,
        default_response: str | None = None,
        response_generator: Callable[[str, str | None], str] | None = None,
        model: str = "mock-llm-v1",
    ) -> None:
        """Initialize mock provider.

        Args:
            default_response: Optional fixed string to return for all calls.
            response_generator: Optional callable receiving (prompt, system_prompt) -> str.
            model: Model name to report in responses.
        """
        self.default_response = default_response
        self.response_generator = response_generator
        self.model = model
        self.call_history: list[dict[str, Any]] = []

    def _create_response(self, prompt: str, system_prompt: str | None) -> LLMResponse:
        self.call_history.append({"prompt": prompt, "system_prompt": system_prompt})

        if self.default_response is not None:
            text = self.default_response
        elif self.response_generator is not None:
            text = self.response_generator(prompt, system_prompt)
        else:
            text = (
                "Based on the provided documentation context, Ridge regression uses an L2 penalty "
                "with the alpha parameter to prevent overfitting. [Source 1]"
            )

        usage = LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
        )
        return LLMResponse(content=text, model=self.model, usage=usage)

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        return self._create_response(prompt, system_prompt)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        return self._create_response(prompt, system_prompt)
