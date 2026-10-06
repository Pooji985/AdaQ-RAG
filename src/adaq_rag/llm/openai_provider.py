"""OpenAI-compatible LLM provider using httpx."""

import logging
import os
from typing import Any
import httpx

from adaq_rag.core.config import get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError
from adaq_rag.llm.models import LLMResponse, LLMUsage

logger = logging.getLogger("adaq_rag.llm.openai")


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider for OpenAI and OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.timeout = timeout

        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY in .env or environment, "
                "or specify LLM_PROVIDER=mock for offline development/testing."
            )

    def _prepare_payload(self, prompt: str, system_prompt: str | None) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            usage_data = data.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            return LLMResponse(content=content, model=self.model, usage=usage)
        except (KeyError, IndexError) as exc:
            raise LLMGenerationError(f"Unexpected response format from OpenAI API: {data}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._prepare_payload(prompt, system_prompt)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGenerationError(
                f"OpenAI API returned error status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI API call failed: {exc}") from exc

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._prepare_payload(prompt, system_prompt)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGenerationError(
                f"OpenAI API returned error status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI API async call failed: {exc}") from exc
