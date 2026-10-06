"""Google Gemini LLM provider using httpx."""

import logging
import os
from typing import Any
import httpx

from adaq_rag.core.config import get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError
from adaq_rag.llm.models import LLMResponse, LLMUsage

logger = logging.getLogger("adaq_rag.llm.gemini")


class GeminiProvider(BaseLLMProvider):
    """LLM provider for Google Gemini REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or (settings.llm_model if "gemini" in settings.llm_model.lower() else "gemini-1.5-flash")
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.timeout = timeout

        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or environment, "
                "or specify LLM_PROVIDER=mock for offline development/testing."
            )

    def _prepare_payload(self, prompt: str, system_prompt: str | None) -> dict[str, Any]:
        contents = [{"parts": [{"text": prompt}]}]
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return payload

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        try:
            candidates = data.get("candidates", [])
            text = candidates[0]["content"]["parts"][0]["text"]
            usage_meta = data.get("usageMetadata", {})
            usage = LLMUsage(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
            )
            return LLMResponse(content=text, model=self.model, usage=usage)
        except (KeyError, IndexError) as exc:
            raise LLMGenerationError(f"Unexpected response format from Gemini API: {data}") from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = self._prepare_payload(prompt, system_prompt)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGenerationError(
                f"Gemini API returned error status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"Gemini API call failed: {exc}") from exc

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = self._prepare_payload(prompt, system_prompt)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return self._parse_response(response.json())
        except httpx.HTTPStatusError as exc:
            raise LLMGenerationError(
                f"Gemini API returned error status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"Gemini API async call failed: {exc}") from exc
