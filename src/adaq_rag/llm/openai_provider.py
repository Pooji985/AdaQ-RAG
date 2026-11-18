"""OpenAI-compatible LLM provider using httpx."""

import asyncio
import logging
import os
import re
import ssl
import time
from typing import Any
import httpx

from adaq_rag.core.config import get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.exceptions import LLMConfigurationError, LLMGenerationError
from adaq_rag.llm.models import LLMResponse, LLMUsage

logger = logging.getLogger("adaq_rag.llm.openai")

TRANSIENT_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    httpx.TimeoutException,
    ssl.SSLError,
)
TRANSIENT_HTTP_STATUS_CODES = (500, 502, 503, 504)


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
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self.timeout = timeout
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")
        if retry_base_delay < 0:
            raise ValueError(f"retry_base_delay must be non-negative, got {retry_base_delay}")
        if retry_max_delay < 0:
            raise ValueError(f"retry_max_delay must be non-negative, got {retry_max_delay}")

        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

        if not self.api_key:
            raise LLMConfigurationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY in .env or environment, "
                "or specify LLM_PROVIDER=mock for offline development/testing."
            )

    @staticmethod
    def _is_tpd_exhaustion(response_text: str) -> bool:
        """Check if HTTP 429 response body indicates daily tokens-per-day (TPD) quota exhaustion."""
        lower = response_text.lower()
        return "tokens per day" in lower or "(tpd)" in lower

    def _extract_retry_delay(self, response: httpx.Response, retry_count: int) -> float:
        """Extract or calculate retry delay for HTTP 429 response.

        Priority order:
        1. 'Retry-After' response header (seconds).
        2. Groq/provider guidance parsed from response body (e.g., 'try again in 12.105s').
        3. Capped exponential backoff: retry_base_delay * (2 ** retry_count).

        Args:
            response: httpx.Response object with status code 429.
            retry_count: Zero-indexed retry attempt number (0 for first retry).

        Returns:
            Delay in seconds (capped at retry_max_delay).
        """
        # 1. Check Retry-After header
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                delay = float(retry_after)
                if delay > 0:
                    return min(delay, self.retry_max_delay)
            except (ValueError, TypeError):
                pass

        # 2. Parse provider retry guidance from error body (e.g. Groq "Please try again in 12.105s.")
        try:
            match = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", response.text, re.IGNORECASE)
            if match:
                delay = float(match.group(1)) + 0.5  # add 0.5s safety buffer
                if delay > 0:
                    return min(delay, self.retry_max_delay)
        except (ValueError, TypeError):
            pass

        # 3. Capped exponential backoff
        exp_delay = self.retry_base_delay * (2 ** retry_count)
        return min(exp_delay, self.retry_max_delay)

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
            choices = data.get("choices", [])
            if not choices:
                raise LLMGenerationError(f"Unexpected response format from OpenAI API (no choices): {data}")
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None or not str(content).strip():
                raise LLMGenerationError("OpenAI API returned an empty or whitespace-only response.")
            usage_data = data.get("usage", {})
            usage = LLMUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            return LLMResponse(content=str(content), model=self.model, usage=usage)
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

        total_attempts = 1 + self.max_retries
        for attempt in range(total_attempts):
            is_last_attempt = attempt >= self.max_retries
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return self._parse_response(response.json())
            except httpx.HTTPStatusError as exc:
                is_rate_limit = exc.response.status_code == 429
                is_transient_server = exc.response.status_code in TRANSIENT_HTTP_STATUS_CODES
                is_tpd = is_rate_limit and self._is_tpd_exhaustion(exc.response.text)

                if is_tpd:
                    logger.error(
                        "Tokens-per-day (TPD) daily quota exhaustion detected for model %s. Aborting retries immediately.",
                        self.model,
                    )
                    raise LLMGenerationError(
                        f"OpenAI API returned error status 429 (TPD quota exhausted): {exc.response.text}"
                    ) from exc

                if is_rate_limit and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = self._extract_retry_delay(exc.response, retry_count=attempt)
                    logger.warning(
                        "Rate limit (429) encountered on model %s. Retrying (%d/%d) in %.2fs...",
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                elif is_transient_server and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    logger.warning(
                        "Transient server error (%d) from %s. Retrying (%d/%d) in %.2fs...",
                        exc.response.status_code,
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                if is_rate_limit or is_transient_server:
                    logger.error(
                        "HTTP error %d retries exhausted (%d/%d) for model %s.",
                        exc.response.status_code,
                        self.max_retries,
                        self.max_retries,
                        self.model,
                    )
                else:
                    logger.error(
                        "Permanent HTTP error %d from %s: %s (will not retry)",
                        exc.response.status_code,
                        self.model,
                        exc.response.text,
                    )

                raise LLMGenerationError(
                    f"OpenAI API returned error status {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except TRANSIENT_NETWORK_ERRORS as exc:
                if not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
                        logger.warning(
                            "Ambiguous timeout (%s: %s) on model %s: request may have reached provider "
                            "(potential duplicate charges). Retrying (%d/%d) in %.2fs...",
                            type(exc).__name__,
                            exc,
                            self.model,
                            retry_num,
                            self.max_retries,
                            delay,
                        )
                    else:
                        logger.warning(
                            "Transient network error (%s: %s) on model %s. Retrying (%d/%d) in %.2fs...",
                            type(exc).__name__,
                            exc,
                            self.model,
                            retry_num,
                            self.max_retries,
                            delay,
                        )
                    time.sleep(delay)
                    continue

                logger.error(
                    "Transient error retries exhausted (%d/%d) for model %s. Final error: %s",
                    self.max_retries,
                    self.max_retries,
                    self.model,
                    exc,
                )
                raise LLMGenerationError(
                    f"OpenAI API call failed after {self.max_retries} retries: {exc}"
                ) from exc
            except LLMGenerationError as exc:
                if "empty or whitespace-only" in str(exc) and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    logger.warning(
                        "Empty response received from model %s. Retrying (%d/%d) in %.2fs...",
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue

                if "empty or whitespace-only" in str(exc):
                    logger.error(
                        "Empty response retries exhausted (%d/%d) for model %s.",
                        self.max_retries,
                        self.max_retries,
                        self.model,
                    )
                raise
            except Exception as exc:
                logger.error("Non-retryable error during OpenAI API call on %s: %s", self.model, exc)
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

        total_attempts = 1 + self.max_retries
        for attempt in range(total_attempts):
            is_last_attempt = attempt >= self.max_retries
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return self._parse_response(response.json())
            except httpx.HTTPStatusError as exc:
                is_rate_limit = exc.response.status_code == 429
                is_transient_server = exc.response.status_code in TRANSIENT_HTTP_STATUS_CODES
                is_tpd = is_rate_limit and self._is_tpd_exhaustion(exc.response.text)

                if is_tpd:
                    logger.error(
                        "Tokens-per-day (TPD) daily quota exhaustion detected for model %s. Aborting retries immediately.",
                        self.model,
                    )
                    raise LLMGenerationError(
                        f"OpenAI API returned error status 429 (TPD quota exhausted): {exc.response.text}"
                    ) from exc

                if is_rate_limit and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = self._extract_retry_delay(exc.response, retry_count=attempt)
                    logger.warning(
                        "Rate limit (429) encountered on model %s. Retrying (%d/%d) in %.2fs...",
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                elif is_transient_server and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    logger.warning(
                        "Transient server error (%d) from %s. Retrying (%d/%d) in %.2fs...",
                        exc.response.status_code,
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if is_rate_limit or is_transient_server:
                    logger.error(
                        "HTTP error %d retries exhausted (%d/%d) for model %s.",
                        exc.response.status_code,
                        self.max_retries,
                        self.max_retries,
                        self.model,
                    )
                else:
                    logger.error(
                        "Permanent HTTP error %d from %s: %s (will not retry)",
                        exc.response.status_code,
                        self.model,
                        exc.response.text,
                    )

                raise LLMGenerationError(
                    f"OpenAI API returned error status {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except TRANSIENT_NETWORK_ERRORS as exc:
                if not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
                        logger.warning(
                            "Ambiguous timeout (%s: %s) on model %s: request may have reached provider "
                            "(potential duplicate charges). Retrying (%d/%d) in %.2fs...",
                            type(exc).__name__,
                            exc,
                            self.model,
                            retry_num,
                            self.max_retries,
                            delay,
                        )
                    else:
                        logger.warning(
                            "Transient network error (%s: %s) on model %s. Retrying (%d/%d) in %.2fs...",
                            type(exc).__name__,
                            exc,
                            self.model,
                            retry_num,
                            self.max_retries,
                            delay,
                        )
                    await asyncio.sleep(delay)
                    continue

                logger.error(
                    "Transient error retries exhausted (%d/%d) for model %s. Final error: %s",
                    self.max_retries,
                    self.max_retries,
                    self.model,
                    exc,
                )
                raise LLMGenerationError(
                    f"OpenAI API async call failed after {self.max_retries} retries: {exc}"
                ) from exc
            except LLMGenerationError as exc:
                if "empty or whitespace-only" in str(exc) and not is_last_attempt:
                    retry_num = attempt + 1
                    delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
                    logger.warning(
                        "Empty response received from model %s. Retrying (%d/%d) in %.2fs...",
                        self.model,
                        retry_num,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if "empty or whitespace-only" in str(exc):
                    logger.error(
                        "Empty response retries exhausted (%d/%d) for model %s.",
                        self.max_retries,
                        self.max_retries,
                        self.model,
                    )
                raise
            except Exception as exc:
                logger.error("Non-retryable error during OpenAI API async call on %s: %s", self.model, exc)
                raise LLMGenerationError(f"OpenAI API async call failed: {exc}") from exc
