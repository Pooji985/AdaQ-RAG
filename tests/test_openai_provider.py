"""Focused unit tests for OpenAICompatibleProvider HTTP 429 retry and backoff behavior."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from adaq_rag.llm.exceptions import LLMGenerationError
from adaq_rag.llm.openai_provider import OpenAICompatibleProvider


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    """Fixture providing OpenAICompatibleProvider with mock API credentials."""
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        base_url="https://api.test.com/v1",
        max_retries=3,
        retry_base_delay=1.0,
        retry_max_delay=30.0,
    )


def test_extract_retry_delay_retry_after_header(provider: OpenAICompatibleProvider) -> None:
    """Verify Retry-After header takes highest priority."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={"retry-after": "15"},
        text='{"error": {"message": "Please try again in 5.0s."}}',
        request=request,
    )

    delay = provider._extract_retry_delay(response, retry_count=0)
    assert delay == 15.0


def test_extract_retry_delay_groq_body_guidance(provider: OpenAICompatibleProvider) -> None:
    """Verify Groq retry guidance in body is parsed when Retry-After header is absent."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={},
        text='{"error": {"message": "Rate limit reached on TPM: Limit 8000. Please try again in 12.105s."}}',
        request=request,
    )

    delay = provider._extract_retry_delay(response, retry_count=0)
    # 12.105 + 0.5s safety buffer = 12.605
    assert delay == pytest.approx(12.605, abs=0.01)


def test_extract_retry_delay_exponential_backoff_fallback(provider: OpenAICompatibleProvider) -> None:
    """Verify fallback to exponential backoff when neither header nor body guidance exists."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={},
        text='{"error": {"message": "Too many requests"}}',
        request=request,
    )

    # retry_count=0 -> 1.0 * (2 ** 0) = 1.0
    assert provider._extract_retry_delay(response, retry_count=0) == 1.0
    # retry_count=1 -> 1.0 * (2 ** 1) = 2.0
    assert provider._extract_retry_delay(response, retry_count=1) == 2.0
    # retry_count=2 -> 1.0 * (2 ** 2) = 4.0
    assert provider._extract_retry_delay(response, retry_count=2) == 4.0


def test_extract_retry_delay_capped_at_max(provider: OpenAICompatibleProvider) -> None:
    """Verify retry delay is capped at retry_max_delay."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={"retry-after": "120"},
        request=request,
    )

    delay = provider._extract_retry_delay(response, retry_count=0)
    assert delay == 30.0  # capped at provider.retry_max_delay


def test_generate_retry_success_after_429(provider: OpenAICompatibleProvider) -> None:
    """Verify synchronous generate() retries on 429 and succeeds on subsequent attempt."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_429 = httpx.Response(
        status_code=429,
        headers={"retry-after": "0.01"},
        text='{"error": {"message": "Rate limited"}}',
        request=request,
    )
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Success answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [resp_429, resp_200]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        res = provider.generate(prompt="Test prompt")

    assert res.content == "Success answer"
    assert res.usage.total_tokens == 15
    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once()


def test_generate_async_retry_success_after_429(provider: OpenAICompatibleProvider) -> None:
    """Verify asynchronous generate_async() retries on 429 and succeeds on subsequent attempt."""
    import asyncio

    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_429 = httpx.Response(
        status_code=429,
        headers={"retry-after": "0.01"},
        text='{"error": {"message": "Rate limited"}}',
        request=request,
    )
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Async success answer"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        },
        request=request,
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = [resp_429, resp_200]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client), patch("asyncio.sleep") as mock_sleep:
        res = asyncio.run(provider.generate_async(prompt="Async test prompt"))

    assert res.content == "Async success answer"
    assert res.usage.total_tokens == 18
    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once()


def test_generate_exhausted_retries_raises_original_429(provider: OpenAICompatibleProvider) -> None:
    """Verify that when 3 retries are exhausted, LLMGenerationError preserves 429 status and response body."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_429 = httpx.Response(
        status_code=429,
        headers={"retry-after": "0.01"},
        text='{"error": {"message": "Persistent rate limit"}}',
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.return_value = resp_429
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep"):
        with pytest.raises(LLMGenerationError) as exc_info:
            provider.generate(prompt="Test prompt")

    # 1 initial + 3 retries = 4 attempts
    assert mock_client.post.call_count == 4
    # Check that error details and chaining are preserved
    assert "429" in str(exc_info.value)
    assert "Persistent rate limit" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


def test_generate_non_429_http_error_does_not_retry(provider: OpenAICompatibleProvider) -> None:
    """Verify non-429 HTTP errors (e.g. 400 Bad Request, 500) fail immediately without retry."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_400 = httpx.Response(
        status_code=400,
        text='{"error": {"message": "Invalid request parameter"}}',
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.return_value = resp_400
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMGenerationError) as exc_info:
            provider.generate(prompt="Test prompt")

    # Fails immediately on first attempt
    assert mock_client.post.call_count == 1
    mock_sleep.assert_not_called()
    assert "400" in str(exc_info.value)
    assert "Invalid request parameter" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


def test_generate_transient_connect_error_retries_and_succeeds(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify transient ConnectError triggers retry and succeeds on subsequent attempt."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Success after connect retry"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [httpx.ConnectError("Connection refused"), resp_200]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        resp = provider.generate(prompt="Test prompt")

    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once_with(1.0)
    assert resp.content == "Success after connect retry"


def test_generate_transient_read_timeout_logs_and_retries(
    provider: OpenAICompatibleProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify ReadTimeout triggers retry and logs ambiguous timeout warning."""
    import logging

    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Success after read timeout"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [httpx.ReadTimeout("Read timed out"), resp_200]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with caplog.at_level(logging.WARNING):
        with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
            resp = provider.generate(prompt="Test prompt")

    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once_with(1.0)
    assert resp.content == "Success after read timeout"
    assert any("Ambiguous timeout" in record.message for record in caplog.records)


def test_generate_transient_retry_exhaustion_raises_llm_error(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify exhausting retries on transient network error raises LLMGenerationError."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectError("getaddrinfo failed")
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMGenerationError, match="failed after 3 retries"):
            provider.generate(prompt="Test prompt")

    # 1 initial + 3 retries = 4 attempts
    assert mock_client.post.call_count == 4
    assert mock_sleep.call_count == 3


def test_generate_async_transient_retry_success(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify asynchronous generate_async retries on transient network error."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_200 = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Async success after network retry"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=request,
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = [httpx.ConnectTimeout("Connect timeout"), resp_200]
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client), patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        import asyncio
        resp = asyncio.run(provider.generate_async(prompt="Async test"))

    assert mock_client.post.call_count == 2
    mock_sleep.assert_awaited_once_with(1.0)
    assert resp.content == "Async success after network retry"


def test_generate_permanent_auth_error_does_not_retry(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify permanent 401 Unauthorized HTTP errors are not retried."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_401 = httpx.Response(
        status_code=401,
        text='{"error": {"message": "Invalid API Key"}}',
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.return_value = resp_401
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMGenerationError, match="401"):
            provider.generate(prompt="Test prompt")

    assert mock_client.post.call_count == 1
    mock_sleep.assert_not_called()


def test_generate_empty_string_retries_and_raises_if_exhausted(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify empty response content triggers retry and raises LLMGenerationError when exhausted."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_empty = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        },
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.return_value = resp_empty
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMGenerationError, match="empty or whitespace-only"):
            provider.generate(prompt="Test prompt")

    # 1 initial + 3 retries = 4 attempts
    assert mock_client.post.call_count == 4
    assert mock_sleep.call_count == 3


def test_generate_whitespace_response_retries_and_succeeds(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify whitespace-only response retries and succeeds when valid answer is returned."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_whitespace = httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": "   \n\t  "}}]},
        request=request,
    )
    resp_valid = httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": "Valid answer text"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.side_effect = [resp_whitespace, resp_valid]
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        resp = provider.generate(prompt="Test prompt")

    assert mock_client.post.call_count == 2
    mock_sleep.assert_called_once_with(1.0)
    assert resp.content == "Valid answer text"


def test_init_rejects_negative_max_retries() -> None:
    """Verify OpenAICompatibleProvider rejects negative max_retries."""
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        OpenAICompatibleProvider(api_key="test-key", max_retries=-1)


def test_init_rejects_negative_retry_base_delay() -> None:
    """Verify OpenAICompatibleProvider rejects negative retry_base_delay."""
    with pytest.raises(ValueError, match="retry_base_delay must be non-negative"):
        OpenAICompatibleProvider(api_key="test-key", retry_base_delay=-0.5)


def test_init_rejects_negative_retry_max_delay() -> None:
    """Verify OpenAICompatibleProvider rejects negative retry_max_delay."""
    with pytest.raises(ValueError, match="retry_max_delay must be non-negative"):
        OpenAICompatibleProvider(api_key="test-key", retry_max_delay=-10.0)


def test_extract_retry_delay_malformed_retry_after_header(provider: OpenAICompatibleProvider) -> None:
    """Verify non-numeric Retry-After header falls back to body or exponential backoff."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={"retry-after": "not-a-number"},
        text='{"error": {"message": "Rate limited without guidance"}}',
        request=request,
    )
    # Should fall back to exponential backoff (retry_count=0 -> 1.0)
    delay = provider._extract_retry_delay(response, retry_count=0)
    assert delay == 1.0


def test_extract_retry_delay_malformed_body_guidance_falls_back(provider: OpenAICompatibleProvider) -> None:
    """Verify malformed non-numeric text matching guidance prefix falls back to exponential backoff."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    response = httpx.Response(
        status_code=429,
        headers={},
        text='{"error": {"message": "Please try again in ...s"}}',
        request=request,
    )
    # Strict regex \d+(?:\.\d+)? won't match "...", so it falls back to backoff
    delay = provider._extract_retry_delay(response, retry_count=1)
    assert delay == 2.0  # 1.0 * (2 ** 1)


def test_extract_retry_delay_integer_seconds_and_spaces(provider: OpenAICompatibleProvider) -> None:
    """Verify integer seconds with or without space before 's' are parsed cleanly."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    # Integer format: 10s -> 10 + 0.5 = 10.5
    resp1 = httpx.Response(
        status_code=429,
        headers={},
        text='{"error": {"message": "Rate limit exceeded. Please try again in 10s."}}',
        request=request,
    )
    assert provider._extract_retry_delay(resp1, retry_count=0) == 10.5

    # Spaced format: 15 s -> 15 + 0.5 = 15.5
    resp2 = httpx.Response(
        status_code=429,
        headers={},
        text='{"error": {"message": "Rate limit exceeded. Please try again in 15 s."}}',
        request=request,
    )
    assert provider._extract_retry_delay(resp2, retry_count=0) == 15.5


def test_is_tpd_exhaustion_detection() -> None:
    """Verify detection of tokens-per-day (TPD) quota exhaustion messages."""
    assert OpenAICompatibleProvider._is_tpd_exhaustion(
        'Rate limit reached on tokens per day (TPD): Limit 200000, Used 198770.'
    )
    assert OpenAICompatibleProvider._is_tpd_exhaustion(
        '{"error":{"message":"Rate limit on tokens per day: limit exceeded"}}'
    )
    assert OpenAICompatibleProvider._is_tpd_exhaustion(
        'Rate limit exceeded (TPD)'
    )
    # Temporary rate limits should NOT be flagged as TPD
    assert not OpenAICompatibleProvider._is_tpd_exhaustion(
        'Rate limit reached on requests per minute (RPM). Please try again in 5s.'
    )
    assert not OpenAICompatibleProvider._is_tpd_exhaustion(
        'Rate limit reached on tokens per minute (TPM). Please try again in 10s.'
    )
    assert not OpenAICompatibleProvider._is_tpd_exhaustion('Too Many Requests')


def test_generate_tpd_exhaustion_aborts_immediately_without_retry(
    provider: OpenAICompatibleProvider,
) -> None:
    """Verify that 429 TPD exhaustion raises LLMGenerationError immediately on first attempt without retrying."""
    request = httpx.Request("POST", "https://api.test.com/v1/chat/completions")
    resp_tpd = httpx.Response(
        status_code=429,
        text='{"error":{"message":"Rate limit reached for model openai/gpt-oss-20b on tokens per day (TPD): Limit 200000, Used 199112."}}',
        request=request,
    )

    mock_client = MagicMock()
    mock_client.post.return_value = resp_tpd
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("httpx.Client", return_value=mock_client), patch("time.sleep") as mock_sleep:
        with pytest.raises(LLMGenerationError) as exc_info:
            provider.generate(prompt="Test prompt")

    # MUST fail on first attempt without any sleep or retry
    assert mock_client.post.call_count == 1
    mock_sleep.assert_not_called()
    assert "TPD quota exhausted" in str(exc_info.value)


