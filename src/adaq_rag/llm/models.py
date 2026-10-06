"""Data models for LLM responses and usage statistics."""

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Token usage statistics for an LLM generation call."""

    prompt_tokens: int = Field(default=0, description="Tokens in input prompt")
    completion_tokens: int = Field(default=0, description="Tokens in output completion")
    total_tokens: int = Field(default=0, description="Total tokens consumed")


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model identifier that produced the response")
    usage: LLMUsage | None = Field(default=None, description="Optional token usage metrics")
