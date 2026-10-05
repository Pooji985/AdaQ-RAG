"""Token estimation and text splitting utilities."""

import re

TOKEN_REGEX = re.compile(r"\w+|[^\w\s]")
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])")


def count_tokens(text: str) -> int:
    """Estimate token count for a text string using word/punctuation tokenization.

    Args:
        text: Input string.

    Returns:
        int: Estimated token count.
    """
    if not text:
        return 0
    return len(TOKEN_REGEX.findall(text))


def split_into_paragraphs(text: str) -> list[str]:
    """Split markdown text into distinct paragraphs or block elements.

    Args:
        text: Input markdown text.

    Returns:
        list[str]: Non-empty paragraphs/blocks.
    """
    raw_blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in raw_blocks if b.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into approximate sentences.

    Args:
        text: Input text block.

    Returns:
        list[str]: List of sentences.
    """
    parts = SENTENCE_SPLIT_REGEX.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
