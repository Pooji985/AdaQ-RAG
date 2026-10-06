"""Exceptions for the query analysis module."""


class QueryAnalysisError(Exception):
    """Base exception for query analysis errors."""


class InvalidQueryError(QueryAnalysisError, ValueError):
    """Raised when an invalid, empty, or whitespace-only query is provided for analysis."""
