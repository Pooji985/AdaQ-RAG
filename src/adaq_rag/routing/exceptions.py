"""Exceptions for the adaptive routing module."""


class RoutingError(Exception):
    """Base exception for routing module errors."""


class InvalidAnalysisResultError(RoutingError, ValueError):
    """Raised when an invalid, null, or malformed QueryAnalysisResult is provided to the router."""
