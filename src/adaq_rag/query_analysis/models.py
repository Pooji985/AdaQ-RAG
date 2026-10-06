"""Data models and taxonomies for Phase 6 Query Analysis Engine."""

from enum import Enum
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Taxonomy of user query intents.

    Categories:
        FACT_LOOKUP: A direct fact, definition, value, property, or specific piece of information.
        EXPLANATION: Asks why/how something works conceptually or requests an explanation.
        PROCEDURE: Asks how to perform, implement, configure, use, or execute something.
        COMPARISON: Compares two or more concepts, methods, algorithms, parameters, or approaches.
        TROUBLESHOOTING: Describes an error, unexpected behavior, failure, incorrect result, or asks how to diagnose/fix a problem.
        MULTI_CONCEPT: Requires connecting multiple related concepts to answer properly.
    """

    FACT_LOOKUP = "FACT_LOOKUP"
    EXPLANATION = "EXPLANATION"
    PROCEDURE = "PROCEDURE"
    COMPARISON = "COMPARISON"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    MULTI_CONCEPT = "MULTI_CONCEPT"


class QueryComplexity(str, Enum):
    """Taxonomy of query complexity representing expected retrieval/evidence requirements.

    Levels:
        SIMPLE: Likely answerable using one focused chunk or single concept.
        MEDIUM: Likely needs multiple related pieces of information or a more detailed explanation/procedure.
        COMPLEX: Likely requires multiple concepts, relationships, cross-section evidence, or troubleshooting reasoning.
    """

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


class QueryAnalysisResult(BaseModel):
    """Structured result of rule-based query intent and complexity analysis.

    Confidence Note:
        The confidence scores (intent_confidence and complexity_confidence) are deterministic
        heuristic metrics in the range [0.0, 1.0] derived from rule match specificity, signal margins,
        and pattern density. They are NOT calibrated statistical probabilities.
    """

    query: str = Field(description="Normalized query string analyzed")
    intent: QueryIntent = Field(description="Classified primary intent category")
    intent_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic confidence score for intent (0.0 - 1.0). Not a calibrated probability.",
    )
    complexity: QueryComplexity = Field(
        description="Classified retrieval and evidence requirement complexity level"
    )
    complexity_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Heuristic confidence score for complexity (0.0 - 1.0). Not a calibrated probability.",
    )
    intent_signals: list[str] = Field(
        default_factory=list,
        description="Diagnostic explanations of matched patterns and signals for intent classification",
    )
    complexity_signals: list[str] = Field(
        default_factory=list,
        description="Diagnostic explanations of scoring and signals for complexity level",
    )
    analyzer_version: str = Field(
        default="1.0.0",
        description="Version identifier of the query analysis engine rule set",
    )
