"""Deterministic, rule-based Query Analysis Engine for AdaQ-RAG."""

from collections import OrderedDict
import re
from typing import Any

from adaq_rag.query_analysis.exceptions import InvalidQueryError
from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)

ANALYZER_VERSION = "1.0.0"

# Curated scikit-learn & Machine Learning domain terms for concept extraction
DOMAIN_ENTITIES: list[str] = [
    # Scalers & Transformers
    "standardscaler",
    "minmaxscaler",
    "robustscaler",
    "maxabsscaler",
    "normalizer",
    "onehotencoder",
    "ordinalencoder",
    "labelencoder",
    "targetencoder",
    "simpleimputer",
    "iterativeimputer",
    "knnimputer",
    "polynomialfeatures",
    "splinetransformer",
    "columntransformer",
    "power_transform",
    "quantiletransformer",
    # Linear Models & Regressors
    "linearregression",
    "logisticregression",
    "ridgeclassifier",
    "ridge",
    "lasso",
    "elasticnet",
    "lars",
    "bayesianridge",
    "sgdclassifier",
    "sgdregressor",
    # Support Vector Machines
    "linearsvc",
    "linearsvr",
    "nusvc",
    "nusvr",
    "oneclasssvm",
    "svc",
    "svr",
    # Trees & Ensembles
    "decisiontreeclassifier",
    "decisiontreeregressor",
    "decision tree",
    "randomforestclassifier",
    "randomforestregressor",
    "random forest",
    "extratreesclassifier",
    "extratreesregressor",
    "gradientboostingclassifier",
    "gradientboostingregressor",
    "gradient boosting",
    "histgradientboostingclassifier",
    "adaboostclassifier",
    "adaboostregressor",
    "votingclassifier",
    "stackingclassifier",
    # Clustering & Decompositions
    "kmeans",
    "minibatchkmeans",
    "dbscan",
    "spectralclustering",
    "agglomerativeclustering",
    "pca",
    "incrementalpca",
    "kernelpca",
    "truncatedsvd",
    "fastica",
    "tsne",
    # Pipelines & Model Selection
    "model selection",
    "preprocessing",
    "pipeline",
    "pipelines",
    "make_pipeline",
    "featureunion",
    "gridsearchcv",
    "randomizedsearchcv",
    "halvinggridsearchcv",
    "cross_val_score",
    "cross_validate",
    "train_test_split",
    "kfold",
    "stratifiedkfold",
    "timeseriessplit",
    "learning_curve",
    "validation_curve",
    "estimators",
    "transformers",
    # Metrics & Evaluation
    "confusion_matrix",
    "classification_report",
    "roc_auc_score",
    "f1_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "mean_squared_error",
    "r2_score",
    # Core ML concepts
    "cross-validation",
    "cross validation",
    "hyperparameter tuning",
    "hyperparameter",
    "hyperparameters",
    "regularization",
    "l1 regularization",
    "l2 regularization",
    "cost-complexity pruning",
    "ccp_alpha",
    "bias-variance tradeoff",
    "bias-variance",
    "data leakage",
    "overfitting",
    "underfitting",
    "feature selection",
    "dimensionality reduction",
    "class imbalance",
    "class weights",
    "sparse matrix",
]

# Sort domain entities by descending length so multi-word entities match before substrings
DOMAIN_ENTITIES.sort(key=len, reverse=True)


class QueryAnalyzer:
    """Deterministic, explainable rule-based Query Analysis Engine.

    Classifies user technical queries into:
        1. QueryIntent (FACT_LOOKUP, EXPLANATION, PROCEDURE, COMPARISON, TROUBLESHOOTING, MULTI_CONCEPT)
        2. QueryComplexity (SIMPLE, MEDIUM, COMPLEX)

    Design Principles:
        - Deterministic & Explainable: Decisions are driven by strict priority rules and diagnostic signals.
        - Fast & Lightweight: Zero external API calls, zero ML inference latency.
        - Testable & Replaceable: Pure input-output interface suitable for downstream adaptive retrieval (Phase 7).

    Rule Priority for Intent Classification:
        When multiple intent signals match, the following deterministic hierarchy applies:
        1. TROUBLESHOOTING: Strongest precedence when error, warning, failure, or bug fix is mentioned.
        2. COMPARISON: Second precedence when comparison, vs, or difference between entities is queried.
        3. MULTI_CONCEPT: Third precedence when multi-entity interaction, chaining, or synergy is queried.
        4. PROCEDURE: Fourth precedence when action-oriented how-to, setup, or code implementation is requested.
        5. EXPLANATION: Fifth precedence when conceptual 'why', mechanism, or intuition is asked.
        6. FACT_LOOKUP: Sixth precedence for direct parameter lookup, definitions, or single fact queries.
    """

    def __init__(self, version: str = ANALYZER_VERSION) -> None:
        self.version = version
        self._init_regex_rules()

    def _init_regex_rules(self) -> None:
        """Pre-compile regex patterns for performance and determinism."""
        # Troubleshooting patterns
        self._pattern_error_explicit = re.compile(
            r"\b(?:valueerror|typeerror|notfittederror|convergencewarning|userwarning|runtimeerror|"
            r"attributeerror|keyerror|indexerror|zerodivisionerror|memoryerror)\b",
            re.IGNORECASE,
        )
        self._pattern_troubleshoot_action = re.compile(
            r"\b(?:how\s+(?:to|do\s+i|can\s+i)\s+(?:fix|resolve|solve|debug|troubleshoot|diagnose|prevent|handle))\b|"
            r"\b(?:troubleshoot|troubleshooting|debug|debugging)\b",
            re.IGNORECASE,
        )
        self._pattern_failure_state = re.compile(
            r"\b(?:why\s+(?:is|are|does|did|am|do)\b.*\b(?:fail|failing|failed|crash|crashes|crashing|broken|throw|throwing|hang|freeze|leak))\b|"
            r"\b(?:why\s+is\s+my\b.*\b(?:fail|failing|broken|throwing|not\s+working))\b|"
            r"\b(?:not\s+working|doesn't\s+work|does\s+not\s+work|fails?\s+to\s+converge|failing\s+to\s+converge|nan\s+values?|infinite\s+loop|memory\s+leak)\b",
            re.IGNORECASE,
        )
        self._pattern_error_keywords = re.compile(
            r"\b(?:error|exception|warning|traceback|stacktrace|bug|crash)\b",
            re.IGNORECASE,
        )

        # Comparison patterns
        self._pattern_comparison = re.compile(
            r"\b(?:diff(?:erence|erences)?\s+between)\b|"
            r"\b(?:compare|comparing|comparison)\b|"
            r"\b(?:vs\.?|versus)\b|"
            r"\b(?:in\s+contrast\s+to|compared\s+(?:to|with)|relative\s+to)\b|"
            r"\b(?:trade-?offs?\s+between|pros\s+and\s+cons|advantages\s+and\s+disadvantages)\b|"
            r"\b(?:which\s+(?:is|one\s+is)\s+(?:better|faster|preferable|more\s+suitable|recommended))\b|"
            r"\b(?:when\s+(?:to|should\s+i)\s+use\b.*\binstead\s+of\b)|"
            r"\b(?:distinguish\s+between|distinction\s+between)\b|"
            r"\b(?:better\s+than|faster\s+than|preferred\s+over)\b",
            re.IGNORECASE,
        )

        # Multi-concept interaction patterns
        self._pattern_multi_concept = re.compile(
            r"\b(?:how\s+do\b.*\b(?:interact|work\s+together|relate))\b|"
            r"\b(?:interact(?:ion|ions)?\s+between)\b|"
            r"\b(?:interplay\s+between|synergy\s+between)\b|"
            r"\b(?:relationship\s+between\b.*\band\b)|"
            r"\b(?:work\s+together)\b|"
            r"\b(?:combine|combining|integrat(?:e|ing|ion\s+of))\b.*\b(?:with|and)\b|"
            r"\b(?:joint\s+effect\s+of|interconnectedness)\b|"
            r"\b(?:end-to-end\s+workflow\s+(?:with|combining))\b",
            re.IGNORECASE,
        )

        # Procedure / How-to patterns
        self._pattern_procedure = re.compile(
            r"\b(?:how\s+(?:to|do\s+i|can\s+i|should\s+i|would\s+i)\s+(?:use|implement|configure|tune|train|fit|apply|create|build|set\s*up|setup|run|pipeline|save|load|export|import|instantiate|call|write))\b|"
            r"\b(?:how\s+(?:to|do\s+i|can\s+i|should\s+i)\b)|"
            r"\b(?:steps\s+to|step-by-step\s+guide|tutorial\s+(?:on|for)|guide\s+(?:to|for)|how-to)\b|"
            r"\b(?:code\s+(?:example|snippet|for)|example\s+of\s+using|sample\s+code|syntax\s+for)\b|"
            r"\b(?:best\s+practices?\s+for\s+(?:implementing|using|configuring))\b",
            re.IGNORECASE,
        )

        # Explanation patterns
        self._pattern_explanation = re.compile(
            r"\b(?:why\s+(?:does|do|is|are|would|did))\b|"
            r"\b(?:explain|explaining|explanation)\b|"
            r"\b(?:how\s+does\b.*\bwork)\b|"
            r"\b(?:intuition\s+(?:behind|for)|rationale\s+(?:for|behind)|theory\s+behind)\b|"
            r"\b(?:mathematical\s+(?:formula|foundation|derivation)|under\s+the\s+hood)\b|"
            r"\b(?:what\s+causes|working\s+principle|underlying\s+mechanism)\b|"
            r"\b(?:what\s+is\s+the\s+(?:intuition|rationale|purpose|idea\s+behind))\b",
            re.IGNORECASE,
        )

        # Fact lookup patterns
        self._pattern_fact_lookup = re.compile(
            r"^what\s+(?:is|are)\s+[a-zA-Z0-9_\-\s]+\??$|"
            r"\b(?:what\s+(?:is|are))\b|"
            r"\b(?:definition\s+of|define)\b|"
            r"\b(?:what\s+is\s+the\s+(?:default\s+value|parameter|attribute|return\s+type|signature))\b|"
            r"\b(?:default\s+(?:value|parameter|setting)|return\s+(?:value|type)|attributes?\s+of)\b|"
            r"\b(?:parameters?\s+(?:of|for)|arguments?\s+(?:of|for)|signature\s+of)\b|"
            r"\b(?:which\s+(?:module|package|class|method|function))\b|"
            r"\b(?:meaning\s+of\s+parameter|what\s+does\s+the\s+[a-zA-Z0-9_]+\s+parameter\s+do)\b",
            re.IGNORECASE,
        )

        # Complexity modifier patterns
        self._pattern_scenario_constraint = re.compile(
            r"\b(?:for\s+(?:large|sparse|high[- ]dimensional|imbalanced)\s+datasets?|"
            r"with\s+(?:outliers|missing\s+values?|nan)|"
            r"in\s+the\s+presence\s+of|under\s+what\s+conditions|when\s+data\s+is)\b",
            re.IGNORECASE,
        )
        self._pattern_compound_clause = re.compile(
            r"\b(?:and\s+(?:also|how|why|explain|when\s+to\s+use)|as\s+well\s+as|along\s+with)\b",
            re.IGNORECASE,
        )
        self._pattern_systemic_integration = re.compile(
            r"\b(?:data\s+leakage|cross-validation\s+leakage|custom\s+(?:scorer|transformer|estimator)|pipeline\s+combining)\b",
            re.IGNORECASE,
        )

    def analyze(self, query: Any) -> QueryAnalysisResult:
        """Analyze a user query to determine intent, complexity, and confidence scores.

        Args:
            query: The user query string to analyze.

        Returns:
            QueryAnalysisResult: The structured classification result with diagnostic signals.

        Raises:
            InvalidQueryError: If the query is not a string, or is empty / whitespace-only.
        """
        # Input validation
        if not isinstance(query, str):
            raise InvalidQueryError(f"Query must be a string, received {type(query).__name__}")

        cleaned_query = query.strip()
        if not cleaned_query:
            raise InvalidQueryError("Query cannot be empty or whitespace-only.")

        # Extract domain entities
        detected_entities = self._detect_entities(cleaned_query)

        # Detect intent and collect intent signals
        intent, intent_conf, intent_signals = self._classify_intent(cleaned_query, detected_entities)

        # Determine complexity and collect complexity signals
        complexity, complexity_conf, complexity_signals = self._assess_complexity(
            cleaned_query, intent, detected_entities
        )

        return QueryAnalysisResult(
            query=cleaned_query,
            intent=intent,
            intent_confidence=intent_conf,
            complexity=complexity,
            complexity_confidence=complexity_conf,
            intent_signals=intent_signals,
            complexity_signals=complexity_signals,
            analyzer_version=self.version,
        )

    def _detect_entities(self, query: str) -> list[str]:
        """Detect unique scikit-learn and machine learning domain entities in query."""
        lower_query = query.lower()
        matched: list[str] = []
        for entity in DOMAIN_ENTITIES:
            # Check for word boundary match where appropriate
            pattern = rf"\b{re.escape(entity)}\b"
            if re.search(pattern, lower_query):
                matched.append(entity)
        return matched

    def _classify_intent(
        self, query: str, detected_entities: list[str]
    ) -> tuple[QueryIntent, float, list[str]]:
        """Classify query intent according to the deterministic priority rule hierarchy.

        Returns:
            tuple: (chosen_intent, heuristic_confidence, diagnostic_signals)
        """
        signals_by_intent: dict[QueryIntent, list[str]] = OrderedDict()

        # Check for pure definition edge case (e.g., "what is ConvergenceWarning?")
        # Pure definition queries for warning/error classes should remain FACT_LOOKUP
        is_pure_definition = bool(re.match(r"^what\s+(?:is|are)\s+[a-zA-Z0-9_\-\s]+\??$", query, re.IGNORECASE))

        # 1. TROUBLESHOOTING signals
        troubleshoot_signals: list[str] = []
        if self._pattern_error_explicit.search(query):
            if not is_pure_definition:
                troubleshoot_signals.append("explicit error/exception/warning class name detected")
        if self._pattern_troubleshoot_action.search(query):
            troubleshoot_signals.append("troubleshooting/fix/debug action phrasing detected")
        if self._pattern_failure_state.search(query):
            troubleshoot_signals.append("failure/crash/failing-state phrasing detected")
        if self._pattern_error_keywords.search(query) and not is_pure_definition:
            troubleshoot_signals.append("error/exception/warning general keyword detected")

        if troubleshoot_signals:
            signals_by_intent[QueryIntent.TROUBLESHOOTING] = troubleshoot_signals

        # 2. COMPARISON signals
        comparison_signals: list[str] = []
        if self._pattern_comparison.search(query):
            comparison_signals.append("comparison/vs/difference indicator detected")
        if len(detected_entities) >= 2 and (" or " in query.lower() or " vs " in query.lower() or " versus " in query.lower()):
            comparison_signals.append(f"entity contrast detected between {detected_entities[:2]}")

        if comparison_signals:
            signals_by_intent[QueryIntent.COMPARISON] = comparison_signals

        # 3. MULTI_CONCEPT signals
        multi_concept_signals: list[str] = []
        if self._pattern_multi_concept.search(query):
            multi_concept_signals.append("multi-concept interaction/interplay phrasing detected")
        if len(detected_entities) >= 3 and (" and " in query.lower() or " with " in query.lower()):
            multi_concept_signals.append(f"3+ distinct domain entities chained: {detected_entities[:3]}")

        if multi_concept_signals:
            signals_by_intent[QueryIntent.MULTI_CONCEPT] = multi_concept_signals

        # 4. PROCEDURE signals
        procedure_signals: list[str] = []
        if self._pattern_procedure.search(query):
            procedure_signals.append("action-oriented implementation/how-to phrasing detected")

        if procedure_signals:
            signals_by_intent[QueryIntent.PROCEDURE] = procedure_signals

        # 5. EXPLANATION signals
        explanation_signals: list[str] = []
        if self._pattern_explanation.search(query):
            explanation_signals.append("conceptual explanation/why/mechanism phrasing detected")

        if explanation_signals:
            signals_by_intent[QueryIntent.EXPLANATION] = explanation_signals

        # 6. FACT_LOOKUP signals
        fact_signals: list[str] = []
        if self._pattern_fact_lookup.search(query) or is_pure_definition:
            fact_signals.append("factual definition/parameter/attribute query pattern detected")
        elif query.lower().startswith("what is") or query.lower().startswith("what are"):
            fact_signals.append("general 'what is/are' formulation detected")

        if fact_signals:
            signals_by_intent[QueryIntent.FACT_LOOKUP] = fact_signals

        # Strict deterministic priority evaluation
        priority_order = [
            QueryIntent.TROUBLESHOOTING,
            QueryIntent.COMPARISON,
            QueryIntent.MULTI_CONCEPT,
            QueryIntent.PROCEDURE,
            QueryIntent.EXPLANATION,
            QueryIntent.FACT_LOOKUP,
        ]

        diagnostic_signals: list[str] = []
        for intent in priority_order:
            if intent in signals_by_intent:
                chosen_intent = intent
                primary_reasons = signals_by_intent[intent]
                diagnostic_signals.append(f"{chosen_intent.value}: {'; '.join(primary_reasons)}")

                # Note subordinated secondary intents for diagnostic explainability
                for other_intent, other_reasons in signals_by_intent.items():
                    if other_intent != chosen_intent:
                        diagnostic_signals.append(
                            f"subordinated {other_intent.value} ({'; '.join(other_reasons)}) via deterministic priority rule"
                        )

                # Confidence calculation
                num_competing = len(signals_by_intent) - 1
                if num_competing == 0:
                    confidence = 0.95
                elif num_competing == 1:
                    confidence = 0.85
                else:
                    confidence = 0.75

                return chosen_intent, confidence, diagnostic_signals

        # Fallback if no specific rule matched
        diagnostic_signals.append("fallback: no distinct intent pattern matched; defaulting to FACT_LOOKUP")
        return QueryIntent.FACT_LOOKUP, 0.50, diagnostic_signals

    def _assess_complexity(
        self, query: str, intent: QueryIntent, detected_entities: list[str]
    ) -> tuple[QueryComplexity, float, list[str]]:
        """Assess query complexity representing expected retrieval and evidence requirements.

        Scoring Model:
            - Baseline evidence score from Intent:
                FACT_LOOKUP: 1.0 (single chunk/definition usually suffices)
                EXPLANATION: 2.0 (mechanics require multi-paragraph context)
                PROCEDURE: 2.0 (recipes/configurations require code & explanation)
                COMPARISON: 2.0 (requires evidence across distinct concepts)
                TROUBLESHOOTING: 2.0 (requires diagnostic symptoms + solution context)
                MULTI_CONCEPT: 3.0 (inherently requires synthesizing multiple sections)
            - Entity modifier:
                0-1 entities: +0.0
                2 entities: +0.5
                3+ entities: +1.5
            - Modifiers:
                Scenario/Constraint patterns (+0.5)
                Compound question clauses (+0.5)
                Systemic integration / leakage (+0.5)

        Thresholds:
            Score < 2.0: SIMPLE
            2.0 <= Score <= 3.0: MEDIUM
            Score > 3.0: COMPLEX

        Returns:
            tuple: (chosen_complexity, heuristic_confidence, diagnostic_signals)
        """
        diagnostic_signals: list[str] = []

        # 1. Baseline evidence score by intent
        base_scores = {
            QueryIntent.FACT_LOOKUP: 1.0,
            QueryIntent.EXPLANATION: 2.0,
            QueryIntent.PROCEDURE: 2.0,
            QueryIntent.COMPARISON: 2.0,
            QueryIntent.TROUBLESHOOTING: 2.0,
            QueryIntent.MULTI_CONCEPT: 3.0,
        }
        score = base_scores.get(intent, 1.0)
        diagnostic_signals.append(f"base_evidence_score: {score:.1f} (intent={intent.value})")

        # 2. Entity count modifier
        entity_count = len(detected_entities)
        if entity_count >= 3:
            entity_mod = 1.5
            diagnostic_signals.append(f"entity_modifier: +{entity_mod:.1f} ({entity_count} entities: {detected_entities[:3]})")
        elif entity_count == 2:
            entity_mod = 0.5
            diagnostic_signals.append(f"entity_modifier: +{entity_mod:.1f} (2 entities: {detected_entities})")
        else:
            entity_mod = 0.0
            if entity_count == 1:
                diagnostic_signals.append(f"entity_modifier: +0.0 (single entity: {detected_entities[0]})")
            else:
                diagnostic_signals.append("entity_modifier: +0.0 (0 detected domain entities)")
        score += entity_mod

        # 3. Scenario / Constraint modifier
        if self._pattern_scenario_constraint.search(query):
            score += 0.5
            diagnostic_signals.append("scenario_modifier: +0.5 (dataset constraints or condition indicators detected)")

        # 4. Compound clause modifier
        if self._pattern_compound_clause.search(query):
            score += 0.5
            diagnostic_signals.append("compound_modifier: +0.5 (compound multi-part query clauses detected)")

        # 5. Systemic integration modifier
        if self._pattern_systemic_integration.search(query):
            score += 0.5
            diagnostic_signals.append("systemic_modifier: +0.5 (cross-cutting integration or leakage detected)")

        diagnostic_signals.append(f"total_evidence_score: {score:.1f}")

        # Map to Complexity Level
        if score < 2.0:
            complexity = QueryComplexity.SIMPLE
            confidence = 0.90 if score <= 1.2 else 0.80
        elif score <= 3.0:
            complexity = QueryComplexity.MEDIUM
            confidence = 0.85 if (2.0 <= score <= 2.8) else 0.75
        else:
            complexity = QueryComplexity.COMPLEX
            confidence = 0.90 if score >= 3.5 else 0.80

        diagnostic_signals.append(f"assigned_complexity: {complexity.value} (confidence={confidence:.2f})")

        return complexity, confidence, diagnostic_signals
