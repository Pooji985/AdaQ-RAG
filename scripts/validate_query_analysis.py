"""Lightweight validation script for Phase 6 QueryAnalyzer against the 60-question evaluation dataset."""

from collections import Counter, defaultdict
from adaq_rag.evaluation.chunking.dataset import EVAL_QUESTIONS
from adaq_rag.query_analysis import QueryAnalyzer


def run_dataset_validation() -> dict:
    analyzer = QueryAnalyzer()
    total_questions = len(EVAL_QUESTIONS)

    intent_counts = Counter()
    complexity_counts = Counter()
    total_intent_conf = 0.0
    total_complexity_conf = 0.0

    breakdown_by_orig = defaultdict(lambda: {"intents": Counter(), "complexities": Counter(), "count": 0})

    for q in EVAL_QUESTIONS:
        result = analyzer.analyze(q.query)
        intent_counts[result.intent.value] += 1
        complexity_counts[result.complexity.value] += 1
        total_intent_conf += result.intent_confidence
        total_complexity_conf += result.complexity_confidence

        orig_cat = q.category
        breakdown_by_orig[orig_cat]["count"] += 1
        breakdown_by_orig[orig_cat]["intents"][result.intent.value] += 1
        breakdown_by_orig[orig_cat]["complexities"][result.complexity.value] += 1

    avg_intent_conf = total_intent_conf / total_questions
    avg_complexity_conf = total_complexity_conf / total_questions

    print("=================================================================")
    print("PHASE 6 QUERY ANALYZER: 60-QUESTION DATASET VALIDATION REPORT")
    print("=================================================================")
    print(f"Total questions analyzed: {total_questions}\n")

    print("--- Intent Distribution ---")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_questions) * 100
        print(f"  {intent:<18}: {count:>2} ({pct:>5.1f}%)")
    print(f"Average Intent Confidence: {avg_intent_conf:.3f}\n")

    print("--- Complexity Distribution ---")
    for comp, count in sorted(complexity_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_questions) * 100
        print(f"  {comp:<18}: {count:>2} ({pct:>5.1f}%)")
    print(f"Average Complexity Confidence: {avg_complexity_conf:.3f}\n")

    print("--- Breakdown by Phase 3 Dataset Category (question_type) ---")
    for orig_cat, data in sorted(breakdown_by_orig.items()):
        print(f"\nPhase 3 Category: {orig_cat} (N={data['count']})")
        print("  Intents     : " + ", ".join(f"{k}={v}" for k, v in data["intents"].items()))
        print("  Complexities: " + ", ".join(f"{k}={v}" for k, v in data["complexities"].items()))

    print("\n=================================================================")
    print("MANUAL SANITY CHECK (REPRESENTATIVE QUERIES)")
    print("=================================================================")

    sanity_queries = [
        "What is StandardScaler?",
        "Why does standardization help machine learning?",
        "How do I use StandardScaler in a pipeline?",
        "What is the difference between Ridge and Lasso?",
        "Why is my StandardScaler pipeline failing?",
        "How do preprocessing, pipelines, and model selection work together?",
    ]

    for sq in sanity_queries:
        res = analyzer.analyze(sq)
        print(f"\nQuery: '{sq}'")
        print(f"  Intent               : {res.intent.value} (conf: {res.intent_confidence:.2f})")
        print(f"  Complexity           : {res.complexity.value} (conf: {res.complexity_confidence:.2f})")
        print(f"  Intent Signals       : {res.intent_signals}")
        print(f"  Complexity Signals   : {res.complexity_signals}")

    return {
        "total_questions": total_questions,
        "intent_counts": dict(intent_counts),
        "complexity_counts": dict(complexity_counts),
        "avg_intent_conf": avg_intent_conf,
        "avg_complexity_conf": avg_complexity_conf,
        "breakdown": {k: dict(v) for k, v in breakdown_by_orig.items()},
    }


if __name__ == "__main__":
    run_dataset_validation()
