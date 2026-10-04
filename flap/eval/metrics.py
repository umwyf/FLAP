"""Evaluation metrics: normalized exact match (main metric), token F1, Hit@k."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence


def normalize_answer(s: Optional[str]) -> str:
    """Lower-case, strip punctuation/articles and collapse whitespace (SQuAD style)."""
    if s is None:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: Optional[str], golden_answers: Sequence[str]) -> float:
    if prediction is None:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(g) for g in golden_answers))


def f1_score(prediction: Optional[str], golden_answers: Sequence[str]) -> float:
    if prediction is None:
        return 0.0
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for g in golden_answers:
        gold_tokens = normalize_answer(g).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evidence_hit(passages: Iterable[str], golden_answers: Sequence[str], k: Optional[int] = None) -> float:
    """Automatic proxy for Hit@k: does any of the top-k passages contain a gold answer?

    The paper reports Hit@k on human-annotated examples; this string-match
    version is a cheap approximation useful for monitoring.
    """
    passages = list(passages)
    if k is not None:
        passages = passages[:k]
    golds = [normalize_answer(g) for g in golden_answers if g]
    for p in passages:
        norm = normalize_answer(p)
        if any(g and g in norm for g in golds):
            return 1.0
    return 0.0


def aggregate(records: Sequence[Dict]) -> Dict[str, float]:
    """Average ``em``, ``f1`` and ``num_search_calls`` over output records."""
    n = len(records)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "avg_search_calls": 0.0}
    summary = {
        "n": n,
        "em": 100.0 * sum(r.get("em", 0.0) for r in records) / n,
        "f1": 100.0 * sum(r.get("f1", 0.0) for r in records) / n,
        "avg_search_calls": sum(r.get("num_search_calls", 0) for r in records) / n,
    }
    if any("first_hit" in r for r in records):
        hits = [r["first_hit"] for r in records if "first_hit" in r]
        summary["first_retrieval_hit"] = 100.0 * sum(hits) / len(hits)
    return summary
