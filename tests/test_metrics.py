import math
import random

import pytest

from src.metrics import (average_precision_at_k, evaluate_run, ndcg_at_k, precision_at_k, recall_at_k,
                         validate_with_ir_measures)

QRELS = {"d1": 2, "d3": 1, "d9": 1, "d12": 2, "d20": 0}
RANK = ["d1", "x", "d3", "y", "z", "d9", "a", "b", "c", "d", "d12"]   # d12 está na posição 11 (fora do Top-10)

def test_precision_recall():
    assert precision_at_k(RANK, QRELS) == 0.3                         # d1, d3, d9 nas 10 primeiras
    assert recall_at_k(RANK, QRELS) == pytest.approx(3 / 4)           # 4 relevantes (grau > 0) no qrels

def test_average_precision_uses_total_relevant_as_denominator():
    ap = (1 / 1 + 2 / 3 + 3 / 6) / 4
    assert average_precision_at_k(RANK, QRELS) == pytest.approx(ap)

def test_ndcg_linear_gain_and_ideal_from_all_judgments():
    dcg = 2 / math.log2(2) + 1 / math.log2(4) + 1 / math.log2(7)
    ideal = [2, 2, 1, 1]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    assert ndcg_at_k(RANK, QRELS) == pytest.approx(dcg / idcg)

def test_unjudged_and_zero_are_not_relevant():
    assert precision_at_k(["d20", "q"], QRELS) == 0.0 and ndcg_at_k(["d20", "q"], QRELS) == 0.0

def test_short_ranking_still_divides_by_10():
    assert precision_at_k(["d1"], QRELS) == 0.1

def test_matches_ir_measures_on_random_runs():
    rng = random.Random(0)
    docs = [f"d{i}" for i in range(60)]
    qrels = {f"q{i}": {d: rng.choice([1, 2]) for d in rng.sample(docs, rng.randint(1, 25))} for i in range(30)}
    run = {q: [(d, 0.0) for d in rng.sample(docs, rng.randint(5, 40))] for q in qrels}
    pq = evaluate_run(run, qrels)
    worst = validate_with_ir_measures(run, qrels, pq)
    assert max(worst.values()) < 1e-9
