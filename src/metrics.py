"""Métricas de avaliação (item 6), implementadas conforme o enunciado e conferidas com a ir_measures.

* Precision@10, Recall@10, AP@10: relevante = grau > 0.
* Recall@10 e AP@10 dividem pelo TOTAL de relevantes da consulta nos qrels (convenção trec_eval).
* AP@10 = soma de P@k nas posições k <= 10 que contêm relevantes, dividida pelo total de relevantes.
* NDCG@10: ganho linear (= grau), desconto 1/log2(k+1), ranking ideal calculado com TODOS os julgamentos.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import pandas as pd

K = 10


def _rel(qrels_q: Mapping[str, int], doc_id: str) -> int:
    return qrels_q.get(doc_id, 0)


def precision_at_k(ranked: Sequence[str], qrels_q: Mapping[str, int], k: int = K) -> float:
    return sum(1 for d in ranked[:k] if _rel(qrels_q, d) > 0) / k


def recall_at_k(ranked: Sequence[str], qrels_q: Mapping[str, int], k: int = K) -> float:
    total = sum(1 for g in qrels_q.values() if g > 0)
    return sum(1 for d in ranked[:k] if _rel(qrels_q, d) > 0) / total if total else 0.0


def average_precision_at_k(ranked: Sequence[str], qrels_q: Mapping[str, int], k: int = K) -> float:
    total = sum(1 for g in qrels_q.values() if g > 0)
    if not total:
        return 0.0
    hits, acc = 0, 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if _rel(qrels_q, d) > 0:
            hits += 1
            acc += hits / i
    return acc / total


def ndcg_at_k(ranked: Sequence[str], qrels_q: Mapping[str, int], k: int = K) -> float:
    dcg = sum(_rel(qrels_q, d) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted((g for g in qrels_q.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(ranked: Sequence[str], qrels_q: Mapping[str, int]) -> dict[str, float]:
    return {"P10": precision_at_k(ranked, qrels_q), "R10": recall_at_k(ranked, qrels_q),
            "AP10": average_precision_at_k(ranked, qrels_q), "NDCG10": ndcg_at_k(ranked, qrels_q)}


def to_id_run(run: Mapping[str, Sequence]) -> dict[str, list[str]]:
    """Aceita rankings como listas de doc_id ou de (doc_id, score)."""
    return {q: [x[0] if isinstance(x, (tuple, list)) else x for x in docs] for q, docs in run.items()}


def evaluate_run(run: Mapping[str, Sequence], qrels: Mapping[str, Mapping[str, int]],
                 query_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Métricas por consulta (uma linha por consulta do split; consulta sem ranking recebe zeros)."""
    ids_run = to_id_run(run)
    rows = []
    for qid in (query_ids if query_ids is not None else sorted(qrels)):
        rows.append({"query_id": qid, **evaluate_query(ids_run.get(qid, []), qrels[qid])})
    return pd.DataFrame(rows)


def mean_metrics(per_query: pd.DataFrame) -> dict[str, float]:
    return {m: float(per_query[m].mean()) for m in ("P10", "R10", "AP10", "NDCG10")}


def validate_with_ir_measures(run: Mapping[str, Sequence], qrels: Mapping[str, Mapping[str, int]],
                              per_query: pd.DataFrame, tol: float = 1e-9) -> dict[str, float]:
    """Compara as métricas implementadas aqui com a ir_measures (trec_eval). Retorna a maior diferença absoluta."""
    import ir_measures
    from ir_measures import AP, P, R, nDCG

    ids_run = to_id_run(run)
    qrel_objs = [ir_measures.Qrel(q, d, g) for q, dq in qrels.items() for d, g in dq.items()]
    # score = -posição, para que a ordem avaliada pela biblioteca seja exatamente a do ranking
    run_objs = [ir_measures.ScoredDoc(q, d, float(-i)) for q, docs in ids_run.items() if q in qrels
                for i, d in enumerate(docs)]
    measures = {"P10": P(cutoff=K), "R10": R(cutoff=K), "AP10": AP(cutoff=K), "NDCG10": nDCG(cutoff=K)}
    lib = {m: {} for m in measures}
    inv = {str(v): k for k, v in measures.items()}
    for metric in ir_measures.iter_calc(list(measures.values()), qrel_objs, run_objs):
        lib[inv[str(metric.measure)]][metric.query_id] = metric.value
    worst = {}
    mine = per_query.set_index("query_id")
    for m in measures:
        diffs = [abs(mine.loc[q, m] - lib[m].get(q, 0.0)) for q in mine.index]
        worst[m] = max(diffs)
    if max(worst.values()) > tol:
        raise AssertionError(f"Métricas divergem da ir_measures: {worst}")
    return worst
