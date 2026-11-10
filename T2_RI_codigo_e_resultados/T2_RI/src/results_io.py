"""Arquivos padronizados de resultados (item 15)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def ranking_frame(run: Mapping[str, Sequence[tuple[str, float]]], qrels: Mapping[str, Mapping[str, int]],
                  top_k: int = 100, query_order: Sequence[str] | None = None, depth_marker: int | None = None,
                  marker_names: tuple[str, str] = ("cross_encoder", "rrf")) -> pd.DataFrame:
    """query_id, doc_id, rank, score, relevance (grau do qrels; 0 se ausente/não julgado) — 100 primeiros por consulta.
    Se `depth_marker` for dado, acrescenta `score_type` (ex.: top-50 = cross_encoder; demais = rrf)."""
    rows = []
    for qid in (query_order if query_order is not None else sorted(run)):
        for rank, (doc_id, score) in enumerate(run[qid][:top_k], start=1):
            row = {"query_id": qid, "doc_id": doc_id, "rank": rank, "score": float(score),
                   "relevance": int(qrels.get(qid, {}).get(doc_id, 0))}
            if depth_marker is not None:
                row["score_type"] = marker_names[0] if rank <= depth_marker else marker_names[1]
            rows.append(row)
    return pd.DataFrame(rows)


def write_ranking_csv(path: Path, run, qrels, top_k: int = 100, query_order=None, depth_marker=None) -> pd.DataFrame:
    df = ranking_frame(run, qrels, top_k, query_order, depth_marker)
    df.to_csv(path, index=False)
    return df


def read_run(path: Path, query_ids: Sequence[str] | None = None) -> dict[str, list[tuple[str, float]]]:
    """Reconstrói um ranking a partir de um results_*.csv. Consultas sem nenhum resultado (ex.: nenhum termo
    existe no índice) não aparecem no CSV; se `query_ids` for dado, elas voltam com ranking vazio."""
    df = pd.read_csv(path).sort_values(["query_id", "rank"])
    run = {q: list(zip(g["doc_id"], g["score"])) for q, g in df.groupby("query_id", sort=False)}
    for q in (query_ids or []):
        run.setdefault(q, [])
    return run


def write_metrics_per_query(path: Path, per_model: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for model, df in per_model.items():
        d = df.copy()
        d.insert(1, "model", model)
        frames.append(d[["query_id", "model", "P10", "R10", "AP10", "NDCG10"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path, index=False)
    return out
