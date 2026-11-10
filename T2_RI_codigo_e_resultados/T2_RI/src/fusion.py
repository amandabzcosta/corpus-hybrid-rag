"""Busca híbrida (item 8): Reciprocal Rank Fusion.  RRF(d) = soma_s 1/(k + rank_s(d)), rank começando em 1.
Documentos ausentes de um ranking não recebem contribuição desse sistema."""
from __future__ import annotations

from typing import Sequence


def rrf(rankings: Sequence[Sequence], k: int = 60, depth: int = 100) -> list[tuple[str, float]]:
    """Funde os `depth` primeiros documentos de cada ranking (listas de doc_id ou de (doc_id, score)).
    Empates: menor posição em algum dos rankings e, por fim, ordem alfabética do doc_id (determinístico)."""
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for r, item in enumerate(ranking[:depth], start=1):
            d = item[0] if isinstance(item, (tuple, list)) else item
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
            best[d] = min(best.get(d, r), r)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


def rrf_run(lexical: dict, semantic: dict, k: int = 60, depth: int = 100) -> dict[str, list[tuple[str, float]]]:
    return {q: rrf([lexical.get(q, []), semantic.get(q, [])], k=k, depth=depth) for q in lexical}
