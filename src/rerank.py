"""Reranking (item 9): o ranking híbrido é o recuperador inicial; os 50 primeiros são reordenados por um
cross-encoder e os demais mantêm a ordem relativa do ranking híbrido."""
from __future__ import annotations

from typing import Mapping, Sequence


def rerank_top_n(query_text: str, hybrid: Sequence[tuple[str, float]], doc_texts: Mapping[str, str],
                 cross_encoder, depth: int = 50, batch_size: int = 32) -> list[tuple[str, float]]:
    """Retorna o ranking completo: top-`depth` reordenado pelo score do cross-encoder (logit) seguido do restante
    do ranking híbrido (com o score RRF original). Empate de score mantém a ordem do híbrido."""
    head, tail = list(hybrid[:depth]), list(hybrid[depth:])
    if not head:
        return list(hybrid)
    scores = cross_encoder.predict([(query_text, doc_texts[d]) for d, _ in head], batch_size=batch_size)
    order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), i))
    return [(head[i][0], float(scores[i])) for i in order] + tail
