from __future__ import annotations

from typing import Mapping, Sequence

def rerank_top_n(query_text: str, hybrid: Sequence[tuple[str, float]], doc_texts: Mapping[str, str],
                 cross_encoder, depth: int = 50, batch_size: int = 32) -> list[tuple[str, float]]:
    head, tail = list(hybrid[:depth]), list(hybrid[depth:])
    if not head:
        return list(hybrid)
    scores = cross_encoder.predict([(query_text, doc_texts[d]) for d, _ in head], batch_size=batch_size)
    order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), i))
    return [(head[i][0], float(scores[i])) for i in order] + tail
