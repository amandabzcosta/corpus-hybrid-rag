from __future__ import annotations

import time

from .inverted_index import BM25Searcher, InvertedIndex
from .preprocess import Preprocessor

class LexicalSystem:
    def __init__(self, doc_texts: dict[str, str], remove_stopwords: bool = True, stemmer: str | None = "porter",
                 k1: float = 1.2, b: float = 0.75) -> None:
        self.pre = Preprocessor(remove_stopwords=remove_stopwords, stemmer=stemmer)
        self.index = InvertedIndex.build(doc_texts.items(), self.pre)
        self.set_params(k1, b)

    def set_params(self, k1: float, b: float) -> None:
        self.k1, self.b = k1, b
        self.searcher = BM25Searcher(self.index, k1, b)

    def search(self, query_text: str, top_k: int = 100) -> list[tuple[str, float]]:
        hits = self.searcher.search(self.pre(query_text), top_k)
        ids = self.index.doc_ids
        return [(ids[d], s) for d, s in hits]

    def run(self, queries: dict[str, str], top_k: int = 100) -> dict[str, list[tuple[str, float]]]:
        return {qid: self.search(text, top_k) for qid, text in queries.items()}

    def timed_run(self, queries: dict[str, str], top_k: int = 100, warmup: int = 5):
        qs = list(queries.items())
        for _, text in qs[:warmup]:
            self.search(text, top_k)
        run, times = {}, []
        for qid, text in qs:
            t0 = time.perf_counter()
            run[qid] = self.search(text, top_k)
            times.append(time.perf_counter() - t0)
        return run, times
