"""Índice invertido implementado do zero (item 2) e ranqueamento BM25 (item 3).

Estruturas de dados
-------------------
postings : dict[str, (array('I'), array('I'))]
           termo -> (doc_idx ordenados, tf correspondentes); a posting list de um termo guarda TF por documento,
           e o DF do termo é o comprimento da lista.
doc_ids  : list[str]    doc_idx -> doc_id externo (ex.: "MED-10")
doc_len  : array('I')   |d| = nº de tokens do documento após o pré-processamento
Estatísticas globais: N, comprimento médio (avgdl), tamanho do vocabulário e total de postings.

Nenhuma consulta percorre a coleção: o ranking é calculado "termo a termo" (term-at-a-time), acumulando
scores apenas dos documentos que aparecem nas posting lists dos termos da consulta.
"""
from __future__ import annotations

import heapq
import math
import pickle
import time
from array import array
from collections import Counter
from typing import Callable, Iterable


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, tuple[array, array]] = {}
        self.doc_ids: list[str] = []
        self.doc_len: array = array("I")
        self.total_tokens: int = 0
        self.preprocess_seconds: float = 0.0   # tokenização + stopwords + stemming dos documentos
        self.structure_seconds: float = 0.0    # construção das posting lists

    # ---------- construção ----------
    def add_document(self, doc_id: str, tokens: list[str]) -> None:
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        self.doc_len.append(len(tokens))
        self.total_tokens += len(tokens)
        for term, tf in Counter(tokens).items():
            plist = self.postings.get(term)
            if plist is None:
                plist = self.postings[term] = (array("I"), array("I"))
            plist[0].append(idx)   # documentos entram em ordem crescente de idx -> lista já ordenada
            plist[1].append(tf)

    @classmethod
    def build(cls, docs: Iterable[tuple[str, str]], preprocess: Callable[[str], list[str]]) -> "InvertedIndex":
        index = cls()
        for doc_id, text in docs:
            t0 = time.perf_counter()
            tokens = preprocess(text)
            t1 = time.perf_counter()
            index.add_document(doc_id, tokens)
            index.preprocess_seconds += t1 - t0
            index.structure_seconds += time.perf_counter() - t1
        return index

    # ---------- consulta às estatísticas ----------
    @property
    def N(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        return self.total_tokens / self.N if self.N else 0.0

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    @property
    def num_postings(self) -> int:
        return sum(len(p[0]) for p in self.postings.values())

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        return len(p[0]) if p else 0

    def tf(self, term: str, doc_idx: int) -> int:
        """TF do termo no documento (busca binária na posting list)."""
        p = self.postings.get(term)
        if not p:
            return 0
        ids, tfs = p
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if ids[mid] < doc_idx:
                lo = mid + 1
            else:
                hi = mid
        return tfs[lo] if lo < len(ids) and ids[lo] == doc_idx else 0

    def stats(self) -> dict:
        return {
            "n_documentos": self.N, "vocabulario": self.vocab_size, "total_postings": self.num_postings,
            "total_tokens": self.total_tokens, "comprimento_medio_doc": round(self.avgdl, 2),
            "tempo_preprocessamento_s": round(self.preprocess_seconds, 3),
            "tempo_estruturas_indice_s": round(self.structure_seconds, 3),
            "tempo_construcao_total_s": round(self.preprocess_seconds + self.structure_seconds, 3),
        }

    # ---------- persistência ----------
    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path) -> "InvertedIndex":
        index = cls()
        with open(path, "rb") as f:
            index.__dict__.update(pickle.load(f))
        return index


class BM25Searcher:
    """BM25 (Robertson/Lucene):  score(q,d) = sum_t qtf * idf(t) * tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl)),
    com idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)). Empates são resolvidos pelo menor doc_idx."""

    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75) -> None:
        self.index, self.k1, self.b = index, k1, b
        avgdl = index.avgdl
        # normalização por documento: depende só de (k1, b, |d|); calculada uma vez no construtor
        self._norm = [k1 * (1.0 - b + b * dl / avgdl) for dl in index.doc_len]
        self._idf: dict[str, float] = {}

    def idf(self, term: str) -> float:
        v = self._idf.get(term)
        if v is None:
            df = self.index.df(term)
            v = self._idf[term] = math.log(1.0 + (self.index.N - df + 0.5) / (df + 0.5)) if df else 0.0
        return v

    def accumulate(self, terms: list[str]) -> dict[int, float]:
        acc: dict[int, float] = {}
        k1p1, norm = self.k1 + 1.0, self._norm
        for term, qtf in Counter(terms).items():
            plist = self.index.postings.get(term)
            if plist is None:
                continue
            w = qtf * self.idf(term) * k1p1
            for d, tf in zip(plist[0], plist[1]):
                acc[d] = acc.get(d, 0.0) + w * tf / (tf + norm[d])
        return acc

    def search(self, terms: list[str], top_k: int = 100) -> list[tuple[int, float]]:
        acc = self.accumulate(terms)
        return heapq.nsmallest(top_k, acc.items(), key=lambda kv: (-kv[1], kv[0]))

    def term_contribution(self, term: str, doc_idx: int) -> float:
        """Contribuição de um termo (qtf=1) ao score de um documento — usada nas análises por consulta."""
        tf = self.index.tf(term, doc_idx)
        return self.idf(term) * tf * (self.k1 + 1.0) / (tf + self._norm[doc_idx]) if tf else 0.0
