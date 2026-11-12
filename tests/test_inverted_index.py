import math
import random
from collections import Counter

from src.inverted_index import BM25Searcher, InvertedIndex
from src.preprocess import Preprocessor

DOCS = {
    "d1": "Probiotics may prevent the common cold in children.",
    "d2": "Cholesterol and statin drugs: do statins cause cancer?",
    "d3": "Probiotics, prebiotics and gut flora; probiotics again.",
    "d4": "Coconut milk, coconut oil and saturated fat.",
}

def build(**kw):
    pre = Preprocessor(**kw)
    return InvertedIndex.build(DOCS.items(), pre), pre

def brute_force_bm25(index_tokens, query_tokens, k1, b):
    N = len(index_tokens)
    avgdl = sum(len(t) for t in index_tokens.values()) / N
    df = Counter(t for toks in index_tokens.values() for t in set(toks))
    scores = {}
    for d, toks in index_tokens.items():
        tf = Counter(toks)
        s = 0.0
        for term, qtf in Counter(query_tokens).items():
            if tf[term]:
                idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
                s += qtf * idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(toks) / avgdl))
        if s > 0:
            scores[d] = s
    return scores

def test_posting_lists_and_stats():
    idx, pre = build(remove_stopwords=True, stemmer="porter")
    assert idx.N == 4 and idx.doc_ids == list(DOCS)
    term = pre("probiotics")[0]
    ids, tfs = idx.postings[term]
    assert list(ids) == [0, 2] and list(tfs) == [1, 2]            # d1 (tf=1) e d3 (tf=2), ordenadas por doc_idx
    assert idx.df(term) == 2 and idx.tf(term, 2) == 2 and idx.tf(term, 1) == 0
    assert idx.num_postings == sum(len(p[0]) for p in idx.postings.values())
    assert idx.total_tokens == sum(idx.doc_len)
    assert abs(idx.avgdl - idx.total_tokens / 4) < 1e-12

def test_bm25_matches_brute_force():
    idx, pre = build(remove_stopwords=True, stemmer="porter")
    tokens = {d: pre(t) for d, t in DOCS.items()}
    for q in ["probiotics common cold", "statins cancer cholesterol", "coconut coconut fat", "unknownterm"]:
        qt = pre(q)
        for k1, b in [(1.2, 0.75), (2.0, 0.5), (0.9, 1.0)]:
            got = {idx.doc_ids[d]: s for d, s in BM25Searcher(idx, k1, b).search(qt, 100)}
            ref = brute_force_bm25(tokens, qt, k1, b)
            assert got.keys() == ref.keys()
            assert all(abs(got[d] - ref[d]) < 1e-9 for d in ref)

def test_ranking_order_and_tie_break():
    idx, pre = build(remove_stopwords=True, stemmer="porter")
    hits = BM25Searcher(idx).search(pre("probiotics"), 10)
    assert [idx.doc_ids[d] for d, _ in hits][0] == "d3"           # tf maior -> primeiro
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)

def test_save_load_roundtrip(tmp_path):
    idx, _ = build()
    idx.save(tmp_path / "i.pkl")
    idx2 = InvertedIndex.load(tmp_path / "i.pkl")
    assert idx2.stats()["vocabulario"] == idx.stats()["vocabulario"] and idx2.doc_ids == idx.doc_ids
