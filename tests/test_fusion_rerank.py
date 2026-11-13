import pytest

from src.fusion import rrf
from src.rerank import rerank_top_n

def test_rrf_hand_computed():
    out = dict(rrf([["a", "b", "c"], ["b", "d", "a"]], k=60))
    assert out["a"] == pytest.approx(1 / 61 + 1 / 63)                # rank 1 e rank 3
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)                # rank 2 e rank 1
    assert out["c"] == pytest.approx(1 / 63)                         # ausente do 2º ranking: sem contribuição
    assert out["d"] == pytest.approx(1 / 62)
    assert [d for d, _ in rrf([["a", "b", "c"], ["b", "d", "a"]], k=60)] == ["b", "a", "d", "c"]

def test_rrf_depth_truncates_each_ranking():
    out = dict(rrf([["a", "b", "c"], ["c", "b", "a"]], k=10, depth=2))      # só os 2 primeiros de cada ranking
    assert out["a"] == pytest.approx(1 / 11)                               # a: rank 1 no 1º ranking; fora do corte no 2º
    assert out["c"] == pytest.approx(1 / 11)                               # c: rank 1 no 2º ranking; fora do corte no 1º
    assert out["b"] == pytest.approx(2 / 12)

class FakeCE:
    def predict(self, pairs, batch_size=32):
        return [float(len(d)) for _, d in pairs]

def test_rerank_reorders_top_n_and_keeps_tail_order():
    hybrid = [(f"d{i}", 1.0 / (i + 1)) for i in range(6)]
    texts = {"d0": "aa", "d1": "aaaa", "d2": "a", "d3": "aaaaaa", "d4": "aaa", "d5": "aaaaa"}
    out = rerank_top_n("q", hybrid, texts, FakeCE(), depth=4)
    assert [d for d, _ in out[:4]] == ["d3", "d1", "d0", "d2"]
    assert [d for d, _ in out[4:]] == ["d4", "d5"]
    assert out[4][1] == hybrid[4][1]

def test_rerank_ties_keep_hybrid_order():
    hybrid = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    out = rerank_top_n("q", hybrid, {"a": "xx", "b": "xx", "c": "xx"}, FakeCE(), depth=3)
    assert [d for d, _ in out] == ["a", "b", "c"]
