from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.numpy_bert import NumpyBert
from src.rag import (CRITERIA, make_blind_sheet, merge_grades, retrieval_at_k, user_prompt_a, user_prompt_b)

ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_at_5():
    qrels = {"d1": 2, "d2": 1, "d7": 1, "d8": 0}
    m = retrieval_at_k(["d1", "x", "d2", "y", "z", "d7"], qrels, 5)
    assert m["Recall@5"] == pytest.approx(2 / 3) and m["Precision@5"] == pytest.approx(2 / 5) and m["Hit@5"] == 1
    assert retrieval_at_k(["x", "y"], qrels, 5)["Hit@5"] == 0


def test_prompts_have_ids_only_in_condition_b():
    docs = [("MED-1", "Title one", "Text one."), ("MED-2", "Title two", "Text two.")]
    assert "[MED-1]" not in user_prompt_a("Q?") and "MED" not in user_prompt_a("Q?")
    b = user_prompt_b("Q?", docs)
    assert "[MED-1] Title one" in b and "[MED-2] Title two" in b and b.rstrip().endswith("Question: Q?")


def test_blind_sheet_and_merge_with_two_annotators():
    rag = pd.DataFrame([{"query_id": q, "query": f"{q}?", "condition": c, "answer": f"ans {q}{c}", "context_doc_ids": "MED-1"}
                        for q in ("Q1", "Q2") for c in ("A", "B")])
    contexts = {q: [("MED-1", "t", "x")] for q in ("Q1", "Q2")}
    sheet, key = make_blind_sheet(rag, contexts, seed=42)
    assert "condition" not in sheet.columns and len(sheet) == 4 and set(key["condition"]) == {"A", "B"}
    s1, s2 = sheet.copy(), sheet.copy()
    for c in CRITERIA:
        s1[c], s2[c] = "2", "1"
    s1["just_fidelidade"] = "ok"
    merged, means, agree = merge_grades(rag, key, [s1, s2])
    assert merged.loc[merged.condition == "A", "rastreabilidade"].isna().all()       # N/A na condição A
    assert means.loc["B", "fidelidade"] == pytest.approx(1.5)
    assert agree["fidelidade"]["concordancia_exata"] == 0.0


def test_numpy_bert_random_weights_shapes():
    rng = np.random.default_rng(0)
    H, I, V = 16, 32, 50
    t = {"embeddings.word_embeddings.weight": rng.normal(size=(V, H)), "embeddings.position_embeddings.weight": rng.normal(size=(20, H)),
         "embeddings.token_type_embeddings.weight": rng.normal(size=(2, H)), "embeddings.LayerNorm.weight": np.ones(H),
         "embeddings.LayerNorm.bias": np.zeros(H), "pooler.dense.weight": rng.normal(size=(H, H)), "pooler.dense.bias": np.zeros(H),
         "classifier.weight": rng.normal(size=(1, H)), "classifier.bias": np.zeros(1)}
    for i in range(2):
        p = f"encoder.layer.{i}."
        for n in ("query", "key", "value"):
            t[p + f"attention.self.{n}.weight"], t[p + f"attention.self.{n}.bias"] = rng.normal(size=(H, H)) * 0.1, np.zeros(H)
        t[p + "attention.output.dense.weight"], t[p + "attention.output.dense.bias"] = rng.normal(size=(H, H)) * 0.1, np.zeros(H)
        t[p + "attention.output.LayerNorm.weight"], t[p + "attention.output.LayerNorm.bias"] = np.ones(H), np.zeros(H)
        t[p + "intermediate.dense.weight"], t[p + "intermediate.dense.bias"] = rng.normal(size=(I, H)) * 0.1, np.zeros(I)
        t[p + "output.dense.weight"], t[p + "output.dense.bias"] = rng.normal(size=(H, I)) * 0.1, np.zeros(H)
        t[p + "output.LayerNorm.weight"], t[p + "output.LayerNorm.bias"] = np.ones(H), np.zeros(H)
    net = NumpyBert({k: v.astype(np.float32) for k, v in t.items()}, heads=4)
    ids = rng.integers(0, V, (3, 7))
    mask = np.ones((3, 7), np.int64)
    mask[1, 5:] = 0
    h = net.forward(ids, mask)
    assert h.shape == (3, 7, H) and net.classify(h).shape == (3,)
    # padding não pode influenciar os tokens reais: o resultado da sequência 1 não depende dos tokens mascarados
    ids2 = ids.copy()
    ids2[1, 5:] = (ids2[1, 5:] + 1) % V
    assert np.allclose(h[1, :5], net.forward(ids2, mask)[1, :5], atol=1e-5)


@pytest.mark.skipif(not (ROOT / "models" / "ce_l6" / "model.safetensors").exists(), reason="pesos locais ausentes")
def test_numpy_bert_reproduces_model_card_scores():
    """Valor de referência do card do cross-encoder ms-marco-MiniLM-L6-v2: [8.607138, -4.320078]."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "models/ce_l6/tokenizer.json"))
    tok.enable_truncation(max_length=512)
    q = "How many people live in Berlin?"
    ps = ["Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
          "Berlin is well known for its museums."]
    enc = tok.encode_batch([(q, p) for p in ps])
    L = max(len(e.ids) for e in enc)
    ids, am, tt = (np.zeros((2, L), np.int64) for _ in range(3))
    for i, e in enumerate(enc):
        ids[i, :len(e.ids)], am[i, :len(e.ids)], tt[i, :len(e.ids)] = e.ids, 1, e.type_ids
    net = NumpyBert.from_safetensors(ROOT / "models/ce_l6/model.safetensors")
    assert np.allclose(net.classify(net.forward(ids, am, tt)), [8.607138, -4.320078], atol=1e-3)
