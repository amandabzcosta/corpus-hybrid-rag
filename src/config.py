"""Configuração central do projeto: caminhos, sementes e constantes definidas no enunciado do T2."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RI_DATA_DIR", ROOT / "data" / "nfcorpus"))
RESULTS_DIR = Path(os.environ.get("RI_RESULTS_DIR", ROOT / "results"))
CACHE_DIR = Path(os.environ.get("RI_CACHE_DIR", ROOT / "cache"))   # embeddings/scores em cache (regeneráveis; não são entregues)
CONFIGS_DIR = ROOT / "configs"

SEED = 42                           # semente única do projeto
CUTOFF = 10                         # P@10, R@10, AP@10, NDCG@10
TOP_K = 100                         # tamanho dos rankings salvos e fundidos (enunciado, itens 8 e 15)
RERANK_DEPTH = 50                   # documentos reordenados pelo cross-encoder (item 9)
RAG_N_QUERIES = 10
RAG_CONTEXT_K = 5                   # Top-5 após o reranking vai para a LLM (item 12)

# k do RRF avaliados no dev. O enunciado exige pelo menos {10, 60, 100}; os demais refinam a busca.
RRF_K_REQUIRED = [10, 60, 100]
RRF_K_GRID = [5, 10, 20, 30, 40, 60, 80, 100, 150, 200]

# As 10 consultas fixas do experimento de RAG (lista do enunciado; "PLAIN270" no PDF = PLAIN-270).
EXPECTED_RAG_IDS = ["PLAIN-68", "PLAIN-2450", "PLAIN-153", "PLAIN-2850", "PLAIN-2750",
                    "PLAIN-270", "PLAIN-2470", "PLAIN-2430", "PLAIN-238", "PLAIN-3342"]

# Estatísticas oficiais do NFCorpus/BEIR usadas para validar que a cópia local é a esperada.
EXPECTED_STATS = {
    "n_docs": 3633, "n_queries": 3237,
    "n_test_queries": 323, "n_dev_queries": 324, "n_train_queries": 2590,
    "n_test_judgments": 12334, "n_dev_judgments": 11385, "n_train_judgments": 110575,
    "n_test_question_queries": 42,
}


def models_config(path: str | os.PathLike | None = None) -> dict:
    """Lê o JSON com os modelos candidatos (bi-encoders e cross-encoders).

    Ordem de busca: argumento -> variável RI_MODELS -> configs/models_hf.json (Hugging Face, padrão).
    """
    p = Path(path or os.environ.get("RI_MODELS") or CONFIGS_DIR / "models_hf.json")
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = str(p)
    return cfg
