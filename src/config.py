from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RI_DATA_DIR", ROOT / "data" / "nfcorpus"))
RESULTS_DIR = Path(os.environ.get("RI_RESULTS_DIR", ROOT / "results"))
CACHE_DIR = Path(os.environ.get("RI_CACHE_DIR", ROOT / "artifacts"))
CONFIGS_DIR = ROOT / "configs"

SEED = 42                           # semente fixa para o projeto
CUTOFF = 10                         # P@10, R@10, AP@10, NDCG@10
TOP_K = 100
RERANK_DEPTH = 50
RAG_N_QUERIES = 10
RAG_CONTEXT_K = 5

RRF_K_REQUIRED = [10, 60, 100]
RRF_K_GRID = [5, 10, 20, 30, 40, 60, 80, 100, 150, 200]

EXPECTED_RAG_IDS = ["PLAIN-68", "PLAIN-2450", "PLAIN-153", "PLAIN-2850", "PLAIN-2750",
                    "PLAIN-270", "PLAIN-2470", "PLAIN-2430", "PLAIN-238", "PLAIN-3342"]

EXPECTED_STATS = {
    "n_docs": 3633, "n_queries": 3237,
    "n_test_queries": 323, "n_dev_queries": 324, "n_train_queries": 2590,
    "n_test_judgments": 12334, "n_dev_judgments": 11385, "n_train_judgments": 110575,
    "n_test_question_queries": 42,
}

def models_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path or os.environ.get("RI_MODELS") or CONFIGS_DIR / "models_hf.json")
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_path"] = str(p)
    return cfg
