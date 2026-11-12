from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CACHE_DIR, RESULTS_DIR, models_config  # noqa: E402
from src.data import doc_text, load_all, split_queries  # noqa: E402

SELECTION_PATH = RESULTS_DIR / "dev_selection.json"

class Ctx:

    def __init__(self) -> None:
        self.corpus, self.queries, self.qrels, self.info = load_all()
        self.doc_ids = list(self.corpus)
        self.doc_texts = [doc_text(self.corpus[d]) for d in self.doc_ids]
        self.docs = dict(zip(self.doc_ids, self.doc_texts))
        self.dev_queries = split_queries(self.queries, self.qrels["dev"])
        self.test_queries = split_queries(self.queries, self.qrels["test"])

def load_selection() -> dict:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8")) if SELECTION_PATH.exists() else {}

def save_selection(update: dict) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    sel = load_selection()
    sel.update(update)
    SELECTION_PATH.write_text(json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8")
    return sel

def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--models", default=None, help="JSON com os modelos candidatos (padrão: configs/models_hf.json "
                                                  "ou a variável RI_MODELS)")
    p.add_argument("--time-budget-min", type=float, default=None,
                   help="orçamento de tempo; ao esgotar, o progresso fica salvo em cache e basta rodar de novo")
    return p

def deadline_from(args) -> float | None:
    return time.time() + 60 * args.time_budget_min if getattr(args, "time_budget_min", None) else None

def find_spec(specs: list[dict], spec_id: str) -> dict:
    for s in specs:
        if s["id"] == spec_id:
            return s
    raise KeyError(f"modelo '{spec_id}' não está no JSON de modelos")

def load_models(args):
    return models_config(args.models)
