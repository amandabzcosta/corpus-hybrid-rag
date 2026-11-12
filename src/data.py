from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import DATA_DIR, EXPECTED_RAG_IDS, EXPECTED_STATS, RAG_N_QUERIES

def _jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def doc_text(doc: dict) -> str:
    return f"{doc.get('title', '')} {doc.get('text', '')}".strip()

def load_corpus(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    return {d["_id"]: {"title": d.get("title", ""), "text": d.get("text", "")}
            for d in _jsonl(Path(data_dir) / "corpus.jsonl")}

def load_queries(data_dir: Path = DATA_DIR) -> dict[str, str]:
    return {q["_id"]: q["text"] for q in _jsonl(Path(data_dir) / "queries.jsonl")}

def load_qrels(split: str, data_dir: Path = DATA_DIR) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(Path(data_dir) / "qrels" / f"{split}.tsv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # cabeçalho: query-id, corpus-id, score
        for row in reader:
            if len(row) >= 3:
                qrels[row[0].strip()][row[1].strip()] = int(row[2])
    return dict(qrels)

def split_queries(queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> dict[str, str]:
    return {q: queries[q] for q in sorted(qrels)}

def rag_query_ids(test_qrels: dict, queries: dict[str, str]) -> list[str]:
    ids = [q for q in sorted(test_qrels.keys()) if queries[q].strip().endswith("?")]
    return random.Random(42).sample(ids, RAG_N_QUERIES)

def validate_dataset(corpus, queries, qrels_by_split, strict: bool = True) -> dict:
    s = EXPECTED_STATS
    got = {
        "n_docs": len(corpus), "n_queries": len(queries),
        "n_test_queries": len(qrels_by_split["test"]), "n_dev_queries": len(qrels_by_split["dev"]),
        "n_train_queries": len(qrels_by_split["train"]),
        "n_test_judgments": sum(len(v) for v in qrels_by_split["test"].values()),
        "n_dev_judgments": sum(len(v) for v in qrels_by_split["dev"].values()),
        "n_train_judgments": sum(len(v) for v in qrels_by_split["train"].values()),
        "n_test_question_queries": sum(queries[q].strip().endswith("?") for q in qrels_by_split["test"]),
    }
    rag = rag_query_ids(qrels_by_split["test"], queries)
    problems = [f"{k}: esperado {s[k]}, obtido {got[k]}" for k in s if got[k] != s[k]]
    if rag != EXPECTED_RAG_IDS:
        problems.append(f"consultas do RAG diferem do enunciado: {rag}")
    if problems and strict:
        raise AssertionError("Dataset diferente do esperado:\n  " + "\n  ".join(problems))
    return {"stats": got, "rag_ids": rag, "problems": problems}

def load_all(data_dir: Path = DATA_DIR, validate: bool = True):
    corpus = load_corpus(data_dir)
    queries = load_queries(data_dir)
    qrels = {sp: load_qrels(sp, data_dir) for sp in ("train", "dev", "test")}
    info = validate_dataset(corpus, queries, qrels, strict=validate)
    return corpus, queries, qrels, info
