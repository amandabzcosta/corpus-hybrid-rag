from __future__ import annotations

import hashlib
import json

from _common import *  # noqa: F401,F403
import pandas as pd

from src.config import CACHE_DIR, RESULTS_DIR
from src.rag import (AnthropicBackend, CRITERIA, HFBackend, ManualBackend, MockBackend, OpenAICompatBackend)

CACHE_FILE = CACHE_DIR / "llm_cache.jsonl"

def main():
    ap = base_parser('RAG, parte 2: executa a LLM nas condições A (sem RAG) e B (com RAG) com o MESMO modelo e parâmetros.')
    ap.add_argument("--backend", required=True, choices=["mock", "manual", "hf", "openai", "anthropic"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--answers-file", default=str(RESULTS_DIR / "rag_answers_manual.jsonl"))
    ap.add_argument("--meta", default="", help="metadados livres do backend manual, formato chave=valor;chave=valor")
    ap.add_argument("--out", default=str(RESULTS_DIR / "rag_experiment.csv"))
    ap.add_argument("--no-cache", action="store_true", help="ignora as respostas guardadas em artifacts/llm_cache.jsonl")
    args = ap.parse_args()
    if args.backend == "mock":
        llm = MockBackend()
    elif args.backend == "manual":
        llm = ManualBackend(args.answers_file, dict(kv.split("=", 1) for kv in args.meta.split(";") if "=" in kv))
    elif args.backend == "hf":
        llm = HFBackend(args.model, args.max_tokens)
    elif args.backend == "openai":
        llm = OpenAICompatBackend(args.model, args.base_url, args.temperature, args.max_tokens)
    else:
        llm = AnthropicBackend(args.model, args.temperature, args.max_tokens)

    params = {k: v for k, v in llm.describe().items() if k in ("backend", "model", "temperature", "max_tokens", "seed")}
    cache = {}
    if CACHE_FILE.exists() and not args.no_cache:
        for line in open(CACHE_FILE, encoding="utf-8"):
            r = json.loads(line)
            cache.setdefault(r["key"], r["answer"])
    rows = []
    for line in open(RESULTS_DIR / "rag_prompts.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if args.backend == "manual":
            llm.key = (p["query_id"], p["condition"])
        key = hashlib.sha256(json.dumps([params, p["system"], p["user"]], sort_keys=True).encode()).hexdigest()
        if key in cache:
            answer = cache[key]
        else:
            answer, meta = llm.generate(p["system"], p["user"])
            if args.backend not in ("mock", "manual") and not args.no_cache:
                CACHE_FILE.parent.mkdir(exist_ok=True)
                with open(CACHE_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "query_id": p["query_id"], "condition": p["condition"], **params,
                                        "answer": answer}, ensure_ascii=False) + "\n")
        rows.append({"query_id": p["query_id"], "query": p["query"], "condition": p["condition"],
                     "context_doc_ids": p["context_doc_ids"], "prompt": p["prompt"], "answer": answer.strip(),
                     **{c: "" for c in CRITERIA}, **{f"justificativa_{c}": "" for c in CRITERIA}})
        print(p["query_id"], p["condition"], "->", answer.strip()[:90].replace("\n", " "), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    (RESULTS_DIR / "rag_llm_config.json").write_text(json.dumps(llm.describe(), ensure_ascii=False, indent=2), encoding="utf-8")
    print("salvo em", args.out, "|", llm.describe())

if __name__ == "__main__":
    main()
