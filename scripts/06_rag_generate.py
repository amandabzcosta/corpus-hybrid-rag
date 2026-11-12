"""RAG, parte 2: executa a LLM nas condições A (sem RAG) e B (com RAG) com o MESMO modelo e parâmetros.
Gera results/rag_experiment.csv (notas e justificativas ficam vazias até a avaliação humana — script 07).

Exemplos:
  python scripts/06_rag_generate.py --backend hf --model Qwen/Qwen2.5-3B-Instruct            # Colab (GPU T4)
  python scripts/06_rag_generate.py --backend anthropic --model <modelo>                      # ANTHROPIC_API_KEY
  python scripts/06_rag_generate.py --backend openai --model gpt-4o-mini                      # OPENAI_API_KEY
  python scripts/06_rag_generate.py --backend openai --base-url http://localhost:11434/v1 --model qwen2.5:7b   # Ollama
  python scripts/06_rag_generate.py --backend manual --answers-file results/rag_answers_manual.jsonl --meta "modelo=..."
"""
from __future__ import annotations

import json

from _common import *  # noqa: F401,F403
import pandas as pd

from src.config import RESULTS_DIR
from src.rag import (AnthropicBackend, CRITERIA, HFBackend, ManualBackend, MockBackend, OpenAICompatBackend)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--backend", required=True, choices=["mock", "manual", "hf", "openai", "anthropic"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--answers-file", default=str(RESULTS_DIR / "rag_answers_manual.jsonl"))
    ap.add_argument("--meta", default="", help="metadados livres do backend manual, formato chave=valor;chave=valor")
    ap.add_argument("--out", default=str(RESULTS_DIR / "rag_experiment.csv"))
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

    rows = []
    for line in open(RESULTS_DIR / "rag_prompts.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if args.backend == "manual":
            llm.key = (p["query_id"], p["condition"])
        answer, meta = llm.generate(p["system"], p["user"])
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
