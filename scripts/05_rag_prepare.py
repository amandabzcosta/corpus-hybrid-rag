from __future__ import annotations

import json

from _common import *  # noqa: F401,F403
import pandas as pd

from src.analysis import rank_map, system_summary
from src.config import RAG_CONTEXT_K, RESULTS_DIR
from src.data import rag_query_ids
from src.rag import SYSTEM_A, SYSTEM_B, full_prompt, retrieval_at_k, user_prompt_a, user_prompt_b
from src.results_io import read_run

def main():
    args = base_parser('RAG, parte 1: contexto Top-5 pós-reranking, métricas @5, prompts A/B e rastreio de falhas.').parse_args()
    ctx = Ctx()
    qids = rag_query_ids(ctx.qrels["test"], ctx.queries)
    runs = {n: read_run(RESULTS_DIR / f"results_{n}.csv", sorted(ctx.qrels["test"])) for n in ("lexical", "semantic", "hybrid", "reranking")}
    rows, prompts, ctxs, traces = [], [], {}, []
    for q in qids:
        question, qrels_q = ctx.queries[q], ctx.qrels["test"][q]
        top = [d for d, _ in runs["reranking"][q][:RAG_CONTEXT_K]]
        docs = [(d, ctx.corpus[d]["title"], ctx.corpus[d]["text"]) for d in top]
        ctxs[q] = docs
        met = retrieval_at_k(top, qrels_q, RAG_CONTEXT_K)
        rows.append({"query_id": q, "query": question, "context_doc_ids": " ".join(top), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()},
                     "relevancias_no_contexto": " ".join(str(qrels_q.get(d, 0)) for d in top)})
        ua, ub = user_prompt_a(question), user_prompt_b(question, docs)
        prompts += [{"query_id": q, "query": question, "condition": "A", "context_doc_ids": "", "system": SYSTEM_A, "user": ua, "prompt": full_prompt(SYSTEM_A, ua)},
                    {"query_id": q, "query": question, "condition": "B", "context_doc_ids": " ".join(top), "system": SYSTEM_B, "user": ub, "prompt": full_prompt(SYSTEM_B, ub)}]
        stage = {n: system_summary(runs[n][q], qrels_q) for n in runs}
        traces.append({"query_id": q, "query": question, "n_relevantes": met["n_relevantes_total"], "contexto": [
            {"doc_id": d, "relevancia": int(qrels_q.get(d, 0)), "titulo": ctx.corpus[d]["title"],
             "rank_lexical": rank_map(runs["lexical"][q]).get(d), "rank_semantico": rank_map(runs["semantic"][q]).get(d),
             "rank_hibrido": rank_map(runs["hybrid"][q]).get(d), "rank_reranking": rank_map(runs["reranking"][q]).get(d)} for d in top],
            "por_etapa": stage})
    df = pd.DataFrame(rows)
    mean = {"query_id": "MÉDIA", "query": "", "context_doc_ids": "", **{c: round(float(df[c].mean()), 4) for c in ("Recall@5", "Precision@5", "Hit@5", "n_relevantes_no_contexto", "n_relevantes_total")}}
    pd.concat([df, pd.DataFrame([mean])], ignore_index=True).to_csv(RESULTS_DIR / "rag_retrieval_metrics.csv", index=False)
    with open(RESULTS_DIR / "rag_prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (RESULTS_DIR / "rag_contexts.json").write_text(json.dumps({q: [{"doc_id": d, "title": t, "text": x} for d, t, x in v] for q, v in ctxs.items()},
                                                              ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / "analysis").mkdir(exist_ok=True)
    (RESULTS_DIR / "analysis" / "rag_failure_traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Rastreio ponta a ponta das 10 consultas do RAG (onde estão os relevantes em cada etapa)", ""]
    for t in traces:
        md += [f"## {t['query_id']} — {t['query']}", f"Relevantes no qrels: {t['n_relevantes']}", ""]
        md += ["| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |", "|---|---|---|---|---|"]
        md += [f"| {n} | {s['primeiro_relevante']} | {s['relevantes_top5']} | {s['relevantes_top10']} | {s['relevantes_top100']} |" for n, s in t["por_etapa"].items()]
        md += ["", "| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |", "|---|---|---|---|---|---|---|"]
        md += [f"| {c['doc_id']} | {c['relevancia']} | {c['rank_lexical'] or '>100'} | {c['rank_semantico'] or '>100'} | {c['rank_hibrido'] or '>200'} | {c['rank_reranking']} | {c['titulo'][:70]} |" for c in t["contexto"]]
        md.append("")
    (RESULTS_DIR / "analysis" / "rag_failure_traces.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(pd.concat([df.drop(columns=["query", "context_doc_ids"]), pd.DataFrame([mean]).drop(columns=["query", "context_doc_ids"])]).to_string(index=False))

if __name__ == "__main__":
    main()
