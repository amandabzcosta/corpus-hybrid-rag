from __future__ import annotations

import random

from _common import *  # noqa: F401,F403
import numpy as np
import pandas as pd

from src.config import CACHE_DIR, RERANK_DEPTH, RESULTS_DIR, RRF_K_GRID, SEED, TOP_K
from src.encoders import CachedCrossEncoder, TimeBudgetExceeded, load_bi_encoder, load_cross_encoder
from src.fusion import rrf
from src.lexical import LexicalSystem
from src.metrics import evaluate_run, mean_metrics
from src.preprocess import english_stopwords, raw_tokens
from src.rerank import rerank_top_n
from src.semantic import SemanticSystem
from src.spelling import SpellCorrector, build_dictionary
from src.typo_experiment import correct_query, make_typo_cases, recovery

PRE_CONFIGS = [(False, None), (True, None), (False, "porter"), (True, "porter"), (True, "snowball")]

def ndcg_of(run, qrels):
    return float(evaluate_run(run, qrels)["NDCG10"].mean())

def stage_lexical(ctx):
    qrels = ctx.qrels["dev"]
    rows = []
    for stop, stem in PRE_CONFIGS:
        lex = LexicalSystem(ctx.docs, stop, stem, 1.2, 0.75)
        m = mean_metrics(evaluate_run(lex.run(ctx.dev_queries), qrels))
        rows.append({"stopwords": stop, "stemmer": stem or "none", "vocab": lex.index.vocab_size,
                     "postings": lex.index.num_postings, **{k: round(v, 4) for k, v in m.items()}})
    pre = pd.DataFrame(rows)
    pre.to_csv(RESULTS_DIR / "dev_preprocessing.csv", index=False)
    print(pre.to_string(index=False))
    best = pre.sort_values(["NDCG10", "AP10"], ascending=False).iloc[0]
    stop, stem = bool(best["stopwords"]), (None if best["stemmer"] == "none" else best["stemmer"])
    lex = LexicalSystem(ctx.docs, stop, stem, 1.2, 0.75)
    grid = []
    for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
        for b in (0.25, 0.5, 0.75, 0.9, 1.0):
            lex.set_params(k1, b)
            m = mean_metrics(evaluate_run(lex.run(ctx.dev_queries), qrels))
            grid.append({"k1": k1, "b": b, **{k: round(v, 4) for k, v in m.items()}})
    grid = pd.DataFrame(grid)
    grid.to_csv(RESULTS_DIR / "dev_bm25_grid.csv", index=False)
    top = grid.sort_values(["NDCG10", "AP10"], ascending=False).iloc[0]
    print(grid.pivot(index="k1", columns="b", values="NDCG10").to_string())
    print("-> melhor:", dict(top))
    save_selection({"preprocessing": {"remove_stopwords": stop, "stemmer": stem},
                    "bm25": {"k1": float(top["k1"]), "b": float(top["b"])},
                    "dev_lexical_ndcg10": float(top["NDCG10"])})

def lexical_from_selection(ctx):
    sel = load_selection()
    pre, bm = sel["preprocessing"], sel["bm25"]
    return LexicalSystem(ctx.docs, pre["remove_stopwords"], pre["stemmer"], bm["k1"], bm["b"])

def stage_spelling(ctx):
    lex = lexical_from_selection(ctx)
    qrels = ctx.qrels["dev"]
    dictionary = build_dictionary(ctx.doc_texts)
    stop = english_stopwords()
    base = evaluate_run(lex.run(ctx.dev_queries), qrels).set_index("query_id")["NDCG10"]
    eligible = [q for q in sorted(ctx.dev_queries) if base[q] > 0]
    cases = make_typo_cases(eligible, ctx.dev_queries, lex, dictionary, stop, seed=SEED)
    print(f"dev: {len(cases)} consultas com erro gerado (de {len(eligible)} elegíveis)")
    err_run = {c["query_id"]: lex.search(c["query_erro"], TOP_K) for c in cases}
    err_ndcg = evaluate_run(err_run, qrels, [c["query_id"] for c in cases]).set_index("query_id")["NDCG10"]
    rows = []
    for k in (2, 3):
        for thr in (0.2, 0.3, 0.4, 0.5):
            for max_edit in (1, 2, 3):
                cor = SpellCorrector(dictionary, k=k, jaccard_threshold=thr, max_edit=max_edit)
                fixed, rec, run = 0, [], {}
                for c in cases:
                    text, changes = correct_query(c["query_erro"], cor)
                    fixed += changes.get(c["termo_erro"]) == c["termo_original"]
                    run[c["query_id"]] = lex.search(text, TOP_K)
                corr = evaluate_run(run, qrels, [c["query_id"] for c in cases]).set_index("query_id")["NDCG10"]
                for c in cases:
                    r = recovery(base[c["query_id"]], err_ndcg[c["query_id"]], corr[c["query_id"]])
                    if r is not None:
                        rec.append(r)
                rows.append({"k": k, "jaccard": thr, "max_edit": max_edit, "n": len(cases),
                             "acerto_termo": round(fixed / len(cases), 4),
                             "ndcg10_original": round(float(base[[c["query_id"] for c in cases]].mean()), 4),
                             "ndcg10_erro": round(float(err_ndcg.mean()), 4), "ndcg10_corrigida": round(float(corr.mean()), 4),
                             "recuperacao_media": round(float(np.mean(rec)), 4) if rec else None})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dev_spelling.csv", index=False)
    print(df.to_string(index=False))
    # empates: preferir a configuração mais conservadora (maior limiar de Jaccard, menor distância de edição, menor k)
    best = df.sort_values(["ndcg10_corrigida", "acerto_termo", "jaccard", "max_edit", "k"], ascending=[False, False, False, True, True]).iloc[0]
    save_selection({"spelling": {"k": int(best["k"]), "jaccard": float(best["jaccard"]), "max_edit": int(best["max_edit"])}})
    print("-> melhor:", dict(best))

def semantic_system(ctx, spec, deadline=None):
    return SemanticSystem(load_bi_encoder(spec), ctx.doc_ids, ctx.doc_texts, spec["id"].replace("@", "_"),
                          deadline=deadline)

def stage_semantic(ctx, mc, deadline):
    rows, infos = [], {}
    for spec in mc["bi_encoders"]:
        sem = semantic_system(ctx, spec, deadline)
        info = sem.build()
        run = sem.run(ctx.dev_queries, TOP_K)
        m = mean_metrics(evaluate_run(run, ctx.qrels["dev"]))
        rows.append({"modelo": spec["id"], "max_seq_length": info["max_seq_length"], "pct_docs_truncados": info["pct_docs_truncated"],
                     "tempo_emb_docs_s": info["embedding_seconds"], **{k: round(v, 4) for k, v in m.items()}})
        infos[spec["id"]] = info
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dev_embeddings.csv", index=False)
    best = df.sort_values(["NDCG10", "AP10"], ascending=False).iloc[0]
    save_selection({"bi_encoder": {"id": best["modelo"]}, "dev_semantic_ndcg10": float(best["NDCG10"])})
    print("-> melhor:", best["modelo"])

def stage_rrf(ctx, mc, deadline):
    sel = load_selection()
    lex = lexical_from_selection(ctx)
    sem = semantic_system(ctx, find_spec(mc["bi_encoders"], sel["bi_encoder"]["id"]), deadline)
    sem.build()
    lrun, srun = lex.run(ctx.dev_queries, TOP_K), sem.run(ctx.dev_queries, TOP_K)
    rows = [{"sistema": "lexical", "k": None, "NDCG10": round(ndcg_of(lrun, ctx.qrels["dev"]), 4)},
            {"sistema": "semantico", "k": None, "NDCG10": round(ndcg_of(srun, ctx.qrels["dev"]), 4)}]
    for k in RRF_K_GRID:
        run = {q: rrf([lrun[q], srun[q]], k=k, depth=TOP_K) for q in ctx.dev_queries}
        m = mean_metrics(evaluate_run(run, ctx.qrels["dev"]))
        rows.append({"sistema": "hibrido_rrf", "k": k, **{kk: round(v, 4) for kk, v in m.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dev_rrf.csv", index=False)
    print(df.to_string(index=False))
    h = df[df.sistema == "hibrido_rrf"]
    best = h.sort_values(["NDCG10", "k"], ascending=[False, True]).iloc[0]
    save_selection({"rrf_k": int(best["k"]), "dev_hybrid_ndcg10": float(best["NDCG10"])})
    print("-> melhor k:", int(best["k"]))

def stage_rerank(ctx, mc, deadline, n_queries: int):
    sel = load_selection()
    lex = lexical_from_selection(ctx)
    sem = semantic_system(ctx, find_spec(mc["bi_encoders"], sel["bi_encoder"]["id"]), deadline)
    sem.build()
    sample = sorted(random.Random(SEED).sample(sorted(ctx.dev_queries), n_queries))
    qs = {q: ctx.dev_queries[q] for q in sample}
    lrun, srun = lex.run(qs, TOP_K), sem.run(qs, TOP_K)
    hyb = {q: rrf([lrun[q], srun[q]], k=sel["rrf_k"], depth=TOP_K) for q in qs}
    qrels = ctx.qrels["dev"]
    rows = [{"modelo": "hibrido (sem reranking)", "n_consultas": len(qs), **{k: round(v, 4) for k, v in mean_metrics(evaluate_run(hyb, qrels, sample)).items()}}]
    for spec in mc["cross_encoders"]:
        ce = CachedCrossEncoder(load_cross_encoder(spec), CACHE_DIR / f"ce_{spec['id']}_dev.tsv", deadline)
        t0 = time.perf_counter()
        run = {q: rerank_top_n(qs[q], hyb[q], ctx.docs, ce, RERANK_DEPTH) for q in sample}
        stats = ce.cache_stats() if hasattr(ce, "cache_stats") else {}
        m = mean_metrics(evaluate_run(run, qrels, sample))
        rows.append({"modelo": spec["id"], "n_consultas": len(qs), **{k: round(v, 4) for k, v in m.items()}})
        print(rows[-1], f"(tempo desta execução: {time.perf_counter() - t0:.0f}s; pares novos: {ce.n_computed})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "dev_reranker.csv", index=False)
    print(df.to_string(index=False))
    best = df[df.modelo != "hibrido (sem reranking)"].sort_values(["NDCG10", "AP10"], ascending=False).iloc[0]
    save_selection({"cross_encoder": {"id": best["modelo"]}, "dev_rerank_queries": sample})
    print("-> melhor:", best["modelo"])

def main():
    ap = base_parser('Ajustes no conjunto DEV (o test nunca é usado para escolher nada).')
    ap.add_argument("--stage", default="all", choices=["all", "lexical", "spelling", "semantic", "rrf", "rerank"])
    ap.add_argument("--rerank-dev-queries", type=int, default=40, help="nº de consultas do dev usadas para escolher o cross-encoder")
    args = ap.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    ctx, mc, deadline = Ctx(), load_models(args), deadline_from(args)
    try:
        for st in (["lexical", "spelling", "semantic", "rrf", "rerank"] if args.stage == "all" else [args.stage]):
            print(f"\n===== estágio: {st} =====", flush=True)
            if st == "lexical":
                stage_lexical(ctx)
            elif st == "spelling":
                stage_spelling(ctx)
            elif st == "semantic":
                stage_semantic(ctx, mc, deadline)
            elif st == "rrf":
                stage_rrf(ctx, mc, deadline)
            else:
                stage_rerank(ctx, mc, deadline, args.rerank_dev_queries)
    except TimeBudgetExceeded as e:
        print("\n[!]", e, "-> rode o mesmo comando para continuar.")
        raise SystemExit(3)

if __name__ == "__main__":
    main()
