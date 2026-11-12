from __future__ import annotations

import json
import platform
import statistics
import time

from _common import *  # noqa: F401,F403
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import CACHE_DIR, RERANK_DEPTH, RESULTS_DIR, TOP_K
from src.data import rag_query_ids
from src.encoders import CachedCrossEncoder, TimeBudgetExceeded, load_cross_encoder
from src.fusion import rrf
from src.lexical import LexicalSystem
from src.metrics import evaluate_run, mean_metrics, validate_with_ir_measures
from src.rerank import rerank_top_n
from src.results_io import write_metrics_per_query, write_ranking_csv
from src.semantic import SemanticSystem
from src.encoders import load_bi_encoder

def environment() -> dict:
    import os
    import subprocess
    import scipy
    info = {"python": platform.python_version(), "so": platform.platform(), "cpu_cores": os.cpu_count(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
    for lib in ("onnxruntime", "tokenizers", "nltk", "sentence_transformers", "transformers", "torch", "ir_measures"):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:
            pass
    ps = lambda cmd: subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()
    try:
        if platform.system() == "Windows":
            info["cpu_model"] = ps("(Get-CimInstance Win32_Processor).Name")
            info["ram_gb"] = round(int(ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")) / 2**30, 1)
            info["gpu"] = ps("(Get-CimInstance Win32_VideoController).Name -join '; '")
        else:
            out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
            info["cpu_model"] = next((l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")), "n/d")
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except Exception:
        pass
    info["dispositivo_modelos"] = "CPU (torch CPU)"
    return info

def main():
    ap = base_parser('Pipeline completo no TEST com as escolhas do DEV: lexical, semântico, híbrido e reranking.')
    ap.add_argument("--limit", type=int, default=None, help="só para testes de fumaça: usa as N primeiras consultas")
    args = ap.parse_args()
    ctx, mc, sel, deadline = Ctx(), load_models(args), load_selection(), deadline_from(args)
    qrels, queries = ctx.qrels["test"], ctx.test_queries
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    qids = sorted(queries)
    pre, bm = sel["preprocessing"], sel["bm25"]
    timings: dict = {"selecao_dev": sel}

    # ---------------- lexical ----------------
    t0 = time.perf_counter()
    lex = LexicalSystem(ctx.docs, pre["remove_stopwords"], pre["stemmer"], bm["k1"], bm["b"])
    timings["indice"] = {**lex.index.stats(), "tempo_construcao_wall_s": round(time.perf_counter() - t0, 3),
                         "preprocessamento": lex.pre.describe(), "bm25": bm}
    run_lex, t_lex = lex.timed_run(queries, TOP_K)
    timings["lexical_ms_por_consulta"] = {"media": 1000 * statistics.mean(t_lex), "mediana": 1000 * statistics.median(t_lex),
                                          "max": 1000 * max(t_lex)}
    print("lexical ok", flush=True)

    # ---------------- semântico ----------------
    spec = find_spec(mc["bi_encoders"], sel["bi_encoder"]["id"])
    sem = SemanticSystem(load_bi_encoder(spec), ctx.doc_ids, ctx.doc_texts, spec["id"].replace("@", "_"), deadline=deadline)
    timings["semantico_docs"] = sem.build()
    run_sem, t_sem = sem.timed_run(queries, TOP_K)
    timings["semantico_ms_por_consulta"] = {"media": 1000 * statistics.mean(t_sem), "mediana": 1000 * statistics.median(t_sem),
                                            "max": 1000 * max(t_sem)}
    print("semântico ok", flush=True)

    # ---------------- híbrido (RRF) ----------------
    k_rrf = sel["rrf_k"]
    run_hyb, t_hyb = {}, []
    for q in qids:
        t0 = time.perf_counter()
        run_hyb[q] = rrf([run_lex[q], run_sem[q]], k=k_rrf, depth=TOP_K)
        t_hyb.append(time.perf_counter() - t0)
    timings["fusao_rrf_ms_por_consulta"] = {"media": 1000 * statistics.mean(t_hyb), "k": k_rrf}

    # ---------------- reranking ----------------
    ce_spec = find_spec(mc["cross_encoders"], sel["cross_encoder"]["id"])
    ce = CachedCrossEncoder(load_cross_encoder(ce_spec), CACHE_DIR / f"ce_{ce_spec['id']}_test.tsv", deadline)
    pairs_all = [(queries[q], ctx.docs[d]) for q in qids for d, _ in run_hyb[q][:RERANK_DEPTH]]
    plen = ce.pair_token_lengths(pairs_all)
    L = ce.max_length
    timings["reranker"] = {"modelo": ce.info(), "pares": len(pairs_all), "pares_truncados": int((plen > L).sum()),
                           "pct_pares_truncados": round(float((plen > L).mean() * 100), 2),
                           "tokens_par_media": round(float(plen.mean()), 1), "tokens_par_p95": float(np.percentile(plen, 95)),
                           "profundidade": RERANK_DEPTH}
    # as 10 consultas do RAG são reordenadas primeiro (seus escores ficam logo em cache); o restante segue a ordem dos ids
    rag_first = set(rag_query_ids(ctx.qrels["test"], ctx.queries))
    order = [q for q in qids if q in rag_first] + [q for q in qids if q not in rag_first]
    try:
        done = {}
        for i, q in enumerate(order):
            done[q] = rerank_top_n(queries[q], run_hyb[q], ctx.docs, ce, RERANK_DEPTH)
            if (i + 1) % 25 == 0:
                print(f"  reranking {i + 1}/{len(order)}  (cache: {len(ce.cache)} pares)", flush=True)
        run_rr = {q: done[q] for q in qids}
    except TimeBudgetExceeded as e:
        print("\n[!]", e, f"-> {len(ce.cache)} pares em cache; rode o mesmo comando para continuar.")
        raise SystemExit(3)
    spp = ce.seconds_per_pair()
    timings["reranker"].update({"segundos_por_par": spp, "ms_por_consulta_50_pares": 1000 * spp * RERANK_DEPTH,
                                "pares_calculados_no_total": ce.stats["pairs"]})
    print("reranking ok", flush=True)

    # ---------------- saídas padronizadas ----------------
    runs = {"lexical": run_lex, "semantic": run_sem, "hybrid": run_hyb, "reranking": run_rr}
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, run in runs.items():
        write_ranking_csv(RESULTS_DIR / f"results_{name}.csv", run, qrels, TOP_K, qids,
                          depth_marker=RERANK_DEPTH if name == "reranking" else None)
    per_model = {"lexical": evaluate_run(run_lex, qrels, qids), "semantic": evaluate_run(run_sem, qrels, qids),
                 "hybrid": evaluate_run(run_hyb, qrels, qids), "hybrid_rerank": evaluate_run(run_rr, qrels, qids)}
    mpq = write_metrics_per_query(RESULTS_DIR / "metrics_per_query.csv", per_model)
    summary = pd.DataFrame([{"model": m, **{k: round(v, 4) for k, v in mean_metrics(df).items()}} for m, df in per_model.items()])
    summary = summary.rename(columns={"AP10": "MAP10"})
    summary.to_csv(RESULTS_DIR / "summary_test.csv", index=False)
    print("\n", summary.to_string(index=False))

    # conferência das métricas com a ir_measures (todas as 4 execuções)
    worst = {}
    for (name, run), (mname, df) in zip(runs.items(), per_model.items()):
        worst[mname] = {k: float(v) for k, v in validate_with_ir_measures(run, qrels, df).items()}
    timings["validacao_ir_measures_max_abs_diff"] = worst

    # significância (Wilcoxon pareado sobre NDCG@10 por consulta)
    nd = {m: df.set_index("query_id")["NDCG10"] for m, df in per_model.items()}
    rows = []
    for a, b in [("semantic", "lexical"), ("hybrid", "lexical"), ("hybrid", "semantic"), ("hybrid_rerank", "hybrid"),
                 ("hybrid_rerank", "lexical"), ("hybrid_rerank", "semantic")]:
        d = (nd[a] - nd[b]).loc[qids]
        stat = wilcoxon(d, zero_method="wilcox") if (d != 0).any() else None
        rows.append({"A": a, "B": b, "dif_media_NDCG10": round(float(d.mean()), 4), "A_melhor": int((d > 0).sum()),
                     "B_melhor": int((d < 0).sum()), "empates": int((d == 0).sum()),
                     "p_wilcoxon": float(stat.pvalue) if stat is not None else None})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    timings["ambiente"] = environment()
    (RESULTS_DIR / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\nResultados em", RESULTS_DIR)

if __name__ == "__main__":
    main()
