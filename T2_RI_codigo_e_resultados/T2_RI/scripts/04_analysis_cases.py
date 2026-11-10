"""Análises por consulta (itens 7 e 10): casos lexical x semântico e casos de reranking, com evidências numéricas.
Seleção dos casos 100% automática, a partir de results/metrics_per_query.csv (nenhuma escolha manual).
Saídas em results/analysis/: lex_vs_sem_cases.{md,json}, rerank_cases.{md,json}, global_evidence.json."""
from __future__ import annotations

import json

from _common import *  # noqa: F401,F403
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis import (lex_sem_case, lex_sem_markdown, query_stems, relevance_coverage, rerank_case, rerank_markdown,
                          top10_displacement)
from src.config import RESULTS_DIR
from src.lexical import LexicalSystem
from src.results_io import read_run

OUT = RESULTS_DIR / "analysis"


def main():
    args = base_parser(__doc__).parse_args()
    ctx, sel = Ctx(), load_selection()
    OUT.mkdir(exist_ok=True)
    pre, bm = sel["preprocessing"], sel["bm25"]
    lex = LexicalSystem(ctx.docs, pre["remove_stopwords"], pre["stemmer"], bm["k1"], bm["b"])
    id2idx = {d: i for i, d in enumerate(lex.index.doc_ids)}
    runs = {n: read_run(RESULTS_DIR / f"results_{n}.csv", sorted(ctx.qrels["test"])) for n in ("lexical", "semantic", "hybrid", "reranking")}
    mpq = pd.read_csv(RESULTS_DIR / "metrics_per_query.csv")
    nd = mpq.pivot(index="query_id", columns="model", values="NDCG10")
    qrels, queries = ctx.qrels["test"], ctx.test_queries

    # ---------- item 7: 3 consultas com maior vantagem do semântico e 3 do lexical ----------
    delta = (nd["semantic"] - nd["lexical"])
    order = sorted(delta.index, key=lambda q: (-delta[q], q))
    pick = [("Vantagem do semântico", q) for q in order[:3]] + [("Vantagem do lexical", q) for q in sorted(delta.index, key=lambda q: (delta[q], q))[:3]]
    cases, md = [], ["# Lexical x Semântico — seis consultas com maior diferença de NDCG@10 (test)", ""]
    for i, (label, q) in enumerate(pick, start=1):
        m = {"lexical": float(nd.loc[q, "lexical"]), "semantico": float(nd.loc[q, "semantic"]), "delta": float(delta[q])}
        case = lex_sem_case(q, queries[q], runs["lexical"][q], runs["semantic"][q], qrels[q], lex, ctx.corpus, id2idx, m)
        case["grupo"] = label
        cases.append(case)
        md.append(lex_sem_markdown(case, f"Caso {i} ({label})"))
    (OUT / "lex_vs_sem_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "lex_vs_sem_cases.md").write_text("\n".join(md), encoding="utf-8")

    # ---------- item 10: dois maiores ganhos, um caso com pouca/nenhuma alteração e um caso de piora ----------
    dr = (nd["hybrid_rerank"] - nd["hybrid"])
    gains = sorted(dr.index, key=lambda q: (-dr[q], q))[:2]
    worst = sorted(dr.index, key=lambda q: (dr[q], q))[0]
    disp = {q: top10_displacement(runs["hybrid"][q], runs["reranking"][q]) for q in dr.index}
    zero = [q for q in dr.index if abs(dr[q]) < 1e-12]
    pool = zero if zero else [min(dr.index, key=lambda q: (abs(dr[q]), q))]
    least = min(pool, key=lambda q: (disp[q]["posicoes_diferentes_top10"], disp[q]["deslocamento_medio_top10"], q))
    rcases, md = [], ["# Reranking — casos selecionados (test)", ""]
    for label, q in [("Maior ganho 1", gains[0]), ("Maior ganho 2", gains[1]), ("Pouca/nenhuma alteração", least), ("Piora", worst)]:
        m = {"hibrido": float(nd.loc[q, "hybrid"]), "reranking": float(nd.loc[q, "hybrid_rerank"]), "delta": float(dr[q])}
        case = rerank_case(q, queries[q], runs["hybrid"][q], runs["reranking"][q], runs["lexical"][q], runs["semantic"][q],
                           qrels[q], lex, ctx.corpus, id2idx, m)
        case["grupo"] = label
        rcases.append(case)
        md.append(rerank_markdown(case, label))
    md.append(f"\nConsultas com ΔNDCG@10 = 0: {len(zero)}; melhoraram: {int((dr > 1e-12).sum())}; pioraram: {int((dr < -1e-12).sum())}.")
    (OUT / "rerank_cases.json").write_text(json.dumps(rcases, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "rerank_cases.md").write_text("\n".join(md), encoding="utf-8")

    # ---------- evidências globais (sustentam afirmações gerais no relatório) ----------
    dl = np.array(lex.index.doc_len, dtype=float)
    def mean_len_top10(run):
        return float(np.mean([dl[id2idx[d]] for q in run for d, _ in run[q][:10]]))
    rel_docs = [d for q in qrels for d, g in qrels[q].items() if g > 0]
    cover, nterms = {}, {}
    for q in qrels:
        st = query_stems(lex, queries[q])
        cover[q] = relevance_coverage(lex, st, qrels[q], id2idx)
        nterms[q] = len(st)
    zero_frac = {q: cover[q]["sem_nenhum_termo"] / max(cover[q]["n_relevantes"], 1) for q in qrels}
    qs = sorted(qrels)
    ev = {"tam_medio_doc_colecao_tokens": float(dl.mean()),
          "tam_medio_doc_relevante_tokens": float(np.mean([dl[id2idx[d]] for d in rel_docs])),
          "tam_medio_top10_por_sistema": {n: mean_len_top10(r) for n, r in runs.items()},
          "frac_media_relevantes_sem_nenhum_termo_da_consulta": float(np.mean(list(zero_frac.values()))),
          "frac_media_relevantes_com_todos_os_termos": float(np.mean([cover[q]["com_todos_os_termos"] / max(cover[q]["n_relevantes"], 1) for q in qs])),
          "n_termos_consulta_media": float(np.mean(list(nterms.values()))),
          "ndcg_medio_lexical_por_n_termos": {int(k): float(v) for k, v in
                                              pd.DataFrame({"n": pd.Series(nterms), "nd": nd["lexical"]}).groupby("n")["nd"].mean().round(4).items()},
          "spearman_delta_sem_lex_vs_frac_relevantes_sem_termo": dict(zip(("rho", "p"), map(float, spearmanr([zero_frac[q] for q in qs], [delta[q] for q in qs])))),
          "spearman_delta_sem_lex_vs_n_termos": dict(zip(("rho", "p"), map(float, spearmanr([nterms[q] for q in qs], [delta[q] for q in qs])))),
          "consultas_semantico_melhor": int((delta > 1e-12).sum()), "consultas_lexical_melhor": int((delta < -1e-12).sum()),
          "consultas_empate": int((delta.abs() <= 1e-12).sum()),
          "reranking": {"melhoraram": int((dr > 1e-12).sum()), "pioraram": int((dr < -1e-12).sum()), "iguais": int((dr.abs() <= 1e-12).sum())}}
    # ---- complementaridade lexical/semântico e margem de ganho do reranking ----
    import math
    def ids(run, q, n):
        return [d for d, _ in run[q][:n]]
    ov, only_l, only_s, both, rec_l, rec_s, rec_u, beat, tie, lose = [], [], [], [], [], [], [], 0, 0, 0
    r50, oracle = [], []
    promoted = demoted = 0
    for q in qs:
        rel = {d for d, g in qrels[q].items() if g > 0}
        l10, s10 = set(ids(runs["lexical"], q, 10)), set(ids(runs["semantic"], q, 10))
        ov.append(len(l10 & s10) / len(l10 | s10) if (l10 | s10) else 0.0)
        l100, s100 = set(ids(runs["lexical"], q, 100)), set(ids(runs["semantic"], q, 100))
        rl, rs = rel & l100, rel & s100
        rec_l.append(len(rl) / len(rel)); rec_s.append(len(rs) / len(rel)); rec_u.append(len(rl | rs) / len(rel))
        u = rl | rs
        if u:
            only_l.append(len(rl - rs) / len(u)); only_s.append(len(rs - rl) / len(u)); both.append(len(rl & rs) / len(u))
        a, b, c = nd.loc[q, "lexical"], nd.loc[q, "semantic"], nd.loc[q, "hybrid"]
        beat += c > max(a, b) + 1e-12; lose += c < min(a, b) - 1e-12; tie += not (c > max(a, b) + 1e-12 or c < min(a, b) - 1e-12)
        h50 = ids(runs["hybrid"], q, 50)
        r50.append(len(rel & set(h50)) / len(rel))
        grades = sorted((qrels[q].get(d, 0) for d in h50), reverse=True)[:10]
        ideal = sorted(qrels[q].values(), reverse=True)[:10]
        oracle.append(sum(g / math.log2(i + 1) for i, g in enumerate(grades, 1)) / sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1)))
        rh = {d: i for i, d in enumerate(ids(runs["hybrid"], q, 50), 1)}
        rr = {d: i for i, d in enumerate(ids(runs["reranking"], q, 50), 1)}
        promoted += sum(1 for d in rel if 11 <= rh.get(d, 999) <= 50 and rr.get(d, 999) <= 10)
        demoted += sum(1 for d in rel if rh.get(d, 999) <= 10 and rr.get(d, 999) > 10)
    ev.update({"overlap_top10_lex_sem_jaccard": float(np.mean(ov)),
               "recall100": {"lexical": float(np.mean(rec_l)), "semantico": float(np.mean(rec_s)), "uniao": float(np.mean(rec_u))},
               "relevantes_achados_so_pelo_lexical": float(np.mean(only_l)), "relevantes_achados_so_pelo_semantico": float(np.mean(only_s)),
               "relevantes_achados_pelos_dois": float(np.mean(both)),
               "hibrido_vs_melhor_isolado": {"melhor_que_ambos": int(beat), "pior_que_ambos": int(lose), "entre_os_dois_ou_igual": int(tie)},
               "rerank_margem": {"recall50_hibrido": float(np.mean(r50)), "ndcg10_oraculo_reordenando_top50": float(np.mean(oracle)),
                                 "relevantes_promovidos_para_top10": int(promoted), "relevantes_rebaixados_do_top10": int(demoted)}})
    (OUT / "global_evidence.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(ev, ensure_ascii=False, indent=1))
    print("\nCasos lexical x semântico:", [(c["grupo"], c["query_id"], round(c["metricas"]["delta"], 3)) for c in cases])
    print("Casos reranking:", [(c["grupo"], c["query_id"], round(c["metricas"]["delta"], 3)) for c in rcases])


if __name__ == "__main__":
    main()
