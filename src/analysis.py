from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

def rank_map(ranking: Sequence) -> dict[str, int]:
    return {(x[0] if isinstance(x, (tuple, list)) else x): i for i, x in enumerate(ranking, start=1)}

def query_stems(lex, text: str) -> list[str]:
    return list(dict.fromkeys(lex.pre(text)))

def query_term_table(lex, text: str) -> list[dict]:
    return [{"termo": t, "df": lex.index.df(t), "idf": round(lex.searcher.idf(t), 2)} for t in query_stems(lex, text)]

def doc_features(lex, doc_id: str, stems: Sequence[str], id2idx: Mapping[str, int]) -> dict:
    idx = id2idx[doc_id]
    tfs = {t: lex.index.tf(t, idx) for t in stems}
    return {"len": int(lex.index.doc_len[idx]), "n_termos": sum(v > 0 for v in tfs.values()),
            "tf": {t: v for t, v in tfs.items() if v}}

def relevance_coverage(lex, stems: Sequence[str], qrels_q: Mapping[str, int], id2idx: Mapping[str, int]) -> dict:
    rel = [d for d, g in qrels_q.items() if g > 0]
    counts = [sum(lex.index.tf(t, id2idx[d]) > 0 for t in stems) for d in rel]
    return {"n_relevantes": len(rel), "sem_nenhum_termo": int(sum(c == 0 for c in counts)),
            "com_todos_os_termos": int(sum(c == len(stems) for c in counts)) if stems else 0,
            "media_termos_presentes": round(float(np.mean(counts)), 2) if counts else 0.0, "n_termos_consulta": len(stems)}

def system_summary(ranking: Sequence, qrels_q: Mapping[str, int]) -> dict:
    ranks = [i for i, x in enumerate(ranking, start=1) if qrels_q.get(x[0] if isinstance(x, (tuple, list)) else x, 0) > 0]
    return {"primeiro_relevante": ranks[0] if ranks else None, "relevantes_top5": sum(r <= 5 for r in ranks),
            "relevantes_top10": sum(r <= 10 for r in ranks), "relevantes_top100": len(ranks)}

def _short(s: str, n: int = 78) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"

def topn_table(ranking: Sequence, other_ranks: Mapping[str, Mapping[str, int]], qrels_q, lex, corpus, id2idx, stems,
               n: int = 5) -> list[dict]:
    rows = []
    for rank, (doc_id, score) in enumerate(ranking[:n], start=1):
        f = doc_features(lex, doc_id, stems, id2idx)
        rows.append({"rank": rank, "doc_id": doc_id, "relevancia": int(qrels_q.get(doc_id, 0)), "score": round(float(score), 4),
                     "tam_doc": f["len"], "termos_presentes": f["n_termos"], "tf": f["tf"],
                     "ranks_outros": {name: rm.get(doc_id) for name, rm in other_ranks.items()},
                     "titulo": _short(corpus[doc_id]["title"])})
    return rows

def lex_sem_case(qid, text, run_lex, run_sem, qrels_q, lex, corpus, id2idx, metrics: Mapping[str, float]) -> dict:
    stems = query_stems(lex, text)
    rm_l, rm_s = rank_map(run_lex), rank_map(run_sem)
    return {"query_id": qid, "query": text, "metricas": metrics, "termos": query_term_table(lex, text),
            "cobertura_relevantes": relevance_coverage(lex, stems, qrels_q, id2idx),
            "resumo": {"lexical": system_summary(run_lex, qrels_q), "semantico": system_summary(run_sem, qrels_q)},
            "top5": {"lexical": topn_table(run_lex, {"semantico": rm_s}, qrels_q, lex, corpus, id2idx, stems),
                     "semantico": topn_table(run_sem, {"lexical": rm_l}, qrels_q, lex, corpus, id2idx, stems)}}

def top10_displacement(before: Sequence, after: Sequence) -> dict:
    b = [x[0] for x in before[:10]]
    ra = rank_map(after)
    changed = sum(1 for i, d in enumerate(b) if after[i][0] != d)
    disp = float(np.mean([abs(ra.get(d, len(after) + 1) - (i + 1)) for i, d in enumerate(b)]))
    return {"posicoes_diferentes_top10": changed, "deslocamento_medio_top10": round(disp, 2),
            "docs_que_saem_do_top10": sum(1 for d in b if ra.get(d, 10 ** 6) > 10)}

def rerank_case(qid, text, hybrid, reranked, run_lex, run_sem, qrels_q, lex, corpus, id2idx, metrics, n: int = 10) -> dict:
    stems = query_stems(lex, text)
    rh, rr, rl, rs = rank_map(hybrid), rank_map(reranked), rank_map(run_lex), rank_map(run_sem)
    union = list(dict.fromkeys([d for d, _ in hybrid[:n]] + [d for d, _ in reranked[:n]]))
    score_rr = dict(reranked)
    moved = []
    for d in union:
        f = doc_features(lex, d, stems, id2idx)
        moved.append({"doc_id": d, "relevancia": int(qrels_q.get(d, 0)), "rank_hibrido": rh.get(d), "rank_reranking": rr.get(d),
                      "delta": (rh.get(d) - rr.get(d)) if rh.get(d) and rr.get(d) else None,
                      "score_ce": round(float(score_rr[d]), 3) if rr.get(d) and rr[d] <= 50 else None,
                      "rank_lexical": rl.get(d), "rank_semantico": rs.get(d), "tam_doc": f["len"],
                      "termos_presentes": f["n_termos"], "titulo": _short(corpus[d]["title"])})
    moved.sort(key=lambda r: (r["rank_reranking"] or 10 ** 6))
    return {"query_id": qid, "query": text, "metricas": metrics, "n_termos_consulta": len(stems),
            "deslocamento": top10_displacement(hybrid, reranked), "documentos": moved,
            "relevantes_top10": {"hibrido": system_summary(hybrid, qrels_q)["relevantes_top10"],
                                 "reranking": system_summary(reranked, qrels_q)["relevantes_top10"]}}

def _fmt_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[k]) for _, k in cols) + " |")
    return "\n".join(out)

def lex_sem_markdown(case: dict, title: str) -> str:
    m, c = case["metricas"], case["cobertura_relevantes"]
    s = [f"### {title}: {case['query_id']} — \"{case['query']}\"", "",
         f"NDCG@10: lexical = {m['lexical']:.4f}, semântico = {m['semantico']:.4f} (Δ = {m['delta']:+.4f}).  "
         f"Relevantes no qrels: {c['n_relevantes']}; sem nenhum termo da consulta: {c['sem_nenhum_termo']}; "
         f"com todos os termos: {c['com_todos_os_termos']}; média de termos presentes: {c['media_termos_presentes']} de {c['n_termos_consulta']}.", "",
         "Termos da consulta (stems): " + ", ".join(f"{t['termo']} (df={t['df']}, idf={t['idf']})" for t in case["termos"]), "",
         f"Resumo: {case['resumo']}", ""]
    for name in ("lexical", "semantico"):
        other = "semantico" if name == "lexical" else "lexical"
        rows = [{**r, "outro": r["ranks_outros"][other] or ">100"} for r in case["top5"][name]]
        s += [f"**Top-5 {name}** (rank no outro sistema = coluna 'rank {other[:3]}')", "",
              _fmt_table(rows, [("#", "rank"), ("doc_id", "doc_id"), ("rel", "relevancia"), ("score", "score"), (f"rank {other[:3]}", "outro"),
                                ("|d|", "tam_doc"), ("termos", "termos_presentes"), ("título", "titulo")]), ""]
    return "\n".join(s)

def rerank_markdown(case: dict, title: str) -> str:
    m, d = case["metricas"], case["deslocamento"]
    s = [f"### {title}: {case['query_id']} — \"{case['query']}\"", "",
         f"NDCG@10: híbrido = {m['hibrido']:.4f}, híbrido+reranking = {m['reranking']:.4f} (Δ = {m['delta']:+.4f}). "
         f"Relevantes no Top-10: {case['relevantes_top10']['hibrido']} → {case['relevantes_top10']['reranking']}. "
         f"Posições do Top-10 com documento diferente: {d['posicoes_diferentes_top10']}; deslocamento médio: {d['deslocamento_medio_top10']}.", "",
         _fmt_table(case["documentos"], [("rank CE", "rank_reranking"), ("rank híb.", "rank_hibrido"), ("Δ", "delta"), ("doc_id", "doc_id"),
                                         ("rel", "relevancia"), ("score CE", "score_ce"), ("rank lex", "rank_lexical"), ("rank sem", "rank_semantico"),
                                         ("|d|", "tam_doc"), ("termos", "termos_presentes"), ("título", "titulo")]), ""]
    return "\n".join(s)
