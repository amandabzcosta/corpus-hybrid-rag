"""Gera as tabelas (.tex), figuras (.pdf) e macros numéricas do relatório a partir de results/.
Tudo em report/auto/. O relatório (report/relatorio.tex) só faz \\input desses arquivos, então os números do texto,
das tabelas e dos CSVs entregues são sempre os mesmos."""
from __future__ import annotations

import json
import re
import unicodedata
from itertools import zip_longest

from _common import ROOT  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

AUTO = ROOT / "report" / "auto"
R = RESULTS_DIR
SYS = {"lexical": "Lexical (BM25)", "semantic": "Semântico", "hybrid": "Híbrido (RRF)", "hybrid_rerank": "Híbrido + reranking"}
MACRO = {"lexical": "Lex", "semantic": "Sem", "hybrid": "Hyb", "hybrid_rerank": "Rr"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def pt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def esc(s) -> str:
    s = str(s)
    for a, b in {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2212": "-"}.items():
        s = s.replace(a, b)
    s = "".join(c for c in unicodedata.normalize("NFKC", s) if ord(c) < 256)
    for a, b in {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}"}.items():
        s = s.replace(a, b)
    return s


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def table(rows, header, colspec, caption, label, font=r"\small", extra_after_header="", pos="t"):
    lines = [f"\\begin{{table}}[{pos}]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", font,
             f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule" + extra_after_header]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def bold_best(vals, nd=4, higher=True):
    best = max(vals) if higher else min(vals)
    return [(f"\\textbf{{{pt(v, nd)}}}" if v == best else pt(v, nd)) for v in vals]


def save(name, text):
    (AUTO / name).write_text(text, encoding="utf-8")


def bootstrap_ci(x, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    AUTO.mkdir(parents=True, exist_ok=True)
    sel = json.loads((R / "dev_selection.json").read_text())
    tm = json.loads((R / "timings.json").read_text())
    summ = pd.read_csv(R / "summary_test.csv").set_index("model")
    mpq = pd.read_csv(R / "metrics_per_query.csv")
    sig = pd.read_csv(R / "significance_tests.csv")
    ge = json.loads((R / "analysis" / "global_evidence.json").read_text())
    M = ""

    # ---------------- macros ----------------
    idx, sd, ce, env = tm["indice"], tm["semantico_docs"], tm["reranker"], tm["ambiente"]
    enc = sd["encoder"]
    M += macro("NDocs", "3.633") + macro("NQueries", "3.237") + macro("NTest", 323) + macro("NDev", 324)
    M += macro("Vocab", f"{idx['vocabulario']:,}".replace(",", ".")) + macro("Postings", f"{idx['total_postings']:,}".replace(",", "."))
    M += macro("TotalTokens", f"{idx['total_tokens']:,}".replace(",", ".")) + macro("AvgDocLen", pt(idx["comprimento_medio_doc"], 1))
    M += macro("IdxPre", pt(idx["tempo_preprocessamento_s"], 2)) + macro("IdxStruct", pt(idx["tempo_estruturas_indice_s"], 2)) + macro("IdxTotal", pt(idx["tempo_construcao_wall_s"], 2))
    M += macro("BmK", pt(sel["bm25"]["k1"], 1)) + macro("BmB", pt(sel["bm25"]["b"], 2))
    M += macro("PreStem", {"snowball": "Snowball (Porter2)", "porter": "Porter", None: "nenhum"}[sel["preprocessing"]["stemmer"]])
    M += macro("LexMs", pt(tm["lexical_ms_por_consulta"]["media"], 2))
    M += macro("EncId", esc(sel["bi_encoder"]["id"])) + macro("EncName", esc(enc["model"]).replace("/", "/\\allowbreak ")) + macro("EncDim", sd["dim"]) + macro("EncMaxLen", sd["max_seq_length"])
    M += macro("EncTrunc", sd["docs_truncated"]) + macro("EncTruncPct", pt(sd["pct_docs_truncated"], 1)) + macro("EncDocSeconds", pt(sd["embedding_seconds"], 0))
    M += macro("EncQueryMs", pt(tm["semantico_ms_por_consulta"]["media"], 1)) + macro("TokMean", pt(sd["tokens_mean"], 0)) + macro("TokMedian", pt(sd["tokens_median"], 0))
    M += macro("TokPNinetyFive", pt(sd["tokens_p95"], 0)) + macro("TokMax", sd["tokens_max"])
    M += macro("RrfK", sel["rrf_k"]) + macro("RrfMs", pt(tm["fusao_rrf_ms_por_consulta"]["media"], 3))
    M += macro("CEName", esc(ce["modelo"]["model"]).replace("/", "/\\allowbreak ")) + macro("CEId", esc(sel["cross_encoder"]["id"])) + macro("CEMaxLen", ce["modelo"]["max_length"])
    M += macro("CEPairs", f"{ce['pares']:,}".replace(",", ".")) + macro("CETruncPairs", ce["pares_truncados"]) + macro("CETruncPct", pt(ce["pct_pares_truncados"], 1))
    M += macro("CEPairTokens", pt(ce["tokens_par_media"], 0)) + macro("CEMsQuery", pt(ce["ms_por_consulta_50_pares"] / 1000, 1)) + macro("CESecPair", pt(ce["segundos_por_par"], 3))
    for m, k in MACRO.items():
        r = summ.loc[m]
        M += macro(f"PTen{k}", pt(r["P10"], 4)) + macro(f"RTen{k}", pt(r["R10"], 4)) + macro(f"MapTen{k}", pt(r["MAP10"], 4)) + macro(f"NdcgTen{k}", pt(r["NDCG10"], 4))
    M += macro("CpuModel", esc(env.get("cpu_model", "n/d"))) + macro("CpuCores", env.get("cpu_cores", "n/d")) + macro("RamGB", pt(env.get("ram_gb", 0), 1))
    M += macro("PyVer", env["python"]) + macro("OrtVer", env.get("onnxruntime", "n/d")) + macro("NumpyVer", env["numpy"]) + macro("NltkVer", env.get("nltk", "n/d"))
    M += macro("AvgRelTest", "38,2")
    M += macro("ZeroOverlapPct", pt(100 * ge["frac_media_relevantes_sem_nenhum_termo_da_consulta"], 1))
    M += macro("AllTermsPct", pt(100 * ge["frac_media_relevantes_com_todos_os_termos"], 1))
    M += macro("LenCollection", pt(ge["tam_medio_doc_colecao_tokens"], 0)) + macro("LenRelevant", pt(ge["tam_medio_doc_relevante_tokens"], 0))
    for k, v in ge["tam_medio_top10_por_sistema"].items():
        M += macro(f"LenTop{MACRO.get(k, k.capitalize())[:3]}", pt(v, 0))
    M += macro("SemWins", ge["consultas_semantico_melhor"]) + macro("LexWins", ge["consultas_lexical_melhor"]) + macro("TieQ", ge["consultas_empate"])
    M += macro("RrBetter", ge["reranking"]["melhoraram"]) + macro("RrWorse", ge["reranking"]["pioraram"]) + macro("RrSame", ge["reranking"]["iguais"])
    ty = json.loads((R / "typo_summary.json").read_text())
    M += macro("TypoOrig", pt(ty["ndcg10_medio"]["original"], 4)) + macro("TypoErr", pt(ty["ndcg10_medio"]["erro"], 4)) + macro("TypoCorr", pt(ty["ndcg10_medio"]["corrigida"], 4))
    M += macro("TypoRecMean", pt(100 * ty["recuperacao_media_por_consulta"], 1)) + macro("TypoRecAgg", pt(100 * ty["recuperacao_agregada"], 1))
    M += macro("TypoNDrop", ty["n_com_queda"]) + macro("TypoNNoDrop", ty["n_erro_nao_reduziu"]) + macro("TypoNExact", ty["correcoes_exatas"]) + macro("TypoEligible", ty["espaco_amostral_elegivel"])
    M += macro("SpK", ty["parametros_correcao"]["k"]) + macro("SpJ", pt(ty["parametros_correcao"]["jaccard"], 1)) + macro("SpEdit", ty["parametros_correcao"]["max_edit"]) + macro("DictSize", f"{ty['tamanho_dicionario']:,}".replace(",", "."))
    rag = pd.read_csv(R / "rag_retrieval_metrics.csv")
    mean = rag[rag.query_id == "MÉDIA"].iloc[0]
    M += macro("RagRecall", pt(float(mean["Recall@5"]), 4)) + macro("RagPrec", pt(float(mean["Precision@5"]), 4)) + macro("RagHit", pt(float(mean["Hit@5"]), 1))
    # evidências globais da análise (script 04) e checagens objetivas do RAG
    ov = ge
    M += macro("OverlapTop", pt(ov["overlap_top10_lex_sem_jaccard"], 2)) + macro("RecLexHundred", pt(ov["recall100"]["lexical"], 3))
    M += macro("RecSemHundred", pt(ov["recall100"]["semantico"], 3)) + macro("RecUnionHundred", pt(ov["recall100"]["uniao"], 3))
    M += macro("OnlyLexPct", pt(100 * ov["relevantes_achados_so_pelo_lexical"], 1)) + macro("OnlySemPct", pt(100 * ov["relevantes_achados_so_pelo_semantico"], 1))
    M += macro("BothPct", pt(100 * ov["relevantes_achados_pelos_dois"], 1))
    M += macro("HybBeatBoth", ov["hibrido_vs_melhor_isolado"]["melhor_que_ambos"]) + macro("HybLoseBoth", ov["hibrido_vs_melhor_isolado"]["pior_que_ambos"])
    M += macro("HybBetween", ov["hibrido_vs_melhor_isolado"]["entre_os_dois_ou_igual"])
    rm = ov["rerank_margem"]
    M += macro("RecFiftyHyb", pt(rm["recall50_hibrido"], 3)) + macro("OracleNdcg", pt(rm["ndcg10_oraculo_reordenando_top50"], 3))
    M += macro("RelPromoted", rm["relevantes_promovidos_para_top10"]) + macro("RelDemoted", rm["relevantes_rebaixados_do_top10"])
    M += macro("SpearmanNTermsRho", pt(ov["spearman_delta_sem_lex_vs_n_termos"]["rho"], 2)) + macro("SpearmanNTermsP", pt(ov["spearman_delta_sem_lex_vs_n_termos"]["p"], 3))
    M += macro("SpearmanZeroRho", pt(ov["spearman_delta_sem_lex_vs_frac_relevantes_sem_termo"]["rho"], 2)) + macro("SpearmanZeroP", pt(ov["spearman_delta_sem_lex_vs_frac_relevantes_sem_termo"]["p"], 2))
    M += macro("AvgTermsQuery", pt(ov["n_termos_consulta_media"], 2))
    lexrun = pd.read_csv(R / "results_lexical.csv")
    nres = lexrun.groupby("query_id").size()
    M += macro("LexNoResult", int(len(mpq[mpq.model == "lexical"]) - len(nres))) + macro("LexShort", int((nres < 100).sum()) + int(len(mpq[mpq.model == "lexical"]) - len(nres)))
    # checagem objetiva das citações nas respostas B
    from src.data import load_qrels
    qr = load_qrels("test")
    exp = pd.read_csv(R / "rag_experiment.csv", dtype=str, keep_default_na=False)
    cited = valid = rel_cited = ctx_rel = n_ctx = 0
    for r in exp[exp.condition == "B"].to_dict("records"):
        ids_ctx = r["context_doc_ids"].split()
        ids_cited = list(dict.fromkeys(re.findall(r"MED-\d+", r["answer"])))
        cited += len(ids_cited); valid += sum(c in ids_ctx for c in ids_cited)
        rel_cited += sum(qr[r["query_id"]].get(c, 0) > 0 for c in ids_cited)
        ctx_rel += sum(qr[r["query_id"]].get(c, 0) > 0 for c in ids_ctx); n_ctx += len(ids_ctx)
    M += macro("RagCited", cited) + macro("RagCitedValid", valid) + macro("RagCitedRel", rel_cited) + macro("RagCtxRel", ctx_rel) + macro("RagCtxN", n_ctx)
    M += macro("RagWordsA", pt(exp[exp.condition == "A"]["answer"].str.split().str.len().mean(), 0)) + macro("RagWordsB", pt(exp[exp.condition == "B"]["answer"].str.split().str.len().mean(), 0))
    save("numbers.tex", M)

    # ---------------- tabelas do dev ----------------
    pre = pd.read_csv(R / "dev_preprocessing.csv")
    rows = [[("sim" if r.stopwords else "não"), {"none": "nenhum", "porter": "Porter", "snowball": "Snowball"}[r.stemmer], f"{r.vocab:,}".replace(",", "."), pt(r.P10), pt(r.R10), pt(r.AP10), pt(r.NDCG10)] for r in pre.itertuples()]
    save("tab_dev_pre.tex", table(rows, ["Stopwords", "Stemming", "Vocab.", "P@10", "R@10", "MAP@10", "NDCG@10"], "llrrrrr",
         "Pré-processamento no \\textbf{dev} (BM25 com $k_1=1{,}2$, $b=0{,}75$).", "tab:dev_pre"))
    g = pd.read_csv(R / "dev_bm25_grid.csv")
    piv = g.pivot(index="k1", columns="b", values="NDCG10")
    best = piv.values.max()
    rows = [[pt(k, 1)] + [(f"\\textbf{{{pt(v)}}}" if v == best else pt(v)) for v in piv.loc[k]] for k in piv.index]
    save("tab_dev_bm25.tex", table(rows, ["$k_1 \\backslash b$"] + [pt(b, 2) for b in piv.columns], "l" + "r" * len(piv.columns),
         "NDCG@10 no \\textbf{dev} para a grade de parâmetros do BM25 (melhor valor em negrito).", "tab:dev_bm25"))
    emb = pd.read_csv(R / "dev_embeddings.csv")
    rows = [[esc(r.modelo), int(r.max_seq_length), pt(r.pct_docs_truncados, 1) + "\\%", pt(r.tempo_emb_docs_s, 0), pt(r.P10), pt(r.R10), pt(r.AP10), pt(r.NDCG10)] for r in emb.itertuples()]
    save("tab_dev_emb.tex", table(rows, ["Modelo@max\\_len", "Max.", "Trunc.", "Tempo (s)", "P@10", "R@10", "MAP@10", "NDCG@10"], "lrrrrrrr",
         "Bi-encoders candidatos no \\textbf{dev} (busca exaustiva por cosseno; ``Trunc.'' = documentos com mais tokens que o limite).", "tab:dev_emb"))
    rr = pd.read_csv(R / "dev_rrf.csv")
    h = rr[rr.sistema == "hibrido_rrf"]
    rows = [[int(r.k), pt(r.P10), pt(r.R10), pt(r.AP10), pt(r.NDCG10)] for r in h.itertuples()]
    base = rr[rr.sistema != "hibrido_rrf"]
    extra = "".join(f"\n\\textit{{{'lexical' if r.sistema == 'lexical' else 'semântico'}}} & \\multicolumn{{3}}{{c}}{{}} & {pt(r.NDCG10)} \\\\" for r in base.itertuples()) + "\n\\midrule"
    save("tab_dev_rrf.tex", table(rows, ["$k$", "P@10", "R@10", "MAP@10", "NDCG@10"], "rrrrr", "Parâmetro $k$ do RRF no \\textbf{dev}.", "tab:dev_rrf", extra_after_header=extra))
    cer = pd.read_csv(R / "dev_reranker.csv")
    rows = [[esc(r.modelo), int(r.n_consultas), pt(r.P10), pt(r.R10), pt(r.AP10), pt(r.NDCG10)] for r in cer.itertuples()]
    save("tab_dev_ce.tex", table(rows, ["Sistema", "Consultas", "P@10", "R@10", "MAP@10", "NDCG@10"], "lrrrrr",
         "Escolha do cross-encoder no \\textbf{dev} (amostra de consultas com semente fixa; reordena o Top-50 do híbrido).", "tab:dev_ce"))
    sp = pd.read_csv(R / "dev_spelling.csv")
    sp = sp[sp.max_edit.isin([1, 2])]
    rows = [[int(r.k), pt(r.jaccard, 1), int(r.max_edit), pt(100 * r.acerto_termo + 1e-9, 1) + "\\%", pt(r.ndcg10_corrigida), pt(100 * r.recuperacao_media, 1) + "\\%"] for r in sp.itertuples()]
    save("tab_dev_spell.tex", table(rows, ["$k$-grama", "Jaccard", "Edição máx.", "Termo corrigido", "NDCG@10 corrig.", "Recuperação"], "rrrrrr",
         f"Parâmetros da correção ortográfica no \\textbf{{dev}} ({int(sp.n.iloc[0])} consultas com um erro gerado; NDCG@10 original = {pt(sp.ndcg10_original.iloc[0])}, com erro = {pt(sp.ndcg10_erro.iloc[0])}).", "tab:dev_spell", font=r"\footnotesize"))

    # ---------------- teste: tabela principal + significância ----------------
    cols = ["P10", "R10", "MAP10", "NDCG10"]
    vals = {c: [summ.loc[m, c] for m in SYS] for c in cols}
    cells = {c: bold_best(vals[c]) for c in cols}
    cis = {m: bootstrap_ci(mpq[mpq.model == m]["NDCG10"]) for m in SYS}
    rows = [[SYS[m]] + [cells[c][i] for c in cols] + [f"[{pt(cis[m][0], 3)}; {pt(cis[m][1], 3)}]"] for i, m in enumerate(SYS)]
    save("tab_test_main.tex", table(rows, ["Sistema", "P@10", "R@10", "MAP@10", "NDCG@10", "IC 95\\% NDCG@10"], "lrrrrr",
         f"Resultados no \\textbf{{test}} ({int(len(mpq)/4)} consultas). IC: bootstrap percentil (10.000 reamostragens) sobre as consultas.", "tab:test_main"))
    nm = {"semantic": "Semântico", "lexical": "Lexical", "hybrid": "Híbrido", "hybrid_rerank": "Híbrido+rerank."}
    rows = [[f"{nm[r.A]} vs.\\ {nm[r.B]}", pt(r.dif_media_NDCG10), f"{r.A_melhor}/{r.B_melhor}/{r.empates}", ("$<0{,}001$" if r.p_wilcoxon < 1e-3 else pt(r.p_wilcoxon, 3))] for r in sig.itertuples()]
    save("tab_sig.tex", table(rows, ["Comparação (A vs.\\ B)", "$\\Delta$ NDCG@10", "A melhor/B melhor/empate", "$p$ (Wilcoxon)"], "lrrr",
         "Diferenças pareadas por consulta no \\textbf{test} (teste de Wilcoxon de postos sinalizados).", "tab:sig"))

    # ---------------- casos lexical x semântico ----------------
    cases = json.loads((R / "analysis" / "lex_vs_sem_cases.json").read_text())
    out = ""
    for i, c in enumerate(cases, start=1):
        rows = []
        empty = {"rank": "-", "doc_id": "-", "relevancia": "-", "score": None, "ranks_outros": {"semantico": None, "lexical": None}, "tam_doc": "-", "termos_presentes": "-"}
        for i5, (a, b) in enumerate(zip_longest(c["top5"]["lexical"], c["top5"]["semantico"]), start=1):
            a, b = a or {**empty, "rank": i5}, b or empty
            rows.append([a["rank"], a["doc_id"], a["relevancia"], pt(a["score"], 2) if a["score"] is not None else "-", a["ranks_outros"]["semantico"] or ">100", a["tam_doc"], a["termos_presentes"],
                         b["doc_id"], b["relevancia"], pt(b["score"], 3) if b["score"] is not None else "-", b["ranks_outros"]["lexical"] or ">100", b["tam_doc"], b["termos_presentes"]])
        m = c["metricas"]
        cov = c["cobertura_relevantes"]
        cap = (f"Caso {i} ({c['grupo'].lower()}): {c['query_id']} --- ``{esc(c['query'])}''. NDCG@10: lexical {pt(m['lexical'], 3)}, semântico {pt(m['semantico'], 3)}. "
               f"Relevantes: {cov['n_relevantes']}; {cov['sem_nenhum_termo']} sem nenhum termo da consulta. Colunas: relevância (0--2), score (BM25 | cosseno), "
               f"posição no outro sistema, $|d|$ = tokens do documento no índice, ter = termos da consulta presentes.")
        out += table(rows, ["\\#", "doc", "rel", "BM25", "r$_s$", "$|d|$", "ter", "doc", "rel", "cos", "r$_l$", "$|d|$", "ter"], "rlrrrrr|lrrrrr", cap, f"tab:case{i}", font=r"\scriptsize",
                     extra_after_header="", pos="!htb")
    save("tab_cases_lexsem.tex", out)

    # ---------------- casos de reranking ----------------
    rc = json.loads((R / "analysis" / "rerank_cases.json").read_text())
    out = ""
    for i, c in enumerate(rc, start=1):
        docs = sorted(c["documentos"], key=lambda d: (-(abs(d["delta"]) if d["delta"] is not None else 0)))
        keep = {d["doc_id"] for d in docs[:8]}
        rows = [[d["rank_reranking"] or "-", d["rank_hibrido"] or "-", (f"{d['delta']:+d}" if d["delta"] is not None else "-"), d["doc_id"], d["relevancia"],
                 pt(d["score_ce"], 2) if d["score_ce"] is not None else "-", d["rank_lexical"] or ">100", d["rank_semantico"] or ">100", d["tam_doc"], d["termos_presentes"]]
                for d in c["documentos"] if d["doc_id"] in keep]
        m = c["metricas"]
        cap = (f"Reranking, caso {i} ({c['grupo'].lower()}): {c['query_id']} --- ``{esc(c['query'])}''. NDCG@10: híbrido {pt(m['hibrido'], 3)} $\\rightarrow$ {pt(m['reranking'], 3)}. "
               f"Documentos do Top-10 (antes ou depois) com maior deslocamento, ordenados pela posição final. $\\Delta$ = posição híbrida $-$ posição final.")
        out += table(rows, ["rank CE", "rank híb.", "$\\Delta$", "doc", "rel", "score CE", "r$_{lex}$", "r$_{sem}$", "$|d|$", "ter"], "rrrlrrrrrr", cap, f"tab:rr{i}", font=r"\scriptsize", pos="!htb")
    save("tab_cases_rerank.tex", out)

    # ---------------- erros ortográficos ----------------
    t = pd.read_csv(R / "typo_experiment.csv")
    rows = []
    for r in t.itertuples():
        rec = r.recuperacao
        try:
            rec = pt(100 * float(rec), 0) + "\\%"
        except (TypeError, ValueError):
            rec = "n/a"
        rows.append([r.query_id, r.tipo_erro, f"\\texttt{{{esc(r.termo_original)}}}", f"\\texttt{{{esc(r.termo_erro)}}}", f"\\texttt{{{esc(r.termo_corrigido) if isinstance(r.termo_corrigido, str) else '-'}}}",
                     pt(r.NDCG10_original, 3), pt(r.NDCG10_erro, 3), pt(r.NDCG10_corrigida, 3), rec])
    save("tab_typo.tex", table(rows, ["Consulta", "Tipo de erro", "Original", "Com erro", "Corrigido", "NDCG orig.", "NDCG erro", "NDCG corr.", "Recup."], "llllllrrr",
         "Experimento de tolerância a erros (sistema lexical, \\textbf{test}). Recup. = (corrigida $-$ erro)/(original $-$ erro).", "tab:typo", font=r"\scriptsize"))

    # ---------------- tempos ----------------
    rows = [["Índice invertido (pré-processamento + estruturas)", f"{pt(idx['tempo_construcao_wall_s'], 2)} s", "uma vez"],
            ["Embeddings dos documentos (3.633)", f"{pt(sd['embedding_seconds'], 0)} s", "uma vez"],
            ["Consulta lexical (pré-proc. + BM25 + Top-100)", f"{pt(tm['lexical_ms_por_consulta']['media'], 2)} ms", "por consulta"],
            ["Consulta semântica (codificação + busca exaustiva)", f"{pt(tm['semantico_ms_por_consulta']['media'], 1)} ms", "por consulta"],
            ["Fusão RRF (100 + 100 documentos)", f"{pt(tm['fusao_rrf_ms_por_consulta']['media'], 3)} ms", "por consulta"],
            ["Reranking (cross-encoder, 50 pares)", f"{pt(ce['ms_por_consulta_50_pares'] / 1000, 1)} s", "por consulta"]]
    save("tab_times.tex", table(rows, ["Etapa", "Tempo", "Frequência"], "lrl", f"Tempos medidos (CPU: {esc(env.get('cpu_model', 'n/d'))}, {env.get('cpu_cores')} núcleo(s), sem GPU).", "tab:times"))
    rows = [["Documentos", f"{idx['n_documentos']:,}".replace(",", ".")], ["Termos do vocabulário", f"{idx['vocabulario']:,}".replace(",", ".")],
            ["Total de postings", f"{idx['total_postings']:,}".replace(",", ".")], ["Tokens indexados", f"{idx['total_tokens']:,}".replace(",", ".")],
            ["Comprimento médio do documento", pt(idx["comprimento_medio_doc"], 1)],
            ["Construção (pré-proc. + estruturas)", f"{pt(idx['tempo_preprocessamento_s'], 2)} s + {pt(idx['tempo_estruturas_indice_s'], 2)} s"],
            ["Tempo médio por consulta", f"{pt(tm['lexical_ms_por_consulta']['media'], 2)} ms"]]
    save("tab_index.tex", table(rows, ["Estatística", "Valor"], "lr", "Índice invertido construído do zero.", "tab:index"))

    # ---------------- RAG: recuperação @5 ----------------
    rows = []
    for d in rag.to_dict("records"):
        if d["query_id"] == "MÉDIA":
            continue
        rows.append([d["query_id"], esc(d["query"]), int(d["n_relevantes_total"]), d["relevancias_no_contexto"], pt(float(d["Recall@5"]), 3), pt(float(d["Precision@5"]), 2), int(d["Hit@5"])])
    rows.append([r"\textbf{Média}", "", "", "", pt(float(mean["Recall@5"]), 3), pt(float(mean["Precision@5"]), 2), pt(float(mean["Hit@5"]), 1)])
    save("tab_rag_ret.tex", table(rows, ["Consulta", "Pergunta", "Rel. total", "Graus no Top-5", "R@5", "P@5", "Hit@5"], "llrlrrr",
         "RAG: recuperação nos Top-5 enviados à LLM (após reranking). ``Graus no Top-5'' = relevância de cada documento do contexto (0 = não relevante/não julgado).", "tab:rag_ret", font=r"\scriptsize"))
    gm = R / "rag_generation_means.csv"
    if gm.exists():
        g = pd.read_csv(gm).set_index("condition")
        rows = [[c, *[pt(float(g.loc[c, k]), 2) if pd.notna(g.loc[c, k]) else "N/A" for k in ("fidelidade", "completude", "utilidade", "rastreabilidade")]] for c in ("A", "B")]
        save("tab_rag_grades.tex", table(rows, ["Condição", "Fidelidade", "Completude", "Utilidade", "Rastreabilidade"], "lrrrr",
             "RAG: médias das notas da avaliação humana (0--2) por condição (A = sem RAG; B = com RAG).", "tab:rag_grades"))
    else:
        save("tab_rag_grades.tex", table([["A (sem RAG)", "\\textit{a preencher}", "\\textit{a preencher}", "\\textit{a preencher}", "N/A"],
                                          ["B (com RAG)", "\\textit{a preencher}", "\\textit{a preencher}", "\\textit{a preencher}", "\\textit{a preencher}"]],
                                         ["Condição", "Fidelidade", "Completude", "Utilidade", "Rastreabilidade"], "lrrrr",
                                         "RAG: médias das notas da avaliação humana (0--2) por condição. \\textbf{PENDENTE}: preencher após a avaliação humana (script 07).", "tab:rag_grades"))

    # ---------------- figuras ----------------
    h = rr[rr.sistema == "hibrido_rrf"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(h.k, h.NDCG10, "o-", color="#1f4e79", label="Híbrido (RRF)")
    for _, r in base.iterrows():
        ax.axhline(r.NDCG10, ls="--", lw=1, color="#c0504d" if r.sistema == "lexical" else "#4f9d4f", label="Lexical" if r.sistema == "lexical" else "Semântico")
    ax.axvline(sel["rrf_k"], color="gray", lw=.8, ls=":")
    ax.set_xlabel("$k$ do RRF"); ax.set_ylabel("NDCG@10 (dev)"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(AUTO / "fig_rrf_k.pdf"); plt.close(fig)

    nd = mpq.pivot(index="query_id", columns="model", values="NDCG10")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    means = [summ.loc[m, "NDCG10"] for m in SYS]
    err = np.array([[summ.loc[m, "NDCG10"] - cis[m][0], cis[m][1] - summ.loc[m, "NDCG10"]] for m in SYS]).T
    ax.bar(range(4), means, yerr=err, capsize=3, color=["#c0504d", "#4f9d4f", "#1f4e79", "#7f6000"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["Lexical", "Semântico", "Híbrido", "Híb.+rerank"], fontsize=7)
    ax.set_ylabel("NDCG@10 (test, IC 95%)"); ax.set_ylim(0.25, max(means) + 0.05)
    ax = axs[1]
    d = nd["semantic"] - nd["lexical"]
    ax.scatter(nd["lexical"], nd["semantic"], s=7, c=np.where(d > 0, "#4f9d4f", np.where(d < 0, "#c0504d", "gray")), alpha=.7)
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    ax.set_xlabel("NDCG@10 lexical"); ax.set_ylabel("NDCG@10 semântico"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(AUTO / "fig_main.pdf"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
    ax = axs[0]
    x = np.arange(len(t)); w = .27
    ax.bar(x - w, t.NDCG10_original, w, label="original", color="#1f4e79"); ax.bar(x, t.NDCG10_erro, w, label="com erro", color="#c0504d"); ax.bar(x + w, t.NDCG10_corrigida, w, label="corrigida", color="#4f9d4f")
    ax.set_xticks(x); ax.set_xticklabels([q.replace("PLAIN-", "") for q in t.query_id], fontsize=6, rotation=45); ax.set_ylabel("NDCG@10 lexical"); ax.legend(frameon=False, fontsize=6)
    ax = axs[1]
    dr = (nd["hybrid_rerank"] - nd["hybrid"]).sort_values().values
    ax.bar(range(len(dr)), dr, width=1.0, color=np.where(dr > 0, "#4f9d4f", np.where(dr < 0, "#c0504d", "gray")))
    ax.set_xlabel("consultas (ordenadas)"); ax.set_ylabel("$\\Delta$NDCG@10 (rerank $-$ híbrido)")
    fig.tight_layout(); fig.savefig(AUTO / "fig_typo_rerank.pdf"); plt.close(fig)
    print("arquivos gerados em", AUTO, sorted(p.name for p in AUTO.iterdir()))


if __name__ == "__main__":
    main()
