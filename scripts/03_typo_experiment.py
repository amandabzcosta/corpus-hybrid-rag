from __future__ import annotations

import json
import random

from _common import *  # noqa: F401,F403
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SEED, TOP_K
from src.encoders import load_bi_encoder
from src.lexical import LexicalSystem
from src.metrics import ndcg_at_k
from src.preprocess import english_stopwords, raw_tokens
from src.semantic import SemanticSystem
from src.spelling import SpellCorrector, build_dictionary, pick_target_token
from src.typo_experiment import correct_query, make_typo_cases, recovery

N_QUERIES = 10

def main():
    args = base_parser('Experimento de tolerância a erros (item 4) em 10 consultas do TEST: original x com erro x corrigida.').parse_args()
    ctx, mc, sel = Ctx(), load_models(args), load_selection()
    pre, bm, sp = sel["preprocessing"], sel["bm25"], sel["spelling"]
    lex = LexicalSystem(ctx.docs, pre["remove_stopwords"], pre["stemmer"], bm["k1"], bm["b"])
    qrels, queries = ctx.qrels["test"], ctx.test_queries
    dictionary = build_dictionary(ctx.doc_texts)                        # termos da coleção antes de stopwords/stemming
    stop = english_stopwords()
    corrector = SpellCorrector(dictionary, k=sp["k"], jaccard_threshold=sp["jaccard"], max_edit=sp["max_edit"])

    # (iii) consultas com NDCG@10 original > 0 no sistema lexical, e com um termo "corrompível"
    nd0 = {q: ndcg_at_k([d for d, _ in lex.search(t, TOP_K)], qrels[q]) for q, t in queries.items()}
    weight = lambda t: lex.searcher.idf(lex.pre.stem(t))
    eligible = [q for q in sorted(queries) if nd0[q] > 0 and pick_target_token(raw_tokens(queries[q]), weight, dictionary, stop)]
    sampled = random.Random(SEED).sample(eligible, N_QUERIES)             # sorteio com semente fixa: sem escolha manual
    cases = make_typo_cases(sampled, queries, lex, dictionary, stop, seed=SEED)
    assert len(cases) == N_QUERIES, "não foi possível gerar erro para todas as consultas sorteadas"

    sem = None
    if sel.get("bi_encoder"):
        spec = find_spec(mc["bi_encoders"], sel["bi_encoder"]["id"])
        sem = SemanticSystem(load_bi_encoder(spec), ctx.doc_ids, ctx.doc_texts, spec["id"].replace("@", "_"))
        sem.build()

    rows = []
    for c in cases:
        q = c["query_id"]
        text_corr, changes = correct_query(c["query_erro"], corrector)
        nd = lambda text: ndcg_at_k([d for d, _ in lex.search(text, TOP_K)], qrels[q])
        n_o, n_e, n_c = nd(c["query_original"]), nd(c["query_erro"]), nd(text_corr)
        rec = recovery(n_o, n_e, n_c)
        assert c["termo_erro"] not in dictionary, "o erro deve gerar termo inexistente no dicionário"
        row = {"query_id": q, "query_original": c["query_original"], "query_erro": c["query_erro"],
               "query_corrigida": text_corr, "tipo_erro": c["tipo_erro"], "NDCG10_original": round(n_o, 4),
               "NDCG10_erro": round(n_e, 4), "NDCG10_corrigida": round(n_c, 4),
               "recuperacao": round(rec, 4) if rec is not None else "erro não reduziu o desempenho",
               "termo_original": c["termo_original"], "termo_erro": c["termo_erro"],
               "termo_corrigido": changes.get(c["termo_erro"], ""), "correcao_exata": changes.get(c["termo_erro"]) == c["termo_original"],
               "correcoes_aplicadas": json.dumps(changes, ensure_ascii=False)}
        if sem is not None:          # opcional: robustez do sistema semântico aos mesmos erros
            nds = lambda text: ndcg_at_k([d for d, _ in sem.search(text, TOP_K)], qrels[q])
            s_o, s_e, s_c = nds(c["query_original"]), nds(c["query_erro"]), nds(text_corr)
            r_s = recovery(s_o, s_e, s_c)
            row.update({"NDCG10_sem_original": round(s_o, 4), "NDCG10_sem_erro": round(s_e, 4), "NDCG10_sem_corrigida": round(s_c, 4),
                        "recuperacao_sem": round(r_s, 4) if r_s is not None else "erro não reduziu o desempenho"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "typo_experiment.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
    print(df[["query_id", "query_original", "query_erro", "tipo_erro", "NDCG10_original", "NDCG10_erro", "NDCG10_corrigida", "recuperacao"]].to_string(index=False))

    valid = [r for r in rows if isinstance(r["recuperacao"], float)]
    summary = {"n_consultas": len(rows), "espaco_amostral_elegivel": len(eligible), "parametros_correcao": sp,
               "tamanho_dicionario": len(dictionary),
               "ndcg10_medio": {k: round(float(df[f"NDCG10_{k}"].mean()), 4) for k in ("original", "erro", "corrigida")},
               "n_com_queda": len(valid), "n_erro_nao_reduziu": len(rows) - len(valid),
               "recuperacao_media_por_consulta": round(float(np.mean([r["recuperacao"] for r in valid])), 4) if valid else None,
               "recuperacao_agregada": round(float((df["NDCG10_corrigida"].mean() - df["NDCG10_erro"].mean()) /
                                                   (df["NDCG10_original"].mean() - df["NDCG10_erro"].mean())), 4),
               "correcoes_exatas": int(df["correcao_exata"].sum()),
               "tipos": df["tipo_erro"].value_counts().to_dict()}
    if sem is not None:
        vs = [r for r in rows if isinstance(r["recuperacao_sem"], float)]
        summary["semantico"] = {"ndcg10_medio": {k: round(float(df[f"NDCG10_sem_{k}"].mean()), 4) for k in ("original", "erro", "corrigida")},
                                "n_com_queda": len(vs),
                                "recuperacao_media_por_consulta": round(float(np.mean([r["recuperacao_sem"] for r in vs])), 4) if vs else None}
    (RESULTS_DIR / "typo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
