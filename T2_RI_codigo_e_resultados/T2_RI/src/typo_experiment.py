"""Experimento de tolerância a erros (item 4): construção dos casos com erro ortográfico e correção.

Protocolo (padronização do enunciado):
 (i)   dicionário = termos da coleção antes de stemming/lematização;
 (ii)  cada erro gera um termo INEXISTENTE no dicionário;
 (iii) só entram consultas com NDCG@10 original > 0 (sistema lexical);
 (iv)  recuperação por consulta = (NDCG_corrigida - NDCG_erro) / (NDCG_original - NDCG_erro) se o denominador > 0.
Decisões do grupo (sem usar qrels): um erro por consulta, no termo de maior IDF (o mais informativo) com >= 5
letras que não seja stopword; a 1ª letra é preservada; os 4 tipos de erro são atribuídos em rodízio.
"""
from __future__ import annotations

import random
from typing import Mapping

from .config import SEED
from .preprocess import raw_tokens
from .spelling import (ERROR_TYPES, SpellCorrector, apply_corrections, make_typo, pick_target_token, replace_word)


def make_typo_cases(query_ids, queries: Mapping[str, str], lex, dictionary, stopwords, seed: int = SEED) -> list[dict]:
    """Um caso por consulta, com tipo de erro atribuído em rodízio (fallback para outro tipo se for impossível)."""
    weight = lambda t: lex.searcher.idf(lex.pre.stem(t))
    cases = []
    for i, qid in enumerate(query_ids):
        text = queries[qid]
        toks = raw_tokens(text)
        target = pick_target_token(toks, weight, dictionary, stopwords)
        if target is None:
            continue
        for j in range(len(ERROR_TYPES)):
            kind = ERROR_TYPES[(i + j) % len(ERROR_TYPES)]
            typo = make_typo(target, kind, random.Random(f"{seed}:{qid}:{kind}"), dictionary)
            if typo is None:
                continue
            text_err = replace_word(text, target, typo)
            if text_err is None:
                continue
            cases.append({"query_id": qid, "query_original": text, "query_erro": text_err,
                          "termo_original": target, "termo_erro": typo, "tipo_erro": kind})
            break
    return cases


def correct_query(text_err: str, corrector: SpellCorrector) -> tuple[str, dict]:
    """Aplica a correção a todos os termos desconhecidos da consulta. Retorna (texto corrigido, {errado: corrigido})."""
    _, changes = corrector.correct_tokens(raw_tokens(text_err))
    return apply_corrections(text_err, changes), changes


def recovery(ndcg_orig: float, ndcg_err: float, ndcg_corr: float):
    """(corrigida - erro) / (original - erro) quando o denominador > 0; caso contrário None (erro não reduziu o desempenho)."""
    den = ndcg_orig - ndcg_err
    return (ndcg_corr - ndcg_err) / den if den > 1e-12 else None
