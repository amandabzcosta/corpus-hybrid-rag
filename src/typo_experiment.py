from __future__ import annotations

import random
from typing import Mapping

from .config import SEED
from .preprocess import raw_tokens
from .spelling import (ERROR_TYPES, SpellCorrector, apply_corrections, make_typo, pick_target_token, replace_word)

def make_typo_cases(query_ids, queries: Mapping[str, str], lex, dictionary, stopwords, seed: int = SEED) -> list[dict]:
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
    _, changes = corrector.correct_tokens(raw_tokens(text_err))
    return apply_corrections(text_err, changes), changes

def recovery(ndcg_orig: float, ndcg_err: float, ndcg_corr: float):
    den = ndcg_orig - ndcg_err
    return (ndcg_corr - ndcg_err) / den if den > 1e-12 else None
