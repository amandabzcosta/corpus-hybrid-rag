"""Tolerância a erros (item 4): correção ortográfica de termos desconhecidos.

Pipeline:  k-gramas (geração de candidatos) -> coeficiente de Jaccard (filtro) -> distância de edição (seleção).
Dicionário: termos da coleção ANTES de stopwords/stemming (padronização do enunciado, item (i)).
Distância de edição: Damerau-Levenshtein (versão "optimal string alignment"): inserção, remoção, substituição
e troca de caracteres adjacentes custam 1 — os quatro tipos de erro pedidos no enunciado.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping

from .preprocess import raw_tokens

ERROR_TYPES = ("substituição", "inserção", "remoção", "troca")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def build_dictionary(texts: Iterable[str]) -> Counter:
    """Dicionário = tokens normalizados da coleção (antes de stopwords/stemming) com frequência na coleção."""
    c: Counter = Counter()
    for t in texts:
        c.update(raw_tokens(t))
    return c


def kgrams(term: str, k: int = 2) -> set[str]:
    padded = f"${term}$"
    return {padded[i:i + k] for i in range(len(padded) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def edit_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (OSA)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


class KGramIndex:
    """Índice k-grama -> termos do dicionário que o contêm."""

    def __init__(self, vocabulary: Mapping[str, int], k: int = 2) -> None:
        self.k = k
        self.terms = sorted(vocabulary)
        self.freq = [vocabulary[t] for t in self.terms]
        self.n_grams: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for tid, term in enumerate(self.terms):
            grams = kgrams(term, k)
            self.n_grams.append(len(grams))
            for g in grams:
                self.postings[g].append(tid)

    def candidates(self, term: str, threshold: float) -> list[tuple[int, float]]:
        """(id do termo, Jaccard) dos termos que compartilham k-gramas e passam do limiar.
        |A ∩ B| é obtido contando em quantas posting lists de k-gramas da consulta o termo aparece."""
        qgrams = kgrams(term, self.k)
        common: Counter = Counter()
        for g in qgrams:
            common.update(self.postings.get(g, ()))
        out = []
        for tid, c in common.items():
            j = c / (len(qgrams) + self.n_grams[tid] - c)
            if j >= threshold:
                out.append((tid, j))
        return out


class SpellCorrector:
    def __init__(self, vocabulary: Mapping[str, int], k: int = 2, jaccard_threshold: float = 0.3,
                 max_edit: int = 2, min_len: int = 3) -> None:
        self.vocab = vocabulary
        self.k, self.jaccard_threshold, self.max_edit, self.min_len = k, jaccard_threshold, max_edit, min_len
        self.kg = KGramIndex(vocabulary, k)

    def is_unknown(self, token: str) -> bool:
        return token.isalpha() and len(token) >= self.min_len and token not in self.vocab

    def suggestions(self, token: str, top_n: int = 3) -> list[dict]:
        scored = []
        for tid, j in self.kg.candidates(token, self.jaccard_threshold):
            cand = self.kg.terms[tid]
            e = edit_distance(token, cand)
            if e <= self.max_edit:
                # menor distância de edição; desempate: maior Jaccard, maior frequência na coleção, ordem alfabética
                scored.append((e, -j, -self.kg.freq[tid], cand))
        scored.sort()
        return [{"termo": c, "edit": e, "jaccard": round(-nj, 4), "freq": -nf} for e, nj, nf, c in scored[:top_n]]

    def correct_tokens(self, tokens: list[str]) -> tuple[list[str], dict[str, str]]:
        """Corrige cada token desconhecido pela melhor sugestão. Retorna (tokens corrigidos, {errado: corrigido})."""
        out, changes = [], {}
        for t in tokens:
            if self.is_unknown(t):
                sug = self.suggestions(t, 1)
                if sug:
                    changes[t] = sug[0]["termo"]
                    t = sug[0]["termo"]
            out.append(t)
        return out, changes


# ---------- geração controlada de erros ortográficos ----------
def make_typo(term: str, kind: str, rng: random.Random, vocabulary: Mapping[str, int], max_tries: int = 200) -> str | None:
    """Gera um erro de tipo `kind` que NÃO exista no dicionário (item (ii) do enunciado). A 1ª letra é preservada
    (erro de digitação mais comum). Retorna None se não for possível."""
    n = len(term)
    for _ in range(max_tries):
        if kind == "inserção":
            pos = rng.randint(1, n)
            cand = term[:pos] + rng.choice(ALPHABET) + term[pos:]
        elif kind == "remoção":
            if n < 4:
                return None
            pos = rng.randint(1, n - 1)
            cand = term[:pos] + term[pos + 1:]
        elif kind == "substituição":
            pos = rng.randint(1, n - 1)
            cand = term[:pos] + rng.choice([c for c in ALPHABET if c != term[pos]]) + term[pos + 1:]
        elif kind == "troca":
            positions = [i for i in range(1, n - 1) if term[i] != term[i + 1]]
            if not positions:
                return None
            pos = rng.choice(positions)
            cand = term[:pos] + term[pos + 1] + term[pos] + term[pos + 2:]
        else:
            raise ValueError(kind)
        if cand != term and cand not in vocabulary:
            return cand
    return None


def pick_target_token(tokens: list[str], weight, vocabulary: Mapping[str, int], stopwords, min_len: int = 5) -> str | None:
    """Termo da consulta a corromper: o de maior peso (IDF), entre os alfabéticos, com >= min_len letras,
    que não sejam stopwords e existam no dicionário. Não usa qrels."""
    cands = [t for t in dict.fromkeys(tokens) if t.isalpha() and len(t) >= min_len and t not in stopwords and t in vocabulary]
    return max(cands, key=lambda t: (weight(t), -tokens.index(t))) if cands else None


def replace_word(text: str, old: str, new: str) -> str | None:
    """Troca `old` por `new` (palavra inteira, sem diferenciar maiúsculas) preservando o formato do texto.
    Retorna None se `old` não for encontrada."""
    pattern = re.compile(r"\b" + re.escape(old) + r"\b", flags=re.IGNORECASE)

    def repl(m: re.Match) -> str:
        w = m.group(0)
        if len(w) > 1 and w.isupper():
            return new.upper()
        return new.capitalize() if w[0].isupper() else new

    out, n = pattern.subn(repl, text)
    return out if n else None


def apply_corrections(text: str, changes: Mapping[str, str]) -> str:
    for wrong, right in changes.items():
        replaced = replace_word(text, wrong, right)
        if replaced is not None:
            text = replaced
    return text
