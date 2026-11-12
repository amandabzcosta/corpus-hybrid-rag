from __future__ import annotations

import re
import unicodedata

from nltk.stem import PorterStemmer, SnowballStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))

def raw_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))

def english_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords
        words = stopwords.words("english")
    except LookupError:  # primeira execução: baixa o recurso
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        words = stopwords.words("english")
    out = set()
    for w in words:
        out.add(normalize(w))
        out.update(raw_tokens(w))
    return frozenset(out)

class Preprocessor:

    def __init__(self, remove_stopwords: bool = True, stemmer: str | None = "porter"):
        if stemmer not in (None, "porter", "snowball"):
            raise ValueError("stemmer deve ser None, 'porter' ou 'snowball'")
        self.remove_stopwords = remove_stopwords
        self.stemmer_name = stemmer
        self.stopwords = english_stopwords() if remove_stopwords else frozenset()
        if stemmer == "porter":
            self._stemmer = PorterStemmer()
        elif stemmer == "snowball":
            self._stemmer = SnowballStemmer("english")
        else:
            self._stemmer = None
        self._cache: dict[str, str] = {}

    def stem(self, token: str) -> str:
        if self._stemmer is None:
            return token
        s = self._cache.get(token)
        if s is None:
            s = self._cache[token] = self._stemmer.stem(token)
        return s

    def __call__(self, text: str) -> list[str]:
        toks = raw_tokens(text)
        if self.remove_stopwords:
            toks = [t for t in toks if t not in self.stopwords]
        return [self.stem(t) for t in toks] if self._stemmer is not None else toks

    def describe(self) -> dict:
        return {"tokenizacao": "regex [a-z0-9]+ sobre texto em minúsculas sem acentos (NFKD)",
                "stopwords": "NLTK (inglês)" if self.remove_stopwords else "não removidas",
                "stemming": self.stemmer_name or "nenhum"}
