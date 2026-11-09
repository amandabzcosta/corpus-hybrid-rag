import re
import unicodedata
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

TOKEN_RE = re.compile(r"[^\W_]+")

def load_stopwords(lang="english", keep=()):
    try:
        words = stopwords.words(lang)
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words(lang)
    return set(words) - set(keep)

def make_stemmer(name):
    if name == "porter":
        return PorterStemmer()
    if name == "snowball":
        return SnowballStemmer("english")
    return None

class Preprocessor:
    def __init__(self, lowercase=True, remove_stopwords=True, stopwords_lang="english",
                 stopwords_keep=(), stemmer="porter", keep_numbers=True):
        self.lowercase = lowercase
        self.keep_numbers = keep_numbers
        self.stopwords = load_stopwords(stopwords_lang, stopwords_keep) if remove_stopwords else set()
        self.stemmer = make_stemmer(stemmer)
        self.stem = lru_cache(maxsize=None)(self.stemmer.stem) if self.stemmer else (lambda t: t)

    def tokenize(self, text):
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        tokens = TOKEN_RE.findall(text)
        if not self.keep_numbers:
            tokens = [t for t in tokens if not t.isdigit()]
        return [t for t in tokens if t not in self.stopwords]

    def __call__(self, text):
        return [self.stem(t) for t in self.tokenize(text)]

    def analyze(self, text):
        return [(t, self.stem(t)) for t in self.tokenize(text)]

def from_config(cfg):
    return Preprocessor(**cfg["preprocessing"])
