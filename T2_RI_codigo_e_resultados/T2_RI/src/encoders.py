"""Backends de bi-encoder e cross-encoder (itens 5 e 9).

kind = "st"    : sentence-transformers (padrão; baixa os modelos do Hugging Face) — recomendado no Colab/PC local.
kind = "onnx"  : onnxruntime + tokenizers, com os pesos do MESMO modelo exportados para ONNX.
kind = "numpy" : encoder BERT em numpy puro (src/numpy_bert.py), lendo model.safetensors.
Os dois últimos existem para ambientes sem PyTorch/Hugging Face; os três expõem a mesma interface.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .config import ROOT


class TimeBudgetExceeded(RuntimeError):
    """Orçamento de tempo da execução esgotado; o progresso já está salvo em cache — execute de novo para continuar."""


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _sha256(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# =====================================================================================================
#  Base para backends locais (tokenização com `tokenizers`, batches ordenados por tamanho)
# =====================================================================================================
class _LocalBert:
    def __init__(self, spec: dict, max_len_key: str, default_max_len: int) -> None:
        from tokenizers import Tokenizer
        self.spec = spec
        self.dir = _resolve(spec["path"])
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.max_len = int(spec.get(max_len_key, default_max_len))
        self.engine = spec["kind"]
        self.token_budget = int(spec.get("token_budget", 6000))   # limita B*L por batch (memória)
        if self.engine == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(spec.get("threads", os.cpu_count() or 1))
            self.onnx_file = self.dir / spec.get("onnx_file", "model.onnx")
            self.sess = ort.InferenceSession(str(self.onnx_file), so, providers=["CPUExecutionProvider"])
            self._names = {i.name for i in self.sess.get_inputs()}
        elif self.engine == "numpy":
            from .numpy_bert import NumpyBert
            self.net = NumpyBert.from_safetensors(self.dir / spec.get("weights_file", "model.safetensors"),
                                                  heads=int(spec.get("heads", 12)))
        else:
            raise ValueError(f"engine inválido: {self.engine}")

    # tokenização ------------------------------------------------------------------------------
    def _tokenize(self, inputs, truncate: bool = True):
        self.tok.no_padding()
        if truncate:
            self.tok.enable_truncation(max_length=self.max_len)   # estratégia longest_first (padrão HF)
        else:
            self.tok.no_truncation()
        return self.tok.encode_batch(inputs)

    def _run(self, encs):
        """Executa um batch de Encoding -> saída bruta do modelo (hidden states ou logits)."""
        n, L = len(encs), max(len(e.ids) for e in encs)
        ids = np.zeros((n, L), np.int64)
        am = np.zeros((n, L), np.int64)
        tt = np.zeros((n, L), np.int64)
        for i, e in enumerate(encs):
            k = len(e.ids)
            ids[i, :k], am[i, :k], tt[i, :k] = e.ids, 1, e.type_ids
        if self.engine == "numpy":
            return self.net.forward(ids, am, tt), am
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self._names:
            feed["token_type_ids"] = tt
        return self.sess.run(None, feed), am

    def _batches(self, encs):
        order = np.argsort([-len(e.ids) for e in encs], kind="stable")      # mais longos primeiro
        i = 0
        while i < len(order):
            L = len(encs[order[i]].ids)
            bs = max(1, min(64, self.token_budget // max(L, 1)))
            yield order[i:i + bs]
            i += bs


# =====================================================================================================
#  Bi-encoders
# =====================================================================================================
class LocalBiEncoder(_LocalBert):
    kind = "local"

    def __init__(self, spec: dict) -> None:
        super().__init__(spec, "max_seq_length", 256)
        self.query_prefix, self.doc_prefix = spec.get("query_prefix", ""), spec.get("doc_prefix", "")
        self.max_seq_length = self.max_len

    def encode(self, texts, batch_size: int = 32, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        pre = self.query_prefix if is_query else self.doc_prefix
        encs = self._tokenize([pre + t for t in texts])
        out = None
        for idx in self._batches(encs):
            raw, am = self._run([encs[i] for i in idx])
            hidden = raw if self.engine == "numpy" else raw[0]            # ONNX: output 0 = token_embeddings
            m = am[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)          # mean pooling
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)   # normalização L2
            if out is None:
                out = np.zeros((len(encs), v.shape[1]), np.float32)
            out[idx] = v
        return out

    def token_lengths(self, texts) -> np.ndarray:
        return np.array([len(e.ids) for e in self._tokenize(list(texts), truncate=False)])

    def info(self) -> dict:
        weights = self.onnx_file if self.engine == "onnx" else self.dir / self.spec.get("weights_file", "model.safetensors")
        import tokenizers
        lib = f"tokenizers {tokenizers.__version__} + " + (f"onnxruntime {__import__('onnxruntime').__version__}"
                                                           if self.engine == "onnx" else "NumpyBert (numpy puro)")
        return {"model": self.spec["name"], "revision": self.spec.get("revision", f"pesos sha256:{_sha256(weights)}"),
                "library": lib, "dim": int(self.spec.get("dim", 384)), "max_seq_length": self.max_seq_length,
                "similarity": "cosseno (produto interno de vetores normalizados)", "normalization": "L2",
                "pooling": "média dos tokens", "source": self.spec.get("source", "")}


class STBiEncoder:
    kind = "st"

    def __init__(self, spec: dict) -> None:
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.model = SentenceTransformer(spec["name"], device=spec.get("device"))
        if spec.get("max_seq_length"):
            self.model.max_seq_length = int(spec["max_seq_length"])
        self.max_seq_length = int(self.model.max_seq_length)
        self.query_prefix, self.doc_prefix = spec.get("query_prefix", ""), spec.get("doc_prefix", "")

    def encode(self, texts, batch_size: int = 32, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        pre = self.query_prefix if is_query else self.doc_prefix
        return np.asarray(self.model.encode([pre + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=show_progress), dtype=np.float32)

    def token_lengths(self, texts) -> np.ndarray:
        ids = self.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
        return np.array([len(x) for x in ids])

    def info(self) -> dict:
        import sentence_transformers
        import torch
        rev = None
        try:
            rev = self.model[0].auto_model.config._commit_hash
        except Exception:
            pass
        return {"model": self.spec["name"], "revision": rev or self.spec.get("revision", "n/d"),
                "library": f"sentence-transformers {sentence_transformers.__version__} (torch {torch.__version__})",
                "dim": int(self.model.get_sentence_embedding_dimension()), "max_seq_length": self.max_seq_length,
                "similarity": "cosseno (produto interno de vetores normalizados)", "normalization": "L2",
                "pooling": "conforme o modelo", "source": "Hugging Face Hub"}


# =====================================================================================================
#  Cross-encoders
# =====================================================================================================
class LocalCrossEncoder(_LocalBert):
    kind = "local"

    def __init__(self, spec: dict) -> None:
        super().__init__(spec, "max_length", 512)
        self.max_length = self.max_len

    def predict(self, pairs, batch_size: int = 32) -> np.ndarray:
        pairs = [(q, d) for q, d in pairs]
        encs = self._tokenize(pairs)
        out = np.zeros(len(encs), np.float32)
        for idx in self._batches(encs):
            raw, _ = self._run([encs[i] for i in idx])
            if self.engine == "numpy":
                out[idx] = self.net.classify(raw)
            else:
                out[idx] = np.asarray(raw[0], dtype=np.float32).reshape(len(idx), -1)[:, 0]   # logit (ativação Identity)
        return out

    def pair_token_lengths(self, pairs) -> np.ndarray:
        return np.array([len(e.ids) for e in self._tokenize([(q, d) for q, d in pairs], truncate=False)])

    def info(self) -> dict:
        weights = self.onnx_file if self.engine == "onnx" else self.dir / self.spec.get("weights_file", "model.safetensors")
        import tokenizers
        lib = f"tokenizers {tokenizers.__version__} + " + (f"onnxruntime {__import__('onnxruntime').__version__}"
                                                           if self.engine == "onnx" else "NumpyBert (numpy puro)")
        return {"model": self.spec["name"], "revision": self.spec.get("revision", f"pesos sha256:{_sha256(weights)}"),
                "library": lib, "max_length": self.max_length, "score": "logit (ativação Identity)",
                "source": self.spec.get("source", "")}


class STCrossEncoder:
    kind = "st"

    def __init__(self, spec: dict) -> None:
        from sentence_transformers import CrossEncoder
        self.spec = spec
        self.model = CrossEncoder(spec["name"], max_length=int(spec.get("max_length", 512)), device=spec.get("device"))
        self.max_length = int(spec.get("max_length", 512))

    def predict(self, pairs, batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.predict([(q, d) for q, d in pairs], batch_size=batch_size,
                                             show_progress_bar=False), dtype=np.float32).reshape(-1)

    def pair_token_lengths(self, pairs) -> np.ndarray:
        pairs = [(q, d) for q, d in pairs]
        ids = self.model.tokenizer([q for q, _ in pairs], [d for _, d in pairs], add_special_tokens=True,
                                   truncation=False)["input_ids"]
        return np.array([len(x) for x in ids])

    def info(self) -> dict:
        import sentence_transformers
        import torch
        rev = None
        try:
            rev = self.model.model.config._commit_hash
        except Exception:
            pass
        return {"model": self.spec["name"], "revision": rev or self.spec.get("revision", "n/d"),
                "library": f"sentence-transformers {sentence_transformers.__version__} (torch {torch.__version__})",
                "max_length": self.max_length, "score": "saída do modelo (logit/ativação padrão do CrossEncoder)",
                "source": "Hugging Face Hub"}


class CachedCrossEncoder:
    """Envolve um cross-encoder com cache em disco (JSONL, append-only) por (modelo, consulta, documento).
    Torna a execução retomável e evita recomputar pares já avaliados (ex.: dev e test compartilham pares)."""

    def __init__(self, inner, path: Path, deadline: float | None = None) -> None:
        self.inner, self.path, self.deadline = inner, Path(path), deadline
        self.cache: dict[str, float] = {}
        self.compute_seconds = 0.0        # tempo efetivo de inferência acumulado nesta execução
        self.n_computed = 0
        self.stats_path = self.path.with_suffix(".stats.json")
        self.stats = json.loads(self.stats_path.read_text()) if self.stats_path.exists() else {"pairs": 0, "seconds": 0.0}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    k, v = line.rstrip("\n").split("\t")
                    self.cache[k] = float(v)

    @staticmethod
    def _key(q: str, d: str) -> str:
        return hashlib.sha1(f"{q}\x00{d}".encode("utf-8")).hexdigest()[:24]

    def predict(self, pairs, batch_size: int = 32, chunk: int = 64) -> np.ndarray:
        pairs = list(pairs)
        keys = [self._key(q, d) for q, d in pairs]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        for s in range(0, len(missing), chunk):
            if self.deadline is not None and time.time() > self.deadline:
                raise TimeBudgetExceeded("orçamento de tempo esgotado (progresso salvo)")
            part = missing[s:s + chunk]
            t0 = time.perf_counter()
            scores = self.inner.predict([pairs[i] for i in part], batch_size=batch_size)
            dt = time.perf_counter() - t0
            self.compute_seconds += dt
            self.n_computed += len(part)
            self.stats["pairs"] += len(part)
            self.stats["seconds"] += dt
            with open(self.path, "a", encoding="utf-8") as f:
                for i, sc in zip(part, scores):
                    self.cache[keys[i]] = float(sc)
                    f.write(f"{keys[i]}\t{float(sc)!r}\n")
            self.stats_path.write_text(json.dumps(self.stats))
        return np.array([self.cache[k] for k in keys], dtype=np.float32)

    def seconds_per_pair(self) -> float:
        return self.stats["seconds"] / self.stats["pairs"] if self.stats["pairs"] else float("nan")

    def __getattr__(self, name):          # delega pair_token_lengths, info, max_length...
        return getattr(self.inner, name)


# =====================================================================================================
def load_bi_encoder(spec: dict):
    return STBiEncoder(spec) if spec["kind"] == "st" else LocalBiEncoder(spec)


def load_cross_encoder(spec: dict):
    return STCrossEncoder(spec) if spec["kind"] == "st" else LocalCrossEncoder(spec)
