"""Encoder BERT (inferência) em numpy puro, lendo pesos .safetensors.

Existe para rodar os modelos MiniLM em ambientes sem PyTorch (ex.: o sandbox em que parte dos experimentos
foi executada). Implementa exatamente a arquitetura BERT usada pelos modelos do projeto
(embeddings -> N camadas de self-attention + FFN com GELU exata -> pooler/classificador opcionais).
A correção foi validada contra o valor de referência publicado no card do cross-encoder (ver tests/).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

_SQRT2 = np.float32(np.sqrt(2.0))


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + erf(x / _SQRT2))


class NumpyBert:
    def __init__(self, tensors: dict[str, np.ndarray], heads: int = 12, eps: float = 1e-12) -> None:
        # aceita prefixo "bert." (modelo de classificação) ou sem prefixo (encoder puro)
        pre = "bert." if any(k.startswith("bert.embeddings") for k in tensors) else ""
        g = lambda name: np.ascontiguousarray(tensors[pre + name], dtype=np.float32)
        self.heads, self.eps = heads, eps
        self.word, self.pos, self.type = (g("embeddings.word_embeddings.weight"), g("embeddings.position_embeddings.weight"),
                                          g("embeddings.token_type_embeddings.weight"))
        self.emb_ln = (g("embeddings.LayerNorm.weight"), g("embeddings.LayerNorm.bias"))
        n_layers = 1 + max(int(k.split("encoder.layer.")[1].split(".")[0]) for k in tensors if "encoder.layer." in k)
        self.hidden = self.word.shape[1]
        self.layers = []
        for i in range(n_layers):
            p = f"encoder.layer.{i}."
            wqkv = np.concatenate([g(p + f"attention.self.{n}.weight").T for n in ("query", "key", "value")], axis=1)
            bqkv = np.concatenate([g(p + f"attention.self.{n}.bias") for n in ("query", "key", "value")])
            self.layers.append({
                "wqkv": np.ascontiguousarray(wqkv), "bqkv": bqkv,
                "wo": np.ascontiguousarray(g(p + "attention.output.dense.weight").T), "bo": g(p + "attention.output.dense.bias"),
                "ln1": (g(p + "attention.output.LayerNorm.weight"), g(p + "attention.output.LayerNorm.bias")),
                "wi": np.ascontiguousarray(g(p + "intermediate.dense.weight").T), "bi": g(p + "intermediate.dense.bias"),
                "wo2": np.ascontiguousarray(g(p + "output.dense.weight").T), "bo2": g(p + "output.dense.bias"),
                "ln2": (g(p + "output.LayerNorm.weight"), g(p + "output.LayerNorm.bias")),
            })
        self.pooler = (np.ascontiguousarray(g("pooler.dense.weight").T), g("pooler.dense.bias")) \
            if (pre + "pooler.dense.weight") in tensors else None
        self.classifier = (np.ascontiguousarray(tensors["classifier.weight"].astype(np.float32).T),
                           tensors["classifier.bias"].astype(np.float32)) if "classifier.weight" in tensors else None

    @classmethod
    def from_safetensors(cls, path, heads: int = 12) -> "NumpyBert":
        from safetensors import safe_open
        with safe_open(str(path), framework="numpy") as f:
            return cls({k: f.get_tensor(k) for k in f.keys()}, heads=heads)

    def forward(self, input_ids: np.ndarray, attention_mask: np.ndarray, token_type_ids: np.ndarray | None = None) -> np.ndarray:
        """Retorna os estados ocultos finais (B, L, H)."""
        B, L = input_ids.shape
        H, A = self.hidden, self.heads
        dh = H // A
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids)
        x = self.word[input_ids] + self.pos[:L][None] + self.type[token_type_ids]
        x = _layer_norm(x, *self.emb_ln, self.eps)
        mask_bias = ((1.0 - attention_mask.astype(np.float32)) * np.float32(-1e9))[:, None, None, :]
        scale = np.float32(1.0 / np.sqrt(dh))
        for ly in self.layers:
            x2 = x.reshape(B * L, H)
            qkv = (x2 @ ly["wqkv"] + ly["bqkv"]).reshape(B, L, 3, A, dh).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                               # (B, A, L, dh)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale + mask_bias
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            ctx = np.matmul(probs, v).transpose(0, 2, 1, 3).reshape(B * L, H)
            x = _layer_norm(x + (ctx @ ly["wo"] + ly["bo"]).reshape(B, L, H), *ly["ln1"], self.eps)
            h = _gelu(x.reshape(B * L, H) @ ly["wi"] + ly["bi"])
            x = _layer_norm(x + (h @ ly["wo2"] + ly["bo2"]).reshape(B, L, H), *ly["ln2"], self.eps)
        return x

    def classify(self, hidden: np.ndarray) -> np.ndarray:
        """Logit do cross-encoder: classificador(tanh(pooler(h[CLS])))  ->  (B,)"""
        assert self.pooler is not None and self.classifier is not None, "modelo sem pooler/classificador"
        pooled = np.tanh(hidden[:, 0] @ self.pooler[0] + self.pooler[1])
        return (pooled @ self.classifier[0] + self.classifier[1])[:, 0]

    @staticmethod
    def mean_pool(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        m = attention_mask[..., None].astype(np.float32)
        return (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
