from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import CACHE_DIR
from .encoders import TimeBudgetExceeded

class SemanticSystem:
    def __init__(self, encoder, doc_ids: list[str], doc_texts: list[str], name: str, cache_dir: Path = CACHE_DIR,
                 chunk_docs: int = 128, deadline: float | None = None) -> None:
        self.encoder, self.doc_ids, self.doc_texts, self.name = encoder, doc_ids, doc_texts, name
        self.chunk_docs, self.deadline = chunk_docs, deadline
        sig = hashlib.sha1(("\x00".join(doc_texts[:50]) + str(len(doc_texts)) + str(encoder.max_seq_length)).encode()).hexdigest()[:10]
        self.dir = Path(cache_dir) / f"emb_{name}_{encoder.max_seq_length}_{sig}"
        self.emb: np.ndarray | None = None
        self.build_info: dict = {}

    def build(self, verbose: bool = True) -> dict:
        final = self.dir / "embeddings.npy"
        if final.exists():
            self.emb = np.load(final)
            self.build_info = json.loads((self.dir / "info.json").read_text())
            self.build_info["encoder"] = self.encoder.info()
            return self.build_info
        self.dir.mkdir(parents=True, exist_ok=True)
        lengths = self.encoder.token_lengths(self.doc_texts)                  # sem truncamento
        order = np.argsort(-lengths, kind="stable")                           # blocos de tamanho parecido
        n_chunks = int(np.ceil(len(order) / self.chunk_docs))
        parts_t = {}
        for c in range(n_chunks):
            part = self.dir / f"part_{c:04d}.npy"
            meta = self.dir / f"part_{c:04d}.json"
            if part.exists() and meta.exists():
                parts_t[c] = json.loads(meta.read_text())["seconds"]
                continue
            if self.deadline is not None and time.time() > self.deadline:
                raise TimeBudgetExceeded(f"embeddings de {self.name}: {c}/{n_chunks} blocos prontos (progresso salvo)")
            idx = order[c * self.chunk_docs:(c + 1) * self.chunk_docs]
            t0 = time.perf_counter()
            vec = self.encoder.encode([self.doc_texts[i] for i in idx], batch_size=32)
            dt = time.perf_counter() - t0
            np.save(part, vec)
            meta.write_text(json.dumps({"seconds": dt, "n": len(idx)}))
            parts_t[c] = dt
            if verbose:
                print(f"  [{self.name}] bloco {c + 1}/{n_chunks} ({dt:.1f}s)", flush=True)
        emb = np.zeros((len(self.doc_ids), np.load(self.dir / "part_0000.npy").shape[1]), np.float32)
        for c in range(n_chunks):
            emb[order[c * self.chunk_docs:(c + 1) * self.chunk_docs]] = np.load(self.dir / f"part_{c:04d}.npy")
        L = self.encoder.max_seq_length
        self.build_info = {
            "embedding_seconds": round(float(sum(parts_t.values())), 2), "n_docs": len(self.doc_ids), "dim": int(emb.shape[1]),
            "max_seq_length": int(L), "docs_truncated": int((lengths > L).sum()),
            "pct_docs_truncated": round(float((lengths > L).mean() * 100), 2),
            "tokens_mean": round(float(lengths.mean()), 1), "tokens_median": float(np.median(lengths)),
            "tokens_p95": float(np.percentile(lengths, 95)), "tokens_max": int(lengths.max()),
            "encoder": self.encoder.info(),
        }
        np.save(final, emb)
        (self.dir / "info.json").write_text(json.dumps(self.build_info, ensure_ascii=False, indent=1))
        for c in range(n_chunks):
            (self.dir / f"part_{c:04d}.npy").unlink()
        self.emb = emb
        return self.build_info

    def encode_query(self, text: str) -> np.ndarray:
        return self.encoder.encode([text], is_query=True)[0]

    def search_vec(self, qvec: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        sims = self.emb @ qvec                                                 # busca exaustiva (cosseno)
        k = min(top_k, len(sims))
        cand = np.argpartition(-sims, k - 1)[:k]
        cand = cand[np.lexsort((cand, -sims[cand]))]                           # empate: menor índice
        return [(self.doc_ids[i], float(sims[i])) for i in cand]

    def search(self, text: str, top_k: int = 100) -> list[tuple[str, float]]:
        return self.search_vec(self.encode_query(text), top_k)

    def run(self, queries: dict[str, str], top_k: int = 100):
        qids = list(queries)
        qv = self.encoder.encode([queries[q] for q in qids], is_query=True)
        return {q: self.search_vec(v, top_k) for q, v in zip(qids, qv)}

    def timed_run(self, queries: dict[str, str], top_k: int = 100, warmup: int = 3):
        items = list(queries.items())
        for _, t in items[:warmup]:
            self.search(t, top_k)
        run, times = {}, []
        for qid, text in items:
            t0 = time.perf_counter()
            run[qid] = self.search(text, top_k)
            times.append(time.perf_counter() - t0)
        return run, times
