from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

CRITERIA = ("fidelidade", "completude", "utilidade", "rastreabilidade")

SYSTEM_A = ("You are a knowledgeable assistant answering questions about nutrition and health. "
            "Answer the question in English, in at most 200 words.")
SYSTEM_B = ("You are a careful assistant answering questions about nutrition and health. "
            "Answer the question in English, in at most 200 words, using ONLY the context documents provided by the user. "
            "Cite the documents you rely on by their identifiers in square brackets, e.g. [MED-123]. "
            "If the context does not contain enough evidence to answer the question (or part of it), say so explicitly "
            "instead of guessing.")

def format_context(docs: Sequence[tuple[str, str, str]]) -> str:
    return "\n\n".join(f"[{doc_id}] {title}\n{text}" for doc_id, title, text in docs)

def user_prompt_a(question: str) -> str:
    return f"Question: {question}"

def user_prompt_b(question: str, docs: Sequence[tuple[str, str, str]]) -> str:
    return f"Context documents:\n\n{format_context(docs)}\n\nQuestion: {question}"

def full_prompt(system: str, user: str) -> str:
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"

def retrieval_at_k(context_doc_ids: Sequence[str], qrels_q: Mapping[str, int], k: int = 5) -> dict:
    ids = list(context_doc_ids)[:k]
    n_rel = sum(1 for g in qrels_q.values() if g > 0)
    hits = sum(1 for d in ids if qrels_q.get(d, 0) > 0)
    return {f"Recall@{k}": hits / n_rel if n_rel else 0.0, f"Precision@{k}": hits / k, f"Hit@{k}": int(hits > 0),
            "n_relevantes_no_contexto": hits, "n_relevantes_total": n_rel}

class LLMBackend:
    name = "base"

    def generate(self, system: str, user: str) -> tuple[str, dict]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name}

class MockBackend(LLMBackend):
    name = "mock"

    def generate(self, system, user):
        return "[MOCK] resposta de teste", {}

class ManualBackend(LLMBackend):
    name = "manual"

    def __init__(self, path: str, meta: dict | None = None) -> None:
        self.answers = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                self.answers[(r["query_id"], r["condition"])] = r["answer"]
        self.meta = meta or {}
        self.key = None

    def generate(self, system, user):
        return self.answers[self.key], {}

    def describe(self):
        return {"backend": "manual", **self.meta}

class OpenAICompatBackend(LLMBackend):
    name = "openai-compat"

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", temperature: float = 0.0,
                 max_tokens: int = 400, seed: int = 42, api_key_env: str = "OPENAI_API_KEY") -> None:
        self.model, self.base_url, self.temperature, self.max_tokens, self.seed = model, base_url.rstrip("/"), temperature, max_tokens, seed
        self.api_key = os.environ.get(api_key_env, "")

    def generate(self, system, user):
        import requests
        r = requests.post(f"{self.base_url}/chat/completions", timeout=600,
                          headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                          json={"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "seed": self.seed,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"], {"usage": j.get("usage", {})}

    def describe(self):
        return {"backend": self.name, "model": self.model, "base_url": self.base_url, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "seed": self.seed, **self.ollama_info()}

    def ollama_info(self) -> dict:
        import requests
        root = self.base_url.removesuffix("/v1")
        try:
            tags = requests.get(f"{root}/api/tags", timeout=10).json()["models"]
            show = requests.post(f"{root}/api/show", json={"model": self.model}, timeout=10).json()
        except Exception:
            return {}
        tag = next((m for m in tags if m["name"] == self.model), {})
        return {"execucao": "local (Ollama)", "ollama_digest": tag.get("digest", ""), "ollama_details": show.get("details", {}),
                "custo": 0}

class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 400) -> None:
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(self, system, user):
        import requests
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=180,
                          headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                          json={"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                                "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        return "".join(b.get("text", "") for b in j["content"]), {"usage": j.get("usage", {})}

    def describe(self):
        return {"backend": self.name, "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens}

class HFBackend(LLMBackend):
    name = "hf"

    def __init__(self, model: str, max_new_tokens: int = 400, seed: int = 42) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.model_name, self.max_new_tokens, self.seed = model, max_new_tokens, seed
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map="auto")
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    def generate(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.lm.device)
        out = self.lm.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), {"prompt_tokens": int(ids.shape[1])}

    def describe(self):
        return {"backend": self.name, "model": self.model_name, "temperature": 0, "do_sample": False,
                "max_new_tokens": self.max_new_tokens, "seed": self.seed, "hardware": self.gpu}

def make_blind_sheet(rag_df: pd.DataFrame, contexts: Mapping[str, list], seed: int = 42):
    order = list(range(len(rag_df)))
    random.Random(seed).shuffle(order)
    sheet, key = [], []
    for n, i in enumerate(order, start=1):
        r = rag_df.iloc[i]
        item = f"R{n:02d}"
        ctx_txt = "\n\n".join(f"[{d}] {t}\n{x}" for d, t, x in contexts[r["query_id"]])
        sheet.append({"item": item, "query_id": r["query_id"], "pergunta": r["query"], "resposta": r["answer"],
                      "top5_contexto_de_referencia": ctx_txt,
                      **{c: "" for c in CRITERIA}, **{f"just_{c}": "" for c in CRITERIA}})
        key.append({"item": item, "query_id": r["query_id"], "condition": r["condition"]})
    return pd.DataFrame(sheet), pd.DataFrame(key)

def _kappa(a: Sequence[float], b: Sequence[float]) -> float | None:
    from sklearn.metrics import cohen_kappa_score
    a, b = list(a), list(b)
    if len(set(a)) < 2 and len(set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2]))

def _read_grades(sheet: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    df = sheet.set_index("item")
    out = df[list(CRITERIA)].apply(pd.to_numeric, errors="coerce")
    for c in CRITERIA:
        out[f"just_{c}"] = df[f"just_{c}"]
    k = key.set_index("item")
    out["condition"] = k.loc[out.index, "condition"].values
    out["query_id"] = k.loc[out.index, "query_id"].values
    return out

def merge_grades(rag_df: pd.DataFrame, key: pd.DataFrame, final: pd.DataFrame, sheets: Mapping[str, pd.DataFrame]):
    rag = rag_df.copy()
    g = _read_grades(final, key).set_index(["query_id", "condition"])
    for c in CRITERIA:
        rag[c] = [g.loc[(q, cond), c] for q, cond in zip(rag.query_id, rag.condition)]
        rag[f"justificativa_{c}"] = [g.loc[(q, cond), f"just_{c}"] for q, cond in zip(rag.query_id, rag.condition)]
    is_a = rag.condition == "A"
    rag.loc[is_a, "rastreabilidade"] = np.nan
    means = rag.groupby("condition")[list(CRITERIA)].mean().round(3)
    rag["rastreabilidade"] = rag["rastreabilidade"].astype(object)
    rag.loc[is_a, "rastreabilidade"] = "N/A"
    rag.loc[is_a, "justificativa_rastreabilidade"] = "N/A: condição A sem contexto externo nem fontes (excluída da média)"
    rag["contexto_externo"] = np.where(is_a, "sem contexto externo", "top-5 após reranking")
    grades = {name: _read_grades(s, key) for name, s in sheets.items()}
    names = list(grades)
    agree = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d1, d2 = grades[names[i]], grades[names[j]].loc[grades[names[i]].index]
            pair = {}
            for c in CRITERIA:
                m = d1[c].notna() & d2[c].notna() & ((d1["condition"] == "B") if c == "rastreabilidade" else True)
                if m.sum():
                    pair[c] = {"n": int(m.sum()), "concordancia_exata": round(float((d1.loc[m, c] == d2.loc[m, c]).mean()), 3),
                               "kappa_ponderado_quadratico": _kappa(d1.loc[m, c], d2.loc[m, c])}
            agree[f"{names[i]} x {names[j]}"] = pair
    return rag, means, agree
