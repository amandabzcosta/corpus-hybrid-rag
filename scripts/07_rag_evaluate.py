"""RAG, parte 3: avaliação humana estruturada (notas 0/1/2) — fidelidade, completude, utilidade, rastreabilidade.

  python scripts/07_rag_evaluate.py make-sheet            # gera planilha cega e embaralhada (cópia por avaliador)
  python scripts/07_rag_evaluate.py merge --sheets avaliador1.csv [avaliador2.csv]
O merge grava as notas em results/rag_experiment.csv (média dos avaliadores), calcula as médias por condição e,
com dois avaliadores, a concordância (exata e kappa quadrático). Rastreabilidade da condição A = N/A."""
from __future__ import annotations

import argparse
import json

from _common import ROOT  # noqa: F401
import pandas as pd

from src.config import RESULTS_DIR
from src.rag import CRITERIA, make_blind_sheet, merge_grades


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["make-sheet", "merge"])
    ap.add_argument("--sheets", nargs="+")
    args = ap.parse_args()
    rag = pd.read_csv(RESULTS_DIR / "rag_experiment.csv", dtype=str, keep_default_na=False)
    if args.cmd == "make-sheet":
        ctx = json.loads((RESULTS_DIR / "rag_contexts.json").read_text(encoding="utf-8"))
        contexts = {q: [(d["doc_id"], d["title"], d["text"]) for d in v] for q, v in ctx.items()}
        sheet, key = make_blind_sheet(rag, contexts, seed=42)
        sheet.to_csv(RESULTS_DIR / "rag_annotation_sheet.csv", index=False)
        key.to_csv(RESULTS_DIR / "rag_annotation_key.csv", index=False)
        print(f"{len(sheet)} itens embaralhados em results/rag_annotation_sheet.csv (NÃO abra a chave antes de avaliar).")
        print("Cada avaliador preenche uma cópia: colunas fidelidade/completude/utilidade/rastreabilidade (0,1,2) e just_*.")
        print("Na planilha, avalie a rastreabilidade só se a resposta citar fontes; a condição A vira N/A no merge.")
        return
    key = pd.read_csv(RESULTS_DIR / "rag_annotation_key.csv", dtype=str)
    sheets = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in args.sheets]
    merged, means, agree = merge_grades(rag, key, sheets)
    merged.to_csv(RESULTS_DIR / "rag_experiment.csv", index=False)
    means.to_csv(RESULTS_DIR / "rag_generation_means.csv")
    (RESULTS_DIR / "rag_agreement.json").write_text(json.dumps(agree, ensure_ascii=False, indent=2), encoding="utf-8")
    print(means.to_string())
    print(json.dumps(agree, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
