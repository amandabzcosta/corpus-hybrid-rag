from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
import pandas as pd

from src.config import RESULTS_DIR
from src.rag import CRITERIA, make_blind_sheet, merge_grades

def main():
    ap = argparse.ArgumentParser(description='RAG, parte 3: avaliação humana estruturada (notas 0/1/2) — fidelidade, completude, utilidade, rastreabilidade.', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["make-sheet", "merge"])
    ap.add_argument("--evaluators", nargs="+", default=["amanda", "leonardo", "avaliador_ia"])
    ap.add_argument("--final", help="planilha com as notas finais (humanas) gravadas no rag_experiment.csv")
    ap.add_argument("--sheets", nargs="+", help="planilhas independentes, para a concordância")
    ap.add_argument("--force", action="store_true", help="make-sheet: sobrescreve planilhas já existentes")
    args = ap.parse_args()
    rag = pd.read_csv(RESULTS_DIR / "rag_experiment.csv", dtype=str, keep_default_na=False)
    if args.cmd == "make-sheet":
        ctx = json.loads((RESULTS_DIR / "rag_contexts.json").read_text(encoding="utf-8"))
        contexts = {q: [(d["doc_id"], d["title"], d["text"]) for d in v] for q, v in ctx.items()}
        existing = [RESULTS_DIR / f"rag_eval_{n}.csv" for n in args.evaluators if (RESULTS_DIR / f"rag_eval_{n}.csv").exists()]
        if existing and not args.force:
            raise SystemExit(f"planilhas já existem (use --force para sobrescrever): {[p.name for p in existing]}")
        sheet, key = make_blind_sheet(rag, contexts, seed=42)
        for name in args.evaluators:
            sheet.to_csv(RESULTS_DIR / f"rag_eval_{name}.csv", index=False)
        key.to_csv(RESULTS_DIR / "rag_annotation_key.csv", index=False)
        print(f"{len(sheet)} itens embaralhados em results/rag_eval_{{{','.join(args.evaluators)}}}.csv (NÃO abra a chave antes de avaliar).")
        print("Cada avaliador preenche uma cópia: colunas fidelidade/completude/utilidade/rastreabilidade (0,1,2) e just_*.")
        print("Na planilha, avalie a rastreabilidade só se a resposta citar fontes; a condição A vira N/A no merge.")
        return
    key = pd.read_csv(RESULTS_DIR / "rag_annotation_key.csv", dtype=str)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    sheets = {Path(p).stem.removeprefix("rag_eval_"): read(p) for p in args.sheets or []}
    if not args.final:
        raise SystemExit("merge: informe --final com a planilha das notas finais")
    merged, means, agree = merge_grades(rag, key, read(args.final), sheets)
    merged.to_csv(RESULTS_DIR / "rag_experiment.csv", index=False)
    means.to_csv(RESULTS_DIR / "rag_generation_means.csv")
    (RESULTS_DIR / "rag_agreement.json").write_text(json.dumps(agree, ensure_ascii=False, indent=2), encoding="utf-8")
    print(means.to_string())
    print(json.dumps(agree, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
