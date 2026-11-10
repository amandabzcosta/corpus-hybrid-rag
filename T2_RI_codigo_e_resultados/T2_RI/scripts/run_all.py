"""Executa o pipeline completo na ordem recomendada, pulando etapas já concluídas (idempotente e retomável).

  python scripts/run_all.py                 # roda tudo o que falta (00 -> 05)
  python scripts/run_all.py --force dev-rerank test    # refaz etapas específicas
  python scripts/run_all.py --list          # mostra o estado de cada etapa
Etapas manuais (LLM e avaliação humana): scripts 06 e 07, ver README."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402

S = ROOT / "scripts"
sel = lambda: json.loads((RESULTS_DIR / "dev_selection.json").read_text()) if (RESULTS_DIR / "dev_selection.json").exists() else {}
STAGES = [
    ("dados", ["00_download_data.py"], lambda: (DATA_DIR / "corpus.jsonl").exists()),
    ("dev-lexical", ["01_dev_tuning.py", "--stage", "lexical"], lambda: "bm25" in sel()),
    ("dev-spelling", ["01_dev_tuning.py", "--stage", "spelling"], lambda: "spelling" in sel()),
    ("dev-semantic", ["01_dev_tuning.py", "--stage", "semantic"], lambda: "bi_encoder" in sel()),
    ("dev-rrf", ["01_dev_tuning.py", "--stage", "rrf"], lambda: "rrf_k" in sel()),
    ("dev-rerank", ["01_dev_tuning.py", "--stage", "rerank"], lambda: "cross_encoder" in sel()),
    ("test", ["02_run_test_pipeline.py"], lambda: (RESULTS_DIR / "timings.json").exists()),
    ("typo", ["03_typo_experiment.py"], lambda: (RESULTS_DIR / "typo_summary.json").exists()),
    ("analise", ["04_analysis_cases.py"], lambda: (RESULTS_DIR / "analysis" / "global_evidence.json").exists()),
    ("rag-prepare", ["05_rag_prepare.py"], lambda: (RESULTS_DIR / "rag_prompts.jsonl").exists()),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", nargs="*", default=[], help="nomes das etapas a refazer")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--rerank-dev-queries", type=int, default=30)
    ap.add_argument("--models", default=None)
    ap.add_argument("--skip-data", action="store_true", help="não executa a etapa 'dados' (dataset já copiado)")
    args = ap.parse_args()
    if args.list:
        for name, _, done in STAGES:
            print(f"{'[ok]' if done() else '[  ]'} {name}")
        return
    for name, cmd, done in STAGES:
        if name == "dados" and args.skip_data:
            continue
        if done() and name not in args.force:
            print(f"== {name}: concluída (pulando)", flush=True)
            continue
        extra = ["--rerank-dev-queries", str(args.rerank_dev_queries)] if name == "dev-rerank" else []
        if args.models and name.startswith(("dev-", "test", "typo")) and name != "dev-lexical":
            extra += ["--models", args.models]
        print(f"\n== {name}: executando {' '.join(cmd)}", flush=True)
        rc = subprocess.call([sys.executable, str(S / cmd[0]), *cmd[1:], *extra], cwd=str(S))
        if rc != 0:
            print(f"[!] etapa '{name}' terminou com código {rc}")
            raise SystemExit(rc)
    print("\nPipeline concluído. Próximos passos manuais: scripts 06 (LLM) e 07 (avaliação humana).")


if __name__ == "__main__":
    main()
