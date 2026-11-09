# TP2 — Recuperação de Informação (SCC0282, ICMC-USP, 2026)

Sistema de RI sobre o **NFCorpus (BEIR)** que combina busca lexical (BM25 sobre um índice invertido próprio), busca semântica (bi-encoder), busca híbrida (RRF), reranking (cross-encoder) e RAG.

## Ambiente

- Python **3.11** (ambiente conda). As versões das bibliotecas estão fixadas em `requirements.txt`.
- Hardware: Intel Core 5 210H (12 threads), 15,6 GB de RAM, NVIDIA RTX 3050 Laptop 6 GB. Embeddings e reranking rodam na **CPU** (torch CPU), por reprodutibilidade.

```bash
conda create -n tp2ri python=3.11
conda activate tp2ri
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # só se a LLM for acessada por API
```

## Configuração

Todos os parâmetros estão em [`config.yaml`](config.yaml).

| Componente | Escolha |
|---|---|
| Modelo lexical | BM25 (k1 = 1.2, b = 0.75) — [a confirmar no dev] |
| Embeddings | [Etapa 6] |
| Cross-encoder | [Etapa 9] |
| LLM | [Etapa 11] |

## Dados e artefatos (fora do Git)

- `data/`: dataset. Para gerar: `python scripts/00_download_data.py` (baixa o zip oficial do BEIR e valida as contagens).
- `artifacts/`: índice serializado, embeddings, cache da LLM. São regenerados pelos scripts 01–08.

## Ordem de execução

| Script | O que faz |
|---|---|
| `00_download_data.py` | download e validação do NFCorpus |
| `01_build_index.py` | índice invertido |
| `02_run_lexical.py` | BM25 → `results/results_lexical.csv` |
| `03_typo_experiment.py` | correção ortográfica → `results/typo_experiment.csv` |
| `04_run_semantic.py` | bi-encoder → `results/results_semantic.csv` |
| `05_eval_lexical_vs_semantic.py` | comparação por consulta |
| `06_tune_and_run_hybrid.py` | RRF → `results/results_hybrid.csv` |
| `07_run_reranking.py` | cross-encoder → `results/results_reranking.csv` |
| `08_rag_generate.py` | respostas A/B da LLM |
| `09_rag_make_blind_sheet.py` | planilha cega para a avaliação humana |
| `10_rag_merge_scores.py` | merge das notas → `results/rag_experiment.csv` |
| `11_make_figures_tables.py` | tabelas e figuras do relatório |

## Estrutura

- `src/`: módulos (dados, pré-processamento, índice, BM25, correção, semântico, híbrido, reranking, RAG, métricas)
- `scripts/`: um script por etapa
- `results/`: CSVs padronizados (versionados e entregues)
- `report/`: relatório e figuras
- `tests/`: testes (`pytest`)
