# T2 — Recuperação de Informação (SCC0282, 2º sem/2026): lexical + semântico + híbrido (RRF) + reranking + RAG no NFCorpus

Pipeline: documentos → pré-processamento e indexação → recuperação lexical + semântica → busca híbrida → reranking → seleção de contexto → LLM → resposta.

## 1. Integrantes
| Nome completo | Nº USP |
|---|---|
| **Amanda Bezerra Costa** | **12680861** |
| **Leonardo Ferreira Runho** | **11832958** |



## 2. Escolhas do projeto (todas feitas no conjunto dev)
| Item | Escolha |
|---|---|
| Modelo lexical | **BM25**, k1 = 2,0 e b = 0,5; índice invertido implementado do zero (`src/inverted_index.py`); stopwords NLTK (inglês) + stemmer Snowball |
| Correção ortográfica | k-gramas (k = 2) → Jaccard ≥ 0,3 → distância de Damerau-Levenshtein ≤ 1 |
| Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2`, 384 dim, **máx. 256 tokens** (78,8% dos documentos truncados), cosseno com vetores L2-normalizados |
| RRF | k = 10 (avaliados 5…200, incluindo 10, 60 e 100); fusão dos 100 primeiros de cada sistema |
| Modelo de reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` sobre o Top-50 híbrido, máx. 512 tokens (9,4% dos pares truncados) |
| LLM do RAG | Claude Sonnet 5.5 via interface de chat (sem API; temperatura não controlável; custo 0) — ver §6 |
| Linguagem / bibliotecas | Python 3.12.3; numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, nltk 3.10.4, onnxruntime 1.31.0, tokenizers 0.23.3, safetensors 0.8.0, ir_measures 0.4.3, matplotlib 3.11.2 |
| Hardware da execução entregue | VM Linux, Intel Xeon 2,10 GHz, **1 vCPU**, 3,9 GB RAM, sem GPU |

Resultados no test (323 consultas): NDCG@10 — lexical 0,3303; semântico 0,3167; híbrido 0,3555; híbrido+reranking 0,3557 (tabela completa, P@10, R@10, MAP@10 e testes de significância no relatório e em `results/summary_test.csv`, `results/significance_tests.csv`).

## 3. Instalação e execução
```bash
pip install -r requirements.txt          # Python >= 3.10
python scripts/run_all.py                # roda 00 -> 05 pulando etapas concluídas (retomável)
python -m pytest -q tests                # 24 testes (índice vs força bruta, métricas vs ir_measures, RRF, ortografia...)
```
`run_all.py` usa `configs/models_hf.json` (sentence-transformers; baixa os pesos do Hugging Face). Ordem recomendada, caso queira rodar manualmente (ou use `notebooks/T2_pipeline.ipynb`, que serve no Colab):

| # | Script | O que faz |
|---|---|---|
| 00 | `00_download_data.py` | baixa o NFCorpus oficial do BEIR, confere md5 e estatísticas, e a lista das 10 consultas do RAG |
| 01 | `01_dev_tuning.py --stage all` | pré-processamento, BM25, ortografia, bi-encoder, k do RRF e cross-encoder, **só no dev** → `results/dev_*.csv`, `dev_selection.json` |
| 02 | `02_run_test_pipeline.py` | lexical, semântico, híbrido, reranking no test → `results_*.csv`, `metrics_per_query.csv`, `timings.json` |
| 03 | `03_typo_experiment.py` | 10 consultas com erro: original × erro × corrigida → `typo_experiment.csv` |
| 04 | `04_analysis_cases.py` | 6 casos lexical×semântico e 4 casos de reranking (seleção automática) → `results/analysis/` |
| 05 | `05_rag_prepare.py` | Top-5 pós-reranking, Recall/Precision/Hit@5, prompts A/B e rastreio de falhas |
| 06 | `06_rag_generate.py --backend ...` | LLM nas condições A e B → `rag_experiment.csv` |
| 07 | `07_rag_evaluate.py make-sheet / merge` | planilha cega e incorporação das notas humanas, médias e concordância |
| 08 | `08_make_report_assets.py` | tabelas, figuras e macros do relatório (`report/auto/`) |

Relatório: `cd report && pdflatex relatorio && bibtex relatorio && pdflatex relatorio && pdflatex relatorio` (precisa de `texlive-lang-portuguese`; ou envie a pasta `report/` ao Overleaf).

Backends de modelos: `kind: "st"` (sentence-transformers, padrão), `"onnx"` (onnxruntime) e `"numpy"` (BERT em numpy puro, `src/numpy_bert.py`). Os dois últimos existem para rodar sem PyTorch/Hugging Face; `configs/models_local.json` é a configuração usada na execução entregue (pesos em `models/`, não incluídos).

## 4. Arquivos de resultados (`results/`)
- `results_lexical.csv`, `results_semantic.csv`, `results_hybrid.csv`, `results_reranking.csv`: `query_id, doc_id, rank, score, relevance` (100 primeiros por consulta; `results_reranking.csv` tem `score_type`: top-50 = logit do cross-encoder, demais = score RRF original). Consultas sem nenhum termo no índice (15) não têm linhas em `results_lexical.csv`.
- `metrics_per_query.csv`: `query_id, model, P10, R10, AP10, NDCG10` (a média de AP10 é o MAP@10 reportado).
- `typo_experiment.csv`: `query_id, query_original, query_erro, query_corrigida, tipo_erro, NDCG10_original, NDCG10_erro, NDCG10_corrigida` + recuperação, termos e (opcional) resultados do sistema semântico.
- `rag_experiment.csv`: `query_id, query, condition (A/B), context_doc_ids, prompt, answer`, as quatro notas e as justificativas (**notas em branco até a avaliação humana**).
- Apoio: `dev_*.csv`, `dev_selection.json`, `summary_test.csv`, `significance_tests.csv`, `timings.json`, `typo_summary.json`, `rag_*` e `analysis/` (evidências por consulta citadas no relatório).
- `rag_annotation_key.csv` revela quem é A ou B: **não abra antes de avaliar**.

## 5. Proveniência, verificações e limitações 
- **Dados:** o servidor oficial do BEIR e o Hugging Face não eram acessíveis no ambiente onde os experimentos rodaram; usei espelhos públicos do NFCorpus no GitHub (duas cópias independentes, idênticas). Validação: 3.633 documentos, 3.237 consultas, 323/324 consultas no test/dev, 12.334 julgamentos no test, 42 perguntas no test e a lista exata das 10 consultas do RAG do enunciado. **Rode `00_download_data.py` com o zip oficial** (confere o md5 publicado pelo BEIR) para confirmar.
- **Modelos:** pesos de espelhos no GitHub. O bi-encoder (ONNX) reproduz os embeddings do `sentence-transformers` (cosseno = 1,000000) e o cross-encoder (NumPy) reproduz os escores de referência do card do modelo (8,607141 e −4,320078 contra 8,607138 e −4,320078). Os backends `st` seguem a API documentada do sentence-transformers, mas **não foram executados** neste ambiente: se algo quebrar no Colab, os backends `onnx`/`numpy` funcionam sem PyTorch.
- **Tempos** valem para 1 vCPU (o reranking leva ~12 s por consulta em NumPy; em GPU é bem menor).
- **Reranking:** escolhido no dev (30 consultas, +0,026), mas no test o NDCG@10 ficou igual ao do híbrido (+0,0002, p = 0,96). É um resultado real, discutido no relatório.
- **RAG:** não havia API de LLM no ambiente. As respostas das 2 condições foram escritas pelo Claude em chat: as da condição A **antes** de qualquer documento recuperado ser visualizado, e as da B só com o contexto fornecido (arquivo `results/rag_answers_manual.jsonl`, reproduzível com `--backend manual`). Não há controle de temperatura/semente. As notas 0–2 **não** foram atribuídas por IA.

## 6. Estrutura
`src/` módulos (dados, pré-processamento, índice invertido, BM25, ortografia, encoders, semântico, fusão, rerank, métricas, análise, RAG) · `scripts/` etapas 00–08 e `run_all.py` · `configs/` modelos · `tests/` · `notebooks/` · `results/` · `report/` (fonte LaTeX, tabelas/figuras geradas e PDF) · `docs/`.
