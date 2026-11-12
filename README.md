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
| LLM do RAG | `qwen2.5:7b` (Q4_K_M, 7,6B, digest `845dbda0…`) local via Ollama 0.40.16, API compatível com OpenAI; temperatura 0, seed 42, `max_tokens` 400, respostas em inglês nas duas condições; custo 0 |
| Linguagem / bibliotecas | Python 3.11.18; torch 2.13.0+cpu, sentence-transformers 6.2.0, transformers 5.22.2, numpy 2.4.6, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.9.1, nltk 3.10.4, ir_measures 0.4.3, matplotlib 3.11.2 (lista completa em `requirements.txt`) |
| Hardware da execução entregue | Windows 11, Intel Core 5 210H (12 threads), 15,6 GB RAM, NVIDIA RTX 3050 Laptop 6 GB. Embeddings e reranking em CPU (torch CPU); a LLM usa a GPU (≈4,2 de 5,2 GB do modelo na VRAM, o resto na CPU) |

Resultados no test (323 consultas): NDCG@10 — lexical 0,3303; semântico 0,3167; híbrido 0,3555; híbrido+reranking 0,3557 (tabela completa, P@10, R@10, MAP@10 e testes de significância no relatório e em `results/summary_test.csv`, `results/significance_tests.csv`).

## 3. Instalação e execução
```bash
conda create -n tp2ri python=3.11 && conda activate tp2ri
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/run_all.py                # roda 00 -> 05 pulando etapas concluídas (retomável)
python -m pytest -q tests                # 27 testes (índice vs força bruta, métricas vs ir_measures, RRF, ortografia...)
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
| 06 | `06_rag_generate.py --backend openai --base-url http://localhost:11434/v1 --model qwen2.5:7b` | LLM nas condições A e B → `rag_experiment.csv` (cache em `artifacts/llm_cache.jsonl`; `--no-cache` gera de novo). Requer `ollama pull qwen2.5:7b` |
| 07 | `07_rag_evaluate.py make-sheet` / `merge --final rag_eval_amanda_final.csv --sheets rag_eval_amanda.csv rag_eval_avaliador_ia.csv ...` | planilhas cegas (uma por avaliador) e incorporação das notas finais humanas, médias por condição e concordância entre avaliadores |
| 08 | `08_make_report_assets.py` | tabelas, figuras e macros do relatório (`report/auto/`) |

Arquivos fora do Git (regenerados pelos scripts): `data/` (dataset, script 00), `artifacts/` (embeddings, scores do cross-encoder e cache da LLM; scripts 01, 02 e 06) e `models/` (pesos dos backends opcionais ONNX/NumPy). Os parâmetros escolhidos no dev ficam em `results/dev_selection.json`, que os scripts 02 a 05 leem.

Relatório: `cd report && pdflatex relatorio && bibtex relatorio && pdflatex relatorio && pdflatex relatorio` (precisa de `texlive-lang-portuguese`; ou envie a pasta `report/` ao Overleaf).

Backends de modelos: `kind: "st"` (sentence-transformers, padrão), `"onnx"` (onnxruntime) e `"numpy"` (BERT em numpy puro, `src/numpy_bert.py`). Os dois últimos existem para rodar sem PyTorch/Hugging Face; `configs/models_local.json` os usa (pesos em `models/`, não incluídos). A execução entregue usa o backend `st`.

## 4. Arquivos de resultados (`results/`)
- `results_lexical.csv`, `results_semantic.csv`, `results_hybrid.csv`, `results_reranking.csv`: `query_id, doc_id, rank, score, relevance` (100 primeiros por consulta; `results_reranking.csv` tem `score_type`: top-50 = logit do cross-encoder, demais = score RRF original). Consultas sem nenhum termo no índice (15) não têm linhas em `results_lexical.csv`.
- `metrics_per_query.csv`: `query_id, model, P10, R10, AP10, NDCG10` (a média de AP10 é o MAP@10 reportado).
- `typo_experiment.csv`: `query_id, query_original, query_erro, query_corrigida, tipo_erro, NDCG10_original, NDCG10_erro, NDCG10_corrigida` + recuperação, termos e (opcional) resultados do sistema semântico.
- `rag_experiment.csv`: `query_id, query, condition (A/B), context_doc_ids, prompt, answer`, as quatro notas, as justificativas e `contexto_externo` (A = "sem contexto externo"; rastreabilidade de A = N/A).
- `rag_eval_<avaliador>.csv`: planilhas cegas de cada avaliador; `rag_eval_avaliador_ia.csv` é a avaliação feita por IA, usada **só** como segundo avaliador para medir a concordância.
- Apoio: `dev_*.csv`, `dev_selection.json`, `summary_test.csv`, `significance_tests.csv`, `timings.json`, `typo_summary.json`, `rag_*` e `analysis/` (evidências por consulta citadas no relatório).
- `rag_annotation_key.csv` revela quem é A ou B: **não abra antes de avaliar**.

## 5. Proveniência, verificações e limitações
- **Dados:** zip oficial do BEIR (md5 `a89dba18a62ef92f7d323ec890a0d38d`, confere com o publicado). Validação: 3.633 documentos, 3.237 consultas, 323/324 consultas no test/dev, 12.334 julgamentos no test, 42 perguntas no test e a lista exata das 10 consultas do RAG do enunciado.
- **Modelos:** pesos do Hugging Face via sentence-transformers (revisões registradas em `results/timings.json`: bi-encoder `1110a243…`, cross-encoder `233902d2…`). Os backends `onnx`/`numpy` reproduzem o sentence-transformers (cosseno = 1,000000; escores do card do cross-encoder 8,607141 e −4,320078) e numa execução anterior em VM deram exatamente as mesmas métricas.
- **Métricas:** conferidas com o ir_measures nos 4 sistemas (diferença máxima 0,0).
- **Reranking:** escolhido no dev com as 324 consultas: o L6 (0,3316) ficou um pouco abaixo do híbrido sem reranking (0,3326), e o L12 foi pior (0,3195). O enunciado exige a etapa, e o L6 foi o melhor cross-encoder. No test: +0,0002 sobre o híbrido (p = 0,96).
- **RAG:** temperatura 0 e seed 42, mas o Ollama não foi totalmente determinístico (o modelo não cabe inteiro nos 6 GB de VRAM): uma 2ª geração sem cache repetiu 8 de 20 respostas palavra por palavra. As respostas avaliadas estão congeladas em `rag_experiment.csv` e em `artifacts/llm_cache.jsonl`.
- **Avaliação do RAG:** as notas finais são humanas. Uma avaliação por IA (Claude, cega à condição pela planilha, embora as citações `[MED-…]` denunciem a condição B) foi usada como segundo avaliador para a concordância humano × IA.

## 6. Estrutura
`src/` módulos (dados, pré-processamento, índice invertido, BM25, ortografia, encoders, semântico, fusão, rerank, métricas, análise, RAG) · `scripts/` etapas 00–08 e `run_all.py` · `configs/` modelos · `tests/` · `notebooks/` · `results/` · `report/` (fonte LaTeX, tabelas/figuras geradas e PDF) · `docs/`.
