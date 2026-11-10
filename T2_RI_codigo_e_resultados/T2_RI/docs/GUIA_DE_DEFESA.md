# Guia para a defesa individual (o enunciado diz o que pode ser pedido)

Onde está cada coisa e o que você deve saber explicar. Rode `python -m pytest -q tests` para ver cada componente verificado.

**Índice invertido (`src/inverted_index.py`)** — `postings: dict termo -> (array doc_idx, array tf)`, `doc_len`, `doc_ids`.
Construção: um laço sobre os documentos; para cada termo, `append` do doc_idx e do TF (os docs entram em ordem, então as listas já saem ordenadas).
Consulta (*term-at-a-time*): para cada termo da consulta, percorre-se **só** a posting list dele somando a contribuição BM25 num dicionário de acumuladores; nunca se varre a coleção.
Perguntas típicas: por que as posting lists estão ordenadas? o que muda se acrescentar um termo à consulta? como o DF é obtido? onde está o tamanho dos documentos?

**BM25 (`BM25Searcher`)** — `idf = ln(1 + (N-df+0,5)/(df+0,5))`; fator de TF `tf(k1+1)/(tf + k1(1-b+b|d|/avgdl))`. `k1` controla a saturação do TF; `b`, a normalização por comprimento (b=0 ignora o comprimento; b=1 normaliza totalmente).
Previsão de efeito: aumentar `b` penaliza mais os documentos longos; aumentar `k1` dá mais peso a repetições do termo.

**Tolerância a erros (`src/spelling.py`)** — k-gramas com `$` nas bordas → índice k-grama→termos → candidatos que compartilham k-gramas → filtro por Jaccard (|A∩B|/|A∪B|, calculado com contagem de k-gramas em comum) → distância de edição Damerau-Levenshtein (troca de adjacentes custa 1) → menor distância; desempate por Jaccard e frequência.

**RRF (`src/fusion.py`)** — `score(d)=Σ 1/(k+rank)`, rank a partir de 1; ausente num ranking = sem contribuição. Exemplo: `rank_lex=1`, `rank_sem=3`, k=60 → `1/61+1/63`. `k` pequeno dá mais peso às primeiras posições; `k` grande aproxima os documentos.
Veja `tests/test_fusion_rerank.py`.

**Semântico (`src/semantic.py`, `src/encoders.py`)** — embeddings L2-normalizados → produto interno = cosseno; busca exaustiva; embeddings dos documentos em cache. Truncamento: o modelo lê no máximo `max_seq_length` tokens; o que passa disso não é visto.

**Reranking (`src/rerank.py`)** — cross-encoder reordena **só o Top-50** do híbrido; o restante mantém a ordem relativa. O NDCG@10 só pode mudar pelos documentos que estão no Top-50.

**Métricas (`src/metrics.py`)** — P@10, R@10 (denominador = total de relevantes), AP@10 (soma de P@k nas posições relevantes ÷ total de relevantes), NDCG@10 (ganho linear, ideal com todos os julgamentos). Conferidas com a `ir_measures` em `tests/test_metrics.py`.

**RAG (`src/rag.py`, scripts 05–07)** — que documentos foram enviados (`results/rag_contexts.json`), onde estavam os relevantes em cada etapa (`results/analysis/rag_failure_traces.md`), por que uma resposta é fiel ou não (planilha de avaliação + justificativas).
