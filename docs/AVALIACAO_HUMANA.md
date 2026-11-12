# Avaliação humana do RAG (item 13) — como preencher

Notas **0, 1 ou 2** para cada critério, com **uma justificativa curta por nota**. Idealmente dois integrantes avaliam **de forma independente** (cada um preenche uma cópia da planilha) — o `merge` calcula a concordância.

| Critério | 0 | 1 | 2 |
|---|---|---|---|
| **Fidelidade** | Afirmações importantes não são sustentadas pelo contexto ou o contradizem | Parcialmente sustentada | As afirmações principais são sustentadas pelo contexto |
| **Completude** | Não responde aos aspectos principais | Responde parcialmente | Cobre adequadamente os aspectos que podem ser respondidos |
| **Utilidade** | Não responde à necessidade da consulta | Parcialmente útil | Responde diretamente à necessidade apresentada |
| **Rastreabilidade** | Não é possível identificar as fontes | Só parte das afirmações pode ser associada às fontes | As afirmações principais podem ser associadas aos doc_ids apresentados |

Regras do enunciado:
- A planilha (`results/rag_annotation_sheet.csv`) está **embaralhada e sem a condição (A/B)**. Não abra `rag_annotation_key.csv` antes de terminar.
- Para **todas** as respostas, a *fidelidade* é julgada contra os **Top-5 documentos** da coluna `top5_contexto_de_referencia` (os mesmos usados na condição B). Na condição A isso significa verificar se o que a LLM disse, sem acesso à coleção, é consistente com essas evidências.
- *Rastreabilidade*: avalie quando a resposta citar fontes; na condição A ela vira **N/A** automaticamente no `merge` (e fica fora da média).
- Escreva a justificativa com fatos (ex.: "afirma X, mas o MED-123 diz o contrário"), não com impressões.

Passos:
```bash
python scripts/07_rag_evaluate.py make-sheet            # gera a planilha cega (já gerada em results/)
# cada avaliador preenche sua cópia (fidelidade, completude, utilidade, rastreabilidade, just_*)
python scripts/07_rag_evaluate.py merge --sheets avaliador1.csv avaliador2.csv
python scripts/08_make_report_assets.py                 # atualiza a tabela de notas do relatório
```
Depois disso, escolha os 3 casos da análise de falhas ponta a ponta (item 14) olhando as médias por consulta e
`results/analysis/rag_failure_traces.md` (onde estão os relevantes em cada etapa).
