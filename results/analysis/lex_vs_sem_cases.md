# Lexical x Semântico — seis consultas com maior diferença de NDCG@10 (test)

### Caso 1 (Vantagem do semântico): PLAIN-3085 — "The Difficulty of Arriving at a Vitamin D Recommendation"

NDCG@10: lexical = 0.0000, semântico = 0.5706 (Δ = +0.5706).  Relevantes no qrels: 2; sem nenhum termo da consulta: 0; com todos os termos: 0; média de termos presentes: 1.0 de 4.

Termos da consulta (stems): difficulti (df=22, idf=5.08), arriv (df=7, idf=6.18), vitamin (df=222, idf=2.79), recommend (df=275, idf=2.58)

Resumo: {'lexical': {'primeiro_relevante': 19, 'relevantes_top5': 0, 'relevantes_top10': 0, 'relevantes_top100': 2}, 'semantico': {'primeiro_relevante': 3, 'relevantes_top5': 2, 'relevantes_top10': 2, 'relevantes_top100': 2}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-3632 | 0 | 12.5265 | >100 | 153 | 1 | An unexpected mortality increase in the United States follows arrival of the … |
| 2 | MED-2765 | 0 | 11.414 | 19 | 160 | 2 | The value of current nutrition information. |
| 3 | MED-919 | 0 | 11.2684 | 5 | 155 | 2 | Evaluation, treatment, and prevention of vitamin D deficiency: an Endocrine S… |
| 4 | MED-4872 | 0 | 11.1646 | 14 | 170 | 2 | Safety considerations and potential interactions of vitamins: should vitamins… |
| 5 | MED-4104 | 0 | 10.3949 | >100 | 275 | 2 | Changes in nutrient intake and dietary quality among participants with type 2… |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-714 | 0 | 0.7008 | 7 | 158 | 2 | Towards prevention of vitamin D deficiency and beyond: knowledge gaps and res… |
| 2 | MED-4566 | 0 | 0.6409 | 59 | 218 | 1 | Taking vitamin D with the largest meal improves absorption and results in hig… |
| 3 | MED-4573 | 2 | 0.635 | 19 | 135 | 1 | Low Vitamin D Status: Definition, Prevalence, Consequences and Correction |
| 4 | MED-4570 | 2 | 0.6172 | 47 | 146 | 1 | Vitamin D supplement doses and serum 25-hydroxyvitamin D in the range associa… |
| 5 | MED-919 | 0 | 0.6105 | 3 | 155 | 2 | Evaluation, treatment, and prevention of vitamin D deficiency: an Endocrine S… |

### Caso 2 (Vantagem do semântico): PLAIN-1537 — "low-carb diets"

NDCG@10: lexical = 0.0948, semântico = 0.6618 (Δ = +0.5670).  Relevantes no qrels: 45; sem nenhum termo da consulta: 16; com todos os termos: 0; média de termos presentes: 0.91 de 3.

Termos da consulta (stems): low (df=726, idf=1.61), carb (df=1, idf=7.79), diet (df=1041, idf=1.25)

Resumo: {'lexical': {'primeiro_relevante': 4, 'relevantes_top5': 1, 'relevantes_top10': 1, 'relevantes_top100': 7}, 'semantico': {'primeiro_relevante': 1, 'relevantes_top5': 4, 'relevantes_top10': 6, 'relevantes_top100': 8}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-4880 | 0 | 25.7822 | 2 | 178 | 3 | Effects of very-low-carbohydrate (horsemeat- or beef-based) diets and restric… |
| 2 | MED-2943 | 0 | 7.4133 | 43 | 252 | 2 | Long-term low-calorie low-protein vegan diet and endurance exercise are assoc… |
| 3 | MED-1714 | 0 | 7.2666 | >100 | 248 | 2 | Long-term low-protein, low-calorie diet and endurance exercise modulate metab… |
| 4 | MED-4515 | 1 | 6.9691 | 1 | 233 | 2 | The effect of a plant-based low-carbohydrate ("Eco-Atkins") diet on body weig… |
| 5 | MED-2458 | 0 | 6.9346 | >100 | 216 | 2 | Manipulating antioxidant intake in asthma: a randomized controlled trial. |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-4515 | 1 | 0.6623 | 4 | 233 | 2 | The effect of a plant-based low-carbohydrate ("Eco-Atkins") diet on body weig… |
| 2 | MED-4880 | 0 | 0.6094 | 1 | 178 | 3 | Effects of very-low-carbohydrate (horsemeat- or beef-based) diets and restric… |
| 3 | MED-4514 | 1 | 0.5962 | 18 | 308 | 2 | Low-carbohydrate diets and all-cause and cause-specific mortality: Two cohort… |
| 4 | MED-1616 | 1 | 0.587 | 43 | 241 | 2 | Very-low-carbohydrate ketogenic diet v. low-fat diet for long-term weight los… |
| 5 | MED-2586 | 1 | 0.5584 | >100 | 238 | 2 | Systematic review and meta-analysis of clinical trials of the effects of low … |

### Caso 3 (Vantagem do semântico): PLAIN-924 — "cocaine"

NDCG@10: lexical = 0.2201, semântico = 0.7779 (Δ = +0.5578).  Relevantes no qrels: 32; sem nenhum termo da consulta: 31; com todos os termos: 1; média de termos presentes: 0.03 de 1.

Termos da consulta (stems): cocain (df=1, idf=7.79)

Resumo: {'lexical': {'primeiro_relevante': 1, 'relevantes_top5': 1, 'relevantes_top10': 1, 'relevantes_top100': 1}, 'semantico': {'primeiro_relevante': 1, 'relevantes_top5': 4, 'relevantes_top10': 7, 'relevantes_top100': 18}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-3044 | 1 | 20.4291 | 1 | 217 | 1 | Cue-induced cocaine craving: neuroanatomical specificity for drug users and d… |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-3044 | 1 | 0.4667 | 1 | 217 | 1 | Cue-induced cocaine craving: neuroanatomical specificity for drug users and d… |
| 2 | MED-3054 | 1 | 0.3844 | >100 | 128 | 0 | Does food addiction exist? A phenomenological discussion based on the psychia… |
| 3 | MED-3055 | 1 | 0.3389 | >100 | 139 | 0 | Food and drug reward: overlapping circuits in human obesity and addiction. |
| 4 | MED-3052 | 1 | 0.3345 | >100 | 139 | 0 | Obesity and addiction: neurobiological overlaps. |
| 5 | MED-4790 | 0 | 0.3133 | >100 | 130 | 0 | The risks of copper toxicity contributing to cognitive decline in the aging p… |

### Caso 4 (Vantagem do lexical): PLAIN-1473 — "kohlrabi"

NDCG@10: lexical = 1.0000, semântico = 0.0000 (Δ = -1.0000).  Relevantes no qrels: 1; sem nenhum termo da consulta: 0; com todos os termos: 1; média de termos presentes: 1.0 de 1.

Termos da consulta (stems): kohlrabi (df=1, idf=7.79)

Resumo: {'lexical': {'primeiro_relevante': 1, 'relevantes_top5': 1, 'relevantes_top10': 1, 'relevantes_top100': 1}, 'semantico': {'primeiro_relevante': None, 'relevantes_top5': 0, 'relevantes_top10': 0, 'relevantes_top100': 0}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-4455 | 1 | 12.5956 | >100 | 118 | 1 | Glucoraphanin and 4-hydroxyglucobrassicin contents in seeds of 59 cultivars o… |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-3947 | 0 | 0.2202 | >100 | 169 | 0 | Hibiscus anthocyanins rich extract-induced apoptotic cell death in human prom… |
| 2 | MED-4660 | 0 | 0.2173 | >100 | 139 | 0 | Variability of iodine content in common commercially available edible seaweed… |
| 3 | MED-3441 | 0 | 0.2156 | >100 | 79 | 0 | Marine edible algae as disease preventers. |
| 4 | MED-3447 | 0 | 0.2128 | >100 | 112 | 0 | Seaweed prevents breast cancer? |
| 5 | MED-5024 | 0 | 0.2125 | >100 | 92 | 0 | Disseminated chlorellosis in a dog. |

### Caso 5 (Vantagem do lexical): PLAIN-2081 — "shelf life"

NDCG@10: lexical = 1.0000, semântico = 0.0000 (Δ = -1.0000).  Relevantes no qrels: 1; sem nenhum termo da consulta: 0; com todos os termos: 1; média de termos presentes: 2.0 de 2.

Termos da consulta (stems): shelf (df=7, idf=6.18), life (df=262, idf=2.63)

Resumo: {'lexical': {'primeiro_relevante': 1, 'relevantes_top5': 1, 'relevantes_top10': 1, 'relevantes_top100': 1}, 'semantico': {'primeiro_relevante': 15, 'relevantes_top5': 0, 'relevantes_top10': 0, 'relevantes_top100': 1}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-4303 | 1 | 17.3552 | 15 | 181 | 2 | Oxidative stability and shelf-life evaluation of selected culinary oils. |
| 2 | MED-1182 | 0 | 12.8995 | >100 | 182 | 2 | Fruit and Soil Quality of Organic and Conventional Strawberry Agroecosystems |
| 3 | MED-2700 | 0 | 12.048 | >100 | 230 | 2 | The effect of blood removal on oxidation and shelf life of broiler breast mea… |
| 4 | MED-2671 | 0 | 10.1531 | 35 | 100 | 2 | Microbiology of fresh and restructured lamb meat: a review. |
| 5 | MED-2074 | 0 | 8.7183 | 86 | 171 | 2 | Modifying the processing and handling of frozen broccoli for increased sulfor… |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-1378 | 0 | 0.4154 | 54 | 135 | 1 | Longevity and diet. Myth or pragmatism? |
| 2 | MED-2422 | 0 | 0.3863 | 61 | 143 | 1 | A statistical regression model for the estimation of acrylamide concentration… |
| 3 | MED-2510 | 0 | 0.3856 | 30 | 121 | 1 | Comparative and meta-analytic insights into life extension via dietary restri… |
| 4 | MED-4715 | 0 | 0.3818 | >100 | 232 | 0 | Effects of date ( Phoenix dactylifera L., Medjool or Hallawi Variety) consump… |
| 5 | MED-1433 | 0 | 0.3732 | >100 | 121 | 0 | Dietary Advanced Glycation End Products and Aging |

### Caso 6 (Vantagem do lexical): PLAIN-1288 — "grapes"

NDCG@10: lexical = 0.8900, semântico = 0.1299 (Δ = -0.7601).  Relevantes no qrels: 196; sem nenhum termo da consulta: 181; com todos os termos: 15; média de termos presentes: 0.08 de 1.

Termos da consulta (stems): grape (df=29, idf=4.81)

Resumo: {'lexical': {'primeiro_relevante': 1, 'relevantes_top5': 4, 'relevantes_top10': 9, 'relevantes_top100': 15}, 'semantico': {'primeiro_relevante': 9, 'relevantes_top5': 0, 'relevantes_top10': 2, 'relevantes_top100': 22}}

**Top-5 lexical** (rank no outro sistema = coluna 'rank sem')

| # | doc_id | rel | score | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-2668 | 1 | 11.6841 | 15 | 108 | 1 | Concord grape juice supplementation and neurocognitive function in human agin… |
| 2 | MED-2669 | 1 | 11.4134 | 13 | 142 | 1 | Concord grape juice supplementation improves memory function in older adults … |
| 3 | MED-2083 | 0 | 11.1106 | 5 | 182 | 1 | Grape juice, but not orange juice or grapefruit juice, inhibits human platele… |
| 4 | MED-1538 | 1 | 11.1032 | 26 | 183 | 1 | A premeal snack of raisins decreases mealtime food intake more than grapes in… |
| 5 | MED-1533 | 1 | 10.6914 | >100 | 183 | 1 | An after-school snack of raisins lowers cumulative food intake in young child… |

**Top-5 semantico** (rank no outro sistema = coluna 'rank lex')

| # | doc_id | rel | score | rank lex | |d| | termos | título |
|---|---|---|---|---|---|---|---|
| 1 | MED-3200 | 0 | 0.4264 | >100 | 142 | 0 | Prospective study of grapefruit intake and risk of breast cancer in postmenop… |
| 2 | MED-398 | 0 | 0.425 | >100 | 90 | 0 | The grapefruit: an old wine in a new glass? Metabolic and cardiovascular pers… |
| 3 | MED-3207 | 0 | 0.425 | >100 | 90 | 0 | The grapefruit: an old wine in a new glass? Metabolic and cardiovascular pers… |
| 4 | MED-3210 | 0 | 0.4243 | >100 | 206 | 0 | The effects of daily consumption of grapefruit on body weight, lipids, and bl… |
| 5 | MED-2083 | 0 | 0.4141 | 3 | 182 | 1 | Grape juice, but not orange juice or grapefruit juice, inhibits human platele… |
