# Rastreio ponta a ponta das 10 consultas do RAG (onde estão os relevantes em cada etapa)

## PLAIN-68 — What is Actually in Chicken Nuggets?
Relevantes no qrels: 55

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 1 | 2 | 3 | 12 |
| semantic | 1 | 1 | 2 | 13 |
| hybrid | 1 | 3 | 3 | 14 |
| reranking | 1 | 1 | 1 | 14 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-2991 | 2 | 1 | 1 | 1 | 1 | The autopsy of chicken nuggets reads "chicken little". |
| MED-1797 | 0 | 23 | 24 | 18 | 2 | Comparative genome analysis with the human genome reveals chicken gene |
| MED-2492 | 0 | 8 | 18 | 7 | 3 | Roxarsone, Inorganic Arsenic, and Other Arsenic Species in Chicken: A  |
| MED-2088 | 0 | 37 | 5 | 8 | 4 | Arsenic species in poultry feather meal. |
| MED-1810 | 0 | 5 | >100 | 14 | 5 | Transmissibility of adenovirus-induced adiposity in a chicken model. |

## PLAIN-2450 — Are Organic Foods Safer?
Relevantes no qrels: 38

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 2 | 1 | 2 | 7 |
| semantic | 3 | 1 | 2 | 8 |
| hybrid | 4 | 1 | 2 | 8 |
| reranking | 2 | 2 | 3 | 8 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-1140 | 0 | 1 | 4 | 1 | 1 | Organic food: buying more safety or just peace of mind? A critical rev |
| MED-1178 | 2 | 2 | 3 | 4 | 2 | Are organic foods safer or healthier than conventional alternatives?:  |
| MED-1144 | 0 | 3 | 2 | 3 | 3 | Perceived risks of conventional and organic produce: pesticides, patho |
| MED-1179 | 0 | 4 | 1 | 2 | 4 | Organic foods: health and environmental advantages and disadvantages. |
| MED-1173 | 2 | 8 | 10 | 6 | 5 | Choice of organic foods is related to perceived consequences for human |

## PLAIN-153 — How Should I Take Probiotics?
Relevantes no qrels: 47

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 1 | 5 | 10 | 17 |
| semantic | 1 | 5 | 9 | 21 |
| hybrid | 1 | 5 | 10 | 21 |
| reranking | 1 | 5 | 9 | 21 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-3679 | 2 | 8 | 1 | 2 | 1 | The impact of meals on a probiotic during transit through a model of t |
| MED-3692 | 1 | 7 | 5 | 6 | 2 | Human gut microbiota and its relationship to health and disease. |
| MED-3695 | 1 | 5 | 8 | 9 | 3 | Probiotics for the prevention of pediatric antibiotic-associated diarr |
| MED-3693 | 1 | 14 | 7 | 10 | 4 | Microbiological evaluation of commercial probiotic products available  |
| MED-3680 | 1 | 3 | 6 | 4 | 5 | Probiotics for preventing acute upper respiratory tract infections. |

## PLAIN-2850 — More Antibiotics In White Meat or Dark Meat?
Relevantes no qrels: 9

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 77 | 0 | 0 | 2 |
| semantic | 14 | 0 | 0 | 2 |
| hybrid | 38 | 0 | 0 | 2 |
| reranking | 16 | 0 | 0 | 2 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-3891 | 0 | >100 | 3 | 8 | 1 | Comparison of the Prevalences and Antimicrobial Resistances of Escheri |
| MED-4358 | 0 | 8 | 44 | 9 | 2 | Bacteriophage biocontrol in animals and meat products |
| MED-4983 | 0 | 4 | >100 | 10 | 3 | Meat intake and mortality: a prospective study of over half a million  |
| MED-2584 | 0 | 1 | 64 | 2 | 4 | Dietary risk factors for colon cancer in a low-risk population. |
| MED-4482 | 0 | 3 | 82 | 5 | 5 | Meat consumption, Cooking Practices, Meat Mutagens and Risk of Prostat |

## PLAIN-2750 — Preventing the Common Cold with Probiotics?
Relevantes no qrels: 59

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 1 | 4 | 7 | 17 |
| semantic | 1 | 4 | 9 | 23 |
| hybrid | 1 | 4 | 9 | 18 |
| reranking | 1 | 4 | 8 | 18 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-3683 | 2 | 1 | 1 | 1 | 1 | Randomised, double-blind and placebo-controlled study using new probio |
| MED-3695 | 1 | 4 | 7 | 4 | 2 | Probiotics for the prevention of pediatric antibiotic-associated diarr |
| MED-3680 | 2 | 2 | 3 | 2 | 3 | Probiotics for preventing acute upper respiratory tract infections. |
| MED-3690 | 1 | 5 | 6 | 5 | 4 | Probiotics as prevention and treatment for diarrhea. |
| MED-4762 | 0 | 3 | 2 | 3 | 5 | Probiotic effects on cold and influenza-like symptom incidence and dur |

## PLAIN-270 — Is Caffeinated Tea Really Dehydrating?
Relevantes no qrels: 52

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 4 | 1 | 1 | 13 |
| semantic | 1 | 1 | 2 | 9 |
| hybrid | 2 | 1 | 1 | 12 |
| reranking | 1 | 2 | 2 | 12 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-4331 | 1 | 4 | 1 | 2 | 1 | Black tea is not significantly different from water in the maintenance |
| MED-5245 | 0 | 1 | 2 | 1 | 2 | Effect of decaffeination of coffee or tea on gastro-oesophageal reflux |
| MED-878 | 0 | >100 | 13 | 38 | 3 | Acute effects of coffee on endothelial function in healthy subjects. |
| MED-1645 | 0 | 2 | 5 | 3 | 4 | The acute effect of green tea consumption on endothelial function in h |
| MED-4439 | 1 | 15 | 67 | 25 | 5 | A case of fatal caffeine poisoning. |

## PLAIN-2470 — Is Milk Good for Our Bones?
Relevantes no qrels: 73

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 2 | 2 | 4 | 26 |
| semantic | 1 | 4 | 8 | 24 |
| hybrid | 1 | 4 | 6 | 24 |
| reranking | 1 | 3 | 5 | 24 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-1337 | 2 | 2 | 1 | 1 | 1 | Milk intake and risk of hip fracture in men and women: a meta-analysis |
| MED-1340 | 2 | 6 | 4 | 3 | 2 | Milk Consumption During Teenage Years and Risk of Hip Fractures in Old |
| MED-1338 | 2 | 54 | 6 | 11 | 3 | Milk intake and risk of mortality and fractures in women and men: coho |
| MED-904 | 0 | 44 | 16 | 25 | 4 | A systematic review and meta-analysis of the effects of pasteurization |
| MED-4902 | 0 | 33 | 56 | 42 | 5 | Addition of milk prevents vascular protective effects of tea. |

## PLAIN-2430 — Preventing Brain Loss with B Vitamins?
Relevantes no qrels: 46

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 1 | 3 | 4 | 11 |
| semantic | 1 | 4 | 6 | 19 |
| hybrid | 1 | 3 | 5 | 18 |
| reranking | 1 | 3 | 4 | 18 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-994 | 2 | 1 | 1 | 1 | 1 | Preventing Alzheimer’s disease-related gray matter atrophy by B-vitami |
| MED-980 | 2 | 2 | 2 | 2 | 2 | Homocysteine-Lowering by B Vitamins Slows the Rate of Accelerated Brai |
| MED-5128 | 0 | 5 | 4 | 3 | 3 | Low vitamin B-12 status and risk of cognitive decline in older adults. |
| MED-986 | 2 | 17 | 3 | 4 | 4 | Effect of homocysteine lowering treatment on cognitive function: a sys |
| MED-4929 | 0 | 30 | 20 | 16 | 5 | Vitamins E and C in the Prevention of Cardiovascular Disease in Men: T |

## PLAIN-238 — How Chemically Contaminated Are We?
Relevantes no qrels: 54

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 17 | 0 | 0 | 3 |
| semantic | 24 | 0 | 0 | 2 |
| hybrid | 16 | 0 | 0 | 4 |
| reranking | 2 | 1 | 1 | 4 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-2391 | 0 | 1 | 2 | 1 | 1 | Perfluorinated Compounds, Polychlorinated Biphenyls, and Organochlorin |
| MED-4739 | 1 | 17 | 26 | 16 | 2 | Nowhere to hide: Chemical toxicants and the unborn child. |
| MED-2493 | 0 | 79 | 19 | 26 | 3 | Environmental toxicants and the developing immune system: a missing li |
| MED-4653 | 0 | 2 | >100 | 9 | 4 | Dietary Intake Is Associated with Phthalate Body Burden in a Nationall |
| MED-2994 | 0 | 29 | 52 | 31 | 5 | Carcinogenicity and regulation of caramel colorings. |

## PLAIN-3342 — Is Coconut Milk Good For You?
Relevantes no qrels: 3

| etapa | 1º relevante | relevantes no Top-5 | Top-10 | Top-100 |
|---|---|---|---|---|
| lexical | 1 | 2 | 2 | 3 |
| semantic | 2 | 2 | 2 | 3 |
| hybrid | 1 | 2 | 2 | 3 |
| reranking | 1 | 2 | 3 | 3 |

| doc_id | rel | rank lex | rank sem | rank híb. | rank CE | título |
|---|---|---|---|---|---|---|
| MED-5014 | 2 | 2 | 4 | 3 | 1 | Dietary intake and the risk of coronary heart disease among the coconu |
| MED-5015 | 0 | 5 | 6 | 4 | 2 | The intravenous use of coconut water. |
| MED-4000 | 0 | 3 | 1 | 2 | 3 | Coconut oil predicts a beneficial lipid profile in pre-menopausal wome |
| MED-4002 | 0 | 8 | 7 | 7 | 4 | Effects of dietary coconut oil on the biochemical and anthropometric p |
| MED-5012 | 2 | 1 | 2 | 1 | 5 | The cholesterol-lowering effect of coconut flakes in humans with moder |
