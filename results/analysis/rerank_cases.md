# Reranking — casos selecionados (test)

### Maior ganho 1: PLAIN-1320 — "Harvard Physicians’ Study II"

NDCG@10: híbrido = 0.0911, híbrido+reranking = 0.5617 (Δ = +0.4706). Relevantes no Top-10: 1 → 3. Posições do Top-10 com documento diferente: 10; deslocamento médio: 10.1.

| rank CE | rank híb. | Δ | doc_id | rel | score CE | rank lex | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9 | 8 | MED-2760 | 1 | 0.755 | 2 | 74 | 319 | 3 | Multivitamins in the Prevention of Cancer in Men: The Physicians’ Health Stud… |
| 2 | 5 | 3 | MED-4929 | 0 | 0.178 | 1 | 61 | 389 | 3 | Vitamins E and C in the Prevention of Cardiovascular Disease in Men: The Phys… |
| 3 | 12 | 9 | MED-2758 | 1 | -2.013 | 4 | 38 | 329 | 3 | Multivitamins in the Prevention of Cardiovascular Disease in Men: The Physici… |
| 4 | 7 | 3 | MED-1986 | 0 | -4.05 | 3 | 32 | 195 | 2 | Long-term morbidity and mortality of overweight adolescents. A follow-up of t… |
| 5 | 19 | 14 | MED-4398 | 0 | -4.215 | 5 | None | 157 | 3 | High school dietary dairy intake and teenage acne. |
| 6 | 22 | 16 | MED-2761 | 1 | -5.364 | 11 | 84 | 205 | 3 | Who uses multivitamins? A cross-sectional study in the Physicians' Health Stu… |
| 7 | 26 | 19 | MED-3932 | 0 | -6.235 | 10 | None | 188 | 3 | Caffeine and risk of Parkinson disease in a large cohort of men and women |
| 8 | 46 | 38 | MED-5124 | 0 | -7.14 | 24 | None | 205 | 2 | Egg Consumption and Cardiovascular Disease and Mortality The Physicians' Heal… |
| 9 | 4 | -5 | MED-761 | 0 | -7.343 | 35 | 2 | 222 | 1 | The counseling practices of internists. |
| 10 | 39 | 29 | MED-1622 | 0 | -7.523 | 21 | None | 184 | 3 | Coffee, caffeine, and risk of completed suicide: results from 3 prospective c… |
| 12 | 10 | -2 | MED-728 | 0 | -7.609 | 9 | 14 | 155 | 2 | Barriers to providing nutrition counseling cited by physicians: a survey of p… |
| 16 | 1 | -15 | MED-1053 | 0 | -8.552 | 6 | 4 | 144 | 2 | Physician disclosure of healthy personal behaviors improves credibility and a… |
| 17 | 3 | -14 | MED-1097 | 0 | -8.663 | 14 | 3 | 86 | 1 | Moses Maimonides: medieval physician and scholar. |
| 19 | 8 | -11 | MED-1545 | 0 | -8.978 | 8 | 15 | 178 | 1 | Physician smoking status, attitudes toward smoking, and cessation advice to p… |
| 20 | 6 | -14 | MED-1543 | 0 | -9.051 | 16 | 6 | 156 | 1 | Patient-related diet and exercise counseling: do providers' own lifestyle hab… |
| 28 | 2 | -26 | MED-2112 | 0 | -9.775 | 17 | 1 | 64 | 2 | How evidence-based medicine biases physicians against nutrition. |

### Maior ganho 2: PLAIN-1473 — "kohlrabi"

NDCG@10: híbrido = 0.6309, híbrido+reranking = 1.0000 (Δ = +0.3691). Relevantes no Top-10: 1 → 1. Posições do Top-10 com documento diferente: 10; deslocamento médio: 25.7.

| rank CE | rank híb. | Δ | doc_id | rel | score CE | rank lex | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | MED-4455 | 1 | 1.221 | 1 | None | 118 | 1 | Glucoraphanin and 4-hydroxyglucobrassicin contents in seeds of 59 cultivars o… |
| 2 | 26 | 24 | MED-4713 | 0 | -4.31 | None | 25 | 119 | 0 | A case of Kombucha tea toxicity. |
| 3 | 23 | 20 | MED-4459 | 0 | -6.17 | None | 22 | 206 | 0 | Impact of Nrf2 on UVB-induced skin inflammation/photoprotection and photoprot… |
| 4 | 48 | 44 | MED-1686 | 0 | -6.726 | None | 47 | 106 | 0 | Cardioprotective properties of kiwifruit. |
| 5 | 30 | 25 | MED-3539 | 0 | -7.218 | None | 29 | 175 | 0 | Effect of kiwifruit consumption on sleep quality in adults with sleep problem… |
| 6 | 27 | 21 | MED-4832 | 0 | -7.908 | None | 26 | 184 | 0 | Effects of kiwifruit consumption on serum lipid profiles and antioxidative st… |
| 7 | 45 | 38 | MED-2135 | 0 | -8.073 | None | 44 | 197 | 0 | Fisetin regulates obesity by targeting mTORC1 signaling. |
| 8 | 9 | 1 | MED-4664 | 0 | -8.363 | None | 8 | 63 | 0 | Iodine toxicity from soy milk and seaweed ingestion is associated with seriou… |
| 9 | 36 | 27 | MED-712 | 0 | -8.418 | None | 35 | 186 | 0 | Chemopreventive properties and molecular mechanisms of the bioactive compound… |
| 10 | 34 | 24 | MED-1864 | 0 | -8.482 | None | 33 | 278 | 0 | Hibiscus sabdariffa L. in the treatment of hypertension and hyperlipidemia: a… |
| 20 | 1 | -19 | MED-3947 | 0 | -9.141 | None | 1 | 169 | 0 | Hibiscus anthocyanins rich extract-induced apoptotic cell death in human prom… |
| 29 | 5 | -24 | MED-3447 | 0 | -9.769 | None | 4 | 112 | 0 | Seaweed prevents breast cancer? |
| 35 | 8 | -27 | MED-5173 | 0 | -10.038 | None | 7 | 33 | 0 | Acute rhabdomyolysis caused by Spirulina (Arthrospira platensis). |
| 40 | 10 | -30 | MED-744 | 0 | -10.226 | None | 9 | 124 | 0 | Therapy with saffron and the goddess at Thera. |
| 41 | 6 | -35 | MED-5024 | 0 | -10.245 | None | 5 | 92 | 0 | Disseminated chlorellosis in a dog. |
| 43 | 7 | -36 | MED-4385 | 0 | -10.384 | None | 6 | 87 | 0 | The influence of dietary lutein and zeaxanthin on visual performance. |
| 44 | 3 | -41 | MED-4660 | 0 | -10.425 | None | 2 | 139 | 0 | Variability of iodine content in common commercially available edible seaweed… |
| 47 | 4 | -43 | MED-3441 | 0 | -10.893 | None | 3 | 79 | 0 | Marine edible algae as disease preventers. |

### Pouca/nenhuma alteração: PLAIN-1557 — "magnesium"

NDCG@10: híbrido = 0.3152, híbrido+reranking = 0.3152 (Δ = +0.0000). Relevantes no Top-10: 3 → 3. Posições do Top-10 com documento diferente: 5; deslocamento médio: 6.1.

| rank CE | rank híb. | Δ | doc_id | rel | score CE | rank lex | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0 | MED-1406 | 0 | 2.19 | 1 | 1 | 196 | 1 | Dietary magnesium intake is inversely associated with mortality in adults at … |
| 2 | 3 | 1 | MED-3918 | 1 | -0.533 | 3 | 3 | 98 | 1 | Herbal infusions as a source of calcium, magnesium, iron, zinc and copper in … |
| 3 | 2 | -1 | MED-4315 | 1 | -1.357 | 4 | 2 | 82 | 1 | Differences in metal content of the heart muscle in death from ischemic heart… |
| 4 | 4 | 0 | MED-2248 | 0 | -1.558 | 2 | 7 | 107 | 1 | Trace element status in healthy subjects switching from a mixed to a lactoveg… |
| 5 | 11 | 6 | MED-5005 | 0 | -2.691 | 7 | None | 259 | 1 | Increased prevalence of constipation in pre-school children is attributable t… |
| 6 | 6 | 0 | MED-1447 | 0 | -3.296 | 5 | None | 203 | 1 | Nutrient intake in the GEICO multicenter trial: the effects of a multicompone… |
| 7 | 27 | 20 | MED-3497 | 0 | -3.8 | 16 | None | 180 | 1 | Evaluation of certain food additives and contaminants. |
| 8 | 41 | 33 | MED-2249 | 0 | -3.904 | 23 | None | 237 | 1 | Zinc and multi-mineral supplementation should mitigate the pathogenic impact … |
| 9 | 9 | 0 | MED-4313 | 1 | -4.039 | 6 | None | 254 | 1 | A vegetarian dietary pattern as a nutrient-dense approach to weight managemen… |
| 10 | 10 | 0 | MED-2380 | 0 | -4.195 | 9 | 92 | 129 | 1 | Nuts, hypertension and endothelial function. |
| 15 | 7 | -8 | MED-1844 | 0 | -5.363 | None | 5 | 167 | 0 | Determination of total aluminum, chromium, copper, iron, manganese, and nicke… |
| 30 | 5 | -25 | MED-1110 | 0 | -7.056 | None | 4 | 139 | 0 | The potential role of curcumin in patients with monoclonal gammopathy of unde… |
| 34 | 8 | -26 | MED-1843 | 0 | -7.97 | None | 6 | 143 | 0 | Aluminium in over-the-counter drugs: risks outweigh benefits? |

### Piora: PLAIN-1679 — "myelopathy"

NDCG@10: híbrido = 0.3869, híbrido+reranking = 0.0000 (Δ = -0.3869). Relevantes no Top-10: 1 → 0. Posições do Top-10 com documento diferente: 10; deslocamento médio: 10.8.

| rank CE | rank híb. | Δ | doc_id | rel | score CE | rank lex | rank sem | |d| | termos | título |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 31 | MED-1105 | 0 | 0.193 | None | 32 | 83 | 0 | A CASE OF MULTIPLE MYELOMA PRESENTING AS A BULLOUS DERMATOSIS |
| 2 | 24 | 22 | MED-1104 | 0 | -0.934 | None | 24 | 62 | 0 | Multiple myeloma : recent progress in diagnosis and treatment. |
| 3 | 8 | 5 | MED-1107 | 0 | -2.38 | None | 8 | 149 | 0 | From myeloma precursor disease to multiple myeloma: new diagnostic concepts a… |
| 4 | 3 | -1 | MED-3193 | 0 | -2.536 | None | 3 | 219 | 0 | 11C-choline vs. 18F-FDG PET/CT in assessing bone involvement in patients with… |
| 5 | 4 | -1 | MED-3787 | 0 | -2.536 | None | 4 | 219 | 0 | 11C-choline vs. 18F-FDG PET/CT in assessing bone involvement in patients with… |
| 6 | 19 | 13 | MED-1111 | 0 | -2.879 | None | 19 | 195 | 0 | Clinical Trials and Observations: Monoclonal gammopathy of undetermined signi… |
| 7 | 10 | 3 | MED-1115 | 0 | -3.464 | None | 10 | 125 | 0 | Disparities in the prevalence, pathogenesis and progression of monoclonal gam… |
| 8 | 14 | 6 | MED-1109 | 0 | -3.824 | None | 14 | 202 | 0 | Risk factors for multiple myeloma: a hospital-based case-control study in Nor… |
| 9 | 2 | -7 | MED-1113 | 0 | -4.114 | None | 2 | 185 | 0 | Monoclonal gammopathy of undetermined significance, smoldering multiple myelo… |
| 10 | 39 | 29 | MED-4312 | 0 | -5.223 | None | 39 | 77 | 0 | Post-epidemic eosinophilia myalgia syndrome associated with L-Tryptophan |
| 11 | 6 | -5 | MED-3317 | 0 | -6.407 | None | 6 | 115 | 0 | Auto-immune polyradiculoneuropathy and a novel IgG biomarker in workers expos… |
| 15 | 7 | -8 | MED-3302 | 0 | -7.577 | None | 7 | 111 | 0 | A clustering of immune-mediated polyradiculoneuropathy among swine abattoir w… |
| 16 | 9 | -7 | MED-4115 | 0 | -7.656 | None | 9 | 61 | 0 | The rise and fall of horror autotoxicus and forbidden clones. |
| 37 | 1 | -36 | MED-5133 | 0 | -9.984 | None | 1 | 89 | 0 | [Floppy baby with macrocytic anemia and vegan mother]. |
| 40 | 5 | -35 | MED-5142 | 1 | -10.203 | None | 5 | 129 | 0 | Irreversible subacute sclerotic combined degeneration of the spinal cord in a… |


Consultas com ΔNDCG@10 = 0: 124; melhoraram: 100; pioraram: 99.