# TabPFN–Deep Conservative Fusion v1

Branch: `experiment/tpcf-v1`; run source commit: `2315ba96d31fb76ba9a97e776739e94787cd11c6`; device: NVIDIA GeForce RTX 3050 Ti Laptop GPU; formal training sum: 1276.6s.

## Results

| model | correct | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 | confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original Query | 556 | 0.9297658862876255 | 0.9140778251271361 | 0.9140778251271361 | 0.9560490697782983 | 0.9297658862876255 | [[62,0,10],[0,198,11],[10,11,296]] |
| TabPFN E4 | 548 | 0.9163879598662207 | 0.9058959712564674 | 0.9013186544732287 | 0.9764191581686475 | 0.9163072846281483 | [[62,0,10],[0,190,19],[8,13,296]] |
| arithmetic_alpha_0.30 | 556 | 0.9297658862876255 | 0.9140778251271361 | 0.9140778251271361 | 0.9827181375006354 | 0.9297658862876255 | [[62,0,10],[0,198,11],[10,11,296]] |
| tpcf_v1_cap020 | 556 | 0.9297658862876255 | 0.9140778251271361 | 0.9140778251271361 | 0.9668304002008151 | 0.9297658862876255 | [[62,0,10],[0,198,11],[10,11,296]] |
| tpcf_v2_cap035 | 556 | 0.9297658862876255 | 0.9140778251271361 | 0.9140778251271361 | 0.97052107495311 | 0.9297658862876255 | [[62,0,10],[0,198,11],[10,11,296]] |

Best bounded gate: **tpcf_v2_cap035** (cap=0.35), Correct=556/598, ACC=0.9297658863, Macro-F1=0.9140778251, BACC=0.9140778251, Probability Macro-AUC=0.9705210750. Decision: **NO_NET_IMPROVEMENT**.

Repairs=0; damages=0; changed=0; exact McNemar p=1. Mean/max gate=0.332287/0.350000; gate>0.05=99.4983%.

## Gate diagnostics by cap

| arm | cap | correct | Weighted-F1 | changed | repairs | damages | Original-only | TPCF-only | McNemar p | mean gate | max gate | gate>0.05 | agree gate | disagree gate | AD gate | CN gate | SMCI gate | TabPFN-only gate | Original-only gate | confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tpcf_v1_cap020 | 0.2 | 556 | 0.9297658862876255 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0.19405037316066964 | 0.19999714195728302 | 0.9983277591973244 | 0.1937222303310921 | 0.19749587287123388 | 0.19657668471336365 | 0.19359900000277888 | 0.19377416671967657 | 0.19990862905979156 | 0.19572651833295823 | [[62,0,10],[0,198,11],[10,11,296]] |
| tpcf_v2_cap035 | 0.35 | 556 | 0.9297658862876255 | 0 | 0 | 0 | 0 | 0 | 1.0 | 0.332287041200979 | 0.34999996423721313 | 0.9949832775919732 | 0.33128811650817375 | 0.3427757504754342 | 0.3402441197799312 | 0.33155344840156975 | 0.33096341734417134 | 0.3499064269390973 | 0.3375465877354145 | [[62,0,10],[0,198,11],[10,11,296]] |

## Required conclusions

1. The best zero-training fusion did **not** produce a net accuracy gain: 556/598, with 0 repairs and 0 damages; its probability AUC rose to 0.9827181375.
2. TabPFN-only-correct samples show a modest confidence separation: TabPFN margin mean 0.4007 versus 0.3056 on Original-only-correct samples, while Original remains highly confident on many TabPFN-only repairs (Original margin mean 0.9477).
3. The bounded gate was not more effective than the best fixed fusion by correct count (556 versus 556).
4. cap=0.35 is preferred by Correct > AUC > Macro-F1; cap=.20/.35 correct counts were 556/556.
5. Boundary net changes: AD/sMCI 0 repairs and 0 damages; CN/sMCI 0 repairs and 0 damages.
6. TPCF did not reach 557 correct subjects.
7. If unsuccessful, the observed reason is: the gate was highly active, but the bounded log-residual crossed no Original decision boundary; failure was neither under-activation nor damage-heavy correction.

No larger cap, class-wise gate, new feature input, new loss, ensemble, or graph branch was tested. If both bounded gates fail, the next recommended route is DIFFormer global relations plus Sparse GCN local relations.
