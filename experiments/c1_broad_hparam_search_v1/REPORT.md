# C1 Broad Hyperparameter Search v1

Decision: **C1_SEARCH_POSITIVE**
Selected trial across all completed experiments: **A012** (Correct > AUC > Macro-F1)
Reached 563/598: **False**

## Best configuration and result

- rank/base lr/weight decay: 8 / 0.0112714160759 / 0.00109176779218
- lambda_aux/adapter LR multiplier/dropout multiplier: 0.5 / 2.0 / 1.1
- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 562/598 / 0.9397993 / 0.9294205 / 0.9286299 / 0.9615248 / 0.9397415
- Confusion matrix: [[64, 0, 8], [0, 200, 9], [7, 12, 298]]
- Repairs/damages/changed vs C1: 20 / 18 / 38
- AD-sMCI/CN-sMCI/AD-CN errors: 15 / 21 / 0
- Predicted AD/CN/sMCI: 71 / 212 / 315
- Parameters: 862971; AD-CN/BACC category-risk flag: False
- Best protected trial for reference: A012

## Fixed-reference comparison

| Reference | Correct | ACC | Macro-F1 | BACC | AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 560/598 | 0.9364548 | 0.9175457 | 0.9163359 | 0.9585608 | 0.9363660 |
| PC-BBF v1 | 561/598 | 0.9381271 | 0.9266945 | 0.9152140 | 0.9701189 | 0.9378308 |
| T1 | 550/598 | 0.9197324 | 0.9029119 | 0.8926874 | 0.9546117 | 0.9194786 |
| T2 | 552/598 | 0.9230769 | 0.9005424 | 0.8930679 | 0.9640011 | 0.9224998 |
| T3 | 546/598 | 0.9130435 | 0.9006004 | 0.8824118 | 0.9582175 | 0.9124829 |
| Best search | 562/598 | 0.9397993 | 0.9294205 | 0.9286299 | 0.9615248 | 0.9397415 |

T1/T2/T3 are comparison-only failed tune arms; T2 was not used as a baseline.

## Ten folds

fold0=83/0.9166667, fold1=67/0.9500000, fold2=306/0.9666667, fold3=106/0.9333333, fold4=63/0.9333333, fold5=268/0.9833333, fold6=117/0.9666667, fold7=228/0.9166667, fold8=85/0.8983051, fold9=84/0.9322034

## Stage A top 10

| Rank | Trial | Correct | AUC | Macro-F1 | BACC | lr | wd | lambda | adapter mult | dropout mult | protected |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | A012 | 562 | 0.9615248 | 0.9294205 | 0.9286299 | 0.011271416 | 0.0010917678 | 0.5 | 2.0 | 1.1 | True |
| 2 | A007 | 560 | 0.9631939 | 0.9169142 | 0.9127578 | 0.0099542972 | 0.0011186181 | 0.75 | 1.0 | 1.1 | True |
| 3 | A001 | 557 | 0.9591818 | 0.9209429 | 0.9165340 | 0.012999365 | 0.00083330487 | 1.0 | 1.0 | 1.1 | True |
| 4 | A022 | 557 | 0.9554593 | 0.9214566 | 0.9211988 | 0.0095789674 | 0.00067927374 | 0.75 | 1.5 | 1.1 | True |
| 5 | A013 | 557 | 0.9540847 | 0.9227469 | 0.9206554 | 0.010682365 | 0.0007446172 | 0.25 | 1.0 | 1.1 | True |
| 6 | A016 | 555 | 0.9592325 | 0.9160384 | 0.9078181 | 0.010117597 | 0.00088954537 | 0.25 | 0.5 | 1.0 | False |
| 7 | A004 | 554 | 0.9638582 | 0.9128714 | 0.9037318 | 0.010925775 | 0.00035728133 | 1.0 | 0.5 | 0.9 | False |
| 8 | A023 | 554 | 0.9619206 | 0.9132043 | 0.9144661 | 0.010691335 | 0.00098592725 | 0.75 | 1.5 | 1.0 | True |
| 9 | A019 | 553 | 0.9583666 | 0.9085710 | 0.9010502 | 0.0097051747 | 0.0008745888 | 0.25 | 1.5 | 1.0 | False |
| 10 | A010 | 553 | 0.9488822 | 0.9129411 | 0.9026803 | 0.0094932259 | 0.0011553792 | 1.0 | 1.0 | 1.1 | False |

## Rank capacity sensitivity

- rank 4: n=3, parameters=[858339], Correct mean/max=555.667/558, AUC mean=0.9591822, BACC mean=0.9197968
- rank 8: n=3, parameters=[862971], Correct mean/max=559.667/562, AUC mean=0.9613002, BACC mean=0.9193072
- rank 12: n=3, parameters=[867603], Correct mean/max=555.333/559, AUC mean=0.9544562, BACC mean=0.9092553
- rank 16: n=3, parameters=[872235], Correct mean/max=554.667/557, AUC mean=0.9618121, BACC mean=0.9106834

Matched rank 4/8/12/16 curves for each Stage B parent are stored in `all_trials.json`.

## Parameter relationships (Stage A)

- base_lr: Pearson with Correct/AUC/BACC = 0.6162977843198463 / 0.3743842468205923 / 0.4919361199599962; grouped means are stored in `all_trials.json`.
- base_weight_decay: Pearson with Correct/AUC/BACC = 0.5238504388410894 / 0.05113302351962485 / 0.41067985908545623; grouped means are stored in `all_trials.json`.
- lambda_aux: Pearson with Correct/AUC/BACC = -0.13387919311691068 / -0.34270000266320194 / -0.09220812121129658; grouped means are stored in `all_trials.json`.
- adapter_lr_multiplier: Pearson with Correct/AUC/BACC = 0.10571175081453858 / -0.00858207560071578 / 0.212615789906921; grouped means are stored in `all_trials.json`.
- dropout_multiplier: Pearson with Correct/AUC/BACC = 0.4176741487106133 / -0.09432712721888203 / 0.3484431893682068; grouped means are stored in `all_trials.json`.

## Provenance and runtime

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- source commit: 13a25931fd8e5a111a7ae1be056f7b1b213c3055
- result commit: the Git commit containing this report (self-referential SHA is intentionally not embedded)
- completed/failed experiments: 33 / 0
- summed fold training seconds / formal-session wall seconds: 13998.641 / 14019.079
- sampler: python_stdlib_deterministic_fallback (Optuna detected: False)
