# Counterfactual Marginal Evidence Guided Dual-Branch Learning v1

This experiment studies the patient-level conditional marginal diagnostic value of modality-private information and uses counterfactual deletion to supervise private-evidence selection in a dual-branch model.

## Decision

- Final decision: **CME_GO**
- C2 method decision: **CME_NO_GAIN**
- C3 status: **SKIP_C3_C2_NOT_PROMISING**
- Best experimental arm by Correct > Probability Macro-AUC > Macro-F1: **C1**

## Pooled OOF metrics

| Model | Params | Correct | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 | Repairs | Damages | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 853131 | 556/598 | 0.9297659 | 0.9140778 | 0.9140778 | 0.9560491 | 0.9297659 | - | - | - |
| C1 | 862971 | 560/598 | 0.9364548 | 0.9175457 | 0.9163359 | 0.9585608 | 0.9363660 | 23 | 19 | 0.643969 |
| C2 | 866444 | 550/598 | 0.9197324 | 0.9081410 | 0.9020171 | 0.9632680 | 0.9195541 | 22 | 28 | 0.4798877 |

## Confusion matrices and fold results

### C1

```text
[61, 0, 11]
[0, 201, 8]
[10, 9, 298]
```

10-fold ACC = 0.9364689 ± 0.0171698 (sample SD); training time = 485.517 s.
Artifacts: `experiments/cme_dual_branch_v1/c1_shared_private_control_2`.

| Fold | Best epoch | Correct | ACC |
|---:|---:|---:|---:|
| 0 | 116 | 55 | 0.9166667 |
| 1 | 153 | 56 | 0.9333333 |
| 2 | 105 | 57 | 0.9500000 |
| 3 | 159 | 55 | 0.9166667 |
| 4 | 103 | 55 | 0.9166667 |
| 5 | 180 | 57 | 0.9500000 |
| 6 | 332 | 58 | 0.9666667 |
| 7 | 181 | 56 | 0.9333333 |
| 8 | 90 | 55 | 0.9322034 |
| 9 | 181 | 56 | 0.9491525 |

### C2

```text
[61, 0, 11]
[0, 194, 15]
[7, 15, 295]
```

10-fold ACC = 0.9197740 ± 0.0218431 (sample SD); training time = 505.032 s.
Artifacts: `experiments/cme_dual_branch_v1/c2_counterfactual_marginal_evidence`.

| Fold | Best epoch | Correct | ACC |
|---:|---:|---:|---:|
| 0 | 78 | 54 | 0.9000000 |
| 1 | 51 | 53 | 0.8833333 |
| 2 | 75 | 56 | 0.9333333 |
| 3 | 106 | 55 | 0.9166667 |
| 4 | 226 | 55 | 0.9166667 |
| 5 | 132 | 54 | 0.9000000 |
| 6 | 168 | 57 | 0.9500000 |
| 7 | 224 | 56 | 0.9333333 |
| 8 | 166 | 54 | 0.9152542 |
| 9 | 58 | 56 | 0.9491525 |

## C1 private-residual diagnostics

- Mean adapter gradient norm: 0.02775639
- Mean Category–Global cosine: 0.5812827
- Private collapse: False

| Modality | Residual mean norm | Private/shared norm ratio |
|---|---:|---:|
| MRI | 3.2676491 | 0.0756563 |
| PET | 3.0607443 | 0.0680928 |
| CSF | 2.1156939 | 0.0535536 |
| Risk | 3.0490732 | 0.0737990 |
| COG | 10.6765237 | 0.2795456 |
| ROI | 3.2241727 | 0.0763696 |

## C2 router and counterfactual utility

### all

Router normalized entropy=0.4980316; mean max weight=0.6862509; collapse=False; positive-utility proportion=0.9795691; mean KL=0.5441095.

| Modality | OOF router weight | Positive utility | Mean deletion loss change |
|---|---:|---:|---:|
| MRI | 0.0757364 | 0.0768048 | 0.0748566 |
| PET | 0.0468809 | 0.0298587 | 0.0268639 |
| CSF | 0.0523730 | 0.0065967 | 0.0048094 |
| Risk | 0.0649180 | 0.0447451 | 0.0434865 |
| COG | 0.6849305 | 0.9621435 | 0.9596915 |
| ROI | 0.0751612 | 0.0223881 | 0.0213303 |

Router weights by true class:

| Class | MRI | PET | CSF | Risk | COG | ROI |
|---|---:|---:|---:|---:|---:|---:|
| AD | 0.0730643 | 0.0550408 | 0.0572910 | 0.0722231 | 0.6713253 | 0.0710555 |
| CN | 0.0682360 | 0.0432988 | 0.0499259 | 0.0543544 | 0.7101393 | 0.0740457 |
| SMCI | 0.0812885 | 0.0473892 | 0.0528694 | 0.0702234 | 0.6714003 | 0.0768291 |

Router weights for Original-relative repairs and damages:

| Group | MRI | PET | CSF | Risk | COG | ROI |
|---|---:|---:|---:|---:|---:|---:|
| repairs | 0.0852453 | 0.0489857 | 0.0475174 | 0.0637492 | 0.6891873 | 0.0653150 |
| damages | 0.0861461 | 0.0612522 | 0.0703881 | 0.0721138 | 0.6279498 | 0.0821499 |

## Required research questions

1. Did ordinary shared/private C1 improve performance? **True**.
2. Did private residuals train and diverge? Gradient=0.02775639; collapse=False.
3. Did C2 outperform C1 under the fixed ranking? **False**.
4. Modalities with positive marginal evidence, high to low: **COG, MRI, Risk, PET, ROI, CSF**.
5. Was the router only COG-dominant? It was strongly COG-dominant but did not meet the predeclared collapse threshold: mean COG weight=0.6849305; collapse=False; non-COG top-selection frequency=0.1053512.
6. Could utility predict repairs/damages? No reliable beneficial separation was established: repair/damage-conditioned weights differed, but net repairs=-6 (22 repairs, 28 damages).
7. Did C2 reach 557 correct? **False**.
8. If C3 ran, did adjacent diagnostic-boundary routers differ? **False**.
9. Which boundary supplied the net gain? C3 was not triggered, so no boundary-specific source can be assigned.
10. Does the result support the hypothesis that private information is not automatically effective complementary information? **True**: C1 showed useful private capacity, while non-collapsed counterfactual routing in C2 reduced net accuracy.

## Protocol and provenance

The six Original Query modal encoders, shared Modal Transformer, three private Query pools, three OVR heads, Message_MLP, historical Global Message, additive fusion, graph-disabled DIFFormer, and classifier were retained. The historical Global module consumes X_gated; keeping this interface was necessary for exact zero-adapter initialization equivalence. Consequently, counterfactual utility measures Category-token marginal evidence while the Global evidence path is fixed.

Branch=experiment/cme-dual-branch-v1; source HEAD=7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840; device=NVIDIA GeForce RTX 3050 Ti Laptop GPU; total wall time=994.105 s.
Original OOF SHA256=a668339f25aa87ce09d5fa04ecbbc01795f0c3cf3f3ced1a3b9ebf8f789e5959.
