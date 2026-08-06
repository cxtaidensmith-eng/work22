# Multimodal / Graph / Classifier Component Localization v1

## Component conclusions

| Component | Conclusion | Evidence |
|---|---|---|
| Multimodal learning | effective | MULTIMODAL_FUSION_EFFECTIVE |
| DIFFormer head | inconclusive | FROZEN_HEAD_INCONCLUSIVE |
| Explicit graph | unnecessary | RELATION_NOT_USEFUL |
| Classification head | sufficient | FROZEN_HEAD_INCONCLUSIVE |

Recommended direction: **KEEP_DIFFORMER_AND_IMPROVE_MULTIMODAL**

## Original anchor

Correct=556/598; ACC=0.9297658863; Macro-F1=0.9140778251; BACC=0.9140778251; Probability Macro-AUC=0.9560490698.

Confusion matrix:

```text
[62, 0, 10]
[0, 198, 11]
[10, 11, 296]
```

## Stage A: modality inference

| Setting | Correct | Delta Correct | ACC | Macro-F1 | BACC | Prob AUC |
|---|---:|---:|---:|---:|---:|---:|
| all_modalities | 556 | 0 | 0.9297659 | 0.9140778 | 0.9140778 | 0.9560491 |
| without_MRI | 534 | -22 | 0.8929766 | 0.8593826 | 0.8348737 | 0.9563566 |
| without_PET | 549 | -7 | 0.9180602 | 0.9004513 | 0.8938094 | 0.9564028 |
| without_CSF | 556 | 0 | 0.9297659 | 0.9140778 | 0.9140778 | 0.9561216 |
| without_Risk | 538 | -18 | 0.8996656 | 0.8786550 | 0.8874508 | 0.9508103 |
| without_COG | 324 | -232 | 0.5418060 | 0.2690605 | 0.3511104 | 0.5442377 |
| without_ROI | 546 | -10 | 0.9130435 | 0.8926171 | 0.8840419 | 0.9573140 |
| MRI_only | 320 | -236 | 0.5351171 | 0.2439629 | 0.3386614 | 0.5222462 |
| PET_only | 317 | -239 | 0.5301003 | 0.2309654 | 0.3333333 | 0.5007623 |
| CSF_only | 317 | -239 | 0.5301003 | 0.2309654 | 0.3333333 | 0.5030117 |
| Risk_only | 317 | -239 | 0.5301003 | 0.2309654 | 0.3333333 | 0.5043444 |
| COG_only | 514 | -42 | 0.8595318 | 0.8164732 | 0.7781546 | 0.9552476 |
| ROI_only | 317 | -239 | 0.5301003 | 0.2309654 | 0.3333333 | 0.5207014 |

## Stage B: frozen heads

| Head | Correct | ACC | Macro-F1 | BACC | Prob AUC | Params |
|---|---:|---:|---:|---:|---:|---:|
| original_joint_difformer | 556 | 0.9297659 | 0.9140778 | 0.9140778 | 0.9560491 | 853131 |
| linear_probe | 549 | 0.9180602 | 0.9035925 | 0.9050870 | 0.9545047 | 291 |
| residual_mlp | 554 | 0.9264214 | 0.9108705 | 0.9083967 | 0.9470486 | 14451 |
| frozen_difformer | 553 | 0.9247492 | 0.9089444 | 0.9032237 | 0.9493782 | 19203 |
| pee_head | 554 | 0.9264214 | 0.9121333 | 0.9078533 | 0.9468694 | 10040 |

Stage B decision: **FROZEN_HEAD_INCONCLUSIVE**

PEE ensemble diagnostics:

- Ensemble: 554/598, ACC=0.9264214
- Pairwise prediction disagreement: 0.0103122
- Pairwise probability cosine: 0.9957687
- Pairwise symmetric KL: 0.0087975

## Stage C: graph localization

Status=completed; decision=RELATION_NOT_USEFUL.

## Stage D: multimodal capacity

Status=completed; decision=MULTIMODAL_FUSION_EFFECTIVE.

## Bottleneck priority

1. Classification head: sufficient (FROZEN_HEAD_INCONCLUSIVE)
2. Explicit graph: unnecessary (RELATION_NOT_USEFUL)
3. Multimodal learning: effective (MULTIMODAL_FUSION_EFFECTIVE)
4. DIFFormer head: inconclusive (FROZEN_HEAD_INCONCLUSIVE)

## Runtime

Branch=experiment/component-localization-v1; source commit=7c38306ff4704c0e2540fa59e283ba0c88d2b779; device=NVIDIA GeForce RTX 3050 Ti Laptop GPU; total seconds=783.397.

## PEE members

- member_0: 553/598, ACC=0.9247492
- member_1: 546/598, ACC=0.9130435
- member_2: 553/598, ACC=0.9247492
- member_3: 549/598, ACC=0.9180602

## Stage C metrics

| Arm | Correct | ACC | Macro-F1 | BACC | Prob AUC | Gamma |
|---|---:|---:|---:|---:|---:|---:|
| C0 Residual MLP | 554 | 0.9264214 | 0.9108705 | 0.9083967 | 0.9470486 | - |
| C1 DIFFormer no graph | 553 | 0.9247492 | 0.9089444 | 0.9032237 | 0.9493782 | - |
| C2 Sparse Residual GCN | 555 | 0.9280936 | 0.9121819 | 0.9094482 | 0.9604608 | 0.0594537 |

## Stage D metrics

| Arm | Correct | ACC | Macro-F1 | BACC | Prob AUC |
|---|---:|---:|---:|---:|---:|
| D0 COG-only MLP | 552 | 0.9230769 | 0.9117884 | 0.9112763 | 0.9688224 |
| D1 Raw-All MLP | 504 | 0.8428094 | 0.8473263 | 0.8499357 | 0.9091078 |
| D2 Current Multimodal | 556 | 0.9297659 | 0.9140778 | 0.9140778 | 0.9560491 |
