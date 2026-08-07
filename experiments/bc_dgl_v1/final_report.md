# Boundary-Conditioned Disentangled Gradient Learning v1

Decision: **BC_DGL_SCREEN_STOP**
Selected arm: **None**
Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Total runtime: 343.502 s

## Screen

| Arm | rho | Parameters | Correct/179 | ACC | Macro-F1 | BACC | AUC | AD-sMCI | CN-sMCI | Time (s) | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BC-DGL-S | 0.00 | 865299 | 158/179 | 0.8826816 | 0.8546524 | 0.8538012 | 0.9357672 | 10 | 11 | 172.12 | False |
| BC-DGL-M | 0.25 | 865299 | 160/179 | 0.8938547 | 0.8776072 | 0.8678363 | 0.9341669 | 7 | 12 | 165.55 | False |

### BC-DGL-S folds

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 4 | 50 | 0.8666667 |
| 7 | 163 | 0.8666667 |
| 8 | 239 | 0.9152542 |

| Modality | AD-sMCI AUC | CN-sMCI AUC | Head gradients nonzero |
|---|---:|---:|---|
| MRI | 0.7483709 | 0.6137009 | True |
| PET | 0.7583960 | 0.6095238 | True |
| CSF | 0.8185464 | 0.6121972 | True |
| Risk | 0.6220551 | 0.6533835 | True |
| COG | 0.9518797 | 0.9373434 | True |
| ROI | 0.8345865 | 0.6596491 | True |

COG-minus-non-COG AUC gap: 0.2515706; effective non-COG modalities: ['MRI', 'PET', 'CSF', 'Risk', 'ROI']

### BC-DGL-M folds

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 4 | 60 | 0.9000000 |
| 7 | 48 | 0.8833333 |
| 8 | 139 | 0.8983051 |

| Modality | AD-sMCI AUC | CN-sMCI AUC | Head gradients nonzero |
|---|---:|---:|---|
| MRI | 0.7538847 | 0.6339181 | True |
| PET | 0.8190476 | 0.7604010 | True |
| CSF | 0.7929825 | 0.5989975 | True |
| Risk | 0.6120301 | 0.6913116 | True |
| COG | 0.9679198 | 0.9555556 | True |
| ROI | 0.8571429 | 0.6835422 | True |

COG-minus-non-COG AUC gap: 0.2414119; effective non-COG modalities: ['MRI', 'PET', 'CSF', 'Risk', 'ROI']

Registered screen ranking: BC-DGL-M > BC-DGL-S.

## Formal

Formal ten-fold training was not run because neither preregistered arm passed the hard screen gate.

Checkpoint readback: `{'passed': True, 'fold': 4, 'rows': 60, 'max_probability_abs_diff': 0.0}`

Mechanism conclusion: BC-DGL-M ranked ahead of the other preregistered routing arm, but neither preserved both hard-fold accuracy and boundary errors.
Next recommendation: Neither preregistered arm passed; do not tune lambda or run formal ten-fold training.
