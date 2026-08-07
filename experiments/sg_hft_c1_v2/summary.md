# Structured Group HFT-C1 v2

Decision: `SG_HFT_STOP`

Branch: `experiment/sg-hft-c1-v2`
Base commit: `0dd9161bdec9c91bb99f67325766f9361cba9b91`
Run commit: `b380b49d98856b94b10533d1093b0566ded2cafd`
Device: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`
Parameters: `875322` (added `12351`).

## Results

Correct `161/179`; ACC `0.8994413`; Macro-F1 `0.8782577`; BACC `0.8749095`; Probability Macro-AUC `0.9379088`; Weighted-F1 `0.8995622`.

Confusion matrix: `[[17, 0, 4], [0, 56, 7], [4, 3, 88]]`.
Repairs/damages: `3/8`.
AD-sMCI / CN-sMCI / AD-CN errors: `8/10/0`.

| Fold | Best epoch | Correct | ACC |
|---:|---:|---:|---:|
| 4 | 214 | 55 | 0.9166667 |
| 7 | 151 | 53 | 0.8833333 |
| 8 | 130 | 53 | 0.8983051 |

## Mechanism

Grouping: deterministic feature-name semantics; no labels or feature values used.

| Modality | Mean ratio | Max ratio | Cap saturation | Gate entropy | Group weights |
|---|---:|---:|---:|---:|---|
| MRI | 0.006326 | 0.017120 | 0.000000 | 1.000000 | cortical_volume_CV=0.5000, surface_area_SA=0.5000, cortical_thickness_TA_TS=0.5000, subcortical_volume_SV=0.5000 |

MRI: group gate did not form reliable selection.
| PET | 0.005324 | 0.017048 | 0.000000 | 1.000000 | cortical_left_uptake=0.5000, cortical_left_size=0.5000, cortical_right_uptake=0.5000, cortical_right_size=0.5000, noncortical_aggregate_uptake=0.5000, noncortical_aggregate_size=0.5000 |

PET: group gate did not form reliable selection.
| ROI | 0.004706 | 0.012199 | 0.000000 | 1.000000 | medial_temporal_structure=0.5000, global_structure=0.5000, ventricular_structure=0.5000, amyloid_AV45=0.5000, glucose_metabolism_FDG=0.5000 |

ROI: group gate did not form reliable selection.

## Conclusion

Exceeded the corresponding C1 correct count: `no`.
Proceed to formal ten-fold training: `no`.
