# HFT-C1-Lite v1

Decision: `SCREEN_STOP`

Branch: `experiment/hft-c1-lite-v1`  
Base commit: `7fb0a9aec28c1a0cfa68aed8bc5470b57ff3a840`  
Run commit: `0dd9161bdec9c91bb99f67325766f9361cba9b91`
Device: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`
Worktree: `D:\Work\WORK2 final\WORK2 final26.7,21\tmp\hft_c1_lite_v1`

## Screen

Correct `161/179`; ACC `0.8994413`; Macro-F1 `0.8784740`; BACC `0.8784740`; Probability Macro-AUC `0.9439129`; Weighted-F1 `0.8994413`.

Confusion matrix: `[[17, 0, 4], [0, 58, 5], [4, 5, 86]]`. Repairs/damages: `4/9`. AD-sMCI/CN-sMCI/total adjacent errors: `8/10/18`; AD-CN errors: `0`.

| Fold | Best epoch | Correct | ACC |
|---:|---:|---:|---:|
| 4 | 155 | 54 | 0.9000000 |
| 7 | 142 | 54 | 0.9000000 |
| 8 | 62 | 53 | 0.8983051 |

Runtime: `169.5` seconds. Parameters: `942747` total/inference (`+79776` vs C1).

## Minimal mechanism

| Modality | Mean ratio | Maximum ratio | Cap saturation | Attention entropy | Top features |
|---|---:|---:|---:|---:|---|
| MRI | 0.003327 | 0.010991 | 0.000000 | 0.999986 | ST54SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16, ST127SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16, ST30SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16, ST39SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16, ST34SA_UCSFFSX_11_02_15_UCSFFSX51_08_01_16 |
| PET | 0.003887 | 0.015083 | 0.000000 | 0.999984 | NON_WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16, RIGHT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16, RIGHT_INF_LAT_VENT_SIZE_UCBERKELEYAV45_10_17_16, LEFT_LATERAL_VENTRICLE_SIZE_UCBERKELEYAV45_10_17_16, RIGHT_CHOROID_PLEXUS_SIZE_UCBERKELEYAV45_10_17_16 |
| CSF | 0.004576 | 0.015212 | 0.000000 | 0.999914 | ABETA_UPENNBIOMK9_04_19_17, PTAU_UPENNBIOMK9_04_19_17, TAU_UPENNBIOMK9_04_19_17 |
| Risk | 0.003771 | 0.011705 | 0.000000 | 0.999988 | AGE_[60.4, 62.4), AGE_[78.4, 80.4), AGE_[80.4, 82.4), PTEDUCAT_[16, 18), PTEDUCAT_[12, 14) |
| ROI | 0.006420 | 0.034028 | 0.000000 | 0.998275 | Hippocampus, FDG, Fusiform, Entorhinal, MidTemp |

## Conclusion

The hard-fold screen does not establish support for the fine-grained-evidence hypothesis.

Exceeded C1 hard-fold correct count: `no`. Proceed to formal ten-fold training: `no`.
