# Frozen Uncertainty-Guided Multimodal Local Evidence Refinement v2 - Final Report

**Decision: NO_NET_IMPROVEMENT**

- Branch: `experiment/um-ler-frozen-v2`
- Source commit: `583e98e2486ed0cea32773b0599785f27dd3bfa7`
- Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU (cuda:0)
- Frozen Original reproduced: True
- Frozen Original correct: 556/598
- Frozen Original confusion: [[62, 0, 10], [0, 198, 11], [10, 11, 296]]
- Frozen Original probability Macro-AUC: 0.9560490698
- Historical prompt probability Macro-AUC: 0.9560491 (matched; legacy adjusted-score AUC was 0.9500702)
- Selected version: `um_ler_frozen_v2`
- Refiner parameters: 3393
- Correct: 556/598
- ACC: 0.9297658863
- Macro-F1: 0.9140778251
- BACC: 0.9140778251
- Probability Macro-AUC: 0.9559611012
- Weighted-F1: 0.9297658863
- Confusion matrix: [[62, 0, 10], [0, 198, 11], [10, 11, 296]]
- Changed / repairs / damages: 0 / 0 / 0
- Gate activation: 168/598 (0.280936)
- Active mean/max g: 0.026051/0.300000
- Completed-fold training time: 439.951 seconds
- Formal wall time: 444.326 seconds

## Attempted versions

| Version | Correct | Macro-F1 | BACC | Probability Macro-AUC | Repairs | Damages | Changed | Gate active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| um_ler_frozen_v2 | 556 | 0.9140778 | 0.9140778 | 0.9559611 | 0 | 0 | 0 | 28.0936% |
| um_ler_frozen_v2_1 | 556 | 0.9140778 | 0.9140778 | 0.9559199 | 0 | 0 | 0 | 27.9264% |
| um_ler_frozen_v2_2 | 556 | 0.9140778 | 0.9140778 | 0.9559493 | 0 | 0 | 0 | 36.7893% |
