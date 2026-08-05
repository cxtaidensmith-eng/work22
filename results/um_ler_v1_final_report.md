# Uncertainty-Guided Multimodal Local Evidence Refinement v1 - Final Report

**Decision: STOP_ROUTE**

- Branch: `experiment/um-ler-v1`
- Source commit: `b1fc0f4b34e57874feacde8805fd1e5251f45118`
- Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU (cuda:0)
- Total formal wall time: 2265.476 seconds
- Completed-fold training time: 1608.749 seconds
- Selected version: `um_ler_v1`
- Parameters: 856524 (+3393 vs Original)
- Correct: 552/598
- ACC: 0.9230769231 (-0.0066889769)
- Macro-F1: 0.9100461738 (-0.0040316262)
- BACC: 0.9087849830 (-0.0052928170)
- Probability Macro-AUC: 0.9602490784 (+0.0041999784)
- Weighted-F1: 0.9230197090 (-0.0067461910)
- OOF confusion matrix: [[62, 0, 10], [0, 196, 13], [9, 14, 294]]
- Historical Original exclusive correct: 24
- UM-LER exclusive correct: 20
- Actual changed predictions: 1
- Original wrong -> new correct: 1
- Original correct -> new wrong: 0
- Gate activation: 2/598 (0.003344)
- Active mean/max g: 0.212576/0.300000

## Per-fold best epoch and ACC

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 0 | 170 | 0.9000000000 |
| 1 | 114 | 0.9500000000 |
| 2 | 184 | 0.9333333333 |
| 3 | 122 | 0.9000000000 |
| 4 | 167 | 0.9500000000 |
| 5 | 177 | 0.9000000000 |
| 6 | 138 | 0.9333333333 |
| 7 | 98 | 0.9166666667 |
| 8 | 64 | 0.8813559322 |
| 9 | 105 | 0.9661016949 |

## Attempted versions

| Version | Correct | ACC | Changes vs default |
|---|---:|---:|---|
| um_ler_v1 | 552 | 0.9230769231 | default |
| um_ler_v1_1 | 552 | 0.9230769231 | {'gate_cap': 0.2} |
| um_ler_v1_2 | 551 | 0.9214046823 | {'neighbor_confidence_threshold': 0.65, 'gate_cap': 0.2} |
