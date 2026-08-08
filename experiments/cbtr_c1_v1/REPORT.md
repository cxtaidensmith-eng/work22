# CBTR-C1 v1 final report

Decision: **NO_GAIN**
v1.1 run: **False**; selected: **formal_v1**.

Correct=547/598; ACC=0.9147157; Macro-F1=0.9023140; BACC=0.8947410; Probability Macro-AUC=0.9490057; Weighted-F1=0.9144779.

Confusion: `[[60, 0, 12], [0, 193, 16], [7, 16, 294]]`.
Repairs/damages/changed vs C1: 14/27/41.
Repairs/damages/changed vs PC-BBF: 16/30/46.
AD-sMCI / CN-sMCI / AD-CN: 19 / 32 / 0.
Predicted AD/CN/sMCI: 67/209/322.
Delta vs C1: `{'correct': -13, 'acc': -0.021739130434782705, 'macro_f1': -0.015231676748321266, 'bacc': -0.021594899126859857, 'macro_auc': -0.009555119795105593, 'weighted_f1': -0.021888053386334483}`.
Delta vs PC-BBF: `{'correct': -14, 'acc': -0.02341137123745829, 'macro_f1': -0.024380481075002236, 'bacc': -0.02047293735405964, 'macro_auc': -0.021113292478934786, 'weighted_f1': -0.023352867060657423}`.
Parameters=866347 (+3376); summed fold training=564.273s, resumed-invocation wall=339.174s.
Observed end-to-end formal wall including interruption and strict resume=655.078s.

The retrieval memory was fold-train-only; test-label mechanism statistics were produced offline after complete OOF assembly.

## Selected per-fold epochs

| Fold | Best epoch | Correct | ACC |
|---:|---:|---:|---:|
| 0 | 235 | 53 | 0.8833333 |
| 1 | 136 | 56 | 0.9333333 |
| 2 | 121 | 58 | 0.9666667 |
| 3 | 98 | 53 | 0.8833333 |
| 4 | 156 | 54 | 0.9000000 |
| 5 | 134 | 56 | 0.9333333 |
| 6 | 107 | 56 | 0.9333333 |
| 7 | 91 | 51 | 0.8500000 |
| 8 | 53 | 54 | 0.9152542 |
| 9 | 49 | 56 | 0.9491525 |
