# CBTR-C1 formal_v1

Source commit: `983ecd05b507085fce86c5bd52122d095694a36b`; cap=0.10.

## Formal ten-fold result

- Correct: 547/598
- ACC: 0.9147157
- Macro-F1: 0.9023140
- BACC: 0.8947410
- Probability Macro-AUC: 0.9490057
- Weighted-F1: 0.9144779
- Confusion: `[[60, 0, 12], [0, 193, 16], [7, 16, 294]]`
- Fold ACC: 0.9147740 +/- 0.0354327 sample SD
- Repairs/damages/changed vs C1: 14/27/41
- Repairs/damages/changed vs PC-BBF: 16/30/46
- AD-sMCI / CN-sMCI / AD-CN: 19 / 32 / 0
- Predicted AD/CN/sMCI: 67/209/322
- Delta vs C1: `{'correct': -13, 'acc': -0.021739130434782705, 'macro_f1': -0.015231676748321266, 'bacc': -0.021594899126859857, 'macro_auc': -0.009555119795105593, 'weighted_f1': -0.021888053386334483}`
- Delta vs PC-BBF: `{'correct': -14, 'acc': -0.02341137123745829, 'macro_f1': -0.024380481075002236, 'bacc': -0.02047293735405964, 'macro_auc': -0.021113292478934786, 'weighted_f1': -0.023352867060657423}`
- Parameters: 866347 (+3376)
- Summed fold training seconds: 564.273
- Observed end-to-end formal wall including interruption and strict resume: 655.078 seconds

## Retrieval mechanism

- Class attention mass: `{'AD': 0.14155262194499513, 'CN': 0.33127355494409044, 'SMCI': 0.5271738223415366}`
- Entropy / effective neighbors: 1.654836 / 5.470150
- Residual/H0 mean / max: 0.081012 / 0.100000
- Cap saturation: 0.558528
- Same-checkpoint R-on/off: `{'logit_max_abs_difference': 0.7394070625305176, 'probability_max_abs_difference': 0.27275004982948303, 'argmax_changed': 2, 'direct_repairs': 2, 'direct_damages': 0, 'direct_changed_subject_indices': [27, 564]}`

Test-label mechanism summaries were computed only after all 598 OOF rows were complete.

## Per-fold selected epochs

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
