# Gate-Protected Private Residual v1

Decision: **GPPR_STOP**

## Performance

- Parameters: 862,977 (+6 vs C1)
- Correct: 551/598
- ACC: 0.9214046823
- Macro-F1: 0.9042587556
- BACC: 0.8942822855
- Probability Macro-AUC: 0.9610283078
- Weighted-F1: 0.9211485251
- Confusion matrix: `[[59, 0, 13], [0, 191, 18], [8, 8, 301]]`
- Fold ACC mean +/- sample SD: 0.9213559322 +/- 0.0162598918
- Training time: 460.500 s

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 0 | 81 | 0.9166666667 |
| 1 | 233 | 0.9333333333 |
| 2 | 44 | 0.9333333333 |
| 3 | 268 | 0.9166666667 |
| 4 | 217 | 0.9166666667 |
| 5 | 91 | 0.9333333333 |
| 6 | 203 | 0.9333333333 |
| 7 | 178 | 0.9166666667 |
| 8 | 136 | 0.8813559322 |
| 9 | 160 | 0.9322033898 |

## Compared with C1

- Correct delta: -9
- ACC delta: -0.0150501672
- Macro-F1 delta: -0.0132869618
- BACC delta: -0.0220536484
- Probability Macro-AUC delta: +0.0024675327
- Repairs / damages: 15 / 24
- AD-sMCI errors: 21
- CN-sMCI errors: 26
- Exact McNemar p: 0.1995908669

## Minimal mechanism diagnostics

- Beta: `{'MRI': 1.0238260567188262, 'PET': 0.9833742141723633, 'CSF': 0.9883606970310211, 'Risk': 1.0114591062068938, 'COG': 1.2113306760787963, 'ROI': 0.9941345453262329}`
- Effective private/shared ratios: `{'MRI': 0.0297941793454811, 'PET': 0.004991410719230771, 'CSF': 0.004409097833558917, 'Risk': 0.017797979060560465, 'COG': 0.33568371683359144, 'ROI': 0.00444486434571445}`
- Ratio delta vs C1: `{'MRI': -0.0458621321045189, 'PET': -0.06310142758076923, 'CSF': -0.04914448563644108, 'Risk': -0.056001070049439536, 'COG': 0.056138070673591445, 'ROI': -0.07192471181428554}`
- Non-COG adapter mean gradient: 0.0032061086
- Category-Global cosine: 0.5465131551
- COG effective residual share: 0.8377456903
- Private collapse: False

Shared/Global computation is the historical C1 path. Only the private residual reads the clean pre-noise, pre-modal-gate encoder token.

Next recommendation: Next route: dual-boundary sMCI multi-prototype; not implemented.
