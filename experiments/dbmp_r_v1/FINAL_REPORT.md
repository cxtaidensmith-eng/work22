# Dual-Boundary sMCI Multi-Prototype Residual v1

Decision: **DBMP_STOP**

## Performance

- Parameters: 862,973 (+2 vs C1)
- Correct: 557/598
- ACC: 0.9314381271
- Macro-F1: 0.9159623069
- BACC: 0.9082910152
- Probability Macro-AUC: 0.9687963400
- Weighted-F1: 0.9313487543
- Confusion matrix: `[[61, 0, 11], [0, 192, 17], [8, 5, 304]]`
- Fold ACC mean +/- sample SD: 0.9314971751 +/- 0.0264976001
- Training time: 498.646 s

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 0 | 92 | 0.9500000000 |
| 1 | 160 | 0.9500000000 |
| 2 | 198 | 0.9500000000 |
| 3 | 86 | 0.9166666667 |
| 4 | 196 | 0.9000000000 |
| 5 | 94 | 0.9166666667 |
| 6 | 143 | 0.9500000000 |
| 7 | 135 | 0.8833333333 |
| 8 | 82 | 0.9322033898 |
| 9 | 87 | 0.9661016949 |

## Compared with C1

- Correct delta: -3
- ACC delta: -0.0050167224
- Macro-F1 delta: -0.0015834105
- BACC delta: -0.0080449187
- Probability Macro-AUC delta: +0.0102355649
- Repairs / damages / changed: 15 / 18 / 33
- AD-sMCI errors: 19
- CN-sMCI errors: 22

## Prototype diagnostics

- gamma_CN / gamma_AD: 0.1018218018 / 0.1052391961
- Prototype cosine: 0.9996613741
- Effective masses sum(w_CN) / sum(w_AD): 117.8535423279 / 167.4464584351
- Mean weights w_CN / w_AD: 0.4131150007 / 0.5868850112
- CN boundary residual mean/max abs: 0.0976823710 / 0.1331540495
- AD boundary residual mean/max abs: 0.0995963880 / 0.1418320239
- Residual actually changed predictions: 0
- Post-warmup cosine>0.98 fraction: 0.9142105263
- Prototypes effectively differentiated: False
- Mechanism conclusion: The two boundary-conditioned sMCI prototypes remained nearly collinear; the bounded residual changed no same-checkpoint held-out argmax, while the altered training trajectory was net harmful relative to C1.

These are boundary-conditioned sMCI prototypes only; they are not claimed to be clinical subtypes or longitudinal disease progression states.

Next recommendation: Boundary-Conditioned Disentangled Gradient Learning; not implemented.
