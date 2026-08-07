# PC-BBF-Safe v1.1 Screen Report

Decision: **SCREEN_STOP**

- Correct: 222/240
- ACC: 0.9250000
- Macro-F1: 0.9125112
- BACC: 0.9104663
- Probability Macro-AUC: 0.9667885
- Weighted-F1: 0.9248416
- Confusion matrix: [[24, 0, 4], [0, 80, 4], [3, 7, 118]]
- Fold ACC mean ± sample SD: 0.9250000 ± 0.0396746
- Versus C1 repairs/damages/changed: 5/10/15
- Versus PC-BBF v1 repairs/damages/changed: 7/9/16
- Versus C1 ΔACC/F1/BACC/AUC: -0.0208333 / -0.0244515 / -0.0223214 / -0.0016180
- Versus PC-BBF v1 ΔACC/F1/BACC/AUC: -0.0083333 / -0.0110223 / -0.0117808 / -0.0080569
- Boundary errors AD–sMCI/CN–sMCI/AD–CN: 7/11/0
- Gate mean/std/min/max: 0.8081295/0.1645606/0.1456053/0.9789385
- Residual/shared mean/max; cap saturation: 0.0943903/0.1000000; 0.7791667
- Non-destructive loss mean/nonzero fraction: 0.001026976/0.4580483
- Formal safe inference changed predictions vs PC-BBF v1: True (16)
- Parameters: 866924
- Training time: 193.503 s
- Reproduction: `python -u -B scripts/run_pc_bbf_safe_v1_1.py screen --device cuda:0`

## Folds

- fold2: best epoch 166, correct 57, ACC 0.9500000
- fold3: best epoch 168, correct 54, ACC 0.9000000
- fold5: best epoch 137, correct 53, ACC 0.8833333
- fold6: best epoch 105, correct 58, ACC 0.9666667
