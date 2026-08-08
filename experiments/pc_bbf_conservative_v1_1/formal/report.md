# PC-BBF-Conservative v1.1 Formal Report

Decision: **PC_BBF_CONSERVATIVE_NO_GAIN**

- PC-BBF v1 base commit: `c3a2c6ce13241522cc598f58705417106ce7721c`
- Configuration: `pc_bbf_cap=0.075`, `safe_loss=False`
- Correct: 555/598
- ACC: 0.9280936
- Macro-F1: 0.9076023
- BACC: 0.8915577
- Probability Macro-AUC: 0.9652186
- Weighted-F1: 0.9272546
- Confusion matrix: [[56, 0, 16], [0, 198, 11], [5, 11, 301]]
- Predicted AD/CN/sMCI: 61 / 209 / 328
- Ten-fold ACC mean +/- sample SD: 0.9280508 +/- 0.0239742
- Versus C1 repairs/damages/changed: 18 / 23 / 41
- Versus PC-BBF v1 repairs/damages/changed: 19 / 25 / 44
- Metric delta versus C1 (Correct/ACC/F1/BACC/AUC): -5 / -0.0083612 / -0.0099434 / -0.0247783 / +0.0066579
- Metric delta versus PC-BBF v1 (Correct/ACC/F1/BACC/AUC): -6 / -0.0100334 / -0.0190922 / -0.0236563 / -0.0049003
- AD-sMCI / CN-sMCI / AD-CN errors: 21 / 22 / 0
- CN->sMCI: 11; sMCI->AD: 5
- Gate mean/std/min/max: 0.6443950 / 0.2808312 / 0.0171369 / 0.9963833
- Residual/shared mean/max: 0.0622752 / 0.0750000
- Cap saturation: 0.5785953
- New-module maximum gradient: 0.2865528
- Parameters: 866924
- Formal training time: 455.841 s
- Reproduction: `python -u -B scripts/run_pc_bbf_conservative_v1_1.py formal --device cuda:0`

## Required questions

1. Predicted sMCI moved from 326 toward 317: **False** (328).
2. CN->sMCI below PC-BBF v1's 12: **True** (11).
3. sMCI->AD remains clearly below C1's 10: **True** (5).
4. Repairs still exceed damages: **False** (18 vs 23).
5. Reached 563/598: **False**.

## Folds

- fold0: best epoch 163, correct 56, ACC 0.9333333
- fold1: best epoch 183, correct 55, ACC 0.9166667
- fold2: best epoch 100, correct 58, ACC 0.9666667
- fold3: best epoch 201, correct 55, ACC 0.9166667
- fold4: best epoch 98, correct 57, ACC 0.9500000
- fold5: best epoch 115, correct 55, ACC 0.9166667
- fold6: best epoch 160, correct 55, ACC 0.9166667
- fold7: best epoch 63, correct 56, ACC 0.9333333
- fold8: best epoch 129, correct 52, ACC 0.8813559
- fold9: best epoch 56, correct 56, ACC 0.9491525

No additional cap was tested. If the target was not reached, the next recommendation is Gradient-Isolated Pairwise Boundary Fusion (GI-PBF); it was not implemented or run here.
