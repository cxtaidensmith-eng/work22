# PC-BBF C1 v1 Screen Report

Decision: **SCREEN_GO**

- Correct: 168/179
- ACC: 0.9385475
- Macro-F1: 0.9274422
- BACC: 0.9048176
- Probability Macro-AUC: 0.9568330
- Weighted-F1: 0.9380028
- Confusion matrix: [[17, 0, 4], [0, 59, 4], [0, 3, 92]]
- Fold ACC mean ± sample SD: 0.9386064 ± 0.0190051
- Repairs / damages / changed: 7 / 5 / 12
- Boundary errors (AD–sMCI / CN–sMCI / AD–CN): 4 / 7 / 0
- Parameters: 866924 (+3953 vs C1)
- Training time: 136.335 s
- Versus C1: Correct +2, ACC +0.0111732, Macro-F1 +0.0307193, BACC +0.0034531, Macro-AUC +0.0135724
- Gate mean/std/min/max: 0.5927198 / 0.2119229 / 0.3284680 / 0.9733830
- Residual/shared mean/max: 0.0525600 / 0.1000000; cap saturation 0.2625698
- New-module max gradients (input/residual/gate): 0.0714362 / 0.2865528 / 0.03634694
- Decision basis: screen thresholds
- Reproduction: `python -u -B scripts/run_pc_bbf_c1_v1.py screen --device cuda:0`

## Folds

- fold4: best epoch 127, ACC 0.9166667
- fold7: best epoch 104, ACC 0.9500000
- fold8: best epoch 125, ACC 0.9491525
