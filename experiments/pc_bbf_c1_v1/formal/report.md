# PC-BBF C1 v1 Formal Report

Decision: **PC_BBF_STOP**

- Correct: 561/598
- ACC: 0.9381271
- Macro-F1: 0.9266945
- BACC: 0.9152140
- Probability Macro-AUC: 0.9701189
- Weighted-F1: 0.9378308
- Confusion matrix: [[61, 0, 11], [0, 197, 12], [4, 10, 303]]
- Fold ACC mean ± sample SD: 0.9381638 ± 0.0235452
- Repairs / damages / changed: 18 / 17 / 35
- Boundary errors (AD–sMCI / CN–sMCI / AD–CN): 15 / 22 / 0
- Parameters: 866924 (+3953 vs C1)
- Training time: 459.153 s
- Versus C1: Correct +1, ACC +0.0016722, Macro-F1 +0.0091488, BACC -0.0011220, Macro-AUC +0.0115582
- Gate mean/std/min/max: 0.6097515 / 0.2288230 / 0.0496208 / 0.9881379
- Residual/shared mean/max: 0.0597464 / 0.1000000; cap saturation 0.3110368
- New-module max gradients (input/residual/gate): 0.08285906 / 0.2865528 / 0.06953
- Decision basis: formal STOP: the 563-subject target was not reached and BACC changed by -0.0011220 versus C1
- Reproduction: `python -u -B scripts/run_pc_bbf_c1_v1.py formal --device cuda:0`

## Folds

- fold0: best epoch 84, ACC 0.9333333
- fold1: best epoch 279, ACC 0.9500000
- fold2: best epoch 87, ACC 0.9666667
- fold3: best epoch 73, ACC 0.9500000
- fold4: best epoch 127, ACC 0.9166667
- fold5: best epoch 116, ACC 0.8833333
- fold6: best epoch 118, ACC 0.9333333
- fold7: best epoch 104, ACC 0.9500000
- fold8: best epoch 125, ACC 0.9491525
- fold9: best epoch 60, ACC 0.9491525
