# C1 Optimizer Tune v1

Decision: **C1_TUNE_NO_GAIN**
Selected arm: **T2** (Correct > AUC > Macro-F1)
Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 552/598 / 0.9230769 / 0.9005424 / 0.8930679 / 0.9640011 / 0.9224998
Confusion matrix: [[57, 0, 15], [0, 200, 9], [9, 13, 295]]
BACC risk (drop >0.005 vs C1): True
Parameters: 862971; total training/wall seconds: 1323.238 / 1330.380
Source/device: b8478d65b3765e83db0c4d6103e876e68ffcdb6b / NVIDIA GeForce RTX 3050 Ti Laptop GPU
Actual C1 base optimizer: lr=0.01, weight_decay=0.0005
EMA protocol: disabled. The requested 20/21 schedule conflicts with the tracked C1=560 runner/config; it was not applied to preserve lr/wd-only attribution.

## Arms

### T1

- lr / weight decay: 0.0075 / 0.0005
- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 550/598 / 0.9197324 / 0.9029119 / 0.8926874 / 0.9546117 / 0.9194786
- Confusion matrix: [[59, 0, 13], [0, 190, 19], [8, 8, 301]]
- Ten-fold ACC mean +/- sample SD: 0.9196893 +/- 0.0272329
- Delta vs C1 (Correct/ACC/F1/BACC/AUC/Weighted-F1): -10 / -0.0167224 / -0.0146338 / -0.0236485 / -0.0039491 / -0.0168874
- Repairs/damages/changed: 14 / 24 / 38
- AD-sMCI / CN-sMCI / AD-CN errors: 21 / 27 / 0
- Predicted AD/CN/sMCI: 67 / 198 / 333
- BACC risk: True; training seconds: 431.456
- Fold best epoch/ACC: 0:81/0.8833333, 1:126/0.9333333, 2:167/0.9666667, 3:125/0.9166667, 4:76/0.9166667, 5:53/0.9000000, 6:115/0.9500000, 7:48/0.9166667, 8:83/0.8813559, 9:85/0.9322034

### T2

- lr / weight decay: 0.01 / 0.00025
- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 552/598 / 0.9230769 / 0.9005424 / 0.8930679 / 0.9640011 / 0.9224998
- Confusion matrix: [[57, 0, 15], [0, 200, 9], [9, 13, 295]]
- Ten-fold ACC mean +/- sample SD: 0.9230508 +/- 0.0163067
- Delta vs C1 (Correct/ACC/F1/BACC/AUC/Weighted-F1): -8 / -0.0133779 / -0.0170033 / -0.0232680 / +0.0054403 / -0.0138662
- Repairs/damages/changed: 18 / 26 / 44
- AD-sMCI / CN-sMCI / AD-CN errors: 24 / 22 / 0
- Predicted AD/CN/sMCI: 66 / 213 / 319
- BACC risk: True; training seconds: 448.654
- Fold best epoch/ACC: 0:186/0.9166667, 1:305/0.9333333, 2:120/0.9500000, 3:68/0.9333333, 4:53/0.9166667, 5:97/0.9166667, 6:86/0.9333333, 7:114/0.9000000, 8:241/0.8983051, 9:50/0.9322034

### T3

- lr / weight decay: 0.0075 / 0.00025
- Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 546/598 / 0.9130435 / 0.9006004 / 0.8824118 / 0.9582175 / 0.9124829
- Confusion matrix: [[57, 0, 15], [0, 192, 17], [3, 17, 297]]
- Ten-fold ACC mean +/- sample SD: 0.9129944 +/- 0.0175172
- Delta vs C1 (Correct/ACC/F1/BACC/AUC/Weighted-F1): -14 / -0.0234114 / -0.0169453 / -0.0339241 / -0.0003432 / -0.0238831
- Repairs/damages/changed: 16 / 30 / 46
- AD-sMCI / CN-sMCI / AD-CN errors: 18 / 34 / 0
- Predicted AD/CN/sMCI: 60 / 209 / 329
- BACC risk: True; training seconds: 443.129
- Fold best epoch/ACC: 0:183/0.9000000, 1:251/0.9000000, 2:61/0.9500000, 3:61/0.9166667, 4:80/0.9000000, 5:194/0.9333333, 6:134/0.9166667, 7:346/0.9166667, 8:166/0.8983051, 9:84/0.8983051
