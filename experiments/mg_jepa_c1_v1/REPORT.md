# MG-JEPA-C1 v1

Decision: **MG_JEPA_NO_GAIN**
Selected version / pretraining epochs: **v1 / 200**
Correct/ACC/Macro-F1/BACC/AUC/Weighted-F1: 529/598 / 0.8846154 / 0.8740037 / 0.8594201 / 0.9278717 / 0.8841008
Confusion matrix: [[58, 0, 14], [0, 176, 33], [7, 15, 295]]
Ten-fold ACC mean +/- sample SD: 0.8844915 +/- 0.0386868
Delta vs C1 (Correct/ACC/F1/BACC/AUC): -31 / -0.0518395 / -0.0435420 / -0.0569159 / -0.0306890
Delta vs PC-BBF (Correct/ACC/F1/BACC/AUC): -32 / -0.0535117 / -0.0526908 / -0.0557939 / -0.0422472
Repairs/damages/changed vs C1: 9 / 40 / 49
AD-sMCI / CN-sMCI / AD-CN errors: 21 / 48 / 0
Predicted AD/CN/sMCI: 65 / 191 / 342
Inference parameters: 862971; pretrain/supervised seconds: 163.126 / 536.394
Source/device/total wall: 4cceb63d1bea8c6cf89a730d8723b9365625e373 / cuda:0 / 705.093s
Final shared/private loss: 0.1572302 / 0.2126099; collapse=False
Pretrained-C1 vs C1 OOF prediction changes: 49
v1.1 run: False ({'correct_is_561_or_562': False, 'repairs_above_damages': False, 'no_ad_cn_error': True, 'bacc_drop_at_most_0p003': False, 'no_collapse': True, 'last20_relative_decline_at_least_2_percent': False})

## Complete-view latent mechanism

- MRI: shared/private cosine 0.986871/0.986863; shared/private batch std 0.331407/0.331493
- PET: shared/private cosine 0.988047/0.987548; shared/private batch std 0.228435/0.228584
- CSF: shared/private cosine 0.985715/0.985125; shared/private batch std 0.196341/0.196397
- Risk: shared/private cosine 0.988791/0.988376; shared/private batch std 0.203940/0.203998
- COG: shared/private cosine 0.991259/0.990860; shared/private batch std 0.775468/0.775491
- ROI: shared/private cosine 0.990421/0.990290; shared/private batch std 0.695860/0.695952

## Per-fold best

- fold 0: best_epoch=149, ACC=0.8833333
- fold 1: best_epoch=341, ACC=0.8833333
- fold 2: best_epoch=219, ACC=0.9333333
- fold 3: best_epoch=317, ACC=0.8500000
- fold 4: best_epoch=379, ACC=0.8500000
- fold 5: best_epoch=225, ACC=0.9333333
- fold 6: best_epoch=255, ACC=0.9000000
- fold 7: best_epoch=129, ACC=0.9166667
- fold 8: best_epoch=273, ACC=0.8135593
- fold 9: best_epoch=197, ACC=0.8813559

Stop after the registered result; do not tune another MG-JEPA setting.
