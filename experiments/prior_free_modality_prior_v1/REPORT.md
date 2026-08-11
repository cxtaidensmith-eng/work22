# Prior-Free Modality Prior v1

- Decision: **PRIOR_FREE_STOP**
- Branch: `experiment/prior-free-modality-prior-v1`
- Source commit: `00c61f99808f321bedfc8d2043cd4b921854034a`
- Result commit: reported in the final Git handoff (self-reference is not embedded).
- Device: `cuda:0` (NVIDIA GeForce RTX 3050 Ti Laptop GPU)
- Protocol: fixed D5/D3 configuration, one seed (0), ten folds, 400 epochs, full batch, single model; graph/EMA/ensemble disabled.
- Selection: ACC > ROC-AUC > Macro-F1 > earliest epoch.

## Results

| Task | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | params total/trainable | train s | infer s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| tadpole_smci_pmci | 513/535 | 0.9588785 | 0.8947392 | 0.6219146 | 0.8637984 | 0.8564626 | 0.9584531 | 0.7333333 | 0.9795918 | `[[480, 10], [12, 33]]` | `{'SMCI': 492, 'PMCI': 43}` | 6/12/18 | 0.2378845 | 0.9589099+/-0.0228315 | 617197/617192 | 567.17 | 0.1432 |
| abide5_ads_cn | 754/864 | 0.8726852 | 0.8734621 | 0.8305801 | 0.8717444 | 0.8714664 | 0.8726343 | 0.8564232 | 0.8865096 | `[[340, 57], [53, 414]]` | `{'ADS': 393, 'CN': 471}` | 37/52/89 | 0.137368 | 0.8727880+/-0.0293508 | 383616/383611 | 579.20 | 0.1389 |

## Task artifacts

- `tadpole_smci_pmci`: `tadpole_smci_pmci_prior_free_oof_predictions.csv`, `tadpole_smci_pmci_fold_results.json`, `tadpole_smci_pmci_checkpoint_manifest.json`, and `tadpole_smci_pmci_effective_gate_noise_manifest.json`.
- `abide5_ads_cn`: `abide5_ads_cn_prior_free_oof_predictions.csv`, `abide5_ads_cn_fold_results.json`, `abide5_ads_cn_checkpoint_manifest.json`, and `abide5_ads_cn_effective_gate_noise_manifest.json`.
- The checkpoint manifests record all 10 completed best-checkpoint hashes per task. Checkpoint binaries remain Git-ignored; all 20 were strict-loaded and replayed against their fold OOF rows.

### tadpole_smci_pmci folds

| Fold | Best epoch | Correct | ACC | AUC | PR-AUC | Macro-F1 | BACC | SEN | SPE | loss first->last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 63 | 52 | 0.9629630 | 0.9714286 | 0.7295238 | 0.9062500 | 0.9795918 | 1.0000000 | 0.9591837 | 2.83416->1.01453 |
| 1 | 147 | 50 | 0.9259259 | 0.8938776 | 0.6297744 | 0.7795918 | 0.7795918 | 0.6000000 | 0.9591837 | 2.85545->1.01479 |
| 2 | 119 | 51 | 0.9444444 | 1.0000000 | 1.0000000 | 0.8688259 | 0.9693878 | 1.0000000 | 0.9387755 | 2.84830->1.01581 |
| 3 | 83 | 51 | 0.9444444 | 0.9469388 | 0.7633333 | 0.7708628 | 0.7000000 | 0.4000000 | 1.0000000 | 2.84029->1.02189 |
| 4 | 22 | 54 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 2.83051->1.01495 |
| 5 | 125 | 51 | 0.9622642 | 0.9642857 | 0.7500000 | 0.8233333 | 0.7500000 | 0.5000000 | 1.0000000 | 2.83242->1.01188 |
| 6 | 52 | 52 | 0.9811321 | 0.9846939 | 0.8928571 | 0.9235209 | 0.8750000 | 0.7500000 | 1.0000000 | 2.84151->1.00313 |
| 7 | 216 | 50 | 0.9433962 | 0.7397959 | 0.5025510 | 0.6851485 | 0.6250000 | 0.2500000 | 1.0000000 | 2.82031->1.00363 |
| 8 | 53 | 52 | 0.9811321 | 1.0000000 | 1.0000000 | 0.9392898 | 0.9897959 | 1.0000000 | 0.9795918 | 2.80823->1.00314 |
| 9 | 72 | 50 | 0.9433962 | 0.9693878 | 0.7750000 | 0.8178694 | 0.8545918 | 0.7500000 | 0.9591837 | 2.83593->1.00276 |

- Metric deltas vs D5: `{"acc": -0.01121495327102806, "bacc": -0.03639455782312939, "macro_f1": -0.037145898907609665, "pr_auc": -0.1102501356625114, "roc_auc": -0.02353741496598638, "sen": -0.06666666666666676, "spe": -0.00612244897959191, "weighted_f1": -0.01133097539915584}`; Correct delta=-6.
- Repair sources: `{"PMCI": 3, "SMCI": 3}`; damage sources: `{"PMCI": 6, "SMCI": 6}`; transitions: `{"PMCI->SMCI": 9, "SMCI->PMCI": 9}`.
- Effective gate: `{"COGNITIVE_TEST": 1.0, "CSF": 1.0, "MRI": 1.0, "RISK_FACTOR": 1.0, "ROI_AVERAGE": 1.0}`; effective noise std: `{"COGNITIVE_TEST": 0.05000000074505806, "CSF": 0.05000000074505806, "MRI": 0.05000000074505806, "RISK_FACTOR": 0.05000000074505806, "ROI_AVERAGE": 0.05000000074505806}`.
- Input norm ratio after/before gate changed from the legacy effective gates (MRI .2689414, CSF .1192029, RISK_FACTOR .1192029, COGNITIVE_TEST .9820138, ROI_AVERAGE .8807970) to exactly 1 for every modality; Category and Global effective scaling are both 1.
- Private/shared ratio mean/max: `{"COGNITIVE_TEST": 0.05687280265847221, "CSF": 0.04959586246404797, "MRI": 0.05427857649046928, "RISK_FACTOR": 0.029453848907724022, "ROI_AVERAGE": 0.08161114057293162}` / `{"COGNITIVE_TEST": 0.30029091238975525, "CSF": 0.37101972103118896, "MRI": 0.3101142644882202, "RISK_FACTOR": 0.656001091003418, "ROI_AVERAGE": 0.7861191630363464}`.
- Category-Global cosine mean: 0.6217829; maximum Private Adapter gradient by modality: `{"MRI": 0.1211545393, "CSF": 0.1431357116, "RISK_FACTOR": 0.1299996674, "COGNITIVE_TEST": 0.1687407792, "ROI_AVERAGE": 0.1965730190}`; private collapse: false.
- Confidence mean max-probability / mean entropy: 0.8385876 / 0.4273412.

### abide5_ads_cn folds

| Fold | Best epoch | Correct | ACC | AUC | PR-AUC | Macro-F1 | BACC | SEN | SPE | loss first->last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 157 | 75 | 0.8620690 | 0.8827128 | 0.8136061 | 0.8611702 | 0.8611702 | 0.8500000 | 0.8723404 | 2.19290->0.35999 |
| 1 | 73 | 76 | 0.8735632 | 0.8920213 | 0.8563975 | 0.8729590 | 0.8736702 | 0.8750000 | 0.8723404 | 2.19374->0.35958 |
| 2 | 170 | 74 | 0.8505747 | 0.8619681 | 0.8464391 | 0.8498606 | 0.8505319 | 0.8500000 | 0.8510638 | 2.19318->0.35947 |
| 3 | 197 | 71 | 0.8160920 | 0.7813830 | 0.8076179 | 0.8118919 | 0.8093085 | 0.7250000 | 0.8936170 | 2.19128->0.36028 |
| 4 | 63 | 78 | 0.9069767 | 0.9203492 | 0.9078127 | 0.9061648 | 0.9061648 | 0.8974359 | 0.9148936 | 2.18607->0.35989 |
| 5 | 91 | 75 | 0.8720930 | 0.8788871 | 0.8484422 | 0.8680798 | 0.8633388 | 0.7692308 | 0.9574468 | 2.18360->0.35993 |
| 6 | 137 | 77 | 0.8953488 | 0.9489907 | 0.9201672 | 0.8952213 | 0.8998909 | 0.9487179 | 0.8510638 | 2.18965->0.36052 |
| 7 | 77 | 79 | 0.9186047 | 0.9472826 | 0.9604940 | 0.9185055 | 0.9206522 | 0.9500000 | 0.8913043 | 2.18700->0.35971 |
| 8 | 15 | 75 | 0.8720930 | 0.9146739 | 0.9010847 | 0.8712400 | 0.8706522 | 0.8500000 | 0.8913043 | 2.18644->0.36014 |
| 9 | 114 | 74 | 0.8604651 | 0.8967391 | 0.8596716 | 0.8597826 | 0.8597826 | 0.8500000 | 0.8695652 | 2.18652->0.35995 |

- Metric deltas vs D3: `{"acc": -0.01736111111111105, "bacc": -0.01794777749610299, "macro_f1": -0.017596180290870622, "pr_auc": -0.019984869401168948, "roc_auc": -0.012802118673779228, "sen": -0.02518891687657432, "spe": -0.010706638115631661, "weighted_f1": -0.017422195396815998}`; Correct delta=-15.
- Repair sources: `{"ADS": 15, "CN": 22}`; damage sources: `{"ADS": 25, "CN": 27}`; transitions: `{"ADS->CN": 47, "CN->ADS": 42}`.
- Effective gate: `{"ANAT": 1.0, "FUNC": 1.0, "MRI": 1.0, "PHENO": 1.0, "fMRI": 1.0}`; effective noise std: `{"ANAT": 0.0, "FUNC": 0.0, "MRI": 0.0, "PHENO": 0.0, "fMRI": 0.0}`.
- Input norm ratio after/before gate changed from the legacy effective gates (PHENO .1192029, ANAT .1192029, FUNC .1192029, MRI .2689414, fMRI .2689414) to exactly 1 for every modality; Category and Global effective scaling are both 1.
- Private/shared ratio mean/max: `{"ANAT": 0.005489925527945161, "FUNC": 0.006645146454684436, "MRI": 0.007120157219469547, "PHENO": 0.00563322901725769, "fMRI": 0.012380167399533093}` / `{"ANAT": 0.01569187082350254, "FUNC": 0.03237389773130417, "MRI": 0.04238395765423775, "PHENO": 0.01787029393017292, "fMRI": 0.13916338980197906}`.
- Category-Global cosine mean: 0.3892077; maximum Private Adapter gradient by modality: `{"PHENO": 0.0105748037, "ANAT": 0.0102314204, "FUNC": 0.0108179022, "MRI": 0.0095959371, "fMRI": 0.0100969020}`; private collapse: false.
- Confidence mean max-probability / mean entropy: 0.9778299 / 0.0928429.

## Required answers

1. Original gates: `sigmoid(modal_gate_logit)`. TAD MRI/CSF/RISK_FACTOR/COGNITIVE_TEST/ROI_AVERAGE = .2689414/.1192029/.1192029/.9820138/.8807970. ABIDE-5 PHENO/ANAT/FUNC/MRI/fMRI = .1192029/.1192029/.1192029/.2689414/.2689414.
2. Original noise factor/effective sigma: TAD MRI .3/.15, CSF .3/.15, RISK_FACTOR .3/.15, COGNITIVE_TEST 0/0, ROI_AVERAGE .05/.025. ABIDE-5 all factors .3 but all effective sigma 0 because task `input_noise_std=0`.
3. After modification all real-modality gates are exactly 1: **True**.
4. Within each task all effective noise values are exactly equal: **True** (TAD all .05; ABIDE-5 all 0).
5. TADPOLE reached or exceeded D5 519/535: **False** (513/535).
6. ABIDE-5 reached or exceeded D3 769/864: **False** (754/864).
7. Hard-classification improvement: **neither task**; Correct deltas are -6 and -15.
8. Safety damage accompanied the change: TAD BACC/AUC/SEN deltas -0.0363946/-0.0235374/-0.0666667; ABIDE-5 BACC/AUC/SEN/SPE deltas -0.0179478/-0.0128021/-0.0251889/-0.0107066.
9. Repairs exceeded damages: **False on both tasks** (TAD 6/12; ABIDE-5 37/52).
10. Parameter counts: total counts are unchanged at 617197 and 383616; effective trainable counts decrease by exactly five to 617192 and 383611 because the legacy five-element gate is frozen; new parameters=0.
11. New trainable gate or inference branch: **none**. The legacy gate tensor is only a frozen compatibility placeholder, is absent from the optimizer, and is bypassed in forward.
12. Expansion to ABIDE or TADPOLE three-class is allowed: **No**; STOP forbids automatic expansion.
13. Final Decision: **PRIOR_FREE_STOP**. The current results depend on the historical handcrafted modality prior; revert this change and do not search gate values or run softmax/dynamic gates.
14. Reproducibility: source `00c61f99808f321bedfc8d2043cd4b921854034a`, result commit reported in the final Git handoff, branch `experiment/prior-free-modality-prior-v1`, device `cuda:0` (RTX 3050 Ti). Commands: `python -u -B scripts/run_prior_free_modality_prior_v1.py inspect`; `python -u -B scripts/run_prior_free_modality_prior_v1.py smoke --device cuda:0`; `python -u -B scripts/run_prior_free_modality_prior_v1.py formal --device cuda:0`.

## Decision reasons

- tadpole_smci_pmci:correct
- tadpole_smci_pmci:bacc
- tadpole_smci_pmci:roc_auc
- tadpole_smci_pmci:sen
- abide5_ads_cn:correct
- abide5_ads_cn:bacc
- abide5_ads_cn:roc_auc
- abide5_ads_cn:sen
- abide5_ads_cn:spe
