# MA-ARA Pressure Validation v1

- Decision: **MA_ARA_PRESSURE_STOP**
- Branch: `experiment/ma-ara-pressure-v1`
- Source commit: `c609967a4d11ec708be8051a0eb4770c1b7c9254`
- Result commit: reported in the final Git handoff (self-reference is not embedded).
- Device: `cuda:0`
- Protocol: one seed (0), ten folds, Scout 60 epochs, fresh Formal 400 epochs, single model, graph/EMA/ensemble disabled.
- Selection: ACC > ROC-AUC > Macro-F1 > earliest epoch.

## Results

| Task | Correct | ACC | ROC-AUC | fold AUC mean+/-SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | params/private | train s | infer s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| tadpole_smci_pmci | 515/535 | 0.9626168 | 0.9377778 | 0.9525510+/-0.0645421 | 0.6914931 | 0.8833711 | 0.8988662 | 0.9633356 | 0.8222222 | 0.9755102 | `[[478, 12], [8, 37]]` | `{'SMCI': 486, 'PMCI': 49}` | 9/13/22 | 0.5234671 | 0.9626136+/-0.0175462 | 617197/8200 | 553.82 | 0.1324 |

### tadpole_smci_pmci folds and mechanism

| Fold | Best epoch | Correct | ACC | AUC | ranks | allocation hash |
|---:|---:|---:|---:|---:|---|---|
| 0 | 51 | 52 | 0.9629630 | 0.9755102 | `[10, 4, 12, 8, 6]` | `9fe46da89b5d06e09b4adbc6998c6d02a9a91f01549be98c72b36b22da05d625` |
| 1 | 72 | 51 | 0.9444444 | 0.8816327 | `[10, 8, 6, 10, 6]` | `782baac9b76def45206a55417a7e59427543441fadaccf463a37a5977795807e` |
| 2 | 82 | 51 | 0.9444444 | 0.9755102 | `[9, 5, 11, 9, 6]` | `93e419e4c9170016a91ac9dbb5aa8644d851616ebe65f563f6e57bf8dfec4f03` |
| 3 | 85 | 52 | 0.9629630 | 0.9632653 | `[10, 8, 8, 9, 5]` | `c20fea22c5a972895cc3af2ab381699d5dbaa013875e1a0e7a63e312aa8828d0` |
| 4 | 54 | 54 | 1.0000000 | 1.0000000 | `[11, 4, 8, 12, 5]` | `370e80474891102e34380dcdf0e06679560fa35776abdd2e4ac31efbdba67d91` |
| 5 | 17 | 50 | 0.9433962 | 0.9897959 | `[4, 8, 10, 8, 10]` | `2a17230a9e510e4f960baac1f865f81dbc45b3f7841430b91cf37303e28bbf44` |
| 6 | 23 | 51 | 0.9622642 | 0.9795918 | `[8, 3, 9, 13, 7]` | `b82865e5e0eb0f6561dcf7e49a5f761e58a199adcc1e34b88a1183731d891bf8` |
| 7 | 192 | 51 | 0.9622642 | 0.7959184 | `[9, 5, 9, 8, 9]` | `dff9376c552a0b18fe5ab89d889c832adc068cd74cb23d391853cc8828fe9017` |
| 8 | 59 | 52 | 0.9811321 | 1.0000000 | `[11, 6, 6, 13, 4]` | `d9a846dee964be1c48adc863952ff4607ab055995202ac265e759a1f641268b6` |
| 9 | 130 | 51 | 0.9622642 | 0.9642857 | `[8, 7, 7, 6, 12]` | `d3e62a0f529e92576e513cd8117ad6aa6788e65ceeff3ef2291909c8f3b510cb` |

- Rank mean by modality: `{"COGNITIVE_TEST": 9.6, "CSF": 5.8, "MRI": 9.0, "RISK_FACTOR": 8.6, "ROI_AVERAGE": 7.0}`
- Rank sample SD by modality: `{"COGNITIVE_TEST": 2.3664319132398464, "CSF": 1.8737959096740262, "MRI": 2.0548046676563256, "RISK_FACTOR": 2.011080417199781, "ROI_AVERAGE": 2.5385910352879693}`
- Rank frequency: `{"COGNITIVE_TEST": {"10": 1, "12": 1, "13": 2, "6": 1, "8": 3, "9": 2}, "CSF": {"3": 1, "4": 2, "5": 2, "6": 1, "7": 1, "8": 3}, "MRI": {"10": 3, "11": 2, "4": 1, "8": 2, "9": 2}, "RISK_FACTOR": {"10": 1, "11": 1, "12": 1, "6": 2, "7": 1, "8": 2, "9": 2}, "ROI_AVERAGE": {"10": 1, "12": 1, "4": 1, "5": 2, "6": 3, "7": 1, "9": 1}}`
- Fold allocation stability (pairwise L1 mean/max): 12.444/24.000
- Importance mean/max by modality: `{"COGNITIVE_TEST": 4.065654401462454e-06, "CSF": 2.1113927156716736e-06, "MRI": 2.0086040563052444e-06, "RISK_FACTOR": 2.9577020520124708e-06, "ROI_AVERAGE": 2.8159077962626204e-06}` / `{"COGNITIVE_TEST": 6.194417659735243e-05, "CSF": 4.218042118435066e-05, "MRI": 2.3124880914082135e-05, "RISK_FACTOR": 6.80213628135869e-05, "ROI_AVERAGE": 4.081508012865145e-05}`
- Private/shared norm ratio mean: `{"COGNITIVE_TEST": 0.2553643877618015, "CSF": 0.1893473609816283, "MRI": 0.1112835946609266, "RISK_FACTOR": 0.17656340175308288, "ROI_AVERAGE": 0.10462915736716241}`
- Adapter maximum gradient: 0.44444859; private collapse: False
- Budget all folds: True; train-only allocation all folds: True; Scout weight transfers: 0
- Metric deltas vs reference: `{"acc": -0.00747663551401867, "bacc": 0.00600907029478448, "macro_f1": -0.017573149942455846, "pr_auc": -0.04067163160019749, "roc_auc": 0.019501133786848146, "sen": 0.022222222222222143, "spe": -0.010204081632653073, "weighted_f1": -0.0064484612876034575}`; Correct delta=-4
| abide5_ads_cn | 756/864 | 0.8750000 | 0.8710754 | 0.8806000+/-0.0506923 | 0.8437912 | 0.8742209 | 0.8743629 | 0.8750229 | 0.8664987 | 0.8822270 | `[[344, 53], [55, 412]]` | `{'ADS': 399, 'CN': 465}` | 33/46/79 | 0.1766106 | 0.8751136+/-0.0413375 | 383616/2900 | 566.51 | 0.1438 |

### abide5_ads_cn folds and mechanism

| Fold | Best epoch | Correct | ACC | AUC | ranks | allocation hash |
|---:|---:|---:|---:|---:|---|---|
| 0 | 39 | 74 | 0.8505747 | 0.8571809 | `[4, 4, 2, 4, 6]` | `86b17750edd8199cfaddf6a0e3c75767ae379d25da9b386638d7768944bbcb21` |
| 1 | 81 | 80 | 0.9195402 | 0.9457447 | `[4, 3, 4, 3, 6]` | `c91db6b3ec9bc2f61c4f150129e6702201de5b6bb2398696f465e30b7d67569f` |
| 2 | 195 | 72 | 0.8275862 | 0.8140957 | `[3, 3, 3, 3, 8]` | `7f9ccb84a186d25068f158c55cd976064fc4b001b1b2ebd43c1332b03755dfea` |
| 3 | 242 | 70 | 0.8045977 | 0.8311170 | `[4, 2, 3, 5, 6]` | `a0dc7a22329d5fd66768220e8b19432d9ba9bfe19b7c2ab78172541883c5ddba` |
| 4 | 66 | 80 | 0.9302326 | 0.9492635 | `[4, 4, 3, 6, 3]` | `d5fd387d8a40e3b2879e53376e8fd4ee649698cd095f2001f5dd1280aba44b70` |
| 5 | 247 | 75 | 0.8720930 | 0.8666121 | `[2, 4, 4, 2, 8]` | `357e02084c4016739c6deed635737f2512ce4c8e93d4ee3c95d19feaee84ed66` |
| 6 | 15 | 75 | 0.8720930 | 0.8936170 | `[7, 1, 2, 5, 5]` | `6624e0df5d664bca520ecf3f1721d9bca810ff818b4235811135e56a12552efc` |
| 7 | 138 | 79 | 0.9186047 | 0.9364130 | `[7, 3, 4, 2, 4]` | `172bde8848e1eb929806a6115f21fdb12268b4c6daacd7cc5227c908f88c424c` |
| 8 | 269 | 77 | 0.8953488 | 0.8880435 | `[3, 3, 4, 5, 5]` | `b08a679daf5215f75c718ce21e60d3e61e2f06f32a1015f3e61af20ee040e29b` |
| 9 | 32 | 74 | 0.8604651 | 0.8239130 | `[4, 4, 2, 7, 3]` | `341d1e149f543c4071e6089322f4903e0949ac2db12cd956f2969c0cded83689` |

- Rank mean by modality: `{"ANAT": 3.1, "FUNC": 3.1, "MRI": 4.2, "PHENO": 4.2, "fMRI": 5.4}`
- Rank sample SD by modality: `{"ANAT": 0.9944289260117531, "FUNC": 0.8755950357709131, "MRI": 1.6865480854231356, "PHENO": 1.6193277068654823, "fMRI": 1.776388345929897}`
- Rank frequency: `{"ANAT": {"1": 1, "2": 1, "3": 4, "4": 4}, "FUNC": {"2": 3, "3": 3, "4": 4}, "MRI": {"2": 2, "3": 2, "4": 1, "5": 3, "6": 1, "7": 1}, "PHENO": {"2": 1, "3": 2, "4": 5, "7": 2}, "fMRI": {"3": 2, "4": 1, "5": 2, "6": 3, "8": 2}}`
- Fold allocation stability (pairwise L1 mean/max): 7.911/16.000
- Importance mean/max by modality: `{"ANAT": 4.529128891942315e-09, "FUNC": 5.155772503150771e-09, "MRI": 1.2914098994420055e-08, "PHENO": 5.219882537300553e-09, "fMRI": 3.168254499407938e-08}` / `{"ANAT": 3.398006594714244e-08, "FUNC": 3.3334988011651405e-08, "MRI": 4.993167462543753e-07, "PHENO": 2.0952581725522757e-08, "fMRI": 7.528944946972133e-07}`
- Private/shared norm ratio mean: `{"ANAT": 0.00842405108269304, "FUNC": 0.00772124114446342, "MRI": 0.008548931358382106, "PHENO": 0.008456438640132546, "fMRI": 0.017940441938117148}`
- Adapter maximum gradient: 0.013562182; private collapse: False
- Budget all folds: True; train-only allocation all folds: True; Scout weight transfers: 0
- Metric deltas vs reference: `{"acc": -0.01504629629629628, "bacc": -0.015051321743914436, "macro_f1": -0.015119642813013368, "pr_auc": -0.006773727593615941, "roc_auc": -0.015188862938850911, "sen": -0.015113350125944502, "spe": -0.01498929336188437, "weighted_f1": -0.015033609750442434}`; Correct delta=-13

## Required answers

1. TADPOLE exceeds D5 519/535: **False** (515/535).
2. ABIDE-5 exceeds D3 769/864: **False** (756/864).
3. Repairs exceed damages on both tasks: **False**.
4. Main modality allocations are given by rank means: TAD `{"COGNITIVE_TEST": 9.6, "CSF": 5.8, "MRI": 9.0, "RISK_FACTOR": 8.6, "ROI_AVERAGE": 7.0}`; ABIDE-5 `{"ANAT": 3.1, "FUNC": 3.1, "MRI": 4.2, "PHENO": 4.2, "fMRI": 5.4}`.
5. Ten-fold stability: TAD pairwise L1 mean 12.444; ABIDE-5 7.911; full frequencies are above.
6. Parameter counts do not exceed references: **True**.
7. Scout weights were not used by Formal or inference: **True**; only train-derived allocations were carried over.
8. MA_ARA_PRESSURE_PASS satisfied: **False**.
9. Expansion to ABIDE/TADPOLE-three-class is recommended only on PASS: **False**; it was not run here.
10. Source `c609967a4d11ec708be8051a0eb4770c1b7c9254`, result commit in final Git handoff, branch `experiment/ma-ara-pressure-v1`, device `cuda:0`. Reproduction: `python -u -B scripts/run_ma_ara_pressure_v1.py inspect`; then `... smoke --device cuda:0`; then `... formal --device cuda:0` from the recorded source commit.

## Decision reasons

- tadpole_smci_pmci: Correct below reference
- abide5_ads_cn: Correct below reference
- abide5_ads_cn: BACC drop exceeds .005
- abide5_ads_cn: ROC-AUC drop exceeds .01
