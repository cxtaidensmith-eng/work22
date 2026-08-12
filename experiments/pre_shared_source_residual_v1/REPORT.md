# PS-SPR v1: Pre-Shared Source-Preserving Residual

- Decision: **PS_SPR_STOP**
- Branch: `experiment/pre-shared-source-residual-v1`
- Source commit: `4cdcfcd4dbd76369ab6cbfc566ca0a26553b9246`
- Result commit: reported in the final Git handoff (self-reference is not embedded).
- Device: `cuda:0` (NVIDIA GeForce RTX 3050 Ti Laptop GPU)
- Protocol: fixed D5/D3 configuration, one seed (0), ten folds, 400 epochs, full batch, single model; graph/EMA/ensemble disabled.
- Selection: ACC > ROC-AUC > Macro-F1 > earliest epoch.

## Results

| Task | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | CM | predicted | repairs/damages/changed | McNemar p | fold ACC mean+/-SD | fold AUC mean+/-SD | params total/trainable | train s | infer s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| tadpole_smci_pmci | 513/535 | 0.9588785 | 0.8794558 | 0.6338803 | 0.8691873 | 0.8766440 | 0.9592834 | 0.7777778 | 0.9755102 | `[[478, 12], [10, 35]]` | `{'SMCI': 488, 'PMCI': 47}` | 2/8/10 | 0.109375 | 0.9589448+/-0.0227423 | 0.9413265+/-0.0596185 | 617197/617197 | 608.51 | 0.1374 |
| abide5_ads_cn | 764/864 | 0.8842593 | 0.8890582 | 0.8374174 | 0.8835799 | 0.8838721 | 0.8843004 | 0.8790932 | 0.8886510 | `[[349, 48], [52, 415]]` | `{'ADS': 401, 'CN': 463}` | 41/46/87 | 0.668285 | 0.8843625+/-0.0392262 | 0.9012076+/-0.0445271 | 383616/383616 | 609.89 | 0.1536 |

### tadpole_smci_pmci folds

| Fold | Best epoch | Correct | ACC | AUC | PR-AUC | Macro-F1 | BACC | SEN | SPE | loss first->last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16 | 51 | 0.9444444 | 0.9877551 | 0.9250000 | 0.8481724 | 0.8795918 | 0.8000000 | 0.9591837 | 2.81157->1.01923 |
| 1 | 64 | 50 | 0.9259259 | 0.8816327 | 0.5259649 | 0.7795918 | 0.7795918 | 0.6000000 | 0.9591837 | 2.81132->1.02081 |
| 2 | 120 | 51 | 0.9444444 | 0.9755102 | 0.8392857 | 0.8688259 | 0.9693878 | 1.0000000 | 0.9387755 | 2.82134->1.01496 |
| 3 | 51 | 51 | 0.9444444 | 0.9714286 | 0.7544444 | 0.8181818 | 0.7897959 | 0.6000000 | 0.9795918 | 2.81102->1.01601 |
| 4 | 61 | 54 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 1.0000000 | 2.81639->1.01545 |
| 5 | 160 | 51 | 0.9622642 | 0.8775510 | 0.6372283 | 0.8233333 | 0.7500000 | 0.5000000 | 1.0000000 | 2.80158->1.00469 |
| 6 | 50 | 51 | 0.9622642 | 0.8418367 | 0.7169118 | 0.8647959 | 0.8647959 | 0.7500000 | 0.9795918 | 2.81593->1.00438 |
| 7 | 39 | 50 | 0.9433962 | 0.8979592 | 0.6142857 | 0.8178694 | 0.8545918 | 0.7500000 | 0.9591837 | 2.80404->1.02542 |
| 8 | 120 | 52 | 0.9811321 | 1.0000000 | 1.0000000 | 0.9392898 | 0.9897959 | 1.0000000 | 0.9795918 | 2.79479->1.02347 |
| 9 | 107 | 52 | 0.9811321 | 0.9795918 | 0.8750000 | 0.9235209 | 0.8750000 | 0.7500000 | 1.0000000 | 2.80825->1.00371 |

- Metric deltas vs D5: `{"acc": -0.01121495327102806, "bacc": -0.016213151927437663, "macro_f1": -0.03175693073379293, "pr_auc": -0.09828450314157422, "roc_auc": -0.03882086167800458, "sen": -0.022222222222222254, "spe": -0.010204081632653073, "weighted_f1": -0.010500634672798381}`; Correct delta=-6.
- Repair sources: `{"PMCI": 2, "SMCI": 0}`; damage sources: `{"PMCI": 3, "SMCI": 5}`; transitions: `{"PMCI->SMCI": 3, "SMCI->PMCI": 7}`.
- Historical gate: `{"COGNITIVE_TEST": 0.9781702160835266, "CSF": 0.11850165575742722, "MRI": 0.2724095284938812, "RISK_FACTOR": 0.126991406083107, "ROI_AVERAGE": 0.8679243922233582}`; historical noise std: `{"COGNITIVE_TEST": 0.0, "CSF": 0.15000000596046448, "MRI": 0.15000000596046448, "RISK_FACTOR": 0.15000000596046448, "ROI_AVERAGE": 0.02500000037252903}`.
- Pre/post shape: `[54, 5, 96]` / `[54, 5, 96]`; source/shared norm ratio: `{"COGNITIVE_TEST": 0.05637382969431398, "CSF": 0.009356336009168445, "MRI": 0.01730789344231823, "RISK_FACTOR": 0.01162557164414023, "ROI_AVERAGE": 0.04319959769410695}`.
- Private/source ratio: `{"COGNITIVE_TEST": 0.14013638542965054, "CSF": 0.2572490029036999, "MRI": 0.1395512118935585, "RISK_FACTOR": 0.21937274262309076, "ROI_AVERAGE": 0.04554399736225605}`; Private/shared ratio mean/max: `{"COGNITIVE_TEST": 0.007932323589920997, "CSF": 0.0025089710485190152, "MRI": 0.0026993206585757433, "RISK_FACTOR": 0.0026099137030541897, "ROI_AVERAGE": 0.002222655015066266}` / `{"COGNITIVE_TEST": 0.12505993247032166, "CSF": 0.009290131740272045, "MRI": 0.011137822642922401, "RISK_FACTOR": 0.014269163832068443, "ROI_AVERAGE": 0.010941744782030582}`.
- Adapter max gradient by modality: `{"COGNITIVE_TEST": 0.08438242971897125, "CSF": 0.06156207621097565, "MRI": 0.05766819790005684, "RISK_FACTOR": 0.06219763308763504, "ROI_AVERAGE": 0.047100622206926346}`; Category-Global cosine mean: 0.5000097.
- Source perturbation: max unchanged P_m error=0, min changed P_n magnitude=2.21e-05; private-only Modal Encoder/Shared Transformer gradient max=0.0014/0.
- Same-checkpoint Private-off probability mean/max change=6.980664e-05/0.009837985; argmax changes=0.
- Parameter delta=0; training/inference time delta=92.537s/0.002037s.
- Descriptive targets (not selection gates): `{"correct_at_least_520": false, "correct_at_least_523": false, "fold_acc_mean_above_0_9757": false, "fold_roc_auc_mean_above_0_9049": true}`.
- Confidence mean max-probability / mean entropy: 0.8317999 / 0.4306516.

### abide5_ads_cn folds

| Fold | Best epoch | Correct | ACC | AUC | PR-AUC | Macro-F1 | BACC | SEN | SPE | loss first->last |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 118 | 76 | 0.8735632 | 0.8765957 | 0.8062405 | 0.8718704 | 0.8699468 | 0.8250000 | 0.9148936 | 2.19933->0.36025 |
| 1 | 98 | 81 | 0.9310345 | 0.9771277 | 0.9756176 | 0.9308059 | 0.9324468 | 0.9500000 | 0.9148936 | 2.19677->0.36017 |
| 2 | 228 | 72 | 0.8275862 | 0.8664894 | 0.7824552 | 0.8272210 | 0.8292553 | 0.8500000 | 0.8085106 | 2.20161->0.36016 |
| 3 | 153 | 71 | 0.8160920 | 0.8172872 | 0.7898570 | 0.8148936 | 0.8148936 | 0.8000000 | 0.8297872 | 2.19842->0.36000 |
| 4 | 170 | 80 | 0.9302326 | 0.9378069 | 0.8824859 | 0.9298913 | 0.9318058 | 0.9487179 | 0.9148936 | 2.19460->0.36006 |
| 5 | 167 | 77 | 0.8953488 | 0.8843426 | 0.8983340 | 0.8941900 | 0.8933442 | 0.8717949 | 0.9148936 | 2.19425->0.35993 |
| 6 | 50 | 76 | 0.8837209 | 0.9236225 | 0.8833018 | 0.8827059 | 0.8827059 | 0.8717949 | 0.8936170 | 2.19419->0.35981 |
| 7 | 106 | 79 | 0.9186047 | 0.9336957 | 0.9347949 | 0.9180618 | 0.9173913 | 0.9000000 | 0.9347826 | 2.19457->0.35951 |
| 8 | 126 | 77 | 0.8953488 | 0.8945652 | 0.8789600 | 0.8941900 | 0.8923913 | 0.8500000 | 0.9347826 | 2.19179->0.35992 |
| 9 | 35 | 75 | 0.8720930 | 0.9005435 | 0.9044181 | 0.8720757 | 0.8755435 | 0.9250000 | 0.8260870 | 2.19327->0.35987 |

- Metric deltas vs D3: `{"acc": -0.0057870370370369795, "bacc": -0.005542101090081353, "macro_f1": -0.005760615134248748, "pr_auc": -0.013147598574218766, "roc_auc": 0.002793974077530015, "sen": -0.0025188916876573986, "spe": -0.008565310492505307, "weighted_f1": -0.005756093269723261}`; Correct delta=-5.
- Repair sources: `{"ADS": 19, "CN": 22}`; damage sources: `{"ADS": 20, "CN": 26}`; transitions: `{"ADS->CN": 42, "CN->ADS": 45}`.
- Historical gate: `{"ANAT": 0.17367978394031525, "FUNC": 0.17196425795555115, "MRI": 0.31970807909965515, "PHENO": 0.15934427082538605, "fMRI": 0.341159462928772}`; historical noise std: `{"ANAT": 0.0, "FUNC": 0.0, "MRI": 0.0, "PHENO": 0.0, "fMRI": 0.0}`.
- Pre/post shape: `[87, 5, 64]` / `[87, 5, 64]`; source/shared norm ratio: `{"ANAT": 0.059675051437828634, "FUNC": 0.062189455290375666, "MRI": 0.11387593875900556, "PHENO": 0.05767627905513766, "fMRI": 0.12428563878298297}`.
- Private/source ratio: `{"ANAT": 0.09323961474001408, "FUNC": 0.09062971137464046, "MRI": 0.04349667653441429, "PHENO": 0.09074503108859062, "fMRI": 0.04280423801392317}`; Private/shared ratio mean/max: `{"ANAT": 0.005553888878785074, "FUNC": 0.005728602432645858, "MRI": 0.0053714524488896135, "PHENO": 0.005294262408278882, "fMRI": 0.005893679009750486}` / `{"ANAT": 0.02820548601448536, "FUNC": 0.02602042816579342, "MRI": 0.030127154663205147, "PHENO": 0.02674298919737339, "fMRI": 0.028357187286019325}`.
- Adapter max gradient by modality: `{"ANAT": 0.013574980199337006, "FUNC": 0.01298862136900425, "MRI": 0.012708465568721294, "PHENO": 0.013530678115785122, "fMRI": 0.012708145193755627}`; Category-Global cosine mean: 0.1731798.
- Source perturbation: max unchanged P_m error=0, min changed P_n magnitude=1.14e-05; private-only Modal Encoder/Shared Transformer gradient max=7e-05/0.
- Same-checkpoint Private-off probability mean/max change=0.0002464451/0.1852207; argmax changes=0.
- Parameter delta=0; training/inference time delta=93.273s/-0.003197s.
- Descriptive targets (not selection gates): `{"correct_at_least_770": false, "correct_at_least_787": false, "fold_acc_mean_above_0_9105": false, "fold_roc_auc_mean_above_0_9099": false}`.
- Confidence mean max-probability / mean entropy: 0.9882892 / 0.0630454.

## Required answers

1. Historical post-shared adapters read `H_m`, the corresponding token after Shared Transformer mixing.
2. PS-SPR reads `E_m`, the last gated/noised, modality-encoded token immediately before the first cross-modal Shared Transformer block.
3. At that capture point `E_m` has not read other modality tokens; this is a graph-origin statement, not statistical independence.
4. Parameter counts are exactly unchanged from post-shared references: **True**.
5. Step-0 logits/probabilities are strictly equal within 1e-7 on both tasks: **True**.
6. Source-preserving graph checks passed for every modality in both tasks: **True**.
7. TADPOLE Correct is 513/535; exceeds 519: **False**.
8. ABIDE-5 Correct is 764/864; exceeds 769: **False**.
9. Combined Correct net change is -11.
10. Repairs exceed damages across both tasks: **False**.
11. BACC/AUC/PR-AUC/SEN/SPE safety: TAD `{"bacc": false, "historical_gate_optimized": true, "historical_gate_trainable": true, "no_class_collapse": true, "pr_auc": false, "private_trained": true, "roc_auc": false, "sen": false, "spe": false}`; ABIDE-5 `{"bacc": false, "historical_gate_optimized": true, "historical_gate_trainable": true, "no_class_collapse": true, "pr_auc": false, "private_trained": true, "roc_auc": true, "sen": true, "spe": true}`.
12. Adapters and corresponding Modal Encoders receive effective gradients: **True**.
    Private collapse by task: TAD=False, ABIDE-5=False.
13. The pre-mixing source-evidence hypothesis is supported only if Decision is PASS: **False**.
14. Expansion to ABIDE and TADPOLE three-class is allowed only on PASS: **False**; this run never auto-expands.
15. Retain D5/D3: **True**.
16. Final Decision: **PS_SPR_STOP**.
17. Reproducibility: source `4cdcfcd4dbd76369ab6cbfc566ca0a26553b9246`, result commit is reported in final handoff, branch `experiment/pre-shared-source-residual-v1`, device `cuda:0`; commands are listed below.

## Reproduction

- The exact source gate must be run from an isolated clean checkout where local branch `experiment/pre-shared-source-residual-v1` points to source commit `4cdcfcd4dbd76369ab6cbfc566ca0a26553b9246`. The later result commit is for reading artifacts and intentionally does not satisfy the source-scope gate.
- The protected sibling reference worktrees `tmp/tad_binary_hparam_search_v1` and `tmp/abide5_hparam_search_v1` must remain at their locked commits and paths.
- `python -u -B scripts/run_pre_shared_source_residual_v1.py inspect`
- `python -u -B scripts/run_pre_shared_source_residual_v1.py smoke --device cuda:0`
- `python -u -B scripts/run_pre_shared_source_residual_v1.py formal --device cuda:0`

## Decision reasons

- tadpole_smci_pmci:Correct declined >=2
- tadpole_smci_pmci:BACC below -0.005
- tadpole_smci_pmci:AUC below -0.01
- abide5_ads_cn:Correct declined >=2
- abide5_ads_cn:BACC below -0.005
- combined Correct did not increase
