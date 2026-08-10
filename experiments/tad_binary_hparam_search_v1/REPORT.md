# TADPOLE Binary Task Hyperparameter Search v1

- Source commit: `0d7a183a7106c673de2817f36c69d780dcd792d5`
- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)
- Branch: `experiment/tad-binary-hparam-search-v1`
- Device: `cuda:0`
- Decision: **TAD_BINARY_NEAR**
- Target status: **TAD_BINARY_TARGET_NOT_REACHED**
- Formal aggregation time: 31.147 seconds

## Locked historical protocol

All trials use TADPOLE SMCI_PMCI (535 subjects: 490 sMCI, 45 pMCI; pMCI positive), five real modalities, seed 0, ten full-batch transductive folds, 400 epochs, global full-dataset historical class weights, criterion_lossv2 with two unnormalized OVR CE terms, label smoothing 0.05, orthogonality rate 0.0001, Adam, grad clip 1, CustomCosineAnnealingLR(T_max=400), test-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch, graph off, EMA off, and no ensemble.
The previous Binary Task-Adaptive Private Residual Calibration experiment is not reused because it changed loss, class-weight scope, and validation protocol.

## Selected configurations

- TUNED_B0: trial `C2`, lr=0.0125, weight_decay=0.00025, dropout=0.67.
- TUNED_B1: trial `D5`, lr=0.0125, weight_decay=0.00025, dropout=0.67, rank=8, adapter LR multiplier=1.0.
- Local LR refinement triggered: `True` (Stage D best entered the registered 519-522 safe-improvement window).

## Primary results

| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Fold PR mean +/- SD | Macro-F1 | BACC | Weighted-F1 | pMCI SEN | sMCI SPE | Params | Train sec | Infer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical B0 | 513/535 | 0.9588785 | 0.9588749 +/- 0.0229204 | 0.9223583 | 0.9620408 +/- 0.0349590 | 0.6378392 | 0.8008690 +/- 0.1544057 | 0.8609142 | 0.8463719 | 0.9580058 | 0.7111111 | 0.9816327 | 608997 | 1085.4 | 0.1879 |
| Historical B1 | 515/535 | 0.9626168 | 0.9626834 +/- 0.0214541 | 0.8794558 | 0.9580612 +/- 0.0314638 | 0.6729809 | 0.7550532 +/- 0.1548557 | 0.8786848 | 0.8786848 | 0.9626168 | 0.7777778 | 0.9795918 | 617197 | 1199.0 | 0.1892 |
| TUNED_B0 | 515/535 | 0.9626168 | 0.9626485 +/- 0.0214204 | 0.8963265 | 0.9691837 +/- 0.0227144 | 0.6734920 | 0.7637944 +/- 0.1711480 | 0.8786848 | 0.8786848 | 0.9626168 | 0.7777778 | 0.9795918 | 608997 | 470.6 | 0.1340 |
| TUNED_B1 | 519/535 | 0.9700935 | 0.9701607 +/- 0.0235002 | 0.9182766 | 0.9630612 +/- 0.0436881 | 0.7321648 | 0.8104569 +/- 0.2041248 | 0.9009443 | 0.8928571 | 0.9697841 | 0.8000000 | 0.9857143 | 617197 | 516.0 | 0.1353 |

### Confusion and prediction counts

- Historical B0: confusion=[[481, 9], [13, 32]] (sMCI,pMCI); TP/FN/TN/FP=32/13/481/9; predicted sMCI/pMCI=494/41.
- Historical B1: confusion=[[480, 10], [10, 35]] (sMCI,pMCI); TP/FN/TN/FP=35/10/480/10; predicted sMCI/pMCI=490/45.
- TUNED_B0: confusion=[[480, 10], [10, 35]] (sMCI,pMCI); TP/FN/TN/FP=35/10/480/10; predicted sMCI/pMCI=490/45.
- TUNED_B1: confusion=[[483, 7], [9, 36]] (sMCI,pMCI); TP/FN/TN/FP=36/9/483/7; predicted sMCI/pMCI=492/43.

## Registered trial table

| Trial | Stage | Arm | LR | WD | Dropout | Rank | Adapter mult | Safe | Correct | ROC-AUC | Fold AUC mean | PR-AUC | Macro-F1 | BACC |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| A1 | A | B0 | 0.0075 | 0.0005 | 0.67 | 0 | 0 | True | 513 | 0.9245805 | 0.9324490 | 0.6972331 | 0.8691873 | 0.8766440 |
| A2 | A | B0 | 0.01 | 0.0005 | 0.67 | 0 | 0 | True | 513 | 0.9223583 | 0.9620408 | 0.6378392 | 0.8609142 | 0.8463719 |
| A3 | A | B0 | 0.0125 | 0.0005 | 0.67 | 0 | 0 | True | 515 | 0.8741497 | 0.9704082 | 0.7150518 | 0.8735583 | 0.8585034 |
| B1 | B | B0 | 0.0125 | 0.00025 | 0.67 | 0 | 0 | True | 515 | 0.8963265 | 0.9691837 | 0.6734920 | 0.8786848 | 0.8786848 |
| B2 | B | B0 | 0.0125 | 0.0005 | 0.67 | 0 | 0 | True | 515 | 0.8741497 | 0.9704082 | 0.7150518 | 0.8735583 | 0.8585034 |
| B3 | B | B0 | 0.0125 | 0.001 | 0.67 | 0 | 0 | True | 515 | 0.8634240 | 0.9331122 | 0.6678754 | 0.8810794 | 0.8887755 |
| C1 | C | B0 | 0.0125 | 0.00025 | 0.55 | 0 | 0 | True | 512 | 0.9151020 | 0.9604082 | 0.7099820 | 0.8671525 | 0.8857143 |
| C2 | C | B0 | 0.0125 | 0.00025 | 0.67 | 0 | 0 | True | 515 | 0.8963265 | 0.9691837 | 0.6734920 | 0.8786848 | 0.8786848 |
| C3 | C | B0 | 0.0125 | 0.00025 | 0.75 | 0 | 0 | True | 515 | 0.8933333 | 0.9595918 | 0.6665360 | 0.8786848 | 0.8786848 |
| D1 | D | B1 | 0.0125 | 0.00025 | 0.67 | 4 | 0.5 | True | 515 | 0.9174150 | 0.9618367 | 0.7545979 | 0.8810794 | 0.8887755 |
| D2 | D | B1 | 0.0125 | 0.00025 | 0.67 | 4 | 1 | True | 514 | 0.9446259 | 0.9683673 | 0.7502516 | 0.8809612 | 0.9079365 |
| D3 | D | B1 | 0.0125 | 0.00025 | 0.67 | 4 | 2 | True | 513 | 0.9331973 | 0.9689796 | 0.6755717 | 0.8764382 | 0.9069161 |
| D4 | D | B1 | 0.0125 | 0.00025 | 0.67 | 8 | 0.5 | False | 515 | 0.8944218 | 0.9546939 | 0.6658839 | 0.8833711 | 0.8988662 |
| D5 | D | B1 | 0.0125 | 0.00025 | 0.67 | 8 | 1 | True | 519 | 0.9182766 | 0.9630612 | 0.7321648 | 0.9009443 | 0.8928571 |
| D6 | D | B1 | 0.0125 | 0.00025 | 0.67 | 8 | 2 | True | 517 | 0.8970068 | 0.9618367 | 0.7003609 | 0.8885623 | 0.8807256 |
| L1 | LOCAL | B1 | 0.01125 | 0.00025 | 0.67 | 8 | 1 | False | 514 | 0.9001361 | 0.9617347 | 0.6611601 | 0.8787045 | 0.8978458 |
| L2 | LOCAL | B1 | 0.01375 | 0.00025 | 0.67 | 8 | 1 | False | 513 | 0.8776644 | 0.9535714 | 0.6677162 | 0.8637984 | 0.8564626 |

## Pairwise comparisons

| Comparison | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tuned_b0_vs_historical_b0 | +2 | 10 | 8 | 18 | +0.0037383 | -0.0260317 | +0.0356528 | +0.0177706 | +0.0323129 | 0.81452942 |
| tuned_b1_vs_historical_b1 | +4 | 10 | 6 | 16 | +0.0074766 | +0.0388209 | +0.0591839 | +0.0222595 | +0.0141723 | 0.45449829 |
| tuned_b1_vs_tuned_b0 | +4 | 7 | 3 | 10 | +0.0074766 | +0.0219501 | +0.0586728 | +0.0222595 | +0.0141723 | 0.34375000 |

## TUNED_B1 mechanism diagnostics

- Private/shared ratio mean/max: 0.1103059 / 3.1193817.
- Category-Global cosine mean: 0.5516195.
- Private collapse: `False`; all folds trained: `True`.
- Parameter increase: 8200 (608997 -> 617197).
- Per-modality ratio mean: `{"COGNITIVE_TEST": 0.10547049483284354, "CSF": 0.04961572287138551, "MRI": 0.04874670824501663, "RISK_FACTOR": 0.30878231902606784, "ROI_AVERAGE": 0.03891415111720562}`.
- Per-modality ratio max: `{"COGNITIVE_TEST": 1.0771219730377197, "CSF": 0.6545774340629578, "MRI": 0.4259990155696869, "RISK_FACTOR": 3.1193816661834717, "ROI_AVERAGE": 0.3557499647140503}`.
- Adapter maximum gradients: `{"private_adapters.0.down.bias": 0.025855038315057755, "private_adapters.0.down.weight": 0.23617923259735107, "private_adapters.0.up.bias": 0.05760420486330986, "private_adapters.0.up.weight": 0.11315993219614029, "private_adapters.1.down.bias": 0.029101043939590454, "private_adapters.1.down.weight": 0.30760830640792847, "private_adapters.1.up.bias": 0.06156489998102188, "private_adapters.1.up.weight": 0.11424951255321503, "private_adapters.2.down.bias": 0.018333490937948227, "private_adapters.2.down.weight": 0.18277981877326965, "private_adapters.2.up.bias": 0.062253616750240326, "private_adapters.2.up.weight": 0.11538179218769073, "private_adapters.3.down.bias": 0.016633648425340652, "private_adapters.3.down.weight": 0.19696739315986633, "private_adapters.3.up.bias": 0.04214617982506752, "private_adapters.3.up.weight": 0.07029659301042557, "private_adapters.4.down.bias": 0.02582528069615364, "private_adapters.4.down.weight": 0.2736918330192566, "private_adapters.4.up.bias": 0.04567548632621765, "private_adapters.4.up.weight": 0.14022690057754517}`.

## DGFMC comparison

- DGFMC fold mean ACC target 97.57%: TUNED_B1=97.0161% -> `False`.
- DGFMC fold mean ROC-AUC target 90.49%: TUNED_B1=96.3061% -> `True`.
- The comparison uses ten-fold mean values, not pooled OOF AUC.

## Answers to the twelve required questions

1. Best base parameters: lr=0.0125, WD=0.00025, dropout=0.67.
2. TUNED_B0: 515/535, ACC=0.9626168, pooled ROC-AUC=0.8963265.
3. Best private configuration: rank=8, adapter LR multiplier=1.0.
4. TUNED_B1: 519/535, ACC=0.9700935, pooled ROC-AUC=0.9182766.
5. Reached 523/535: `False`.
6. TUNED_B1 fold mean ACC exceeds 97.57%: `False` (0.9701607).
7. TUNED_B1 fold mean ROC-AUC exceeds 90.49%: `True` (0.9630612).
8. TUNED_B1 minority metrics: TP=36, SEN=0.8000000, BACC=0.8928571, PR-AUC=0.7321648; registered safety=True.
9. Main gain source: `private adapter`.
10. TUNED_B1 truly exceeds TUNED_B0: `True`; paired repairs/damages=7/3.
11. Model-structure change still needed: `True`.
12. Final Decision: `TAD_BINARY_NEAR`; target status `TAD_BINARY_TARGET_NOT_REACHED`.

## Reproduction

Run from a clean local branch named `experiment/tad-binary-hparam-search-v1` at source commit `0d7a183a7106c673de2817f36c69d780dcd792d5`:

```text
python -u -B scripts/run_tad_binary_hparam_search_v1.py inspect
python -u -B scripts/run_tad_binary_hparam_search_v1.py smoke --device cuda:0
python -u -B scripts/run_tad_binary_hparam_search_v1.py search --device cuda:0
python -u -B scripts/run_tad_binary_hparam_search_v1.py formal --device cuda:0
```
