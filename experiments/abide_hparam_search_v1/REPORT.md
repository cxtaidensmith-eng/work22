# ABIDE Binary Task Hyperparameter Search v1

- Source commit: `c9c7a289145a87fbab355e85828fa5f62f9819b0`
- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)
- Branch: `experiment/abide-hparam-search-v1`
- Device: `cuda:0`
- Decision: **ABIDE_PRIVATE_NO_GAIN**
- Target status: **ABIDE_TARGET_NOT_REACHED**
- Formal aggregation time: 29.625 seconds

## Locked protocol

All trials use ABIDE ADS_CN (871 subjects: 403 ADS/ASD positive at class index 0 and 468 CN negative at class index 1), four real modalities, seed 0, ten full-batch transductive folds, 400 epochs, historical global full-dataset class weights, criterion_lossv2 with main weighted CE plus two unnormalized weighted OVR CE terms, label smoothing 0.05, orthogonality 0, Adam, grad clip 1, CustomCosineAnnealingLR(T_max=400, eta_min=0.0001), test-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch, graph off, EMA off, and no ensemble.
The calibration/nested-CV experiment is not reused because it changed loss, class-weight scope, and epoch selection.

## Selected configurations

- TUNED_B0: `C2`, lr=0.00625, WD=0.002, dropout=0.45.
- TUNED_B1: `D3`, lr=0.00625, WD=0.002, dropout=0.45, rank=4, adapter LR multiplier=2.0.
- Registered local LR refinement: `False` (Stage D Correct <= 787).

## Primary results

| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Fold PR mean +/- SD | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Params | Train sec | Infer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical B0 | 761/871 | 0.8737084 | 0.8737069 +/- 0.0291827 | 0.8829240 | 0.9021973 +/- 0.0384672 | 0.8401998 | 0.8773392 +/- 0.0523791 | 0.8732390 | 0.8740350 | 0.8738147 | 0.8784119 | 0.8696581 | 333033 | 1124.9 | 0.2043 |
| Historical B1 | 758/871 | 0.8702641 | 0.8702847 +/- 0.0432827 | 0.8750557 | 0.8969333 +/- 0.0443104 | 0.8289356 | 0.8876276 +/- 0.0418571 | 0.8699899 | 0.8716915 | 0.8704354 | 0.8908189 | 0.8525641 | 337417 | 1212.0 | 0.2060 |
| TUNED_B0 | 777/871 | 0.8920781 | 0.8920977 +/- 0.0291827 | 0.9009512 | 0.9135141 +/- 0.0290895 | 0.8666854 | 0.8945352 +/- 0.0462381 | 0.8913146 | 0.8907844 | 0.8919944 | 0.8734491 | 0.9081197 | 333033 | 475.2 | 0.1359 |
| TUNED_B1 | 775/871 | 0.8897819 | 0.8898250 +/- 0.0349931 | 0.8828657 | 0.9117305 +/- 0.0295463 | 0.8405878 | 0.8858059 +/- 0.0530399 | 0.8889137 | 0.8881307 | 0.8896466 | 0.8660050 | 0.9102564 | 335353 | 517.1 | 0.1395 |

### Confusion and prediction counts

- Historical B0: confusion=[[354, 49], [61, 407]] in [ADS,CN] order; TP/FN/TN/FP=354/49/407/61; predicted ADS/CN=415/456.
- Historical B1: confusion=[[359, 44], [69, 399]] in [ADS,CN] order; TP/FN/TN/FP=359/44/399/69; predicted ADS/CN=428/443.
- TUNED_B0: confusion=[[352, 51], [43, 425]] in [ADS,CN] order; TP/FN/TN/FP=352/51/425/43; predicted ADS/CN=395/476.
- TUNED_B1: confusion=[[349, 54], [42, 426]] in [ADS,CN] order; TP/FN/TN/FP=349/54/426/42; predicted ADS/CN=391/480.

## Registered trials

| Trial | Stage | Arm | LR | WD | Dropout | Rank | Adapter mult | Static safe | Correct | ROC-AUC | Fold AUC mean | Macro-F1 | BACC |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| A1 | A | B0 | 0.00375 | 0.001 | 0.45 | 0 | 0 | True | 762 | 0.8840958 | 0.8907592 | 0.8738448 | 0.8730356 |
| A2 | A | B0 | 0.005 | 0.001 | 0.45 | 0 | 0 | True | 761 | 0.8829240 | 0.9021973 | 0.8732390 | 0.8740350 |
| A3 | A | B0 | 0.00625 | 0.001 | 0.45 | 0 | 0 | True | 767 | 0.8884435 | 0.9013700 | 0.8797051 | 0.8790667 |
| B1 | B | B0 | 0.00625 | 0.0005 | 0.45 | 0 | 0 | True | 772 | 0.8879080 | 0.8952497 | 0.8855986 | 0.8852702 |
| B2 | B | B0 | 0.00625 | 0.001 | 0.45 | 0 | 0 | True | 767 | 0.8884435 | 0.9013700 | 0.8797051 | 0.8790667 |
| B3 | B | B0 | 0.00625 | 0.002 | 0.45 | 0 | 0 | True | 777 | 0.9009512 | 0.9135141 | 0.8913146 | 0.8907844 |
| C1 | C | B0 | 0.00625 | 0.002 | 0.35 | 0 | 0 | True | 774 | 0.8981172 | 0.9042999 | 0.8876379 | 0.8865454 |
| C2 | C | B0 | 0.00625 | 0.002 | 0.45 | 0 | 0 | True | 777 | 0.9009512 | 0.9135141 | 0.8913146 | 0.8907844 |
| C3 | C | B0 | 0.00625 | 0.002 | 0.55 | 0 | 0 | True | 776 | 0.8913994 | 0.9130785 | 0.8902209 | 0.8898883 |
| D1 | D | B1 | 0.00625 | 0.002 | 0.45 | 4 | 0.5 | True | 766 | 0.8787194 | 0.8973335 | 0.8785724 | 0.8779983 |
| D2 | D | B1 | 0.00625 | 0.002 | 0.45 | 4 | 1 | True | 769 | 0.8831228 | 0.8988998 | 0.8818201 | 0.8806865 |
| D3 | D | B1 | 0.00625 | 0.002 | 0.45 | 4 | 2 | True | 775 | 0.8828657 | 0.9117305 | 0.8889137 | 0.8881307 |
| D4 | D | B1 | 0.00625 | 0.002 | 0.45 | 8 | 0.5 | True | 772 | 0.8900288 | 0.9131895 | 0.8853211 | 0.8842363 |
| D5 | D | B1 | 0.00625 | 0.002 | 0.45 | 8 | 1 | True | 768 | 0.8862378 | 0.9038110 | 0.8811813 | 0.8815136 |
| D6 | D | B1 | 0.00625 | 0.002 | 0.45 | 8 | 2 | True | 771 | 0.8888279 | 0.9134349 | 0.8848822 | 0.8862696 |

## Per-fold selected epochs

### TUNED_B0

| Fold | Best epoch | Correct | ACC | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0 | 207 | 77 | 0.8750000 | 0.8902439 |
| 1 | 102 | 81 | 0.9310345 | 0.9363733 |
| 2 | 102 | 80 | 0.9195402 | 0.9125133 |
| 3 | 105 | 74 | 0.8505747 | 0.8627660 |
| 4 | 202 | 79 | 0.9080460 | 0.9343085 |
| 5 | 253 | 81 | 0.9310345 | 0.9680851 |
| 6 | 68 | 76 | 0.8735632 | 0.8920213 |
| 7 | 107 | 75 | 0.8620690 | 0.9159574 |
| 8 | 135 | 78 | 0.8965517 | 0.9180851 |
| 9 | 228 | 76 | 0.8735632 | 0.9047872 |

### TUNED_B1

| Fold | Best epoch | Correct | ACC | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0 | 242 | 75 | 0.8522727 | 0.8764920 |
| 1 | 126 | 82 | 0.9425287 | 0.9490986 |
| 2 | 205 | 78 | 0.8965517 | 0.9294804 |
| 3 | 263 | 74 | 0.8505747 | 0.8739362 |
| 4 | 127 | 81 | 0.9310345 | 0.9382979 |
| 5 | 261 | 81 | 0.9310345 | 0.9494681 |
| 6 | 123 | 75 | 0.8620690 | 0.8808511 |
| 7 | 103 | 75 | 0.8620690 | 0.9154255 |
| 8 | 170 | 78 | 0.8965517 | 0.9119681 |
| 9 | 286 | 76 | 0.8735632 | 0.8922872 |

## Paired comparisons

| Comparison | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tuned_b0_vs_historical_b0 | +16 | 43 | 27 | 70 | +0.0183697 | +0.0180272 | +0.0264856 | +0.0180756 | +0.0167494 | 0.07223793 |
| tuned_b1_vs_historical_b1 | +17 | 51 | 34 | 85 | +0.0195178 | +0.0078100 | +0.0116522 | +0.0189238 | +0.0164392 | 0.08205430 |
| tuned_b1_vs_tuned_b0 | -2 | 31 | 33 | 64 | -0.0022962 | -0.0180855 | -0.0260976 | -0.0024009 | -0.0026537 | 0.90065325 |

## TUNED_B1 mechanism diagnostics

- Private/shared ratio mean/max: 0.0646797 / 0.7462565.
- Per-modality ratio mean: `{"ANAT": 0.07679566033184529, "Correlation": 0.05453234203159809, "FUNC": 0.06890136804431676, "PHENO": 0.05848925393074751}`.
- Per-modality ratio max: `{"ANAT": 0.7462564706802368, "Correlation": 0.2675062119960785, "FUNC": 0.44147124886512756, "PHENO": 0.3852912485599518}`.
- Adapter maximum gradients: `{"private_adapters.0.down.bias": 0.0032279230654239655, "private_adapters.0.down.weight": 0.0062576644122600555, "private_adapters.0.up.bias": 0.13015328347682953, "private_adapters.0.up.weight": 0.004905348177999258, "private_adapters.1.down.bias": 0.0048756166361272335, "private_adapters.1.down.weight": 0.007676133420318365, "private_adapters.1.up.bias": 0.3011397123336792, "private_adapters.1.up.weight": 0.006230068393051624, "private_adapters.2.down.bias": 0.0044170827604830265, "private_adapters.2.down.weight": 0.015119727700948715, "private_adapters.2.up.bias": 0.2076619416475296, "private_adapters.2.up.weight": 0.018914341926574707, "private_adapters.3.down.bias": 0.0070923068560659885, "private_adapters.3.down.weight": 0.01677273027598858, "private_adapters.3.up.bias": 0.14187751710414886, "private_adapters.3.up.weight": 0.01125749945640564}`.
- Private collapse: `False`; all folds trained: `True`.
- Category-Global cosine mean: 0.1020081.
- Parameter increase: 2320 (333033 -> 335353).

## DGFMC comparison

- DGFMC fold mean ACC 90.82%: TUNED_B1=88.9825% -> `False`.
- DGFMC fold mean ROC-AUC 90.84%: TUNED_B1=91.1731% -> `True`.
- The comparison uses ten-fold mean +/- sample SD, not pooled OOF ACC/AUC.

## Answers to the fourteen required questions

1. Best base LR/WD/dropout: 0.00625 / 0.002 / 0.45.
2. TUNED_B0: 777/871, ACC=0.8920781, AUC=0.9009512.
3. Best private rank/multiplier: 4 / 2.0.
4. TUNED_B1: 775/871, ACC=0.8897819, AUC=0.8828657.
5. TUNED_B1 exceeds TUNED_B0: `False`; repairs/damages=31/33.
6. Historical private negative gain reversed: `False`.
7. Reached 792/871: `False`.
8. Fold mean ACC exceeds 90.82%: `False` (0.8898250).
9. Fold mean AUC exceeds 90.84%: `True` (0.9117305).
10. SEN and SPE both improve from TUNED_B0: `False`.
11. Main source of Correct gain: `base tuning or neither`.
12. Continue a separate ABIDE-5 tuning run: `True`.
13. Evidence is sufficient to modify the unified architecture now: `False`; finish ABIDE-5 first.
14. Final Decision: `ABIDE_PRIVATE_NO_GAIN`.

## Reproduction

Run from a clean local branch named `experiment/abide-hparam-search-v1` at source commit `c9c7a289145a87fbab355e85828fa5f62f9819b0`:

```text
python -u -B scripts/run_abide_hparam_search_v1.py inspect
python -u -B scripts/run_abide_hparam_search_v1.py smoke --device cuda:0
python -u -B scripts/run_abide_hparam_search_v1.py search --device cuda:0
python -u -B scripts/run_abide_hparam_search_v1.py formal --device cuda:0
```
