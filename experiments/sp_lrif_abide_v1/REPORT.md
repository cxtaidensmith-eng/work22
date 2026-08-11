# SP-LRIF-ABIDE v1

- Source commit: `6c1400078a5efc4a6e3f69c962c78f6b6f3be2dd`
- Parent result commit: `02e9caefa3ca89a3e5f13e892049c6b9e0c5ca9b`
- Result commit: reported in the final Git handoff (a commit cannot embed its own SHA)
- Branch: `experiment/sp-lrif-abide-v1`
- Device: `cuda:0`
- Decision: **SP_LRIF_ABIDE_NO_GAIN**

## Locked protocol

ABIDE ADS_CN; 871 subjects; ADS/ASD positive at index 0 and CN negative at index 1; four real modalities; seed 0; ten full-batch transductive folds; 400 epochs; lr 0.00625; weight decay 0.002; dropout 0.45; private rank 4; private LR multiplier 2; historical global class weights and criterion_lossv2; Adam; grad clip 1; unchanged CustomCosine schedule; ACC > ROC-AUC > Macro-F1 > earliest checkpoint; graph/EMA/ensemble off.

Only the final Category + Global add was extended to `C + G + delta` with the preregistered biasless rank-4 SP-LRIF formula. No search or second fusion was run.

## Results

| Model | Correct/N | ACC | pooled AUC | fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Params | Train sec | Infer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TUNED_B0 | 777/871 | 0.8920781 | 0.9009512 | 0.9135141 +/- 0.0290895 | 0.8666854 | 0.8913146 | 0.8907844 | 0.8919944 | 0.8734491 | 0.9081197 | 333033 | 475.2 | 0.1359 |
| D3 | 775/871 | 0.8897819 | 0.8828657 | 0.9117305 +/- 0.0295463 | 0.8405878 | 0.8889137 | 0.8881307 | 0.8896466 | 0.8660050 | 0.9102564 | 335353 | 517.1 | 0.1395 |
| SP-LRIF | 780/871 | 0.8955224 | 0.8881545 | 0.9156108 +/- 0.0426658 | 0.8406569 | 0.8948817 | 0.8946788 | 0.8954941 | 0.8833747 | 0.9059829 | 336377 | 538.9 | 0.1559 |

SP-LRIF confusion [ADS,CN]: `[[356, 47], [44, 424]]`; predicted ADS/CN=400/471.
Fold ACC mean +/- sample SD: 0.8955590 +/- 0.0292108.

## Per-fold best checkpoints

| Fold | Best epoch | Correct | ACC | AUC |
|---:|---:|---:|---:|---:|
| 0 | 207 | 76 | 0.8636364 | 0.8845355 |
| 1 | 121 | 79 | 0.9080460 | 0.9517497 |
| 2 | 223 | 80 | 0.9195402 | 0.9554613 |
| 3 | 281 | 74 | 0.8505747 | 0.8617021 |
| 4 | 278 | 79 | 0.9080460 | 0.9263298 |
| 5 | 203 | 81 | 0.9310345 | 0.9656915 |
| 6 | 388 | 75 | 0.8620690 | 0.8478723 |
| 7 | 94 | 78 | 0.8965517 | 0.9260638 |
| 8 | 312 | 81 | 0.9310345 | 0.9521277 |
| 9 | 135 | 77 | 0.8850575 | 0.8845745 |

## Subject-aligned comparisons

| Reference | Correct delta | Repairs | Damages | Changed | ACC delta | AUC delta | PR delta | F1 delta | BACC delta | SEN delta | SPE delta | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vs_TUNED_B0 | +3 | 33 | 30 | 63 | +0.0034443 | -0.0127967 | -0.0260285 | +0.0035671 | +0.0038944 | +0.0099256 | -0.0021368 | 0.80130649 |
| vs_D3 | +5 | 32 | 27 | 59 | +0.0057405 | +0.0052889 | +0.0000690 | +0.0059680 | +0.0065481 | +0.0173697 | -0.0042735 | 0.60292320 |

## Minimal mechanism diagnostics

- Initial logits max difference vs D3: 0.
- Category-Global cosine mean: 0.2065913.
- delta/(C+G) norm ratio mean/max: 0.0470393 / 1.5466464.
- Agreement/disagreement mean norm: 0.0290466 / 0.9253603.
- SP maximum gradients: `{"sp_lrif.proj_c.weight": 0.035543330013751984, "sp_lrif.proj_diff.weight": 0.07284703105688095, "sp_lrif.proj_g.weight": 0.0539802685379982, "sp_lrif.proj_out.weight": 0.07905123382806778}`.
- Every SP parameter changed in every fold: `True`; delta collapse: `False`.
- Same-checkpoint delta-disabled probability max/mean difference: 0.4127240 / 0.0009768; argmax changed=0.
- Added parameters: 1024 (335353 -> 336377); formal aggregation wall=443.356s.

## Answers to the twelve required questions

1. Exceeded D3 775/871: `True` (780/871).
2. Exceeded TUNED_B0 777/871: `True`.
3. Reached the 781 structural GO line: `False`.
4. Reached the 792 external target: `False`.
5. Gain comes from hard classification, not only AUC: `True`.
6. ASD SEN and CN SPE are both safe: `True`.
7. Repairs exceed damages vs B0: `True` (33/30).
8. Agreement and disagreement both activated: `True`.
9. Delta changed predictions: `False` (0).
10. Added parameters/inference overhead: 1024 parameters; SP-LRIF total infer=0.1559s versus D3 recorded 0.1395s.
11. Final Decision: `SP_LRIF_ABIDE_NO_GAIN`.
12. Worth extending to three other tasks: `False`; this run does not execute them.

## Reproduction

Run from a clean local branch named `experiment/sp-lrif-abide-v1` at source commit `6c1400078a5efc4a6e3f69c962c78f6b6f3be2dd`:

```text
python -u -B scripts/run_sp_lrif_abide_v1.py inspect
python -u -B scripts/run_sp_lrif_abide_v1.py smoke --device cuda:0
python -u -B scripts/run_sp_lrif_abide_v1.py formal --device cuda:0
```
