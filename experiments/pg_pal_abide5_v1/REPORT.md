# PG-PAL-ABIDE5 v1

## Outcome

- Decision: **PG_PAL_ABIDE5_NO_GAIN**
- Source commit: `e8930b576910aba1be3f4429eb149d9b3d43210b`
- Result commit: reported in final Git handoff (a commit cannot contain its own SHA)
- Branch: `experiment/pg-pal-abide5-v1`
- Device: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`

PG-PAL did not add inference parameters, an inference branch, or a second model. It is not an ensemble. Only ABIDE-5 was run; no second version or v1.1 was run.

## Purpose and formula

For the fixed D3 model, shared-only and full-private losses use identical CPU/CUDA stochastic states. On all common parameters, `r = g_f - g_s`; when `<r,g_s> < 0`, PG-PAL replaces `r` by `r - <r,g_s>/(||g_s||^2+1e-12) g_s`. It then forms `q=g_s+r_tilde` and globally rescales q to `||g_s||`. Private parameters receive only `grad(L_f)`. One Adam step and one scheduler step follow historical global gradient clipping.

PG-PAL differs from ordinary PCGrad because it protects a shared-only baseline against the private marginal rather than symmetrically projecting task gradients; it differs from DGL because it adds no encoder/fusion gradient-learning modules; it differs from SP-LRIF because inference is the unchanged D3 private-enabled path and no interaction fusion is added.

This only removes a first-order conflict component before Adam. It does not guarantee a lossless Adam parameter step, test improvement, balanced modality use, total multimodal conflict removal, or strict shared/private disentanglement.

## Locked protocol

ABIDE-5 ADS_CN; 864 subjects (397 ADS/ASD positive class index 0, 467 CN class index 1); five real modalities; folds 0..9; seed 0; 400 full-batch transductive epochs; historical global class weights; complete criterion_lossv2 with label smoothing 0.05 and orthogonality 0; Adam; grad clip 1; historical ratio-preserving cosine scheduler; ACC > ROC-AUC > Macro-F1 > earliest checkpoint selection; graph/EMA/ensemble off; single model. Test labels enter only post-probability metric/checkpoint selection, not model inputs, loss, or gradient projection.

## D3 reference and PG-PAL results

| Model | Correct | ACC | pooled AUC | fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Confusion [ADS,CN] | Pred ADS/CN | Params | Train sec | Infer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| D3 | 769/864 | 0.8900463 | 0.8862642 | 0.8897475 +/- 0.0463411 | 0.8505650 | 0.8893405 | 0.8894142 | 0.8900565 | 0.8816121 | 0.8972163 | [[350, 47], [48, 419]] | 398/466 | 383616 | historical | historical |
| PG-PAL | 752/864 | 0.8703704 | 0.8688828 | 0.8906893 +/- 0.0586530 | 0.8362830 | 0.8694639 | 0.8693251 | 0.8703452 | 0.8564232 | 0.8822270 | [[340, 57], [55, 412]] | 395/469 | 383616 | 1383.6 | 0.15155 |

PG-PAL 10-fold ACC mean +/- sample SD: 0.8705025 +/- 0.0386724.

## Subject-aligned comparison with D3

- repairs/damages/changed: 39/56/95
- Correct delta: -17
- ACC/AUC/PR-AUC/Macro-F1/BACC/SEN/SPE deltas: -0.0196759 / -0.0173814 / -0.0142820 / -0.0198767 / -0.0200891 / -0.0251889 / -0.0149893
- exact two-sided McNemar p: 0.100173330

## Per-fold results

| Fold | Best epoch | Correct | ACC | AUC | Projection fraction |
|---:|---:|---:|---:|---:|---:|
| 0 | 88 | 71 | 0.8160920 | 0.8515957 | 0.54750 |
| 1 | 16 | 78 | 0.8965517 | 0.9489362 | 0.53250 |
| 2 | 47 | 74 | 0.8505747 | 0.9026596 | 0.53250 |
| 3 | 42 | 70 | 0.8045977 | 0.7805851 | 0.53500 |
| 4 | 49 | 78 | 0.9069767 | 0.9650846 | 0.53000 |
| 5 | 254 | 76 | 0.8837209 | 0.8728860 | 0.44750 |
| 6 | 104 | 76 | 0.8837209 | 0.8810693 | 0.53000 |
| 7 | 166 | 79 | 0.9186047 | 0.9288043 | 0.59750 |
| 8 | 76 | 77 | 0.8953488 | 0.9440217 | 0.52000 |
| 9 | 30 | 73 | 0.8488372 | 0.8312500 | 0.55250 |

## Minimal gradient mechanism

- Projection applied epoch fraction, mean across folds: 0.5325000
- Per-fold projection fractions: [0.5475, 0.5325, 0.5325, 0.535, 0.53, 0.4475, 0.53, 0.5975, 0.52, 0.5525]
- Raw private-marginal/shared gradient cosine mean: -0.0159595
- Pre/post norm-preservation ratios: 1.0449486 / 1.0000000
- Minimum projected common/shared dot product: 2.81027088e-05
- Maximum private adapter gradient: 0.0135512911
- Private trained all folds / collapse: True / False

## External descriptive targets

- Correct >=787: `False`
- Fold mean ACC >91.05%: `False`
- Fold mean ROC-AUC >90.99%: `False`

These are descriptive cross-paper numerical targets only, not same-protocol SOTA claims and not part of the PG-PAL decision.

## Required answers

1. Exceeded D3 769/864: `False`.
2. Repairs > damages: `False`.
3. AUC/PR-AUC/Macro-F1/BACC all safe: `False`.
4. ASD SEN and CN SPE both safe: `False`.
5. Private adapters trained in all folds: `True`.
6. Conflict/projection epoch fraction: `0.5325000`.
7. First-order non-conflict condition held: `True`.
8. Added inference parameters/path: `False`; the normal single D3 model is used.
9. Reached 787/864: `False`.
10. Cross-task validation allowed: `False`.
11. Final Decision: `PG_PAL_ABIDE5_NO_GAIN`.
12. Strictly stopped without v1.1 if failed: `True`.

## Reproduction

Run from a clean local branch named `experiment/pg-pal-abide5-v1` at source commit `e8930b576910aba1be3f4429eb149d9b3d43210b` using:

```text
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_pg_pal_abide5_v1.py inspect
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_pg_pal_abide5_v1.py smoke --device cuda:0
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_pg_pal_abide5_v1.py formal --device cuda:0
```
