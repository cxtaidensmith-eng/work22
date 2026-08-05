# TabPFN-3 Bounded Inference and Calibration Tuning v1 — 10-Fold Final Report

## Executive conclusion

This bounded experiment completed exactly the two permitted adjustments: TabPFN internal member expansion and outer-train-only class-logit bias calibration. E4 improved the locked E2 baseline from 545 to 548 correct subjects and therefore passed the predeclared gate for E8. E8 was run, but fell to 544 correct subjects. The fixed ACC-first candidate rule consequently selected E4 for calibration.

Strict 5-fold cross-fitted calibration within each outer-train partition produced 547/598 correct subjects, ACC 0.914716, Macro-F1 0.901548, BACC 0.897776, probability Macro-AUC 0.976716, and Weighted-F1 0.914626. Calibration was one correct subject worse than the selected uncalibrated E4 and remained below both Original Query and SEPS-Q on ACC and BACC. It passed the AUC threshold but failed the predeclared ACC and BACC minimums.

**Teacher qualification: STOP.** It does not meet the minimum, strong, or ideal teacher standard.

> 在有限内部成员扩展和严格训练折内类别校准后，TabPFN-3仍未达到T-MEL软概率教师标准，因此停止原始TabPFN概率蒸馏方案，不再继续调参。

No E16/E32, alternate checkpoint, feature-subsampling search, AutoTabPFN, PHE, HPO extension, fine-tuning, embedding, probability fusion, rank distillation, or T-MEL implementation was run.

## Git lineage and isolation

- Repository: `https://github.com/cxtaidensmith-eng/work22`
- Base branch: `origin/experiment/tabpfn-full-feature-minimal-v1`
- Base commit: `77f3943f4c5deb9193123ad8d6f0c933da5ecc0d`
- Experiment branch: `experiment/tabpfn-bounded-tuning-v1`
- Isolated worktree: `D:\Work\WORK2 final\WORK2 final26.7,21\tmp\tabpfn_bounded_tuning_v1`
- Initial experiment-worktree status: clean
- Original dirty worktree: not used for code edits or experiment execution

The base commit was verified to contain the complete 360-feature E2 run, 598-subject OOF predictions, feature-coverage audit, report, and independent recomputation. Those E2 artifacts were reused read-only and were not overwritten.

## Locked environment, checkpoint, and data

- Conda environment: `D:\Anaconda\envs\work22-tabpfn-v1`
- Python 3.11.15
- `tabpfn==8.2.0`
- `torch==2.5.1+cu121`; CUDA build 12.1; `cuda:0`
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU; driver 546.30; 4096 MiB
- Checkpoint: `tabpfn-v3-classifier-v3_default.ckpt`, stored outside the repository
- Checkpoint SHA256: `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`
- Checkpoint revision: `24a16a89d245878b846555110985634aa2e656d7`
- Dataset/task: TADPOLE, AD/CN/SMCI, 598 subjects, 360 features
- Class order/counts: `0=AD, 1=CN, 2=SMCI`; 72/209/317
- Folds/seed: fold0..9, seed 0
- CSV SHA256: `2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042`
- Modal dictionary SHA256: `5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273`
- Fold-manifest file SHA256: `0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106`
- Test-fold assignment SHA256: `8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9`

The 360-column order, preprocessing, fold assignment, class mapping, checkpoint, package versions, fit mode, memory mode, inference precision, and metric definitions remained locked. The complete environment freeze is stored in each arm's `environment.json`.

## Bounded model protocol and coverage

E4 and E8 used the same locked constructor except for the requested `n_estimators`:

```text
model_path=<locked checkpoint path>
n_estimators=4 or 8
auto_scale_n_estimators=True
random_state=0
device="cuda:0"
fit_mode="low_memory"
memory_saving_mode=True
inference_precision="auto"
show_progress_bar=False
```

Every outer fold used a fresh `TabPFNClassifier`. E4 had requested/effective counts 4/4 and E8 had 8/8. In each fold there was one classifier and one hash-locked checkpoint; the 4 or 8 components were TabPFN internal feature-subspace inference members, not separately trained external models. Every member used checkpoint model index 0.

For both arms, every fold's internal-member feature union was exactly all 360 input features, with zero uncovered features. MRI, PET, CSF, Risk, Cognitive, and ROI union coverage were each 100%. Each member used 200 features. The per-fold coverage SHA256 was `1d5b61ce7611a2b0eaff4399bf84546e1bd58182a264e227678643dac9a72809` for E4 and `685c9ae55cf79813ae3a54365b8155afc0b85844fdb3b545bb052c70bd66b455` for E8.

Both isolated fold0 smoke tests passed before formal evaluation:

| Arm | Requested/effective | ACC | Probability Macro-AUC | Union | Uncovered |
|---|---:|---:|---:|---:|---:|
| E4 smoke | 4/4 | 0.916667 | 0.980945 | 360 | 0 |
| E8 smoke | 8/8 | 0.916667 | 0.983947 | 360 | 0 |

Smoke outputs were kept separate from formal summaries.

## Uncalibrated formal results and E8 gate

| Candidate | Requested/effective | Correct | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 locked baseline | 1/2 | 545 | 0.911371 | 0.896275 | 0.886343 | 0.976829 | 0.911108 |
| E4 | 4/4 | **548** | **0.916388** | **0.905896** | **0.901319** | 0.976419 | **0.916307** |
| E8 | 8/8 | 544 | 0.909699 | 0.897551 | 0.893534 | **0.978498** | 0.909612 |

E4 passed gate condition 1 because its correct count was 548, above the required 546. The immutable gate artifact therefore recorded `RUN_E8`; condition 2 was not needed. E8 was run only for that reason. E16 and E32 were not run.

E4 versus E2 had 542 both correct, 6 E4-only correct, 3 E2-only correct, 47 both wrong, 9 discordant, and exact McNemar p=0.5078125. E8 versus E4 had 541 both correct, 3 E8-only correct, 7 E4-only correct, 47 both wrong, 10 discordant, and p=0.34375.

### Fold mean ± sample SD and pooled OOF

| Arm | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| E2 mean ± SD | 0.911328 ± 0.029749 | 0.894854 ± 0.043886 | 0.887665 ± 0.056569 | 0.977568 ± 0.012212 | 0.910226 ± 0.030332 |
| E2 pooled | 0.911371 | 0.896275 | 0.886343 | 0.976829 | 0.911108 |
| E4 mean ± SD | 0.916328 ± 0.025274 | 0.905210 ± 0.039984 | 0.901785 ± 0.049172 | 0.976343 ± 0.012420 | 0.915845 ± 0.025591 |
| E4 pooled | 0.916388 | 0.905896 | 0.901319 | 0.976419 | 0.916307 |
| E8 mean ± SD | 0.909661 ± 0.027741 | 0.897093 ± 0.041488 | 0.893898 ± 0.049747 | 0.978787 ± 0.011862 | 0.909383 ± 0.028178 |
| E8 pooled | 0.909699 | 0.897551 | 0.893534 | 0.978498 | 0.909612 |

The complete fold-level tables, confusion matrices, per-class metrics, probabilities, predictions, labels, indices, runtimes, CUDA memory, member indices, and feature coverage are stored with each arm.

## Fixed candidate selection

The persisted lexicographic rule ranked candidates by pooled OOF ACC, then BACC, Macro-F1, probability Macro-AUC, and finally fewer requested estimators. It ranked E4 first, E2 second, and E8 third. E4 was therefore selected with requested/effective `n_estimators=4/4`.

This selection follows the project's historical exploratory model-selection convention. It is explicitly exploratory because all completed candidates were ranked on outer OOF metrics; the report therefore retains every candidate instead of presenting E4 as a confirmatory winner.

## Strict outer-train cross-fitted calibration

For each outer fold, only the outer-train samples and labels were used to choose the bias. A fixed `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` produced cross-fitted probabilities for the complete outer-train partition. Each inner fold used a fresh E4 classifier trained on inner-train and predicted only inner-validation. The outer-test probability was the already saved uncalibrated E4 probability; its label was not available to bias selection.

The fixed transformation was:

```text
p'_c = softmax(log(max(p_c, 1e-12)) + b_c)
b_SMCI = 0
b_AD, b_CN in {-1.0, -0.9, ..., 0.9, 1.0}
```

All 441 bias pairs were evaluated separately within every outer fold. The immutable order was ACC, BACC, Macro-F1, smallest Euclidean bias norm, then numeric `(b_AD, b_CN)` order. There was no expanded range, finer step, local search, or selection using outer-test metrics.

Independent recomputation verified all 50 inner folds, exact inner-validation union equal to each outer-train set, pairwise-disjoint inner validation, one cross-fitted probability per outer-train subject, and zero outer-test indices in any inner train or validation set. Each grid had exactly 441 rows; all ten selected biases recomputed exactly; no bias was reused across outer folds. Thus, no TabPFN in-sample probability or outer-test label contributed to calibration.

| Outer fold | b_AD | b_CN | b_SMCI |
|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.4 | 0.2 | 0.0 |
| 2 | -0.2 | 0.1 | 0.0 |
| 3 | -0.3 | 0.0 | 0.0 |
| 4 | 0.3 | 0.2 | 0.0 |
| 5 | 0.0 | -0.1 | 0.0 |
| 6 | 0.0 | 0.0 | 0.0 |
| 7 | 0.1 | -0.1 | 0.0 |
| 8 | 0.0 | 0.1 | 0.0 |
| 9 | -0.1 | 0.0 | 0.0 |

The first calibration invocation was interrupted by the command time limit after completing a deterministic prefix. A complete rerun was then performed without changing the script or protocol. For outer folds 0–8, the interrupted and complete runs had identical selected biases and identical cross-fitted-probability SHA256 values. The incomplete staging copy was removed after that comparison; only the complete formal rerun is included here.

## Calibrated 10-fold results

| Fold | Test n | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 0.916667 | 0.902670 | 0.879288 | 0.980945 | 0.915528 |
| 1 | 60 | 0.950000 | 0.928756 | 0.931548 | 0.972337 | 0.949797 |
| 2 | 60 | 0.966667 | 0.974096 | 0.979167 | 0.997256 | 0.966887 |
| 3 | 60 | 0.900000 | 0.902212 | 0.878472 | 0.975513 | 0.899095 |
| 4 | 60 | 0.883333 | 0.829431 | 0.799107 | 0.961131 | 0.880082 |
| 5 | 60 | 0.900000 | 0.877693 | 0.889385 | 0.981448 | 0.901057 |
| 6 | 60 | 0.916667 | 0.919577 | 0.937004 | 0.973197 | 0.916561 |
| 7 | 60 | 0.916667 | 0.934785 | 0.937004 | 0.967491 | 0.916974 |
| 8 | 59 | 0.864407 | 0.830005 | 0.830005 | 0.960113 | 0.864407 |
| 9 | 59 | 0.932203 | 0.911124 | 0.920161 | 0.993665 | 0.933482 |
| Fold mean | — | 0.914661 | 0.901035 | 0.898114 | 0.976310 | 0.914387 |
| Sample SD | — | 0.030203 | 0.045231 | 0.054218 | 0.012403 | 0.030684 |
| Pooled OOF | 598 | **0.914716** | **0.901548** | **0.897776** | **0.976716** | **0.914626** |

Correct count was 547/598. Probability Macro-AUC uses multiclass OVR macro averaging from the three saved probability columns in fixed AD/CN/SMCI order.

### Pooled confusion matrix and per-class metrics

Rows are true classes and columns are predicted classes in AD, CN, SMCI order.

| True / Predicted | AD | CN | SMCI |
|---|---:|---:|---:|
| AD | 61 | 0 | 11 |
| CN | 0 | 192 | 17 |
| SMCI | 9 | 14 | 294 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AD | 0.871429 | 0.847222 | 0.859155 | 72 |
| CN | 0.932039 | 0.918660 | 0.925301 | 209 |
| SMCI | 0.913043 | 0.927445 | 0.920188 | 317 |

Relative to selected E4, recall changed by -0.013889 for AD, +0.009569 for CN, and -0.006309 for SMCI. The fixed class-sacrifice definition required an ACC gain plus a recall loss greater than 0.02; it was not triggered because ACC did not improve and no recall loss exceeded 0.02.

## Unified comparison

| Model | Correct | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original Query | 556 | 0.929766 | 0.914078 | 0.914078 | 0.956049 | 0.929766 |
| SEPS-Q v1 | 553 | 0.924749 | 0.913866 | 0.908206 | 0.953290 | 0.924631 |
| E2 | 545 | 0.911371 | 0.896275 | 0.886343 | 0.976829 | 0.911108 |
| E4 selected, uncalibrated | 548 | 0.916388 | 0.905896 | 0.901319 | 0.976419 | 0.916307 |
| E8 | 544 | 0.909699 | 0.897551 | 0.893534 | 0.978498 | 0.909612 |
| E4 outer-train calibrated | 547 | 0.914716 | 0.901548 | 0.897776 | 0.976716 | 0.914626 |

| Calibrated delta vs | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| E2 | +0.003344 | +0.005273 | +0.011433 | -0.000112 | +0.003518 |
| Selected E4 | -0.001672 | -0.004348 | -0.003543 | +0.000297 | -0.001681 |
| Original Query | -0.015050 | -0.012530 | -0.016302 | +0.020667 | -0.015139 |
| SEPS-Q v1 | -0.010033 | -0.012318 | -0.010431 | +0.023426 | -0.010005 |

E4 was the best bounded uncalibrated configuration. E8 increased probability AUC but reduced correct classifications by four relative to E4. Strict calibration improved E4's CN count but reduced the overall correct count by one.

## Subject-level paired analysis

All comparisons aligned exactly the same 598 unique subject indices, fold assignments, and truths.

| Comparison | Both correct | Calibrated/tuned only | Other only | Both wrong | Discordant | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| E4 vs E2 | 542 | 6 | 3 | 47 | 9 | 0.507812 |
| E8 vs E4 | 541 | 3 | 7 | 47 | 10 | 0.343750 |
| Calibrated vs selected E4 | 545 | 2 | 3 | 48 | 5 | 1.000000 |
| Calibrated vs E2 | 540 | 7 | 5 | 46 | 12 | 0.774414 |
| Calibrated vs Original Query | 525 | 22 | 31 | 20 | 53 | 0.271679 |
| Calibrated vs SEPS-Q v1 | 527 | 20 | 26 | 25 | 46 | 0.461391 |

| Calibrated comparison by true class | AD exclusive (cal/other) | CN exclusive (cal/other) | SMCI exclusive (cal/other) |
|---|---:|---:|---:|
| vs selected E4 | 0/1 | 2/0 | 0/2 |
| vs E2 | 2/0 | 4/0 | 1/5 |
| vs Original Query | 5/6 | 5/11 | 12/14 |
| vs SEPS-Q v1 | 3/4 | 7/8 | 10/14 |

Complete corrected/broken subject rows and class-stratified paired counts are stored in `paired_rows_*.csv` and `paired_summary_*.json`.

## Runtime and GPU memory

| Stage | Wall time | Summed fit time | Summed probability time | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|
| E4 formal | 29.534 s | 7.527 s | 16.717 s | 778,541,056 bytes |
| E8 formal | 46.049 s | 8.300 s | 31.441 s | 778,541,056 bytes |
| Calibration formal | 194.397 s | 37.738 s | 79.474 s | 712,462,848 bytes |

Calibration fit/predict sums cover the 50 inner classifiers; its wall time also includes loading, grid evaluation, validation, and artifact writing. No external ensemble, multi-seed run, or model fusion was used.

## Integrity and independent recomputation

- E4, E8, and calibrated OOF each contain 598 rows and 598 unique subjects.
- Each OOF equals the concatenation of its ten saved outer-fold prediction files.
- Saved predictions and probability argmax agree; probabilities are finite and each row sums to approximately one.
- Per-fold and pooled metrics, confusion matrices, and class reports recompute from disk.
- E4/E8 gate and exploratory candidate selection recompute exactly.
- All 50 inner splits, cross-fitted probabilities, 441-row grids, selected biases, and outer-test transforms independently validate.
- Outer-test leakage count is zero in every fold; outer-test labels were not used for bias selection.
- All ten folds used distinct bias selection and complete validation; no fold ACC fell below the fixed anomalous-collapse boundary of 0.80.
- Independent validation artifact: `experiments/tabpfn_bounded_tuning_v1/recompute_validation.json`; overall `passed=true`.
- E2 OOF SHA256: `03732120ebebde721fd06c6336b751a8726b02fc93562c228bd64c3ab0086336`
- E4 OOF SHA256: `d02f93e6ee3eab0eebf7aa5cb24308cf4d48501024faa5b2ce5ddb23a4a19f82`
- E8 OOF SHA256: `955853a9ef4df6bf3a61aa44f5abe60512ddd1289168fb4375b2e3f275aaf758`
- Calibrated OOF SHA256: `f49f6b2eb1ca1d6a2a4b29d1018f4de22cdddfa80c027461eb1181734f297087`
- Original Query OOF SHA256: `86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565`
- SEPS-Q v1 OOF SHA256: `1286231fd513d811e4828319ca1210b6c76635c029107f497356e7e577a94e68`
- Runner SHA256: `a1185cf14972198db17a4d72873a01975b482ba66d6cd2596f2a6470aa218314`
- Calibrator SHA256: `f67375cfe23eaa970d820e3d213a50ff72ed19b792ce304c98a4e7857c23e4ad`
- Recompute script SHA256: `e31cf1f6aae36f41007edff88798c75fe7eb0f7a52917a11c28f584772d5de98`

The checkpoint, model cache, API key, token, cookie, authentication files, Conda environment, and virtual environment remain outside Git and are not included in these artifacts.

## Teacher qualification and STOP decision

| Criterion | Required | Observed | Pass |
|---|---:|---:|---:|
| Minimum ACC | ≥ 0.9200 | 0.914716 | No |
| Minimum BACC | ≥ 0.9000 | 0.897776 | No |
| Minimum probability Macro-AUC | ≥ 0.9700 | 0.976716 | Yes |
| No anomalous fold | all complete and ACC ≥ 0.80 | minimum 0.864407 | Yes |
| Meaningful nonzero exclusive correct | at least one vs Original or SEPS | 22 / 20 | Yes |
| Strong ACC | ≥ 0.9247492 | 0.914716 | No |
| Strong BACC | ≥ 0.9082064 | 0.897776 | No |
| Ideal ACC near Original | near 0.9297659 | 0.914716 | No |

The fixed STOP reasons are: calibrated ACC below 0.9200, calibrated BACC below 0.9000, and paired correct-only counts favoring both Original Query and SEPS-Q by at least five under the predeclared operational rule. Probability AUC remained strong, but it did not translate into the required label performance. The minimum, strong, and ideal teacher thresholds are all false.

**Final decision: STOP.** No further TabPFN tuning or downstream teacher/distillation experiment is authorized by this result.

## Artifact map

- E4/E8 runner: `scripts/run_tabpfn_bounded_tuning_v1.py`
- Strict calibrator: `scripts/calibrate_tabpfn_bounded_tuning_v1.py`
- Independent recomputation: `scripts/recompute_tabpfn_bounded_tuning_v1.py`
- E4 smoke/formal artifacts: `experiments/tabpfn_bounded_tuning_v1/e4/`
- E8 smoke/formal artifacts: `experiments/tabpfn_bounded_tuning_v1/e8/`
- Automatic E8 gate: `experiments/tabpfn_bounded_tuning_v1/e4/formal/e8_gate_decision.json`
- Exploratory candidate selection: `experiments/tabpfn_bounded_tuning_v1/calibration/candidate_selection.json`
- Calibration manifest and aggregate: `experiments/tabpfn_bounded_tuning_v1/calibration/manifest.json`, `aggregate_summary.json`
- Per-fold inner splits, cross-fitted probabilities, 441-row grids, biases, outer-test raw/calibrated predictions, corrected/broken rows, and validation: `experiments/tabpfn_bounded_tuning_v1/calibration/fold_00/` through `fold_09/`
- Calibrated pooled OOF and metrics: `experiments/tabpfn_bounded_tuning_v1/calibration/oof_predictions.csv`, `oof_metrics.json`
- Selected biases: `experiments/tabpfn_bounded_tuning_v1/calibration/selected_biases.csv`
- Teacher qualification: `experiments/tabpfn_bounded_tuning_v1/calibration/teacher_qualification.json`
- Independent validation and artifact hashes: `experiments/tabpfn_bounded_tuning_v1/recompute_validation.json`
