# TabPFN-3 Full-Feature Minimal-Coverage v1 — 10-Fold Final Report

## Executive conclusion

The minimal correction succeeded technically and materially improved TabPFN-3, but it did not establish a sufficient teacher baseline for T-MEL.

Changing only `auto_scale_n_estimators=False` to `True` caused TabPFN 8.2.0 to scale the requested `n_estimators=1` to an effective `n_estimators_=2`. In every fold, the two internal feature-subspace inference members used 200 original features each, shared the same TabPFN-3 checkpoint and pretrained weights, overlapped on 40 features, and jointly covered exactly all 360 original features with zero omissions.

Compared with the prior 200-feature single-estimator run, pooled OOF ACC improved from 0.7642140 to 0.9113712 (+0.1471572), while probability Macro-AUC improved from 0.9120035 to 0.9768288 (+0.0648253). The paired comparison corrected 101 prior errors while breaking 13 prior correct predictions (exact McNemar p=4.774e-18). This shows that omission of 160 features was a major cause of the previous failure.

However, full-feature TabPFN remained below Original Query by 0.0183946 ACC and 0.0277348 BACC, and below SEPS-Q v1 by 0.0133779 ACC and 0.0218634 BACC. Its higher probability Macro-AUC did not translate into sufficient argmax classification performance. It therefore does not meet the GO threshold or the conditional-retention ACC level of approximately 0.92.

**Decision: STOP.**

在完整覆盖全部360维特征后，TabPFN-3仍不具备作为T-MEL教师的性能基础，因此停止TabPFN教师和蒸馏路线。

No T-MEL, distillation, fusion, alternate checkpoint, additional estimator, feature-subsampling strategy, extension, PHE, AutoTabPFN, fine-tuning, or hyperparameter-search experiment was run.

## Locked lineage and protocol

- Branch: `experiment/tabpfn-full-feature-minimal-v1`
- Base branch: `experiment/tabpfn-baseline-v1`
- Base commit: `57ecc323edf9d24d32d8be7c0eb1b612e4cca74a`
- Worktree: `D:\Work\WORK2 final\WORK2 final26.7,21\tmp\tabpfn_full_feature_minimal_v1`
- Dataset: TADPOLE AD/CN/SMCI, 598 source rows, 360 features, class counts 72/209/317
- Fold assignment: locked fold0..9 manifest, seed 0; each source row appears once in test and nine times in train
- CSV SHA256: `2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042`
- Modal dictionary SHA256: `5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273`
- Fold-manifest file SHA256: `0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106`
- Test-fold assignment SHA256: `8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9`
- Checkpoint: `tabpfn-v3-classifier-v3_default.ckpt`, outside the repository
- Checkpoint SHA256: `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988`
- Official revision: `24a16a89d245878b846555110985634aa2e656d7`
- Environment: Python 3.11.15, `tabpfn==8.2.0`, `torch==2.5.1+cu121`, CUDA build 12.1
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver 546.30, 4096 MiB

The inherited baseline runner SHA256 was `86e0666add6dbe99863d59496d94dec80c6098896d2f63ed480e5962593b3a3f`. The static protocol-delta audit passed and found exactly one constructor difference:

```diff
- auto_scale_n_estimators=False
+ auto_scale_n_estimators=True
```

The remaining locked constructor arguments were `model_path=<hash-locked absolute checkpoint>`, `n_estimators=1`, `random_state=0`, `device="cuda:0"`, `fit_mode="low_memory"`, `memory_saving_mode=True`, `inference_precision="auto"`, and `show_progress_bar=False`. No external standardization, imputation, one-hot encoding, PCA, feature selection, extensions, PHE, AutoTabPFN, or hyperparameter search was applied.

## Model semantics and feature-coverage proof

The precise model description is:

> One TabPFN-3 checkpoint and one TabPFNClassifier with two automatically scaled internal feature-subspace inference members, used solely to cover all 360 input features.

- External model count: 1
- Checkpoint count: 1
- Independently trained model count: 0
- Seed ensemble: false
- Probability fusion with Original Query or SEPS-Q: false
- Internal feature-subspace member count: 2
- Requested/effective estimators: 1/2
- Both members used checkpoint model index 0 and the same checkpoint SHA256
- Each member covered 200 original features; intersection 40; union 360; uncovered 0; repeated coverage 40
- Per-fold coverage SHA256: `a85a5b961e4d3b816bd82b264691cdf73dc202d2006629eea44ed08e2b5bafc2`
- Pooled feature-coverage SHA256: `dc469f67a425bb59ba14d481a8768c563f3690b59f3a0aaef73851a4405845f5`

| Coverage | MRI | PET | CSF | Risk | Cognitive | ROI | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Member 0 | 78 | 83 | 1 | 17 | 14 | 7 | 200 |
| Member 1 | 79 | 80 | 2 | 23 | 12 | 4 | 200 |
| Union | 138 | 150 | 3 | 36 | 24 | 9 | 360 |
| Union rate | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

All 10 formal folds independently passed `n_estimators_==2`, ensemble-config count 2, exact union `0..359`, zero uncovered features, and 100% union coverage for every modality. The two per-member index CSVs and the complete per-feature coverage CSV are saved for every fold.

## Smoke test

The separate fold0 smoke passed before the formal run. It used CUDA, the locked checkpoint, an effective two-member configuration, a 60×3 finite probability matrix in class order AD/CN/SMCI with row sums approximately one, and full 360-feature coverage. Its metrics were ACC 0.9000000 and probability Macro-AUC 0.9713279. Smoke artifacts are isolated under `experiments/tabpfn_full_feature_minimal_v1/smoke/` and are excluded from all formal summaries.

## Formal 10-fold results

| Fold | Test n | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 60 | 0.900000 | 0.842713 | 0.806708 | 0.971328 | 0.891919 |
| 1 | 60 | 0.950000 | 0.928756 | 0.931548 | 0.977924 | 0.949797 |
| 2 | 60 | 0.966667 | 0.973232 | 0.968254 | 0.997698 | 0.966338 |
| 3 | 60 | 0.900000 | 0.902212 | 0.878472 | 0.981745 | 0.899095 |
| 4 | 60 | 0.900000 | 0.854878 | 0.809524 | 0.963770 | 0.895544 |
| 5 | 60 | 0.900000 | 0.877693 | 0.889385 | 0.981820 | 0.901057 |
| 6 | 60 | 0.900000 | 0.905877 | 0.921131 | 0.979766 | 0.899539 |
| 7 | 60 | 0.900000 | 0.922287 | 0.926587 | 0.968200 | 0.900660 |
| 8 | 59 | 0.864407 | 0.829762 | 0.824885 | 0.959716 | 0.864831 |
| 9 | 59 | 0.932203 | 0.911124 | 0.920161 | 0.993714 | 0.933482 |
| Fold mean | — | 0.911328 | 0.894854 | 0.887665 | 0.977568 | 0.910226 |
| Sample SD | — | 0.029749 | 0.043886 | 0.056569 | 0.012212 | 0.030332 |
| Pooled OOF | 598 | **0.911371** | **0.896275** | **0.886343** | **0.976829** | **0.911108** |

Probability Macro-AUC is the same-scale `sklearn.metrics.roc_auc_score`, multiclass OVR, macro average, computed from the three saved probability columns in fixed class order AD/CN/SMCI.

### Pooled confusion matrix

Rows are true classes and columns are predicted classes in AD, CN, SMCI order.

| True / Predicted | AD | CN | SMCI |
|---|---:|---:|---:|
| AD | 59 | 0 | 13 |
| CN | 0 | 188 | 21 |
| SMCI | 8 | 11 | 298 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AD | 0.880597 | 0.819444 | 0.848921 | 72 |
| CN | 0.944724 | 0.899522 | 0.921569 | 209 |
| SMCI | 0.897590 | 0.940063 | 0.918336 | 317 |

The formal wall time was 22.877 s. Summed per-fold fit time was 7.685 s, summed `predict_proba` time was 10.211 s, and summed per-fold total time was 17.927 s. Peak allocated CUDA memory was 778,541,056 bytes (742.47 MiB).

## Four-model comparison

All metrics below were recomputed from hash-locked OOF probabilities and predictions using the common definition.

| Model | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Original Query | 0.929766 | 0.914078 | 0.914078 | 0.956049 | 0.929766 |
| SEPS-Q v1 | 0.924749 | 0.913866 | 0.908206 | 0.953290 | 0.924631 |
| TabPFN single-estimator v1 | 0.764214 | 0.766483 | 0.761022 | 0.912003 | 0.764274 |
| TabPFN full-feature minimal v1 | 0.911371 | 0.896275 | 0.886343 | 0.976829 | 0.911108 |

| Full-feature delta vs | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Single-estimator TabPFN | +0.147157 | +0.129793 | +0.125321 | +0.064825 | +0.146834 |
| Original Query | -0.018395 | -0.017803 | -0.027735 | +0.020780 | -0.018658 |
| SEPS-Q v1 | -0.013378 | -0.017591 | -0.021863 | +0.023539 | -0.013523 |

The result demonstrates that complete feature coverage substantially corrected the prior baseline, especially in discrimination as measured by probability AUC. It did not reach Original Query or SEPS-Q on the label-based metrics needed for the proposed teacher role.

## Paired error analysis

All three comparisons align the same 598 unique `subject_index` values, identical fold assignments, and identical truths.

| Comparison | Both correct | Full-feature only correct | Other only correct | Both wrong | Discordant | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| vs single-estimator TabPFN | 444 | 101 | 13 | 40 | 114 | 4.774e-18 |
| vs Original Query | 524 | 21 | 32 | 21 | 53 | 0.168978 |
| vs SEPS-Q v1 | 524 | 21 | 29 | 24 | 50 | 0.322236 |

Against the prior TabPFN run, the second internal member corrected 5 AD, 42 CN, and 54 SMCI cases, while breaking 1 AD, 8 CN, and 4 SMCI cases. This is a large, statistically decisive improvement and directly supports the feature-omission diagnosis.

Against Original Query, full-feature TabPFN uniquely corrected 4 AD, 3 CN, and 14 SMCI cases, but Original Query uniquely corrected 7 AD, 13 CN, and 12 SMCI cases. Against SEPS-Q, full-feature TabPFN uniquely corrected 3 AD, 6 CN, and 12 SMCI cases, while SEPS-Q uniquely corrected 6 AD, 11 CN, and 12 SMCI cases. These discordances demonstrate some complementarity, but the balance is insufficient to offset the lower ACC/BACC, and neither comparison justifies declaring a teacher baseline.

Complete subject-level paired rows, corrected and broken indices, class-stratified counts, and exact McNemar results are saved in the formal artifact directory.

## Integrity and reproducibility validation

- Formal OOF shape: 598 rows × 3 probability columns
- Unique source-row indices: 598; unique shuffled positions: 598
- OOF equals the concatenation of the ten saved fold prediction files
- Prediction equals probability argmax for every row
- Probabilities are finite; row-sum range 0.999999917 to 1.000000088
- OOF metrics and confusion matrix recompute exactly from disk
- Every per-fold metric recomputes exactly from disk
- Fresh classifier UUID count: 10
- All folds have effective `n_estimators_=2`, exact feature union `0..359`, and zero uncovered features
- All six modality union coverage rates are 100% in all folds
- Independent readback validation: passed
- Formal OOF SHA256: `03732120ebebde721fd06c6336b751a8726b02fc93562c228bd64c3ab0086336`
- Original Query OOF SHA256: `86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565`
- SEPS-Q v1 OOF SHA256: `1286231fd513d811e4828319ca1210b6c76635c029107f497356e7e577a94e68`
- Prior TabPFN OOF SHA256: `f6a667371a8d5c91939c78a533f31e7c950eadfe9bac51f2fa8de256af6e4f04`

The checkpoint remains outside the repository. No checkpoint, model cache, API key, token, cookie, credential file, Conda environment, virtual environment, or authentication material is included in the experiment artifacts.

## Artifact map

- Runner: `scripts/run_tabpfn_full_feature_minimal_v1.py`
- Independent recomputation: `scripts/recompute_tabpfn_full_feature_minimal_v1.py`
- Smoke: `experiments/tabpfn_full_feature_minimal_v1/smoke/`
- Formal manifest/environment/protocol audit: `experiments/tabpfn_full_feature_minimal_v1/formal/`
- Per-fold probabilities, predictions, labels, indices, metrics, model audits, member indices, and coverage: `experiments/tabpfn_full_feature_minimal_v1/formal/fold_00/` through `fold_09/`
- Pooled OOF and metrics: `experiments/tabpfn_full_feature_minimal_v1/formal/oof_predictions.csv`, `oof_metrics.json`
- Coverage summary: `feature_coverage_summary.json`, `pooled_feature_coverage.csv`
- Paired comparisons: `paired_summary_*.json`, `paired_rows_*.csv`
- Readback verification: `readback_validation.json`
