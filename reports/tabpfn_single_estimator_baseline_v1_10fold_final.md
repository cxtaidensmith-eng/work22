# TabPFN Single-Estimator Baseline v1 — 10-Fold Final Report

## 1. Experiment status

Status: **COMPLETE / VALIDATED / STOP**.

The locked TabPFN-3 single-estimator baseline was run on all ten fixed TADPOLE folds after a separate fold0 CUDA smoke test. All fold outputs and the pooled out-of-fold (OOF) output were read back from disk and exactly recomputed. No hyperparameter search, model fusion, extensions, PHE, AutoTabPFN, feature engineering, or follow-on T-MEL experiment was performed.

The final research decision is **STOP for promotion to a T-MEL teacher** under this protocol.

## 2. Git lineage

- Repository: `cxtaidensmith-eng/work22`
- Branch: `experiment/tabpfn-baseline-v1`
- Base branch: `origin/experiment/query-pool-component-sharing-v1`
- Base commit: `765720c1c0a3b2263502e9bf662fe33de7e45428`
- Isolated worktree: `D:\Work\WORK2 final\WORK2 final26.7,21\tmp\tabpfn_baseline_v1`
- The original dirty worktree was not used for model execution and was not modified.
- The formal run began from base HEAD `765720c1c0a3b2263502e9bf662fe33de7e45428`; the runner, artifacts, and this report were committed only after validation.

## 3. Locked data and folds

| Item | Locked value |
|---|---|
| Dataset/task | TADPOLE / AD_CN_SMCI |
| Input matrix | 598 rows × 360 features |
| Label vector | 598 |
| Class order | `0=AD`, `1=CN`, `2=SMCI` |
| Class counts | AD 72, CN 209, SMCI 317 |
| CSV SHA256 | `2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042` |
| Modal dictionary SHA256 | `5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273` |
| Fold-manifest file SHA256 | `0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106` |
| Fold-manifest canonical SHA256 | `be020ede2a03d59dbd5e8bbf715cb4d097f8398f9824c122521c3240e00b6515` |
| Shuffled sample-order SHA256 | `7ef70c38282cd3c3fa2fa95f41d5d7dfd3ee456d7ac58a25f80e2d7c378df668` |
| Shuffled label-order SHA256 | `40aa3114eeb832ce1b80a8706b47e50f99da7a48030d40560ef56164e109662a` |
| Test-fold assignment SHA256 | `8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9` |

Static data checks found no NaN or infinity, no duplicate feature rows, no duplicate feature names, and exact modal-dictionary coverage of all 360 columns. Three features are globally constant and were retained unchanged: indices 299, 318, and 326. Fold7 training also has feature 305 constant; it was likewise retained.

No RID column exists in this preprocessed CSV. Therefore, the committed protocol proves 598 unique source CSV rows (`subject_index=0..597`) and one OOF prediction per source row, but it does not independently verify person-level/RID uniqueness. This limitation is inherited from the locked historical manifest.

### Fold integrity

| Fold | Train/test | Train AD/CN/SMCI | Test AD/CN/SMCI | Split SHA256 |
|---:|---:|---:|---:|---|
| 0 | 538/60 | 64/188/286 | 8/21/31 | `1adda8298733c24259acfa957378b356f52db9545ae9c40ef8889ee97bf3ff04` |
| 1 | 538/60 | 65/188/285 | 7/21/32 | `529f52f8102c68c35ce93fe397fd1e8be83edc6ddf1ae9c2981b099e5e4b638e` |
| 2 | 538/60 | 65/188/285 | 7/21/32 | `45901ce4e5beb6a5c30ca6e7dd63f69f3bef8cf9cd26bd9ad13c805a834d98a1` |
| 3 | 538/60 | 65/188/285 | 7/21/32 | `af1cd9f0c1c662e18e4827e01525e573ca7755e762858c9423f36c467c9ad727` |
| 4 | 538/60 | 65/188/285 | 7/21/32 | `f927bc45ad33595185b6ace11d1b1deb0fbbb43f2fe4239056b3b4c1097ef7a0` |
| 5 | 538/60 | 65/188/285 | 7/21/32 | `7b18387f37130eb297c904390f0a762788d3ec34d59820dff40e5126e2cddf49` |
| 6 | 538/60 | 65/188/285 | 7/21/32 | `7edb8fbd66273445117ecbf87b01106f796bcf571452e0335d2fef069155d4a4` |
| 7 | 538/60 | 65/188/285 | 7/21/32 | `cc05b052a3c49b11ffc913a2d2a548dd400265e1cb8088c37e5378eee7767aa6` |
| 8 | 539/59 | 65/188/286 | 7/21/31 | `b50979aea7b950e281ec0e4147d698a508f32c0f1314b39a2132117755234795` |
| 9 | 539/59 | 64/189/286 | 8/20/31 | `f1395a882a4f6c518e17eeac3d37500a94d855534d5b8966eaa8421c3162ff12` |

The ten test sets are pairwise disjoint, their union is exactly all 598 source rows, and every source row is test once and train nine times.

## 4. Environment and checkpoint

| Item | Value |
|---|---|
| Conda environment | `D:\Anaconda\envs\work22-tabpfn-v1` |
| Python | 3.11.15 |
| `tabpfn` | 8.2.0 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA build | 12.1 |
| cuDNN | 9.1.0 |
| NVIDIA driver | 546.30 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB |
| `torch.cuda.is_available()` | `True` |
| Checkpoint | `tabpfn-v3-classifier-v3_default.ckpt` |
| Checkpoint size | 212,804,803 bytes |
| Checkpoint SHA256 | `d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988` |
| Official source | `Prior-Labs/tabpfn_3`, revision `24a16a89d245878b846555110985634aa2e656d7` |
| License | Accepted by the user and verified before execution |

The checkpoint, token, authentication cache, and Conda environment all remained outside the repository. No credential value was written to a result, report, or log. The complete frozen package list is in both smoke and formal `environment.json` files.

## 5. Single-estimator proof

Every fold created a fresh `TabPFNClassifier` inside the fold loop with:

```text
model_path=<explicit absolute path to the SHA-locked checkpoint>
n_estimators=1
auto_scale_n_estimators=False
random_state=0
device="cuda:0"
fit_mode="low_memory"
memory_saving_mode=True
inference_precision="auto"
show_progress_bar=False
```

All ten post-fit audits independently confirmed:

- requested `n_estimators == 1`;
- effective `n_estimators_ == 1`;
- `auto_scale_n_estimators is False`;
- exactly one `ensemble_configs_` entry;
- exactly one loaded checkpoint model;
- effective devices exactly `("cuda:0",)`;
- class order exactly `[0, 1, 2]`;
- ten distinct fold-local classifier UUIDs.

The effective preprocessor was `squashing_scaler_default`. The official checkpoint caps one estimator at 200 original features. Consequently, each fold received the complete unchanged 360-column matrix, but the single estimator internally selected the same deterministic 200-column subset (index-list SHA256 `782f2347d24c7dd03a8ecd55273f41478a510172c28a64f368f743a54e74b7bb`). This is checkpoint-internal subsampling, not external feature selection. Automatic expansion to two estimators was explicitly disabled and did not occur. The single forward estimator therefore must not be described as covering all 360 columns.

## 6. Fixed protocol

- Seed 0; folds 0 through 9 from the locked committed manifest.
- No external standardization, imputation, PCA, feature selection, one-hot encoding, or feature reordering.
- No inner validation or hyperparameter search.
- No extensions, PHE, AutoTabPFN, embedding extraction, test-time ensemble, or probability fusion.
- Fresh classifier for every fold; no cross-fold model or preprocessing state.
- The only model calls per fold were `fit(X_train, y_train)` and `predict_proba(X_test)`.
- Test labels were used only after prediction for evaluation and were never passed to the classifier.
- Predictions were derived as the fixed `[AD, CN, SMCI]` probability argmax; `predict()` was not called.
- Macro-AUC is multiclass OVR macro AUC computed from predicted probabilities.

TabPFN performs checkpoint-internal preprocessing as part of its inference architecture. No project-side preprocessing was added to the already processed CSV.

## 7. Smoke test

The separate fold0 smoke completed before formal execution and is excluded from every formal aggregate.

- ACC: 0.7666667
- probability OVR Macro-AUC: 0.9152739
- total fit plus prediction time: 4.05 s
- output probability shape: 60 × 3
- maximum probability row-sum error: below `1e-7`
- all values finite; fixed class order and `n_estimators_=1` verified
- saved CSV probabilities were read with round-trip parsing and all five metrics plus the confusion matrix were exactly reproduced

## 8. Formal ten-fold results

| Fold | ACC | Macro-F1 | BACC | Macro-AUC | Weighted-F1 | Total s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.766667 | 0.727917 | 0.700205 | 0.915274 | 0.762293 | 4.077 |
| 1 | 0.850000 | 0.849317 | 0.847222 | 0.931365 | 0.849731 | 1.293 |
| 2 | 0.750000 | 0.784188 | 0.800099 | 0.876895 | 0.746645 | 1.350 |
| 3 | 0.716667 | 0.763200 | 0.747520 | 0.882857 | 0.719838 | 1.414 |
| 4 | 0.683333 | 0.655860 | 0.615079 | 0.880798 | 0.679997 | 1.346 |
| 5 | 0.733333 | 0.735407 | 0.763393 | 0.910471 | 0.733199 | 1.382 |
| 6 | 0.800000 | 0.808442 | 0.805060 | 0.933557 | 0.798864 | 1.363 |
| 7 | 0.733333 | 0.748148 | 0.768849 | 0.911427 | 0.733333 | 1.302 |
| 8 | 0.745763 | 0.738535 | 0.744496 | 0.911147 | 0.745514 | 1.344 |
| 9 | 0.864407 | 0.839912 | 0.834409 | 0.979656 | 0.864741 | 1.240 |

### Fold mean ± sample standard deviation

| Metric | Mean ± sample SD | Min–max | Pooled OOF |
|---|---:|---:|---:|
| ACC | 0.764350 ± 0.057674 | 0.683333–0.864407 | 0.764214 |
| Macro-F1 | 0.765093 ± 0.057857 | 0.655860–0.849317 | 0.766483 |
| BACC | 0.762633 ± 0.068131 | 0.615079–0.847222 | 0.761022 |
| Macro-AUC | 0.913345 ± 0.030674 | 0.876895–0.979656 | 0.912003 |
| Weighted-F1 | 0.763415 ± 0.057978 | 0.679997–0.864741 | 0.764274 |

## 9. Pooled 598-row OOF result

| Metric | Value |
|---|---:|
| ACC | **0.7642140468** |
| Macro-F1 | **0.7664825798** |
| BACC | **0.7610217930** |
| Macro-AUC, probability OVR | **0.9120034763** |
| Weighted-F1 | **0.7642741002** |

Confusion matrix, rows=true and columns=predicted in `[AD, CN, SMCI]` order:

```text
[[ 55,   0,  17],
 [  0, 154,  55],
 [ 13,  56, 248]]
```

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AD | 0.808824 | 0.763889 | 0.785714 | 72 |
| CN | 0.733333 | 0.736842 | 0.735084 | 209 |
| SMCI | 0.775000 | 0.782334 | 0.778650 | 317 |

The OOF probability matrix is 598 × 3, contains no NaN/Inf, and has maximum absolute probability row-sum error `9.63e-8`.

## 10. Comparison with Original Query

The Original Query OOF is an external local, hash-pinned reference from the original dirty worktree; it was read only and was not copied, modified, staged, or claimed as branch-owned input. Its SHA256 is `86055aeb17db0620465350862469bf4342bcddd54fc81c5e1b1d6ba3fe738565`. Alignment by `subject_index` proved all 598 indices, truths, and fold assignments identical.

| Metric | TabPFN | Original Query | TabPFN − Original |
|---|---:|---:|---:|
| ACC | 0.7642140 | 0.9297659 | -0.1655518 |
| Macro-F1 | 0.7664826 | 0.9140778 | -0.1475952 |
| BACC | 0.7610218 | 0.9140778 | -0.1530560 |
| Macro-AUC, common probability OVR | 0.9120035 | 0.9560491 | -0.0440456 |
| Weighted-F1 | 0.7642741 | 0.9297659 | -0.1654918 |

The historical Original Query report's AUC is 0.9500702 because it used adjusted logits, not probabilities. That historical value is retained for provenance but is not treated as a same-score-scale probability comparison.

## 11. Comparison with SEPS-Q v1

The committed SEPS-Q OOF SHA256 is `1286231fd513d811e4828319ca1210b6c76635c029107f497356e7e577a94e68`. All 598 indices, truths, and fold assignments aligned exactly.

| Metric | TabPFN | SEPS-Q v1 | TabPFN − SEPS-Q |
|---|---:|---:|---:|
| ACC | 0.7642140 | 0.9247492 | -0.1605351 |
| Macro-F1 | 0.7664826 | 0.9138657 | -0.1473831 |
| BACC | 0.7610218 | 0.9082064 | -0.1471846 |
| Macro-AUC, common probability OVR | 0.9120035 | 0.9532899 | -0.0412865 |
| Weighted-F1 | 0.7642741 | 0.9246314 | -0.1603573 |

The historical SEPS-Q report's AUC is 0.9571291 because it used adjusted logits. Under the common probability definition it is 0.9532899. TabPFN is below SEPS-Q under both definitions, so the GO/STOP decision is unaffected by the historical AUC-score mismatch.

## 12. Error complementarity

### TabPFN versus Original Query

- both correct: 442
- TabPFN only correct: 15
- Original Query only correct: 114
- both wrong: 27
- discordant source rows: 129
- exact two-sided McNemar p-value: `5.0461e-20`
- by true class, TabPFN-only versus Original-only: AD 3/10, CN 4/48, SMCI 8/56
- Original errors corrected by TabPFN: source indices `4, 16, 33, 120, 203, 234, 270, 350, 378, 401, 433, 473, 492, 522, 538`
- 114 Original-correct rows broken by TabPFN are enumerated in `paired_summary_original_query.json` and `paired_rows_original_query.csv`.

### TabPFN versus SEPS-Q v1

- both correct: 441
- TabPFN only correct: 16
- SEPS-Q only correct: 112
- both wrong: 29
- discordant source rows: 128
- exact two-sided McNemar p-value: `6.3792e-19`
- by true class, TabPFN-only versus SEPS-Q-only: AD 3/10, CN 4/43, SMCI 9/59
- SEPS-Q errors corrected by TabPFN: source indices `47, 63, 68, 226, 234, 245, 259, 317, 357, 378, 393, 487, 522, 523, 593, 594`
- 112 SEPS-Q-correct rows broken by TabPFN are enumerated in `paired_summary_seps_q_v1.json` and `paired_rows_seps_q_v1.csv`.

Although a small complementary subset exists, both paired comparisons overwhelmingly favor the historical models; this is not meaningful complementarity in support of promotion to a teacher.

## 13. Runtime and memory

- formal wall time: 21.095 s
- sum of fold-local fit time: 8.113 s
- sum of `predict_proba` time: 7.982 s
- sum of fold-local fit plus prediction time: 16.111 s
- maximum recorded CUDA allocated memory: 778,541,056 bytes (742.47 MiB)

The first fold includes initial checkpoint/model warm-up overhead. No CUDA OOM occurred.

## 14. Integrity verification

All hard checks passed:

- exact data, modal dictionary, fold-manifest, assignment, label-order, sample-order, and ten split hashes;
- 598 OOF rows, 598 unique `subject_index` values, 598 unique shuffled positions, and every fold present;
- OOF exactly equals the sorted concatenation of the ten saved fold CSVs;
- probabilities finite and normalized within tolerance;
- every saved prediction equals fixed-class-order probability argmax;
- every fold and pooled metric plus confusion matrix exactly reproduced after CSV round-trip reading;
- ten unique classifier UUIDs and all single-estimator gates true;
- Original Query and SEPS-Q comparisons aligned one-to-one on `subject_index` with identical truth/fold values;
- result inventory contains no `.ckpt`, `.pt`, virtual environment, authentication token, cookie, or model cache.

The replay result is in `experiments/tabpfn_baseline_v1/formal/readback_validation.json` and includes SHA256 values for result-bearing artifacts (validation JSON files are deliberately excluded from their own hash inventory).

## 15. Limitations

1. The locked identity is a source CSV row index, not a verified RID.
2. This is intentionally a single-estimator test. The official checkpoint internally selected only 200 of the 360 supplied columns because automatic estimator expansion was disabled. It does not characterize multi-estimator TabPFN performance.
3. Internal TabPFN preprocessing is part of the official model interface; no alternative preprocessing policy was tested.
4. CUDA `inference_precision="auto"` resolved to FP16 autocast. Fixed seeds do not guarantee bitwise equality across different hardware/software stacks.
5. No tuning, extension, ensemble, PHE, feature selection, fusion, or alternate checkpoint was evaluated.
6. The Original Query OOF reference is local and untracked in the original dirty worktree, although its exact SHA and alignment were locked and validated. The branch is self-contained for TabPFN and SEPS-Q artifacts, but not for regenerating the Original Query predictions.
7. Historical Original/SEPS-Q AUCs used adjusted logits, whereas the mandated TabPFN AUC uses probabilities. This report separately provides common probability AUCs and does not silently mix the two definitions.

## 16. GO/STOP decision and research judgment

Decision: **STOP**.

- TabPFN ACC 0.7642140 is below Original Query 0.9297659.
- TabPFN probability Macro-AUC 0.9120035 is below both SEPS-Q's common probability AUC 0.9532899 and its historical adjusted-score AUC 0.9571291.
- BACC is lower by 0.1530560 versus Original Query and by 0.1471846 versus SEPS-Q.
- Paired discordance significantly favors Original Query (114 vs 15) and SEPS-Q (112 vs 16), not TabPFN.

**TabPFN可作为强基线，但当前没有证据支持将其作为T-MEL教师。**

No T-MEL, distillation, fusion, or additional ablation should be started automatically from this result.

## 17. Artifact index

### Code

- `scripts/run_tabpfn_baseline_v1.py`
- `scripts/recompute_tabpfn_baseline_v1.py`

### Smoke (excluded from formal results)

- `experiments/tabpfn_baseline_v1/smoke/manifest.json`
- `experiments/tabpfn_baseline_v1/smoke/environment.json`
- `experiments/tabpfn_baseline_v1/smoke/data_preflight.json`
- `experiments/tabpfn_baseline_v1/smoke/run.txt`
- `experiments/tabpfn_baseline_v1/smoke/smoke_summary.json`
- `experiments/tabpfn_baseline_v1/smoke/fold_00/{predictions.csv,metrics.json,model_audit.json,readback_validation.json}`

### Formal

- `experiments/tabpfn_baseline_v1/formal/manifest.json`
- `experiments/tabpfn_baseline_v1/formal/environment.json`
- `experiments/tabpfn_baseline_v1/formal/data_preflight.json`
- `experiments/tabpfn_baseline_v1/formal/run.txt`
- `experiments/tabpfn_baseline_v1/formal/fold_00` through `fold_09`, each containing `predictions.csv`, `metrics.json`, `model_audit.json`, and `readback_validation.json`
- `experiments/tabpfn_baseline_v1/formal/oof_predictions.csv`
- `experiments/tabpfn_baseline_v1/formal/oof_metrics.json`
- `experiments/tabpfn_baseline_v1/formal/oof_confusion_matrix.csv`
- `experiments/tabpfn_baseline_v1/formal/fold_metrics.csv`
- `experiments/tabpfn_baseline_v1/formal/metrics_summary.csv`
- `experiments/tabpfn_baseline_v1/formal/aggregate_summary.json`
- `experiments/tabpfn_baseline_v1/formal/paired_rows_original_query.csv`
- `experiments/tabpfn_baseline_v1/formal/paired_summary_original_query.json`
- `experiments/tabpfn_baseline_v1/formal/paired_rows_seps_q_v1.csv`
- `experiments/tabpfn_baseline_v1/formal/paired_summary_seps_q_v1.json`
- `experiments/tabpfn_baseline_v1/formal/readback_validation.json`

### Report

- `reports/tabpfn_single_estimator_baseline_v1_10fold_final.md`
