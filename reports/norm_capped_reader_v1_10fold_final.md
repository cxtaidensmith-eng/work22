# Norm-Capped Reader v1 - 10-Fold Final Report

Decision: **RCAP did not reach the target; stop the SEPS/Reader/Graph improvement route and retain Original Query.**

## Locked configuration

- Arm: RCAP (`low_rank_reader=True`, `class_graph=False`)
- Reader rank: 8; per-subject, per-class residual/shared norm cap: 0.5
- TADPOLE AD_CN_SMCI; folds 0..9; seed 0; 400 epochs; full-batch transductive; single model
- Original loss, Adam, CustomCosineAnnealingLR(T_max=400), and historical best-epoch ordering retained
- Execution source commit: `af2957a68cbcc926a84d34de020b83b42417ddda`

## Pooled OOF result

- Parameters: 785187
- Correct: 548 / 598
- ACC: 0.9163880
- Macro-F1: 0.9011053
- BACC: 0.8974227
- Probability Macro-AUC: 0.9482169
- Weighted-F1: 0.9161872
- Training time: 459.20 s
- Total active runtime: 463.38 s
- Confusion matrix (rows truth, columns prediction; AD/CN/SMCI):

```text
[60, 0, 12]
[0, 196, 13]
[9, 16, 292]
```

## Fold results

| Fold | Best epoch | ACC | Probability Macro-AUC |
|---:|---:|---:|---:|
| 0 | 159 | 0.933333 | 0.954937 |
| 1 | 52 | 0.866667 | 0.946923 |
| 2 | 68 | 0.933333 | 0.977141 |
| 3 | 165 | 0.950000 | 0.977219 |
| 4 | 60 | 0.916667 | 0.966765 |
| 5 | 142 | 0.950000 | 0.982412 |
| 6 | 97 | 0.950000 | 0.966804 |
| 7 | 70 | 0.866667 | 0.939728 |
| 8 | 168 | 0.864407 | 0.919821 |
| 9 | 181 | 0.932203 | 0.971678 |

Mean +/- sample SD:

| Metric | Mean +/- SD | Pooled |
|---|---:|---:|
| acc | 0.916328 +/- 0.036302 | 0.916388 |
| macro_f1 | 0.901554 +/- 0.045284 | 0.901105 |
| bacc | 0.898180 +/- 0.042402 | 0.897423 |
| probability_macro_auc | 0.960343 +/- 0.019838 | 0.948217 |
| weighted_f1 | 0.916046 +/- 0.036006 | 0.916187 |

## Comparisons

| Reference | Delta ACC | Delta Macro-F1 | Delta BACC | Delta probability AUC |
|---|---:|---:|---:|---:|
| Original Query | -0.013378 | -0.012973 | -0.016655 | -0.007832 |
| R | -0.010033 | -0.016287 | -0.018448 | -0.007194 |
| SEPS-Q | -0.008361 | -0.012760 | -0.010784 | -0.005073 |

Exact McNemar:

| Comparison | RCAP only correct | Other only correct | Discordant | p-value |
|---|---:|---:|---:|---:|
| RCAP vs Original Query | 24 | 32 | 56 | 0.349682 |
| RCAP vs R | 22 | 28 | 50 | 0.479888 |

## Fold0 cap mechanism

```json
{
  "reader_norm_cap": 0.5,
  "classes": {
    "AD": {
      "mean_uncapped_residual_to_shared_ratio": 9.084585189819336,
      "mean_capped_residual_to_shared_ratio": 0.5,
      "maximum_capped_residual_to_shared_ratio": 0.5000000596046448,
      "capped_subject_fraction": 1.0
    },
    "CN": {
      "mean_uncapped_residual_to_shared_ratio": 19.11554527282715,
      "mean_capped_residual_to_shared_ratio": 0.5,
      "maximum_capped_residual_to_shared_ratio": 0.5000000596046448,
      "capped_subject_fraction": 1.0
    },
    "SMCI": {
      "mean_uncapped_residual_to_shared_ratio": 11.768207550048828,
      "mean_capped_residual_to_shared_ratio": 0.5,
      "maximum_capped_residual_to_shared_ratio": 0.5000000596046448,
      "capped_subject_fraction": 1.0
    }
  },
  "maximum_capped_ratio": 0.5000000596046448
}
```

No other cap value, rank/k search, R/G/RG rerun, multi-seed, ensemble, graph, TabPFN, or additional ablation was run.
