# Original Query + Global Ordinal Auxiliary — 10-Fold Final Report

**Decision: STOP**

## Protocol

TADPOLE / AD_CN_SMCI; folds 0..9; seed=0; 400 epochs; full-batch transductive; single model; no ensemble. The Original Query classification logits remain the inference output. A Global-only ordinal auxiliary loss uses CN < sMCI < AD with lambda=0.2.

## Pooled OOF result

- Parameters: 853230
- Correct: 551/598
- ACC: 0.9214046823 (-0.0083612177 vs Original)
- Macro-F1: 0.9112153077 (-0.0028624923)
- BACC: 0.9055599718 (-0.0085178282)
- Probability Macro-AUC: 0.9495575708 (-0.0064915292)
- Weighted-F1: 0.9212890496 (-0.0084768504)
- Fold ACC: 0.9213276836 ± 0.0298756916 (sample SD)
- Final threshold mean: tau1=-0.4583690286, tau2=2.1087387085

Confusion-matrix order: ['AD', 'CN', 'SMCI']

```text
[62, 0, 10]
[0, 192, 17]
[7, 13, 297]
```

## Per-fold best result

| Fold | Best epoch | ACC |
|---:|---:|---:|
| 0 | 71 | 0.9000000000 |
| 1 | 59 | 0.9166666667 |
| 2 | 291 | 0.9500000000 |
| 3 | 49 | 0.9666666667 |
| 4 | 82 | 0.9000000000 |
| 5 | 66 | 0.9500000000 |
| 6 | 392 | 0.9166666667 |
| 7 | 164 | 0.9166666667 |
| 8 | 79 | 0.8644067797 |
| 9 | 157 | 0.9322033898 |

## Runtime and provenance

- Training time: 498.528 s
- Total run time: 503.202 s
- Branch: `experiment/global-ordinal-aux-v1`
- Source/base commit: `765720c1c0a3b2263502e9bf662fe33de7e45428`
- Command: `D:\Anaconda\envs\work22-tabpfn-v1\python.exe scripts\run_global_ordinal_aux_v1.py`
