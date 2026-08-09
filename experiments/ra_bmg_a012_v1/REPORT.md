# RA-BMG-A012 v1

Decision: `RA_BMG_NO_GAIN`. Retained model: `A012`.

## Formal result

Correct=560/598; ACC=0.9364548; Macro-F1=0.9268046; BACC=0.9265269; Probability Macro-AUC=0.9628727; Weighted-F1=0.9363927; confusion=[[64, 0, 8], [0, 200, 9], [7, 14, 296]]; predicted={'AD': 71, 'CN': 214, 'SMCI': 313}.
AD->sMCI=8; sMCI->AD=7; CN->sMCI=9; sMCI->CN=14; AD-CN=0; AD-sMCI=15; CN-sMCI=23.
Fold ACC mean +/- sample SD=0.9363842 +/- 0.0261091; folds (Correct/ACC): 0:55/0.9166667, 1:57/0.9500000, 2:58/0.9666667, 3:55/0.9166667, 4:56/0.9333333, 5:59/0.9833333, 6:57/0.9500000, 7:55/0.9166667, 8:53/0.8983051, 9:55/0.9322034.
Versus A012 repairs/damages/changed=0/2/2 (net=-2); sources={'AD_to_CN': 0, 'AD_to_SMCI': 0, 'CN_to_AD': 0, 'CN_to_SMCI': 0, 'SMCI_to_AD': 0, 'SMCI_to_CN': 0}/{'AD_to_CN': 0, 'AD_to_SMCI': 0, 'CN_to_AD': 0, 'CN_to_SMCI': 0, 'SMCI_to_AD': 0, 'SMCI_to_CN': 2}.
Versus C1 repairs/damages/changed=20/20/40 (net=0).

## Fixed mechanism diagnostics

Train-only mapped C1 CN-margin Pearson mean=0.9892124; sign agreement mean=0.9936851; normalized RMSE mean=0.1353464; full-logit cosine mean=0.9869046.
All SVD/scales finite=True; any test label used for mapping=False.
A012 AD:sMCI conditional log-odds max error=5.86e-14; RA-BMG/C1 CN-prediction agreement=0.9531773.
Storage: A012 parameters=862971; mapped head buffers=147 coefficients/588 bytes; inference coefficients=863118; state tensor bytes/fold=4888696; artifact bytes total=49494300.
Construction time=4.304s; one-forward inference time sum=0.158429s; GPU=NVIDIA GeForce RTX 3050 Ti Laptop GPU; CPU=AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD.

## Required answers

1. Closed-form alignment was measured without model selection: Pearson=0.9892124, sign agreement=0.9936851, NRMSE=0.1353464.
2. Mapped C1 CN-margin direction agreement was 0.9936851 on train rows.
3. A012 AD:sMCI conditional odds were strictly preserved (max error 5.86e-14).
4. RA-BMG did not reach 564/598 (actual 560/598).
5. Boundary exchange is shown by repairs/damages sources above; CN-sMCI ended at 23 and AD-sMCI at 15.
6. AD-CN errors=0.
7. Final inference uses one A012 backbone; C1 is absent from every artifact state.
8. No test label was used to fit T, c, variance scales, or the mapped head.
9. Compared with raw weight soup=502, representation alignment plus boundary-selective coupling produced 560/598 and avoided that collapse.
10. Final retention: A012.

## Reproduction and Git

Source commit=`eb6bac6e024818290b6664bbfa5c5cff02294816`; result commit=SELF_REFERENCE_NOT_EMBEDDED; use the Git commit tracking this summary; branch=`experiment/ra-bmg-a012-v1`.

```text
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_ra_bmg_a012_v1.py formal --device cuda:0
```

Next recommendation: `DB-MoLA-A012: dual-boundary low-rank feature experts`. It was not implemented or run.
