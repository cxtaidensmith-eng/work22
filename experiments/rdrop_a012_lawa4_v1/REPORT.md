# R-Drop A012 + same-trajectory LAWA-4 v1

Decision: `R_DROP_NO_GAIN`; selected: `v1/raw_rdrop`.
Branch=`experiment/rdrop-a012-lawa4-v1`; source commit=`e2545d183ae75b39dd888e879696e8878862a922`; device=cuda:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU); formal invocation wall time=56.5s; summed fold training time=1280.1s.

## v1 (rdrop_lambda=0.3)

### raw_rdrop

Correct=554/598; ACC=0.9264214; Macro-F1=0.9114846; BACC=0.9026451; Probability Macro-AUC=0.9667385; Weighted-F1=0.9261682; confusion=[[60, 0, 12], [0, 194, 15], [7, 10, 300]]; predicted={'AD': 67, 'CN': 204, 'SMCI': 327}; repairs/damages/changed=14/22/36; AD–sMCI/CN–sMCI/AD–CN=19/25/0
Metric delta vs A012: {'correct': -8.0, 'acc': -0.013377926421404673, 'macro_f1': -0.017935983711283576, 'bacc': -0.025984847087287766, 'macro_auc': 0.005213703727642738, 'weighted_f1': -0.013573337300950894}.
Fold ACC mean ± sample SD=0.9264124 ± 0.0308657; best-epoch mean L_symKL=0.043751234.
Fold best epoch/ACC: 0:197/0.9333333, 1:87/0.9166667, 2:102/0.9666667, 3:257/0.9000000, 4:192/0.9000000, 5:60/0.9166667, 6:106/0.9666667, 7:71/0.9166667, 8:110/0.8813559, 9:107/0.9661017.
Parameters=862971; single_model=True; ensemble=False.

### rdrop_lawa4

Correct=554/598; ACC=0.9264214; Macro-F1=0.9108113; BACC=0.9067666; Probability Macro-AUC=0.9573884; Weighted-F1=0.9263369; confusion=[[61, 0, 11], [0, 195, 14], [9, 10, 298]]; predicted={'AD': 70, 'CN': 205, 'SMCI': 323}; repairs/damages/changed=16/24/40; AD–sMCI/CN–sMCI/AD–CN=20/24/0
Metric delta vs A012: {'correct': -8.0, 'acc': -0.013377926421404673, 'macro_f1': -0.018609239849778025, 'bacc': -0.02186337054758114, 'macro_auc': -0.0041364838917661295, 'weighted_f1': -0.013404597616012759}.
Fold ACC mean ± sample SD=0.9263559 ± 0.0310572; best-epoch mean L_symKL=0.045195144.
Fold best epoch/ACC: 0:219/0.9166667, 1:98/0.9166667, 2:96/0.9666667, 3:258/0.9000000, 4:196/0.9166667, 5:47/0.9333333, 6:107/0.9666667, 7:70/0.9333333, 8:65/0.8644068, 9:108/0.9491525.
Parameters=862971; single_model=True; ensemble=False.

## v1.1 gate

Triggered=False; reason=v1 preliminary decision was R_DROP_NO_GAIN; the only allowed v1.1 gate did not open; replacement=False; replacement reason=v1.1 was not triggered.

## Final

Selected result: Correct=554/598; ACC=0.9264214; Macro-F1=0.9114846; BACC=0.9026451; Probability Macro-AUC=0.9667385; Weighted-F1=0.9261682; confusion=[[60, 0, 12], [0, 194, 15], [7, 10, 300]]; predicted={'AD': 67, 'CN': 204, 'SMCI': 327}; repairs/damages/changed=14/22/36; AD–sMCI/CN–sMCI/AD–CN=19/25/0.
Next recommendation: `A012-SAM-ON`. It was not implemented or run.

## Reproduction

```text
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_rdrop_a012_lawa4_v1.py smoke --device cuda:0
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_rdrop_a012_lawa4_v1.py formal --device cuda:0
```
