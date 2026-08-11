# SP-LRIF Cross-Task Validation v1

Fixed structure: the exact ABIDE SP-LRIF formula, interaction rank 4, zero-initialized bias-free output projection. No search or second version was run.

Source commit: `063bff69a47e471dd2bf5688c3a35ee15f3c36de`  
Decision: `SP_LRIF_CROSS_TASK_STOP`  
Formal wall time: 1683.664 s

## Results

### tad_binary

- Correct: 515/535 (reference 519, delta -4)
- Labels: `ACC_NEGATIVE` / `RANKING_NEGATIVE`; task target `BELOW_TARGET`, target safety=False
- ACC=0.9626168; Macro-F1=0.8786848; BACC=0.8786848; Weighted-F1=0.9626168
- pooled ROC-AUC=0.8917460; PR-AUC=0.6776322; SEN=0.7777778; SPE=0.9795918
- Confusion=[[480, 10], [10, 35]]; predicted={'SMCI': 490, 'PMCI': 45}
- Fold ROC-AUC=0.9720408 +/- 0.0333548; fold ACC=0.9626834 +/- 0.0277654
- Metric deltas vs D5: ROC-AUC -0.0265306; PR-AUC -0.0545325; Macro-F1 -0.0222595; BACC -0.0141723
- Fold best `(fold:epoch:Correct:ACC:AUC)`: `0:74:52:.9629630:.9795918`, `1:59:50:.9259259:.9061224`, `2:89:51:.9444444:.9918367`, `3:133:51:.9444444:.9346939`, `4:78:54:1.0000000:1.0000000`, `5:16:51:.9622642:.9897959`, `6:65:52:.9811321:.9795918`, `7:104:49:.9245283:.9387755`, `8:118:53:1.0000000:1.0000000`, `9:52:52:.9811321:1.0000000`
- repairs/damages/changed=6/10/16; exact McNemar p=0.4544983
- Parameter count=618733; train=548.575s; inference=0.138906s
- Mechanism: cosine=0.6088836; delta ratio mean/max=0.0039514/0.0700484; agreement/disagreement norm=0.3806420/1.0242947; disabled-delta argmax changes=0
- Mechanism gates: all four projections changed=True; collapse=False; disabled-delta probability max/mean difference=0.0285913/0.0002495; max gradients `(c,g,diff,out)`=0.0341482/0.0255319/0.0077096/0.0430594

### abide5

- Correct: 757/864 (reference 769, delta -12)
- Labels: `ACC_NEGATIVE` / `RANKING_NEGATIVE`; task target `BELOW_TARGET`, target safety=False
- ACC=0.8761574; Macro-F1=0.8756896; BACC=0.8769438; Weighted-F1=0.8763074
- pooled ROC-AUC=0.8794977; PR-AUC=0.8463864; SEN=0.8866499; SPE=0.8672377
- Confusion=[[352, 45], [62, 405]]; predicted={'ADS': 414, 'CN': 450}
- Fold ROC-AUC=0.8865756 +/- 0.0400386; fold ACC=0.8763031 +/- 0.0371808
- Metric deltas vs D3: ROC-AUC -0.0067665; PR-AUC -0.0041785; Macro-F1 -0.0136509; BACC -0.0124704
- Fold best `(fold:epoch:Correct:ACC:AUC)`: `0:82:72:.8275862:.8500000`, `1:169:78:.8965517:.8882979`, `2:31:73:.8390805:.8861702`, `3:181:71:.8160920:.8087766`, `4:25:78:.9069767:.9312602`, `5:50:76:.8837209:.8993453`, `6:199:76:.8837209:.8581560`, `7:177:79:.9186047:.9282609`, `8:214:79:.9186047:.9345109`, `9:30:75:.8720930:.8809783`
- repairs/damages/changed=41/53/94; exact McNemar p=0.2564423
- Parameter count=384640; train=566.568s; inference=0.150759s
- Mechanism: cosine=0.0853272; delta ratio mean/max=0.2759766/1.5645914; agreement/disagreement norm=4.5678009/2.6554050; disabled-delta argmax changes=0
- Mechanism gates: all four projections changed=True; collapse=False; disabled-delta probability max/mean difference=0.0394887/0.0006533; max gradients `(c,g,diff,out)`=0.0695894/0.0427746/0.0391714/0.0305568

### tad_triclass

- Correct: 563/598 (reference 562, delta +1)
- Labels: `ACC_POSITIVE` / `RANKING_MIXED`; task target `ACC_POSITIVE`, target safety=False
- ACC=0.9414716; Macro-F1=0.9298221; BACC=0.9219819; Weighted-F1=0.9412494
- Probability Macro-AUC=0.9694026; confusion=[[62, 0, 10], [0, 199, 10], [5, 10, 302]]; predicted={'AD': 67, 'CN': 209, 'SMCI': 322}
- Boundary errors={'AD_SMCI': 15, 'CN_SMCI': 20, 'AD_CN': 0}; fold ACC=0.9414689 +/- 0.0239981
- Metric deltas vs A012: Probability Macro-AUC +0.0078777; Macro-F1 +0.0004016; BACC -0.0066481. Boundary deltas: AD-SMCI 0; CN-SMCI -1; AD-CN 0.
- Fold best `(fold:epoch:Correct:ACC)`: `0:111:54:.9000000`, `1:86:58:.9666667`, `2:100:57:.9500000`, `3:179:57:.9500000`, `4:261:56:.9333333`, `5:38:57:.9500000`, `6:190:58:.9666667`, `7:109:55:.9166667`, `8:143:54:.9152542`, `9:176:57:.9661017`
- repairs/damages/changed=15/14/29; exact McNemar p=1
- Parameter count=864507; train=538.518s; inference=0.142349s
- Mechanism: cosine=0.5816474; delta ratio mean/max=0.0167776/0.4658572; agreement/disagreement norm=1.0902753/1.1609733; disabled-delta argmax changes=1
- Mechanism gates: all four projections changed=True; collapse=False; disabled-delta probability max/mean difference=0.6258264/0.0015083; max gradients `(c,g,diff,out)`=0.0228287/0.0250114/0.0311810/0.0638968

### Existing ABIDE result

- Pre-SP best: 777/871; SP-LRIF: 780/871; net +3; `ACC_POSITIVE` / `RANKING_MIXED`.

## Cross-task decision

New-task deltas: {'tad_binary': -4, 'abide5': -12, 'tad_triclass': 1}; new-task sum=-15; four-task sum=-12.

The STOP rule is independently triggered by two new tasks losing Correct and by BACC drops over 0.01 on TAD binary and ABIDE-5. No task collapsed, all SP/private parameters trained, and tri-class AD-CN errors remained zero.

## Cross-paper numerical context

These values are descriptive only because fold/preprocessing equivalence is not established: DGFMC reports fold-mean ACC/AUC of 97.57%/90.49% for TADPOLE binary, 91.05%/90.99% for ABIDE-5, and 94.15%/94.37% for TADPOLE tri-class. No superiority or significance claim is made.

## Required answers

1. TADPOLE binary Correct improved: no.
2. ABIDE-5 Correct improved: no.
3. TADPOLE tri-class reached 563/564: yes.
4. New tasks with hard gains: 1.
5. Four-task total Correct net: -12.
6. Yes: the existing ABIDE gain is `RANKING_MIXED` because hard accuracy improved while pooled ROC-AUC and PR-AUC fell. Among the three new tasks, tri-class gained one Correct with AUC/F1 up but BACC down; the two binary tasks lost Correct.
7. Agreement and disagreement activated in all three new tasks and the existing ABIDE task; every projection tensor had nonzero gradients/changes and no delta collapse occurred.
8. Disabled-delta argmax changes were 0 for TAD binary, 0 for ABIDE-5, 1 for tri-class, and 0 for existing ABIDE. Thus the binary/ABIDE changes are trajectory-mediated; tri-class has one direct-influence case, but it is not claimed as a direct repair without subject-level causal attribution.
9. Cross-task effectiveness: False.
10. SP_LRIF_CROSS_TASK_SUPPORTED met: False.
11. No. `SP_LRIF_CROSS_TASK_STOP` means it should not be presented as a unified final-paper fusion module, and the formula must not be tuned further.
12. Retain D5 for TAD binary, D3 for ABIDE-5, and A012 for TAD tri-class. The prior ABIDE SP result remains a task-specific result, not evidence for universal transfer.

## Reproduction

Training reproduction must use a clean local branch named `experiment/sp-lrif-cross-task-v1` pointing at source commit `063bff69a47e471dd2bf5688c3a35ee15f3c36de` (not the later result commit):

```text
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_sp_lrif_cross_task_v1.py inspect
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_sp_lrif_cross_task_v1.py smoke --device cuda:0
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_sp_lrif_cross_task_v1.py formal --task all --device cuda:0
```
