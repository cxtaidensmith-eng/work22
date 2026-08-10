# Binary Task-Adaptive Private Residual Calibration v1

Decision: **PRIVATE_RESIDUAL_STOP**

Stop the cross-dataset private-residual route.

## Stage 0: corrected binary objective

The historical `criterion_lossv2` accumulated both OVR CE terms into an unnormalized sum and did not consume the declared ABIDE `0.2/mean` setting. This experiment leaves historical runners untouched and uses `L_main + 0.2 * mean(L_ovr0,L_ovr1) + historical_orthogonality` in its own runner. Class weights are computed only from the current training mask. The deterministic float64 unit check has absolute error below 1e-8.

## Formal results

| Task / variant | Correct | ACC | BACC | Macro-F1 | ROC-AUC | PR-AUC | SEN | SPE | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tadpole_smci_pmci / B0 | 497/535 | 0.9289720 | 0.8098639 | 0.7865751 | 0.8652381 | 0.5162370 | 0.6666667 | 0.9530612 | 0.9315788 |
| tadpole_smci_pmci / B1_Tuned | 484/535 | 0.9046729 | 0.7461451 | 0.7212090 | 0.8542857 | 0.4154489 | 0.5555556 | 0.9367347 | 0.9093229 |
| abide_ads_cn / B0 | 723/871 | 0.8300804 | 0.8294734 | 0.8292428 | 0.8312072 | 0.7724139 | 0.8213400 | 0.8376068 | 0.8301353 |
| abide_ads_cn / B1_Tuned | 699/871 | 0.8025258 | 0.8064171 | 0.8025131 | 0.8469889 | 0.8031787 | 0.8585608 | 0.7542735 | 0.8026315 |
| abide5_ads_cn / B0 | 710/864 | 0.8217593 | 0.8222806 | 0.8211113 | 0.8351555 | 0.7962805 | 0.8287154 | 0.8158458 | 0.8219836 |
| abide5_ads_cn / B1_Transfer | 710/864 | 0.8217593 | 0.8194489 | 0.8201392 | 0.8347888 | 0.7840152 | 0.7909320 | 0.8479657 | 0.8215222 |
| abide5_ads_cn / B1_Tuned | 688/864 | 0.7962963 | 0.7953279 | 0.7951007 | 0.8204413 | 0.7740049 | 0.7833753 | 0.8072805 | 0.7963688 |

Confusion matrices, predicted counts, fold statistics, selected epochs, parameters and timing:

- `tadpole_smci_pmci/B0`: confusion=[[467, 23], [15, 30]]; predicted={'SMCI': 482, 'PMCI': 53}; fold ACC=0.9290007 +/- 0.0400668; epochs=[95, 38, 171, 125, 27, 24, 213, 156, 155, 65]; params=[608997, 608997, 608997, 608997, 608997, 608997, 608997, 608997, 608997, 608997]; train=109.825s; inference=0.159789s.
- `tadpole_smci_pmci/B1_Tuned`: confusion=[[459, 31], [20, 25]]; predicted={'SMCI': 479, 'PMCI': 56}; fold ACC=0.9046122 +/- 0.0464913; epochs=[145, 76, 50, 227, 31, 60, 96, 148, 129, 31]; params=[617197, 613337, 617197, 617197, 617197, 613337, 613337, 613337, 617197, 617197]; train=117.029s; inference=0.179440s.
- `abide_ads_cn/B0`: confusion=[[331, 72], [76, 392]]; predicted={'ADS': 407, 'CN': 464}; fold ACC=0.8300940 +/- 0.0421793; epochs=[241, 205, 252, 170, 217, 311, 212, 197, 245, 192]; params=[333033, 333033, 333033, 333033, 333033, 333033, 333033, 333033, 333033, 333033]; train=211.260s; inference=0.166929s.
- `abide_ads_cn/B1_Tuned`: confusion=[[346, 57], [115, 353]]; predicted={'ADS': 461, 'CN': 410}; fold ACC=0.8025078 +/- 0.1226093; epochs=[188, 120, 244, 212, 239, 297, 300, 317, 281, 177]; params=[337417, 337417, 335353, 337417, 337417, 335353, 335353, 337417, 335353, 335353]; train=239.775s; inference=0.171227s.
- `abide5_ads_cn/B0`: confusion=[[329, 68], [86, 381]]; predicted={'ADS': 415, 'CN': 449}; fold ACC=0.8217455 +/- 0.0421738; epochs=[108, 116, 197, 132, 325, 212, 69, 30, 159, 127]; params=[380716, 380716, 380716, 380716, 380716, 380716, 380716, 380716, 380716, 380716]; train=131.678s; inference=0.163014s.
- `abide5_ads_cn/B1_Transfer`: confusion=[[314, 83], [71, 396]]; predicted={'ADS': 385, 'CN': 479}; fold ACC=0.8218391 +/- 0.0440200; epochs=[188, 120, 244, 212, 239, 297, 300, 317, 281, 177]; params=[386196, 386196, 383616, 386196, 386196, 383616, 383616, 386196, 383616, 383616]; train=255.409s; inference=0.171382s.
- `abide5_ads_cn/B1_Tuned`: confusion=[[311, 86], [90, 377]]; predicted={'ADS': 401, 'CN': 463}; fold ACC=0.7962042 +/- 0.0630807; epochs=[255, 269, 271, 132, 80, 203, 30, 75, 190, 238]; params=[386196, 383616, 386196, 383616, 383616, 386196, 386196, 383616, 386196, 386196]; train=186.833s; inference=0.174533s.

## Paired corrected-loss comparisons

- `tadpole_smci_pmci/B1_Tuned_vs_B0`: Correct delta=-13; repairs/damages/changed=8/21/29; BACC=-0.0637188; Macro-F1=-0.0653662; ROC-AUC=-0.0109524; PR-AUC=-0.1007881; exact McNemar p=0.024119545.
- `abide_ads_cn/B1_Tuned_vs_B0`: Correct delta=-24; repairs/damages/changed=50/74/124; BACC=-0.0230562; Macro-F1=-0.0267298; ROC-AUC=+0.0157817; PR-AUC=+0.0307648; exact McNemar p=0.038448497.
- `abide5_ads_cn/B1_Transfer_vs_B0`: Correct delta=+0; repairs/damages/changed=68/68/136; BACC=-0.0028317; Macro-F1=-0.0009721; ROC-AUC=-0.0003668; PR-AUC=-0.0122653; exact McNemar p=1.
- `abide5_ads_cn/B1_Tuned_vs_B0`: Correct delta=-22; repairs/damages/changed=63/85/148; BACC=-0.0269527; Macro-F1=-0.0260106; ROC-AUC=-0.0147142; PR-AUC=-0.0222755; exact McNemar p=0.083966617.
- `abide5_ads_cn/B1_Tuned_vs_B1_Transfer`: Correct delta=-22; repairs/damages/changed=53/75/128; BACC=-0.0241209; Macro-F1=-0.0250385; ROC-AUC=-0.0143474; PR-AUC=-0.0100103; exact McNemar p=0.063008783.

## Nested selections and mechanisms

- `tadpole_smci_pmci/B1_Tuned`: frequency={'rank=8,multiplier=0.5': 3, 'rank=4,multiplier=1.0': 2, 'rank=8,multiplier=1.0': 2, 'rank=8,multiplier=2.0': 1, 'rank=4,multiplier=2.0': 2}; ratio mean/max=0.0907621/2.1587374; per-modality mean={'MRI': 0.03215000443160534, 'CSF': 0.23938420114573092, 'RISK_FACTOR': 0.09737556318286807, 'COGNITIVE_TEST': 0.03684331092517823, 'ROI_AVERAGE': 0.048057553358376026}; per-modality max={'MRI': 0.1261081099510193, 'CSF': 2.1587374210357666, 'RISK_FACTOR': 0.8339499831199646, 'COGNITIVE_TEST': 0.21006254851818085, 'ROI_AVERAGE': 0.21061964333057404}; Category-Global cosine=0.4596928; max adapter grad=0.1132487; private-off probability diff=0.01422691; direct changed/repairs/damages=0/0/0; active=True; collapse=False.
- `abide_ads_cn/B1_Tuned`: frequency={'rank=8,multiplier=0.5': 4, 'rank=8,multiplier=1.0': 1, 'rank=4,multiplier=2.0': 2, 'rank=4,multiplier=0.5': 2, 'rank=4,multiplier=1.0': 1}; ratio mean/max=0.0255132/0.2125426; per-modality mean={'PHENO': 0.02754235751926899, 'ANAT': 0.02275083865970373, 'FUNC': 0.024251528922468422, 'Correlation': 0.027508051693439485}; per-modality max={'PHENO': 0.12919881939888, 'ANAT': 0.13192126154899597, 'FUNC': 0.11200752854347229, 'Correlation': 0.2125426083803177}; Category-Global cosine=0.1571632; max adapter grad=0.03415306; private-off probability diff=0.05905752; direct changed/repairs/damages=0/0/0; active=True; collapse=False.
- `abide5_ads_cn/B1_Transfer`: frequency={'rank=8,multiplier=0.5': 4, 'rank=8,multiplier=1.0': 1, 'rank=4,multiplier=2.0': 2, 'rank=4,multiplier=0.5': 2, 'rank=4,multiplier=1.0': 1}; ratio mean/max=0.0128830/0.1984839; per-modality mean={'PHENO': 0.010155145218595863, 'ANAT': 0.010186181729659438, 'FUNC': 0.008661735546775162, 'MRI': 0.008026938699185849, 'fMRI': 0.027385019278153778}; per-modality max={'PHENO': 0.06687863171100616, 'ANAT': 0.06181856617331505, 'FUNC': 0.04846221208572388, 'MRI': 0.04188620299100876, 'fMRI': 0.19848386943340302}; Category-Global cosine=0.1245642; max adapter grad=0.007982031; private-off probability diff=0.09508115; direct changed/repairs/damages=0/0/0; active=True; collapse=False.
- `abide5_ads_cn/B1_Tuned`: frequency={'rank=8,multiplier=2.0': 1, 'rank=4,multiplier=2.0': 2, 'rank=8,multiplier=0.5': 2, 'rank=4,multiplier=1.0': 1, 'rank=4,multiplier=0.5': 1, 'rank=8,multiplier=1.0': 3}; ratio mean/max=0.0215177/0.5158086; per-modality mean={'PHENO': 0.011592167709022761, 'ANAT': 0.009405839419923723, 'FUNC': 0.012798198871314526, 'MRI': 0.009783595614135266, 'fMRI': 0.06400894783437253}; per-modality max={'PHENO': 0.07029750198125839, 'ANAT': 0.0431060828268528, 'FUNC': 0.06371733546257019, 'MRI': 0.09286628663539886, 'fMRI': 0.515808641910553}; Category-Global cosine=0.1569232; max adapter grad=0.01923662; private-off probability diff=0.02757481; direct changed/repairs/damages=0/0/0; active=True; collapse=False.

## Legacy reference (not used for this decision)

- `tadpole_smci_pmci` legacy B0 Correct=513, BACC=0.8463719, Macro-F1=0.8609142; corrected B0 delta Correct=-16, BACC=-0.0365079, Macro-F1=-0.0743390.
- `abide_ads_cn` legacy B0 Correct=761, BACC=0.8740350, Macro-F1=0.8732390; corrected B0 delta Correct=-38, BACC=-0.0445616, Macro-F1=-0.0439962.
- `abide5_ads_cn` legacy B0 Correct=768, BACC=0.8872108, Macro-F1=0.8879281; corrected B0 delta Correct=-58, BACC=-0.0649302, Macro-F1=-0.0668168.

## Required answers

1. **Why did OVR not follow the declaration, and how was it fixed?** The historical loss hard-coded a sum of two OVR losses at coefficient 1. The new experiment-local loss explicitly averages the two losses and multiplies by 0.2; historical defaults remain unchanged.
2. **Did B0 change after correction?** Yes/no by task is quantified above against commit `e7d18205ddfefaf898a81c999cce15b348331206`; current corrected B0 Correct values are TAD=497, ABIDE=723, ABIDE-5=710.
3. **Most common TAD configuration?** ['rank=8,multiplier=0.5']; full frequency={'rank=8,multiplier=0.5': 3, 'rank=4,multiplier=1.0': 2, 'rank=8,multiplier=1.0': 2, 'rank=8,multiplier=2.0': 1, 'rank=4,multiplier=2.0': 2}.
4. **Most common ABIDE configuration?** ['rank=8,multiplier=0.5']; full frequency={'rank=8,multiplier=0.5': 4, 'rank=8,multiplier=1.0': 1, 'rank=4,multiplier=2.0': 2, 'rank=4,multiplier=0.5': 2, 'rank=4,multiplier=1.0': 1}.
5. **Did ABIDE turn from negative to net gain?** Correct delta=-24, repairs/damages=50/74; no.
6. **Did ABIDE configuration transfer directly to ABIDE-5?** Transfer Correct delta=+0, BACC delta=-0.0028317; no.
7. **Was ABIDE-5 Tuned clearly better than Transfer?** Tuned-vs-Transfer Correct delta=-22, BACC delta=-0.0241209; no.
8. **Did the TAD pMCI benefit remain?** TP delta=-5, sensitivity delta=-0.1111111, PR-AUC delta=-0.1007881.
9. **Direct inference versus trajectory effect?** TAD direct private-off changed/repairs/damages=0/0/0, while full B0-to-B1 trajectory repairs/damages/changed=8/21/29; analogous values are recorded for every task.
10. **Final Decision?** `PRIVATE_RESIDUAL_STOP`.
11. **Continue Category + Global fusion?** No unified Category-Global fusion experiment is authorized by this result.

## Reproduction and Git handoff

Training source commit: `76ecfaa5aa7891bac84c3f3a213eeba4e7f7e111`. The eventual result commit is for reading results; strict reruns must use a clean local branch named `experiment/binary-task-adapter-calibration-v1` pointing at the source commit.
Historical evidence root: `D:\Work\WORK2 final\WORK2 final26.7,21`.
Python: `D:\Anaconda\envs\work22-tabpfn-v1\python.exe`; torch=2.5.1+cu121; CUDA=12.1; GPU=NVIDIA GeForce RTX 3050 Ti Laptop GPU.

Commands (from the clean source worktree):
```powershell
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts\run_binary_task_adapter_calibration_v1.py inspect --historical-root 'D:\Work\WORK2 final\WORK2 final26.7,21'
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts\run_binary_task_adapter_calibration_v1.py smoke --device cuda:0 --historical-root 'D:\Work\WORK2 final\WORK2 final26.7,21'
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts\run_binary_task_adapter_calibration_v1.py search --device cuda:0 --historical-root 'D:\Work\WORK2 final\WORK2 final26.7,21'
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts\run_binary_task_adapter_calibration_v1.py formal --device cuda:0 --historical-root 'D:\Work\WORK2 final\WORK2 final26.7,21'
```

Result commit SHA is reported in the final Git handoff because a commit cannot contain its own SHA.
