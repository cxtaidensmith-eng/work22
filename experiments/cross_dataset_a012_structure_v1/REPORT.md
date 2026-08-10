# A012/C1 Cross-Dataset Structural Generalization v1

- Source commit: `17004d29714471b5e1548e04bac0072e854e7e18`
- Result commit: reported in the final Git handoff after this report and its result files are committed (self-referential SHA is intentionally not embedded)
- Branch: `experiment/cross-dataset-a012-structure-v1`
- Device: `cuda:0`
- Execution environment: Python `3.11.15`; PyTorch `2.5.1+cu121`; torch CUDA `12.1`; GPU `NVIDIA GeForce RTX 3050 Ti Laptop GPU`
- Decision: **CROSS_DATASET_MIXED**
- Formal wall time: 7039.2 seconds
- B0 is fully rerun Original; B1 is the rank-8 A012 structural branch with private-adapter LR=2x.
- All results are single-model seed-0, ten-fold, 400-epoch OOF evaluations; no ensemble is used.

## Fixed protocol disclosure

The paired arms preserve the last formal historical protocol: offline preprocessed inputs, global full-dataset class weights, and held-out-fold checkpoint selection by ACC > ROC-AUC > Macro-F1 > earliest epoch. Probabilities are materialized before the held-out truth is read for that fixed selection. No test-driven hyperparameter, feature, threshold, calibration, or model search is performed.

The ABIDE task name `ADS_CN` retains the repository's historical `ADS` spelling. Here ADS is the ASD/autism-spectrum class: raw label 1 maps to class index 0 and is positive; raw label 2 maps to CN at class index 1. No external subject identifier is claimed. ABIDE and ABIDE-5 share 64 feature columns and 841 feature-identical rows, but are trained and reported separately.

The historical three-seed NPZ artifacts are provenance only. They are not reused because they do not lock a training source commit or durable external IDs and do not match this one-seed protocol.

## Dataset and formal protocol

| Task | N / class counts (index order) | Modalities | LR | WD | Dropout | Modal transformer | DIFFormer | Loss | Scheduler |
|---|---|---:|---:|---:|---:|---|---|---|---|
| TADPOLE SMCI_PMCI | 535 / SMCI=490, PMCI=45 | 5 | 0.01 | 0.0005 | 0.67 | L2/H4, hidden=96, noise=0.05, drop_path=0.05 | L2/H1, alpha=0.1, beta=0.1, simple, graph=False | global weighted CE(ls=0.05) + 2 weighted OVR CE sum + orth=0.0001 | CustomCosineAnnealingLR(T_max=400, eta_min=0.0001); B1 adapter LR=2x ratio-preserved |
| ABIDE ADS_CN | 871 / ADS=403, CN=468 | 4 | 0.005 | 0.001 | 0.45 | L3/H4, hidden=64, noise=0.0, drop_path=0.05 | L2/H1, alpha=0.15, beta=0.1, simple, graph=False | global weighted CE(ls=0.05) + 2 weighted OVR CE sum + orth=0.0 | CustomCosineAnnealingLR(T_max=400, eta_min=0.0001); B1 adapter LR=2x ratio-preserved |
| ABIDE-5 ADS_CN | 864 / ADS=397, CN=467 | 5 | 0.005 | 0.0005 | 0.45 | L3/H4, hidden=64, noise=0.0, drop_path=0.05 | L2/H2, alpha=0.15, beta=0.1, sigmoid, graph=False | global weighted CE(ls=0.05) + 2 weighted OVR CE sum + orth=0.0 | CustomCosineAnnealingLR(T_max=400, eta_min=0.0001); B1 adapter LR=2x ratio-preserved |

All tasks use their locked offline processed inputs, global historical class weights, full-batch transductive folds 0..9, seed 0, 400 epochs, Adam, grad clip 1, EMA off, graph disabled, tau 0, and one fresh model/criterion/optimizer/scheduler per fold. ABIDE/ABIDE-5 config fields that declared auxiliary weight 0.2/mean were ineffective in the historical main; the actual two OVR losses are summed at coefficient 1 and are preserved here.

## Formal results

| Task | Arm | Correct/N | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | SEN | SPE | Params | Fold ACC mean +/- SD | Train sec | Mean full-graph inference sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TADPOLE SMCI_PMCI | B0 | 513/535 | 0.9588785 | 0.9223583 | 0.6378392 | 0.8609142 | 0.8463719 | 0.9580058 | 0.7111111 | 0.9816327 | 608997 | 0.9588749 +/- 0.0229204 | 1085.4 | 0.018786 |
| TADPOLE SMCI_PMCI | B1 | 515/535 | 0.9626168 | 0.8794558 | 0.6729809 | 0.8786848 | 0.8786848 | 0.9626168 | 0.7777778 | 0.9795918 | 617197 | 0.9626834 +/- 0.0214541 | 1199.0 | 0.018916 |
| ABIDE ADS_CN | B0 | 761/871 | 0.8737084 | 0.8829240 | 0.8401998 | 0.8732390 | 0.8740350 | 0.8738147 | 0.8784119 | 0.8696581 | 333033 | 0.8737069 +/- 0.0291827 | 1124.9 | 0.020430 |
| ABIDE ADS_CN | B1 | 758/871 | 0.8702641 | 0.8750557 | 0.8289356 | 0.8699899 | 0.8716915 | 0.8704354 | 0.8908189 | 0.8525641 | 337417 | 0.8702847 +/- 0.0432827 | 1212.0 | 0.020604 |
| ABIDE-5 ADS_CN | B0 | 768/864 | 0.8888889 | 0.9066931 | 0.8906037 | 0.8879281 | 0.8872108 | 0.8887688 | 0.8664987 | 0.9079229 | 380716 | 0.8889735 +/- 0.0534292 | 1131.7 | 0.019522 |
| ABIDE-5 ADS_CN | B1 | 766/864 | 0.8865741 | 0.8960890 | 0.8701144 | 0.8860244 | 0.8867685 | 0.8866657 | 0.8891688 | 0.8843683 | 386196 | 0.8867014 +/- 0.0451331 | 1252.5 | 0.019424 |

### Confusion matrices and prediction counts

- TADPOLE SMCI_PMCI B0: class order=['SMCI', 'PMCI']; confusion=[[481, 9], [13, 32]]; predicted={'SMCI': 494, 'PMCI': 41}; TP/FN/TN/FP=32/13/481/9; SEN is positive-class sensitivity and SPE is negative-class specificity.
- TADPOLE SMCI_PMCI B1: class order=['SMCI', 'PMCI']; confusion=[[480, 10], [10, 35]]; predicted={'SMCI': 490, 'PMCI': 45}; TP/FN/TN/FP=35/10/480/10; SEN is positive-class sensitivity and SPE is negative-class specificity.
- ABIDE ADS_CN B0: class order=['ADS', 'CN']; confusion=[[354, 49], [61, 407]]; predicted={'ADS': 415, 'CN': 456}; TP/FN/TN/FP=354/49/407/61; SEN is positive-class sensitivity and SPE is negative-class specificity.
- ABIDE ADS_CN B1: class order=['ADS', 'CN']; confusion=[[359, 44], [69, 399]]; predicted={'ADS': 428, 'CN': 443}; TP/FN/TN/FP=359/44/399/69; SEN is positive-class sensitivity and SPE is negative-class specificity.
- ABIDE-5 ADS_CN B0: class order=['ADS', 'CN']; confusion=[[344, 53], [43, 424]]; predicted={'ADS': 387, 'CN': 477}; TP/FN/TN/FP=344/53/424/43; SEN is positive-class sensitivity and SPE is negative-class specificity.
- ABIDE-5 ADS_CN B1: class order=['ADS', 'CN']; confusion=[[353, 44], [54, 413]]; predicted={'ADS': 407, 'CN': 457}; TP/FN/TN/FP=353/44/413/54; SEN is positive-class sensitivity and SPE is negative-class specificity.

## Paired B1 minus B0 changes

| Task | Correct delta | ACC delta | AUC delta | F1 delta | BACC delta | Weighted-F1 delta | SEN delta | SPE delta | PR-AUC delta | Repairs | Damages | Changed | Majority-pred delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TADPOLE SMCI_PMCI | +2 | +0.0037383 | -0.0429025 | +0.0177706 | +0.0323129 | +0.0046111 | +0.0666667 | -0.0020408 | +0.0351416 | 6 | 4 | 10 | -4 |
| ABIDE ADS_CN | -3 | -0.0034443 | -0.0078683 | -0.0032492 | -0.0023435 | -0.0033792 | +0.0124069 | -0.0170940 | -0.0112642 | 41 | 44 | 85 | -13 |
| ABIDE-5 ADS_CN | -2 | -0.0023148 | -0.0106042 | -0.0019036 | -0.0004423 | -0.0021031 | +0.0226700 | -0.0235546 | -0.0204893 | 40 | 42 | 82 | -20 |

## TADPOLE minority-class priority view

| Arm | TP | FN | TN | FP | BACC | Macro-F1 | pMCI SEN | PR-AUC | ROC-AUC | ACC | Predicted pMCI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 32 | 13 | 481 | 9 | 0.8463719 | 0.8609142 | 0.7111111 | 0.6378392 | 0.9223583 | 0.9588785 | 41 |
| B1 | 35 | 10 | 480 | 10 | 0.8786848 | 0.8786848 | 0.7777778 | 0.6729809 | 0.8794558 | 0.9626168 | 45 |

## Fold selections

### TADPOLE SMCI_PMCI

| Fold | B0 best epoch | B0 Correct | B0 ACC | B1 best epoch | B1 Correct | B1 ACC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 130 | 51 | 0.9444444 | 109 | 51 | 0.9444444 |
| 1 | 234 | 50 | 0.9259259 | 175 | 50 | 0.9259259 |
| 2 | 301 | 52 | 0.9629630 | 34 | 52 | 0.9629630 |
| 3 | 71 | 52 | 0.9629630 | 60 | 52 | 0.9629630 |
| 4 | 55 | 54 | 1.0000000 | 9 | 53 | 0.9814815 |
| 5 | 110 | 50 | 0.9433962 | 79 | 51 | 0.9622642 |
| 6 | 45 | 52 | 0.9811321 | 57 | 52 | 0.9811321 |
| 7 | 28 | 50 | 0.9433962 | 43 | 51 | 0.9622642 |
| 8 | 46 | 52 | 0.9811321 | 219 | 53 | 1.0000000 |
| 9 | 43 | 50 | 0.9433962 | 46 | 50 | 0.9433962 |

### ABIDE ADS_CN

| Fold | B0 best epoch | B0 Correct | B0 ACC | B1 best epoch | B1 Correct | B1 ACC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 259 | 77 | 0.8750000 | 299 | 75 | 0.8522727 |
| 1 | 94 | 78 | 0.8965517 | 232 | 77 | 0.8850575 |
| 2 | 90 | 78 | 0.8965517 | 111 | 80 | 0.9195402 |
| 3 | 71 | 74 | 0.8505747 | 135 | 69 | 0.7931034 |
| 4 | 287 | 77 | 0.8850575 | 274 | 79 | 0.9080460 |
| 5 | 217 | 75 | 0.8620690 | 264 | 76 | 0.8735632 |
| 6 | 174 | 75 | 0.8620690 | 150 | 76 | 0.8735632 |
| 7 | 296 | 73 | 0.8390805 | 66 | 72 | 0.8275862 |
| 8 | 249 | 81 | 0.9310345 | 257 | 81 | 0.9310345 |
| 9 | 234 | 73 | 0.8390805 | 243 | 73 | 0.8390805 |

### ABIDE-5 ADS_CN

| Fold | B0 best epoch | B0 Correct | B0 ACC | B1 best epoch | B1 Correct | B1 ACC |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 170 | 73 | 0.8390805 | 144 | 74 | 0.8505747 |
| 1 | 120 | 85 | 0.9770115 | 169 | 81 | 0.9310345 |
| 2 | 162 | 74 | 0.8505747 | 30 | 73 | 0.8390805 |
| 3 | 51 | 71 | 0.8160920 | 285 | 71 | 0.8160920 |
| 4 | 102 | 82 | 0.9534884 | 40 | 81 | 0.9418605 |
| 5 | 91 | 75 | 0.8720930 | 48 | 76 | 0.8837209 |
| 6 | 228 | 75 | 0.8720930 | 79 | 78 | 0.9069767 |
| 7 | 171 | 81 | 0.9418605 | 192 | 79 | 0.9186047 |
| 8 | 59 | 78 | 0.9069767 | 90 | 80 | 0.9302326 |
| 9 | 22 | 74 | 0.8604651 | 27 | 73 | 0.8488372 |

## B1 mechanism audit

| Task | Params B0 -> B1 | Private/shared mean | Private/shared max | Category-Global cosine | Private trained all folds |
|---|---:|---:|---:|---:|---:|
| TADPOLE SMCI_PMCI | 608997 -> 617197 (+8200) | 0.0744016 | 0.7248978 | 0.7809656 | True |
| ABIDE ADS_CN | 333033 -> 337417 (+4384) | 0.0149291 | 0.1186154 | 0.1735645 | True |
| ABIDE-5 ADS_CN | 380716 -> 386196 (+5480) | 0.0096079 | 0.0856841 | 0.2078714 | True |

### TADPOLE SMCI_PMCI per-modality private residual

| Modality | Residual norm mean | Private/shared ratio mean | Private/shared ratio max |
|---|---:|---:|---:|
| MRI | 13.5339860 | 0.1468807 | 0.7248978 |
| CSF | 2.2628918 | 0.0409092 | 0.2928768 |
| RISK_FACTOR | 5.9491868 | 0.0687761 | 0.3479802 |
| COGNITIVE_TEST | 3.2405302 | 0.0349850 | 0.5049902 |
| ROI_AVERAGE | 6.8303554 | 0.0804569 | 0.2449483 |

- Per-adapter-tensor maximum gradients: `{"private_adapters.0.down.bias": 0.03778884559869766, "private_adapters.0.down.weight": 0.19583313167095184, "private_adapters.0.up.bias": 0.06314778327941895, "private_adapters.0.up.weight": 0.1288263201713562, "private_adapters.1.down.bias": 0.0362430065870285, "private_adapters.1.down.weight": 0.18901175260543823, "private_adapters.1.up.bias": 0.06468749046325684, "private_adapters.1.up.weight": 0.1388089954853058, "private_adapters.2.down.bias": 0.0595579594373703, "private_adapters.2.down.weight": 0.3067428469657898, "private_adapters.2.up.bias": 0.06486830860376358, "private_adapters.2.up.weight": 0.08205855637788773, "private_adapters.3.down.bias": 0.0460507869720459, "private_adapters.3.down.weight": 0.22055268287658691, "private_adapters.3.up.bias": 0.05321032553911209, "private_adapters.3.up.weight": 0.12128444015979767, "private_adapters.4.down.bias": 0.03350242227315903, "private_adapters.4.down.weight": 0.16136035323143005, "private_adapters.4.up.bias": 0.05912626534700394, "private_adapters.4.up.weight": 0.13245339691638947}`
- Private collapse on any fold: `False`

### ABIDE ADS_CN per-modality private residual

| Modality | Residual norm mean | Private/shared ratio mean | Private/shared ratio max |
|---|---:|---:|---:|
| PHENO | 0.1211503 | 0.0142099 | 0.0965300 |
| ANAT | 0.0895914 | 0.0096277 | 0.0474515 |
| FUNC | 0.1451641 | 0.0171029 | 0.0932867 |
| Correlation | 0.1441855 | 0.0187759 | 0.1186154 |

- Per-adapter-tensor maximum gradients: `{"private_adapters.0.down.bias": 0.0038738735020160675, "private_adapters.0.down.weight": 0.012711405754089355, "private_adapters.0.up.bias": 0.01810353994369507, "private_adapters.0.up.weight": 0.008909801952540874, "private_adapters.1.down.bias": 0.003250584937632084, "private_adapters.1.down.weight": 0.013701509684324265, "private_adapters.1.up.bias": 0.019404293969273567, "private_adapters.1.up.weight": 0.009160912595689297, "private_adapters.2.down.bias": 0.00670072715729475, "private_adapters.2.down.weight": 0.012517783790826797, "private_adapters.2.up.bias": 0.021585235372185707, "private_adapters.2.up.weight": 0.007672770880162716, "private_adapters.3.down.bias": 0.005270424298942089, "private_adapters.3.down.weight": 0.013054242357611656, "private_adapters.3.up.bias": 0.022431842982769012, "private_adapters.3.up.weight": 0.007363179232925177}`
- Private collapse on any fold: `False`

### ABIDE-5 ADS_CN per-modality private residual

| Modality | Residual norm mean | Private/shared ratio mean | Private/shared ratio max |
|---|---:|---:|---:|
| PHENO | 0.1576710 | 0.0078453 | 0.0418807 |
| ANAT | 0.1621111 | 0.0079183 | 0.0396357 |
| FUNC | 0.1684677 | 0.0081362 | 0.0460643 |
| MRI | 0.1783167 | 0.0087241 | 0.0453272 |
| fMRI | 0.3191546 | 0.0154158 | 0.0856841 |

- Per-adapter-tensor maximum gradients: `{"private_adapters.0.down.bias": 0.0012454120442271233, "private_adapters.0.down.weight": 0.002922282787039876, "private_adapters.0.up.bias": 0.013492479920387268, "private_adapters.0.up.weight": 0.007579118944704533, "private_adapters.1.down.bias": 0.0010605527786538005, "private_adapters.1.down.weight": 0.002753324806690216, "private_adapters.1.up.bias": 0.013547150418162346, "private_adapters.1.up.weight": 0.007817862555384636, "private_adapters.2.down.bias": 0.0007847600500099361, "private_adapters.2.down.weight": 0.0025373301468789577, "private_adapters.2.up.bias": 0.012981196865439415, "private_adapters.2.up.weight": 0.005437931045889854, "private_adapters.3.down.bias": 0.0036284879315644503, "private_adapters.3.down.weight": 0.006550205871462822, "private_adapters.3.up.bias": 0.01271090842783451, "private_adapters.3.up.weight": 0.008305719122290611, "private_adapters.4.down.bias": 0.0024149627424776554, "private_adapters.4.down.weight": 0.010483781807124615, "private_adapters.4.up.bias": 0.012649749405682087, "private_adapters.4.up.weight": 0.008233563974499702}`
- Private collapse on any fold: `False`

## Decision audit

- Improved tasks (strict BACC>0 or Macro-F1>0): tadpole_smci_pmci
- Marked-decline tasks (BACC or Macro-F1 delta < -0.005): none
- TADPOLE pMCI TP: 32 -> 35; added FN=-3; predicted pMCI=45
- ABIDE imbalance flags: {"abide5_ads_cn": false, "abide_ads_cn": false}
- Stop reasons: none
- Final decision: **CROSS_DATASET_MIXED**

## Answers to the ten required questions

1. TADPOLE SMCI_PMCI versus Original: TADPOLE SMCI_PMCI: Correct +2, BACC +0.0323129, Macro-F1 +0.0177706, ROC-AUC -0.0429025.
2. ABIDE versus Original: ABIDE ADS_CN: Correct -3, BACC -0.0023435, Macro-F1 -0.0032492, ROC-AUC -0.0078683.
3. ABIDE-5 versus Original: ABIDE-5 ADS_CN: Correct -2, BACC -0.0004423, Macro-F1 -0.0019036, ROC-AUC -0.0106042.
4. Hard-classification gain rather than AUC-only gain: `True`. Repairs/damages are TADPOLE SMCI_PMCI 6/4; ABIDE ADS_CN 41/44; ABIDE-5 ADS_CN 40/42.
5. Majority-class bias audit: TADPOLE SMCI_PMCI majority-pred delta=-4, bias-vs-truth=+0; ABIDE ADS_CN majority-pred delta=-13, bias-vs-truth=-25; ABIDE-5 ADS_CN majority-pred delta=-20, bias-vs-truth=-10. Preregistered ABIDE imbalance flags={"abide5_ads_cn": false, "abide_ads_cn": false}, TAD collapse=False.
6. Modalities with activated private residual norm >=1e-6: TADPOLE SMCI_PMCI=MRI,CSF,RISK_FACTOR,COGNITIVE_TEST,ROI_AVERAGE; ABIDE ADS_CN=PHENO,ANAT,FUNC,Correlation; ABIDE-5 ADS_CN=PHENO,ANAT,FUNC,MRI,fMRI. Every adapter tensor's maximum gradient is listed above and in mechanism_diagnostics.json.
7. Parameter increases: TADPOLE SMCI_PMCI +8200 (1.346%); ABIDE ADS_CN +4384 (1.316%); ABIDE-5 ADS_CN +5480 (1.439%).
8. All three tasks satisfy CROSS_DATASET_GO: `False`; final decision is `CROSS_DATASET_MIXED`.
9. Continue replacing direct Category + Global addition: `False`. Only a GO records SP-LRIF-A012, and this run does not implement it.
10. If fusion work is not justified: Private residuals trained and activated on every fold, so a failure to pass GO is more consistent with dataset/task-specific utility than with a disabled private mechanism.

## Reproduction

The committed runner intentionally accepts only the locked source commit as executable training source. After the result commit is published, reproduce in a fresh clone and point the local experiment branch at the source commit below; running `formal` directly from the result commit is expected to be rejected by the source-scope safety gate. The locked legacy provenance artifacts are local rather than Git-tracked, so all three commands explicitly reuse the verified historical evidence root from this run.

```text
git clone https://github.com/cxtaidensmith-eng/work22.git work22-cross-dataset-repro
cd work22-cross-dataset-repro
git switch -C experiment/cross-dataset-a012-structure-v1 17004d29714471b5e1548e04bac0072e854e7e18
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts/run_cross_dataset_a012_structure_v1.py inspect --historical-root "D:\Work\WORK2 final\WORK2 final26.7,21"
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts/run_cross_dataset_a012_structure_v1.py smoke --device cuda:0 --historical-root "D:\Work\WORK2 final\WORK2 final26.7,21"
& 'D:\Anaconda\envs\work22-tabpfn-v1\python.exe' -u -B scripts/run_cross_dataset_a012_structure_v1.py formal --device cuda:0 --historical-root "D:\Work\WORK2 final\WORK2 final26.7,21"
```
