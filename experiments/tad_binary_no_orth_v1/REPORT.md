# TADPOLE Binary Orthogonality Loss Removal v1

- Decision: **NO_ORTH_NO_GAIN**
- Branch: experiment/tad-binary-no-orth-v1
- Source commit: afd98765a99bc22bde19bbcfab13460cfc082c51
- Result commit: reported in the final Git handoff.
- Device: cuda:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU)
- Protocol: D5 except orthogonality_rate=0.0; seed 0, ten folds, 400 epochs, full-batch transductive, single model.
- Loss: main weighted CE + two unnormalized weighted OVR CE terms. Raw orthogonality is diagnostic-only and never enters an optimizer step.
- Confusion order: [[TN, FP], [FN, TP]], pMCI positive.

## Pooled OOF result

| Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC | Weighted-F1 | pMCI SEN | sMCI SPE | Confusion | Predicted sMCI/pMCI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 511/535 | 0.9551402 | 0.9108390 | 0.6212843 | 0.8544218 | 0.8544218 | 0.9551402 | 0.7333333 | 0.9755102 | [[478, 12], [12, 33]] | 490/45 |

- TP/FN=33/12; TN/FP=478/12.
- Fold ACC mean +/- sample SD: 0.9551712 +/- 0.0199335.
- Fold ROC-AUC mean +/- sample SD: 0.9698980 +/- 0.0420787.
- Parameters total/trainable: 617197/617197.
- Training/inference seconds: 540.51/0.1288.

## Folds

| Fold | Best epoch | Correct | ACC | ROC-AUC | PR-AUC | Macro-F1 | BACC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12 | 52 | 0.9629630 | 0.9755102 | 0.8211111 | 0.8897959 | 0.8897959 |
| 1 | 30 | 50 | 0.9259259 | 0.8571429 | 0.3811111 | 0.7795918 | 0.7795918 |
| 2 | 32 | 50 | 0.9259259 | 0.9836735 | 0.9111111 | 0.8358663 | 0.9591837 |
| 3 | 49 | 52 | 0.9629630 | 0.9795918 | 0.8583333 | 0.8650000 | 0.8000000 |
| 4 | 17 | 53 | 0.9814815 | 1.0000000 | 1.0000000 | 0.9493908 | 0.9897959 |
| 5 | 10 | 51 | 0.9622642 | 1.0000000 | 1.0000000 | 0.8895833 | 0.9795918 |
| 6 | 14 | 51 | 0.9622642 | 0.9846939 | 0.8541667 | 0.8233333 | 0.7500000 |
| 7 | 38 | 50 | 0.9433962 | 0.9693878 | 0.6083333 | 0.7705628 | 0.7397959 |
| 8 | 113 | 52 | 0.9811321 | 0.9948980 | 0.9500000 | 0.9235209 | 0.8750000 |
| 9 | 161 | 50 | 0.9433962 | 0.9540816 | 0.7708333 | 0.7705628 | 0.7397959 |

## Paired with D5

- Repairs/damages/changed: 6/14/20; Correct delta=-8.
- Metric deltas: {"acc": -0.01495327102803734, "bacc": -0.03843537414965992, "macro_f1": -0.046522500872224426, "pr_auc": -0.11088046082974112, "roc_auc": -0.007437641723356037, "sen": -0.06666666666666676, "spe": -0.010204081632653073, "weighted_f1": -0.014643878686362966}.
- Exact two-sided McNemar p: 0.1153183.
- Repair sources: {"PMCI": 3, "SMCI": 3}.
- Damage sources: {"PMCI": 6, "SMCI": 8}.
- Prediction transitions: {"PMCI->SMCI": 9, "SMCI->PMCI": 11}.

## Loss diagnostic

- Old representation shapes: [[535, 96], [535, 96]]; similarity matrix: [535, 535].
- Raw/weighted old orthogonality: 3958.3466797/0.3958347.
- Weighted orth/supervised loss ratio: 0.1638566.
- Weighted orth/supervised gradient-norm ratio: 1.0419483.
- Formal no-orth maximum absolute contribution: 0.0.
- Per-fold first/best/last main CE, OVR0 and OVR1: `{"0": {"best": {"main_ce": 0.6374474763870239, "ovr_0": 0.6639692783355713, "ovr_1": 0.5323264002799988}, "first": {"main_ce": 0.8868284821510315, "ovr_0": 0.7560681104660034, "ovr_1": 0.7728416323661804}, "last": {"main_ce": 0.34336674213409424, "ovr_0": 0.3387318551540375, "ovr_1": 0.33848124742507935}}, "1": {"best": {"main_ce": 0.3987252116203308, "ovr_0": 0.4915729761123657, "ovr_1": 0.4740174114704132}, "first": {"main_ce": 0.8876925110816956, "ovr_0": 0.7593425512313843, "ovr_1": 0.768448531627655}, "last": {"main_ce": 0.33969855308532715, "ovr_0": 0.3377343416213989, "ovr_1": 0.3378596007823944}}, "2": {"best": {"main_ce": 0.4130117893218994, "ovr_0": 0.4187023937702179, "ovr_1": 0.4387318193912506}, "first": {"main_ce": 0.9027694463729858, "ovr_0": 0.7498905658721924, "ovr_1": 0.7728492617607117}, "last": {"main_ce": 0.3394927680492401, "ovr_0": 0.33773910999298096, "ovr_1": 0.3377368748188019}}, "3": {"best": {"main_ce": 0.42625534534454346, "ovr_0": 0.4906379282474518, "ovr_1": 0.545788049697876}, "first": {"main_ce": 0.8952630162239075, "ovr_0": 0.7508591413497925, "ovr_1": 0.769063413143158}, "last": {"main_ce": 0.3402503728866577, "ovr_0": 0.33771052956581116, "ovr_1": 0.3377203941345215}}, "4": {"best": {"main_ce": 0.49722519516944885, "ovr_0": 0.5175710320472717, "ovr_1": 0.5105092525482178}, "first": {"main_ce": 0.8992965221405029, "ovr_0": 0.7535305619239807, "ovr_1": 0.7677280306816101}, "last": {"main_ce": 0.3391820788383484, "ovr_0": 0.3377074897289276, "ovr_1": 0.3380137085914612}}, "5": {"best": {"main_ce": 0.5307317972183228, "ovr_0": 0.8640413880348206, "ovr_1": 0.7801723480224609}, "first": {"main_ce": 0.8921602368354797, "ovr_0": 0.7533602118492126, "ovr_1": 0.7602238059043884}, "last": {"main_ce": 0.3358592987060547, "ovr_0": 0.33500435948371887, "ovr_1": 0.3340529799461365}}, "6": {"best": {"main_ce": 0.49235039949417114, "ovr_0": 0.6613479852676392, "ovr_1": 0.7142871022224426}, "first": {"main_ce": 0.89711993932724, "ovr_0": 0.7547906041145325, "ovr_1": 0.768186092376709}, "last": {"main_ce": 0.3454430103302002, "ovr_0": 0.33960962295532227, "ovr_1": 0.3414267599582672}}, "7": {"best": {"main_ce": 0.42553919553756714, "ovr_0": 0.4723204970359802, "ovr_1": 0.48258909583091736}, "first": {"main_ce": 0.89057856798172, "ovr_0": 0.7538362741470337, "ovr_1": 0.7637884020805359}, "last": {"main_ce": 0.33563750982284546, "ovr_0": 0.3337877094745636, "ovr_1": 0.3339000344276428}}, "8": {"best": {"main_ce": 0.3523947596549988, "ovr_0": 0.35208237171173096, "ovr_1": 0.35263824462890625}, "first": {"main_ce": 0.8749922513961792, "ovr_0": 0.7534346580505371, "ovr_1": 0.7705280780792236}, "last": {"main_ce": 0.34623706340789795, "ovr_0": 0.33990806341171265, "ovr_1": 0.3431035280227661}}, "9": {"best": {"main_ce": 0.36379581689834595, "ovr_0": 0.366122841835022, "ovr_1": 0.36566638946533203}, "first": {"main_ce": 0.8891076445579529, "ovr_0": 0.7585312128067017, "ovr_1": 0.7647800445556641}, "last": {"main_ce": 0.3355315327644348, "ovr_0": 0.33396726846694946, "ovr_1": 0.3345709443092346}}}`.
- NaN or gradient abnormality: false.
- Descriptive targets: >=520=False; >=523=False; fold mean ACC>97.57%=False; fold mean ROC-AUC>90.49%=True.

## Required answers

1. The old orthogonality produces an N by N matrix: **yes**, 535 by 535.
2. It contains cross-subject terms: **yes**, including off-diagonal subject pairs.
3. It contains test rows: **yes**; the raw term is computed from full-dataset embeddings and receives no train mask.
4. Old weighted orth/supervised scale: loss ratio=0.1638566, gradient-norm ratio=1.0419483.
5. Parameters and forward at step 0 are identical to D5: **yes**; max logit difference=0.0.
6. NO_ORTH Correct: **511/535**.
7. It exceeds D5 519/535: **False**.
8. Repairs exceed damages: **False** (6/14).
9. pMCI TP/SEN/PR-AUC: 33/0.7333333/0.6212843; registered preservation gates=False/False/False.
10. It reaches 523/535: **False**.
11. Retain NO_ORTH as the new reference: **False**.
12. Allow the next pre-shared source residual experiment: **False**; it is not executed here.
13. Final Decision: **NO_ORTH_NO_GAIN**.
14. Source=afd98765a99bc22bde19bbcfab13460cfc082c51; result commit is reported in Git handoff; branch=experiment/tad-binary-no-orth-v1; device=cuda:0; commands are listed below.

## Decision reasons

- correct_at_least_520
- repairs_gt_damages
- bacc_at_least_0_8898571
- roc_auc_at_least_0_9132766
- pr_auc_at_least_0_7221648
- pmci_tp_at_least_36
- pmci_sen_decline_at_most_0_01
- smci_spe_decline_at_most_0_01

Decision conclusion: Complete removal of the legacy orthogonality loss did not improve D5; retain D5 as the performance reference and do not automatically search the orth rate or run a corrected per-subject version.

## Reproduction

Run these commands from a clean worktree whose same-named branch points exactly to source commit `afd98765a99bc22bde19bbcfab13460cfc082c51`; the result commit is for reading artifacts and intentionally fails the source-scope gate.
    python -u -B scripts/run_tad_binary_no_orth_v1.py inspect
    python -u -B scripts/run_tad_binary_no_orth_v1.py smoke --device cuda:0
    python -u -B scripts/run_tad_binary_no_orth_v1.py formal --device cuda:0

No orth-rate search, corrected orthogonality, second version, other task, or v1.1 was run.
