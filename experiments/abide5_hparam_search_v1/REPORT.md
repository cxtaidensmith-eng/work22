# ABIDE-5 Binary Task Hyperparameter Search v1

- Source commit: `35bcf76af36719b96ae81ec2f5015256ee43f728`
- Result commit: reported in final Git handoff
- Branch: `experiment/abide5-hparam-search-v1`
- Device: `cuda:0`
- Decision: **ABIDE5_PRIVATE_POSITIVE**
- Private status: **ABIDE5_PRIVATE_SUPPORTED**
- Retained model: `D3` (B1)
- Formal aggregation: 26.170 seconds

## Locked protocol

ABIDE-5 ADS_CN only: 864 subjects (397 ADS/ASD positive index 0, 467 CN negative index 1), five real modalities, seed 0, ten full-batch transductive folds, 400 epochs, historical global class weights, criterion_lossv2 main weighted CE plus two unnormalized weighted OVR CE terms, label smoothing 0.05, orthogonality 0, Adam, grad clip 1, historical cosine scheduler, graph/EMA/ensemble off, and test-fold best selection by ACC > ROC-AUC > Macro-F1 > earliest.

## Selected configurations

- TUNED_B0: `A2`, lr=0.005, WD=0.0005, dropout=0.45.
- TUNED_B1: `D3`, lr=0.005, WD=0.0005, dropout=0.45, rank=4, multiplier=2.0.
- LOCAL triggered: `False` (Current best Correct <= 782).

## Results

| Model | Correct/N | ACC | Fold ACC mean +/- SD | ROC-AUC | Fold AUC mean +/- SD | PR-AUC | Macro-F1 | BACC | Weighted-F1 | ASD SEN | CN SPE | Confusion [ADS,CN] | Pred ADS/CN | Params | Train sec | Infer sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Historical B0 | 768/864 | 0.8888889 | 0.8889735 +/- 0.0534292 | 0.9066931 | 0.9011509 +/- 0.0479674 | 0.8906037 | 0.8879281 | 0.8872108 | 0.8887688 | 0.8664987 | 0.9079229 | [[344, 53], [43, 424]] | 387/477 | 380716 | 1131.7 | 0.1952 |
| Historical B1 | 766/864 | 0.8865741 | 0.8867014 +/- 0.0451331 | 0.8960890 | 0.9082004 +/- 0.0546506 | 0.8701144 | 0.8860244 | 0.8867685 | 0.8866657 | 0.8891688 | 0.8843683 | [[353, 44], [54, 413]] | 407/457 | 386196 | 1252.5 | 0.1942 |
| TUNED_B0 | 768/864 | 0.8888889 | 0.8889735 +/- 0.0534292 | 0.9066931 | 0.9011509 +/- 0.0479674 | 0.8906037 | 0.8879281 | 0.8872108 | 0.8887688 | 0.8664987 | 0.9079229 | [[344, 53], [43, 424]] | 387/477 | 380716 | 1131.7 | 0.1952 |
| TUNED_B1 | 769/864 | 0.8900463 | 0.8901363 +/- 0.0394193 | 0.8862642 | 0.8897475 +/- 0.0463411 | 0.8505650 | 0.8893405 | 0.8894142 | 0.8900565 | 0.8816121 | 0.8972163 | [[350, 47], [48, 419]] | 398/466 | 383616 | 516.6 | 0.1568 |

## Registered trials

| Trial | Stage | Arm | LR | WD | Dropout | Rank | Mult | Safe | Correct | AUC | F1 | BACC |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| A1 | A | B0 | 0.00375 | 0.0005 | 0.45 | 0 | 0 | False | 761 | 0.8782005 | 0.8799299 | 0.8797162 |
| A2 | A | B0 | 0.005 | 0.0005 | 0.45 | 0 | 0 | True | 768 | 0.9066931 | 0.8879281 | 0.8872108 |
| A3 | A | B0 | 0.00625 | 0.0005 | 0.45 | 0 | 0 | False | 760 | 0.8808893 | 0.8789655 | 0.8794006 |
| A4 | A | B0 | 0.0075 | 0.0005 | 0.45 | 0 | 0 | False | 750 | 0.8709810 | 0.8670805 | 0.8668062 |
| B1 | B | B0 | 0.005 | 0.00025 | 0.45 | 0 | 0 | False | 754 | 0.8734190 | 0.8719828 | 0.8724103 |
| B2 | B | B0 | 0.005 | 0.0005 | 0.45 | 0 | 0 | True | 768 | 0.9066931 | 0.8879281 | 0.8872108 |
| B3 | B | B0 | 0.005 | 0.001 | 0.45 | 0 | 0 | True | 766 | 0.8947783 | 0.8856420 | 0.8850695 |
| B4 | B | B0 | 0.005 | 0.0015 | 0.45 | 0 | 0 | True | 767 | 0.8852367 | 0.8867360 | 0.8859514 |
| C1 | C | B0 | 0.005 | 0.0005 | 0.35 | 0 | 0 | False | 755 | 0.8667253 | 0.8729355 | 0.8727259 |
| C2 | C | B0 | 0.005 | 0.0005 | 0.45 | 0 | 0 | True | 768 | 0.9066931 | 0.8879281 | 0.8872108 |
| C3 | C | B0 | 0.005 | 0.0005 | 0.55 | 0 | 0 | True | 767 | 0.8931143 | 0.8868327 | 0.8863289 |
| D1 | D | B1 | 0.005 | 0.0005 | 0.45 | 4 | 0.5 | True | 766 | 0.8907896 | 0.8858671 | 0.8860134 |
| D2 | D | B1 | 0.005 | 0.0005 | 0.45 | 4 | 1 | True | 763 | 0.8860889 | 0.8823071 | 0.8822351 |
| D3 | D | B1 | 0.005 | 0.0005 | 0.45 | 4 | 2 | True | 769 | 0.8862642 | 0.8893405 | 0.8894142 |
| D4 | D | B1 | 0.005 | 0.0005 | 0.45 | 8 | 0.5 | True | 758 | 0.8875614 | 0.8764569 | 0.8763154 |
| D5 | D | B1 | 0.005 | 0.0005 | 0.45 | 8 | 1 | True | 767 | 0.8887157 | 0.8868327 | 0.8863289 |
| D6 | D | B1 | 0.005 | 0.0005 | 0.45 | 8 | 2 | True | 766 | 0.8960890 | 0.8860244 | 0.8867685 |

## Pairwise comparisons

- TUNED_B0 vs Historical B0: repairs/damages/changed=0/0/0, exact McNemar p=1.00000000.
- TUNED_B1 vs Historical B1: repairs/damages/changed=41/38/79, exact McNemar p=0.82214424.
- TUNED_B1 vs TUNED_B0: repairs/damages/changed=40/39/79, exact McNemar p=1.00000000.

## DGFMC target

- Correct >=787: `False`.
- Retained fold mean ACC >91.05%: `False`.
- Retained fold mean ROC-AUC >90.99%: `False`.

## Private mechanism

- TUNED_B1 mechanism: `{"adapter_lr_multiplier": 2.0, "adapter_max_gradient_by_tensor": {"private_adapters.0.down.bias": 0.0012060124427080154, "private_adapters.0.down.weight": 0.0028228273149579763, "private_adapters.0.up.bias": 0.019781889393925667, "private_adapters.0.up.weight": 0.005173949524760246, "private_adapters.1.down.bias": 0.0007912796572782099, "private_adapters.1.down.weight": 0.0016891093691810966, "private_adapters.1.up.bias": 0.01766117289662361, "private_adapters.1.up.weight": 0.004971039481461048, "private_adapters.2.down.bias": 0.0012883159797638655, "private_adapters.2.down.weight": 0.003253018483519554, "private_adapters.2.up.bias": 0.016848500818014145, "private_adapters.2.up.weight": 0.007004443556070328, "private_adapters.3.down.bias": 0.0007951558800414205, "private_adapters.3.down.weight": 0.0017904011765494943, "private_adapters.3.up.bias": 0.028108445927500725, "private_adapters.3.up.weight": 0.002149460604414344, "private_adapters.4.down.bias": 0.0005865329876542091, "private_adapters.4.down.weight": 0.0027467363979667425, "private_adapters.4.up.bias": 0.02298160456120968, "private_adapters.4.up.weight": 0.002469930797815323}, "adapter_rank": 4, "category_global_cosine_mean": 0.1345327191054821, "parameter_count_b0": 380716, "parameter_count_b1": 383616, "parameter_delta": 2900, "private_collapse_any_fold": false, "private_shared_ratio_max": 0.047280263155698776, "private_shared_ratio_max_by_modality": {"ANAT": 0.02707696706056595, "FUNC": 0.03783052787184715, "MRI": 0.03528645262122154, "PHENO": 0.034203801304101944, "fMRI": 0.047280263155698776}, "private_shared_ratio_mean": 0.006108896341174841, "private_shared_ratio_mean_by_modality": {"ANAT": 0.005308685079216957, "FUNC": 0.005879081226885319, "MRI": 0.00527657640632242, "PHENO": 0.006111053912900388, "fMRI": 0.007969085569493472}, "private_trained_all_folds": true}`.

## Conclusion

- B0 improved over history: `False`.
- Private residual adds hard-classification gain over TUNED_B0: `True`.
- Final Decision: `ABIDE5_PRIVATE_POSITIVE`; private status `ABIDE5_PRIVATE_SUPPORTED`.

## Artifact semantics

- ADS/ASD is positive class index 0. The exported `positive_probability` column is therefore an exact copy of `probability_0`.
- A task-profile export correction changed only this redundant CSV column; `probability_0`, `probability_1`, predictions, checkpoints, metrics, comparisons, and model selection are unchanged.

## Reproduction

Run a clean local branch named `experiment/abide5-hparam-search-v1` at corrected source commit `111f5dcd8350729bfbd227f8116ce158a259bbea`:

```text
python -u -B scripts/run_abide5_hparam_search_v1.py inspect
python -u -B scripts/run_abide5_hparam_search_v1.py smoke --device cuda:0
python -u -B scripts/run_abide5_hparam_search_v1.py search --device cuda:0
python -u -B scripts/run_abide5_hparam_search_v1.py formal --device cuda:0
```
