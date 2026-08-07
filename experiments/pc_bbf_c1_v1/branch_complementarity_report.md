# PC-BBF v1 Stage-A Category/Global Complementarity

Decision: **BRANCH_COMPLEMENTARITY_GO**

C1 was not retrained and PC-BBF was not implemented in this stage. Each formal C1 checkpoint was loaded strictly, and Y/G are the exact two inputs to the historical Y + G add.

## Checkpoint replay and representation contract

- C1 source HEAD: `90326eef6ab1a8a4111e71a33280f8f7113c1ea7`
- Strict fold loads: `True`
- Predictions matched formal C1 OOF: `True`
- Maximum raw-logit difference: `0.000e+00`
- Maximum probability difference: `0.000e+00`
- Maximum `H_fused - (Y + G)` difference: `0.000e+00`

## Pooled OOF metrics

- C1: Correct=560/598, ACC=0.9364548, Macro-F1=0.9175457, BACC=0.9163359, Probability Macro-AUC=0.9585608, Weighted-F1=0.9363660, Confusion=[[61, 0, 11], [0, 201, 8], [10, 9, 298]]
- Category-only probe: Correct=534/598, ACC=0.8929766, Macro-F1=0.8681681, BACC=0.8574291, Probability Macro-AUC=0.9384635, Weighted-F1=0.8923774, Confusion=[[54, 0, 18], [0, 189, 20], [11, 15, 291]]
- Global-only probe: Correct=497/598, ACC=0.8311037, Macro-F1=0.8170408, BACC=0.8254534, Probability Macro-AUC=0.9211897, Weighted-F1=0.8312349, Confusion=[[57, 0, 15], [0, 182, 27], [19, 40, 258]]

## Complementarity

- Category-only correct / Global wrong: 69
- Global-only correct / Category wrong: 32
- C1 errors fixed by Category: 3
- C1 errors fixed by Global: 11
- C1 errors fixed by either branch: 14
- Oracle(C1, Category, Global): 574/598
- Category/Global prediction disagreement: 101/598 (16.890%)

## Decision criteria

- oracle_at_least_565: `True`
- category_has_at_least_3_exclusive_correct: `True`
- global_has_at_least_3_exclusive_correct: `True`
- at_least_5_c1_errors_fixed_by_either: `True`

If the decision is `BRANCH_COMPLEMENTARITY_STOP`, PC-BBF must not be implemented or trained; the prescribed next recommendation is MR-HGR-C1 multi-relation graph.
