# BP-ADS v1 formal report

## Protocol

Source commit: `371df39117b4d56f3277b9101bec1d95d41e221e`. Dataset=TADPOLE AD_CN_SMCI; folds=0..9; seed=0; 400 epochs/fold; full-batch transductive; graph disabled.
All formal candidates are independently inferable single models (`single_model=True`, `ensemble=False`). Test labels were excluded from specialist loss and inputs.

## Stage A — artifact alignment and complementarity

All 598 subject IDs were unique and aligned explicitly by `subject_index`; labels and folds matched. C1 and A012 do not fail on exactly the same subjects: both correct=542, C1-only correct=18, A012-only correct=20, both wrong=18, changed predictions=38.

## Stage B — boundary feasibility

- B1_c1_protect_cn_a012_ads: Correct=564/598, ACC=0.9431438, Macro-F1=0.9294628, BACC=0.9312763, Probability Macro-AUC=0.9630258, Weighted-F1=0.9431858; confusion=[[64, 0, 8], [0, 201, 8], [9, 9, 299]]; AD–sMCI=17, CN–sMCI=17, AD–CN=0; predicted={'AD': 73, 'CN': 210, 'SMCI': 315}; vs C1 repairs/damages/changed=9/5/14; gate=PASS.
- B2_c1_protect_cn_pc_ads: Correct=564/598, ACC=0.9431438, Macro-F1=0.9279445, BACC=0.9205420, Probability Macro-AUC=0.9669302, Weighted-F1=0.9428447; confusion=[[61, 0, 11], [0, 201, 8], [6, 9, 302]]; AD–sMCI=17, CN–sMCI=17, AD–CN=0; predicted={'AD': 67, 'CN': 210, 'SMCI': 321}; vs C1 repairs/damages/changed=11/7/18; gate=FAIL.
- B3_a012_protect_ad_c1_cn_smci: Correct=565/598, ACC=0.9448161, Macro-F1=0.9331138, BACC=0.9323279, Probability Macro-AUC=0.9630550, Weighted-F1=0.9447569; confusion=[[64, 1, 7], [0, 201, 8], [7, 10, 300]]; AD–sMCI=14, CN–sMCI=18, AD–CN=1; predicted={'AD': 71, 'CN': 212, 'SMCI': 315}; vs C1 repairs/damages/changed=8/3/12; gate=FAIL.

Selected direction: `B1_c1_protect_cn_a012_ads`. C1-protected CN + A012 AD/sMCI reached 564/598; PC-BBF substitution reached 564/598.

## Stage C — model soup

- S1_probability_only (c1+a012): probability feasibility only, Correct=561/598, ACC=0.9381271, Macro-F1=0.9259566, BACC=0.9306131, Probability Macro-AUC=0.9761066, Weighted-F1=0.9382701; confusion=[[65, 0, 7], [0, 199, 10], [10, 10, 297]]; AD–sMCI=17, CN–sMCI=20, AD–CN=0; predicted={'AD': 75, 'CN': 209, 'SMCI': 314}; exact checkpoint rerun=forbidden.
- S2_probability_only (a012+a007): probability feasibility only, Correct=556/598, ACC=0.9297659, Macro-F1=0.9135093, BACC=0.9110431, Probability Macro-AUC=0.9763225, Weighted-F1=0.9296180; confusion=[[61, 0, 11], [0, 199, 10], [9, 12, 296]]; AD–sMCI=20, CN–sMCI=22, AD–CN=0; predicted={'AD': 70, 'CN': 211, 'SMCI': 317}; exact checkpoint rerun=forbidden.
- S3_probability_only (c1+a012+a007): probability feasibility only, Correct=569/598, ACC=0.9515050, Macro-F1=0.9360790, BACC=0.9376207, Probability Macro-AUC=0.9823766, Weighted-F1=0.9515647; confusion=[[64, 0, 8], [0, 203, 6], [9, 6, 302]]; AD–sMCI=17, CN–sMCI=12, AD–CN=0; predicted={'AD': 73, 'CN': 209, 'SMCI': 316}; exact checkpoint rerun=allowed.

Formal S3 weight soup: Correct=502/598, ACC=0.8394649, Macro-F1=0.8311202, BACC=0.8466536, Probability Macro-AUC=0.9401724, Weighted-F1=0.8396308; confusion=[[61, 0, 11], [0, 185, 24], [19, 42, 256]]; AD–sMCI=30, CN–sMCI=66, AD–CN=0; predicted={'AD': 80, 'CN': 227, 'SMCI': 291}; vs C1 repairs/damages/changed=11/69/80; retained=False. It is one averaged state dict, not a probability ensemble.

## Stage D — frozen-C1 boundary specialist

Correct=560/598, ACC=0.9364548, Macro-F1=0.9175457, BACC=0.9163359, Probability Macro-AUC=0.9592249, Weighted-F1=0.9363660; confusion=[[61, 0, 11], [0, 201, 8], [10, 9, 298]]; AD–sMCI=21, CN–sMCI=17, AD–CN=0; predicted={'AD': 71, 'CN': 210, 'SMCI': 317}; vs C1 repairs/damages/changed=0/0/0; vs A012=18/20/38; retained=False.
Fold ACC=0.9364689 ± 0.0171698 sample SD; training time=89.6s; maximum expert gradient=0.05604036.
The model has 862971 frozen C1 parameters + 3089 expert parameters = 866060 total. C1-protected CN prediction changes=0; internal max probability difference=0.0; external CSV max difference=0.

## Required questions

1. C1 and A012 complementarity is not confined to the same subjects: C1 uniquely repairs 18 and A012 uniquely repairs 20; 18 are wrong in both.
2. C1-protected CN + A012 AD/sMCI reached 564/598 (gate PASS); the PC specialist reached 564/598 (gate FAIL).
3. The formal weight soup did not outperform A012 in Correct (502 vs 562).
4. The trained specialist reached 560/598 versus the zero-training splice upper-bound 564/598; it did not fully reproduce that gain.
5. CN predictions were strictly protected: 0 protected prediction changes and internal probability max difference 0.0.
6. Relative to C1, formal BP-ADS changed AD–sMCI errors from 21 to 21 and produced 0 AD–CN errors.
7. Yes. Both formal candidates are single models with `ensemble=False`; BP-ADS is one frozen C1 backbone plus one 3,089-parameter expert.
8. Final selection: `A012` (locked_reference); Correct=562/598, ACC=0.9397993, Macro-F1=0.9294205, BACC=0.9286299, Probability Macro-AUC=0.9615248, Weighted-F1=0.9397415; confusion=[[64, 0, 8], [0, 200, 9], [7, 12, 298]]; AD–sMCI=15, CN–sMCI=21, AD–CN=0; predicted={'AD': 71, 'CN': 212, 'SMCI': 315}.

## Decision

`STOP_STATIC_ROUTE`

锁定A012为当前最佳静态横断面模型，不再扩大静态网络与超参数搜索；后续提升需要纵向访视、转换时间或新的生物标志物监督。

## Reproduction

```text
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_bp_ads_v1.py smoke --device cuda:0
"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_bp_ads_v1.py formal --device cuda:0
```
