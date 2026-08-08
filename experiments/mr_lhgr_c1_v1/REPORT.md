# MR-LHGR-C1 v1

Decision: **MR_LHGR_NO_GAIN**
Selected version/cap: **v1 / 0.10**
Parameters: 873723 (10752 graph); total training time 576.938s
Pooled Correct/ACC/F1/BACC/AUC/Weighted-F1: 558/598 / 0.9331104 / 0.9175701 / 0.9054466 / 0.9609350 / 0.9326418
Confusion matrix: [[59, 0, 13], [0, 198, 11], [5, 11, 301]]
Ten-fold ACC mean +/- sample SD: 0.9331638 +/- 0.0247638
Delta vs Original/C1/PC-BBF ACC: +0.0033445 / -0.0033445 / -0.0050167
Repairs/damages/changed vs C1: 14 / 16 / 30
AD-sMCI / CN-sMCI / AD-CN errors: 18 / 22 / 0
Cap saturation: 0.7842810; R_graph/H0 mean/max: 0.0894498/0.1000000
Edge dependence ratio/logit max/prob mean/argmax changed: 1.0000000 / 0.9482386 / 0.002957073 / 3
Edge Jaccard mean/min/max: 0.0267794 / 0.0104547 / 0.0919369
v1.1 run: False ({'correct_is_561_or_562': False, 'repairs_above_damages': False, 'adjacent_boundary_total_below_c1_38': False, 'no_ad_cn_error': True, 'cap_saturation_at_least_0p70': True, 'graph_relation_dependence_valid': True})

## Per-modality low/high contribution

- MRI: degree 14.097; low/high RMS 0.420807/0.420807; delta norms 5.678707/7.254562
- PET: degree 13.395; low/high RMS 0.409751/0.409751; delta norms 4.295342/5.268249
- CSF: degree 10.120; low/high RMS 0.430561/0.430561; delta norms 4.789138/4.997907
- Risk: degree 10.244; low/high RMS 0.427477/0.427477; delta norms 4.673799/4.925250
- COG: degree 12.619; low/high RMS 0.306436/0.306436; delta norms 3.593725/4.130902
- ROI: degree 11.154; low/high RMS 0.389199/0.389199; delta norms 4.895439/5.407979

Target not reached; stop this experiment without another cap or model.
