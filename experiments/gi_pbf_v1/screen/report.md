# GI-PBF v1 Screen Report

Decision: **SCREEN_STOP**

- Actual C1 base commit: `90326eef6ab1a8a4111e71a33280f8f7113c1ea7`
- Corrected: 226/240
- ACC / Macro-F1 / BACC / AUC / Weighted-F1: 0.9416667 / 0.9222449 / 0.9208829 / 0.9648764 / 0.9417422
- Confusion: [[24, 0, 4], [0, 80, 4], [4, 2, 122]]
- Base trajectory reproduced: True
- Repairs / damages / changed vs C1: 1 / 1 / 2
- AD-sMCI / CN-sMCI / AD-CN errors: 8 / 6 / 0
- Parameters: 866069
- Training seconds: 197.186
- Reproduction: `"D:\Anaconda\envs\work22-tabpfn-v1\python.exe" -u -B scripts/run_gi_pbf_v1.py screen --device cuda:0`

## Fold best epochs

- fold4: base 55@103; corrected 55@101
- fold5: base 57@180; corrected 57@180
- fold6: base 58@332; corrected 58@330
- fold7: base 56@181; corrected 56@181

## Correction diagnostics

{
  "ad_smci": {
    "raw_delta_mean": -1.5234901773743332,
    "raw_delta_abs_mean": 5.459410574877014,
    "raw_delta_abs_max": 19.637174606323242,
    "raw_delta_std": 6.814711476200125,
    "delta_mean": 0.0004415108676766977,
    "delta_abs_mean": 0.015777933171436113,
    "delta_abs_max": 0.22939766943454742,
    "delta_std": 0.027720870193725402,
    "uncertainty_mean": 0.04527755665282408,
    "uncertainty_min": 0.008806470781564713,
    "uncertainty_max": 0.6297826170921326,
    "tanh_saturation_fraction": 0.7625,
    "gradient_max": 0.028207454830408096
  },
  "cn_smci": {
    "raw_delta_mean": -1.5574822250753642,
    "raw_delta_abs_mean": 3.671476253370444,
    "raw_delta_abs_max": 12.683084487915039,
    "raw_delta_std": 4.190697831259654,
    "delta_mean": -0.0019722314051856906,
    "delta_abs_mean": 0.009721072245641456,
    "delta_abs_max": 0.12226054072380066,
    "delta_std": 0.013819971863580648,
    "uncertainty_mean": 0.028607261782356848,
    "uncertainty_min": 0.005818298552185297,
    "uncertainty_max": 0.666802704334259,
    "tanh_saturation_fraction": 0.5791666666666667,
    "gradient_max": 0.07361166179180145
  },
  "input_projection_gradient_max": 0.12863905727863312
}

Next recommendation: MR-LHGR: modality-specific multi-relation low/high-frequency graph residual (not implemented or run).
