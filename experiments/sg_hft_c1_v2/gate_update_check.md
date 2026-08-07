# SG-HFT-C1 v2 Gate Parameter Update Check

- Decision: `BUG_FIXED_BUT_BRANCH_INACTIVE`
- Branch: `experiment/sg-hft-c1-v2-gatecheck`
- Run commit: `b380b49d98856b94b10533d1093b0566ded2cafd`
- Runtime: `7.167 s`
- Checkpoint SHA256: `a08a3296abe20bcac93a47c6640af982146f5df301234085b6b6d07f92ae5371`

## Required summary

- Optimizer coverage: `1.000000`
- Checkpoint SG-HFT key coverage: `1.000000`
- Active-step optimizer-state coverage: `1.000000`
- Manual pooled/logit diff: `9.09494701773e-13` / `0`
- Root cause: `optimizer-driven branch collapse from zero-alpha warmup plus coupled L2 decay`
- Action: `minimal fix passed but micro-training gate failed; stopped before screen`

## Checkpoint family deltas

| Modality | Family | max abs delta | L2 delta | relative L2 delta |
|---|---|---:|---:|---:|
| MRI | group_encoder | 0.357232958078 | 6.96594285965 | 0.999998699401 |
| MRI | gate | 0.331708371639 | 1.77316582203 | 1.00000033615 |
| MRI | adapter | 0.101773522794 | 1.19513070583 | 1.0127720311 |
| PET | group_encoder | 0.353488564491 | 7.36896848679 | 0.999999870582 |
| PET | gate | 0.333971142769 | 1.68921399117 | 0.999999929429 |
| PET | adapter | 0.101840376854 | 1.18349039555 | 1.00728763465 |
| ROI | group_encoder | 0.955356955528 | 7.69619035721 | 0.997310673197 |
| ROI | gate | 0.252624124289 | 1.70479404926 | 0.999999580444 |
| ROI | adapter | 0.101490415633 | 1.2045699358 | 1.01616423448 |

## Active-step family gradients and updates

| Modality | Family | gradient norm | gradient finite | max abs update | relative L2 update |
|---|---|---:|---|---:|---:|
| MRI | group_encoder | 3.30883995048e-05 | True | 0.00999945402145 | 0.0641390235312 |
| MRI | gate | 1.28862473048e-06 | True | 0.00999939441681 | 0.221337476204 |
| MRI | adapter | 0.0105846002698 | True | 0.00999996997416 | 0.24929225546 |
| PET | group_encoder | 7.58245878387e-05 | True | 0.00999942421913 | 0.0623219132981 |
| PET | gate | 1.61461878179e-06 | True | 0.00999939441681 | 0.232305874565 |
| PET | adapter | 0.0113779203966 | True | 0.00999997183681 | 0.250227500784 |
| ROI | group_encoder | 4.85897253384e-05 | True | 0.00999981164932 | 0.0404698445105 |
| ROI | gate | 1.39471057992e-06 | True | 0.00999920070171 | 0.230058447554 |
| ROI | adapter | 0.00717811845243 | True | 0.00999995134771 | 0.248043605299 |

## Checkpoint gate outputs

- MRI: raw-logit group std=`0`, gate group std=`0`
  - cortical_volume_CV: raw=`0.0000002384`, sigmoid=`0.5000000596`
  - surface_area_SA: raw=`0.0000002384`, sigmoid=`0.5000000596`
  - cortical_thickness_TA_TS: raw=`0.0000002384`, sigmoid=`0.5000000596`
  - subcortical_volume_SV: raw=`0.0000002384`, sigmoid=`0.5000000596`
- PET: raw-logit group std=`0`, gate group std=`0`
  - cortical_left_uptake: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - cortical_left_size: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - cortical_right_uptake: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - cortical_right_size: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - noncortical_aggregate_uptake: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - noncortical_aggregate_size: raw=`0.0000000000`, sigmoid=`0.5000000000`
- ROI: raw-logit group std=`0`, gate group std=`0`
  - medial_temporal_structure: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - global_structure: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - ventricular_structure: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - amyloid_AV45: raw=`0.0000000000`, sigmoid=`0.5000000000`
  - glucose_metabolism_FDG: raw=`0.0000000000`, sigmoid=`0.5000000000`

## Manual PET gate perturbation

- Selected raw-logit diff: `0.10000000149`
- Selected sigmoid-gate diff: `0.0249791145325`
- Normalized-weight diff: `0.00688134133816`
- Checkpoint PET token absolute max: `6.41368933429e-06`
- Checkpoint PET token cross-group max deviation: `7.45785655454e-11`
- Checkpoint pooled/delta/logit diff: `9.09494701773e-13` / `0` / `0`
- Fresh-init pooled/delta/logit control: `0.00769528746605` / `4.00783028454e-05` / `3.93390655518e-06`

## Root cause and action boundary

The scorer hook changes the selected raw gate logit and its normalized weight, but the checkpoint PET group tokens are numerically indistinguishable. The failure is therefore not a hook error or gate-normalization cancellation. It is an optimizer-driven branch collapse caused by zero-alpha warmup combined with coupled L2 decay.

The minimal repair preserves the alpha schedule and optimizer configuration while keeping disabled SG parameters outside the loss graph. This prevents coupled weight decay from updating them during alpha-zero warmup.

## Minimal-fix validation

- Epoch-1 SG gradients None: `60/60`
- Epoch-1 SG parameters bitwise unchanged: `60/60`
- Twenty-step continuous 9-family gradients: `False`
- Modalities passing gate deviation/std/ratio thresholds: `0` / `0` / `0`
- Post-micro PET logit perturbation diff: `4.76837158203e-07`
- Micro-training passed: `False`

No checkpoint was saved and this script did not start the difficult-fold screen or any 400-epoch training.

The PET perturbation was injected into the actual scorer output by a temporary forward hook and removed immediately after the diagnostic forward.

## Post-micro mechanism

| Modality | gate max deviation | gate group std | mean ratio | max ratio | finite |
|---|---:|---:|---:|---:|---|
| MRI | 0.000675082206726 | 5.87134127272e-06 | 0.00301240081899 | 0.00355750741437 | True |
| PET | 0.000815391540527 | 4.30757836511e-06 | 0.00284060882404 | 0.00330547709018 | True |
| ROI | 5.42402267456e-05 | 2.33082428167e-05 | 0.00293757161126 | 0.00338876526803 | True |
