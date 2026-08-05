# Class-Private Evidence Reader and Graph v1 — 10-Fold Final Report

## Executive result

The fixed selection rule chose **R**. 全部低于Original：本路线当前不能替代Original。

All three arms used the locked TADPOLE 598×360 data, fold0..9, seed 0, 400 epochs, full-batch transductive training, one fresh model/criterion/Adam/scheduler per fold, and the unchanged SEPS-Q loss and historical best-epoch ordering. Original Query, SEPS-Q, and TabPFN were not rerun.

## Lineage and locked protocol

- Base: `origin/experiment/query-pool-component-sharing-v1` at `765720c1c0a3b2263502e9bf662fe33de7e45428`
- Branch: `experiment/class-private-evidence-graph-v1`
- Worktree: `D:\Work\WORK2 final\WORK2 final26.7,21\tmp\class_private_evidence_graph_v1`
- Feature CSV SHA256: `2f8efe85c2154d785dc361bc60553c9d983ac1990cee13e38b935b4623787042`
- Modal dictionary SHA256: `5e72aa0b9268b54e3f447059a728eed96696c97f5783184ffb0c615102f90273`
- Fold manifest SHA256: `0f8964a2009a3660d76c99fa9446f9147a51630083a79d11437a31544438c106`
- Fold split assignment SHA256: `8b9d2d49ac6f7100c08a9ca213fa62cb65eb4e725131112b17b7ae4baa16b4e9`
- Old graph path remains disabled: `graph_use_graph=False`, `adj_mode=none`, old label-graph controls zero.
- Best epoch ordering: ACC, historical adjusted-score Macro-AUC, then Macro-F1. All reported AUC values below are probability Macro-AUC (multiclass OVR macro).

## Core results

| Arm | Parameters | ACC | Macro-F1 | BACC | Probability Macro-AUC | Weighted-F1 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R | 785187 | 0.926421 | 0.917393 | 0.915871 | 0.955411 | 0.926425 | 439.41 |
| G | 787761 | 0.911371 | 0.901212 | 0.901742 | 0.955104 | 0.911415 | 480.70 |
| RG | 794697 | 0.914716 | 0.905364 | 0.898863 | 0.953153 | 0.914550 | 503.56 |
| Original Query | 853131 | 0.929766 | 0.914078 | 0.914078 | 0.956049 | 0.929766 | — |
| SEPS-Q | 778251 | 0.924749 | 0.913866 | 0.908206 | 0.953290 | 0.924631 | — |
| TabPFN E4 | — | 0.916388 | 0.905896 | 0.901319 | 0.976419 | 0.916307 | — |

## Fold summaries

### R

| Fold | Best epoch | ACC | Probability Macro-AUC |
|---:|---:|---:|---:|
| 0 | 81 | 0.933333 | 0.934821 |
| 1 | 72 | 0.900000 | 0.955422 |
| 2 | 99 | 0.933333 | 0.969640 |
| 3 | 199 | 0.916667 | 0.980173 |
| 4 | 302 | 0.900000 | 0.908710 |
| 5 | 210 | 0.966667 | 0.992363 |
| 6 | 144 | 0.933333 | 0.980687 |
| 7 | 160 | 0.900000 | 0.961552 |
| 8 | 118 | 0.915254 | 0.971777 |
| 9 | 59 | 0.966102 | 0.975838 |

Mean ± sample SD:

| Metric | Mean ± SD | Pooled OOF |
|---|---:|---:|
| acc | 0.926469 ± 0.025059 | 0.926421 |
| macro_f1 | 0.916618 ± 0.038603 | 0.917393 |
| bacc | 0.915431 ± 0.043601 | 0.915871 |
| probability_macro_auc | 0.963098 ± 0.024848 | 0.955411 |
| weighted_f1 | 0.926359 ± 0.025100 | 0.926425 |

Pooled confusion matrix (rows true, columns predicted; AD/CN/SMCI):

```text
[64, 0, 8]
[0, 192, 17]
[8, 11, 298]
```

### G

| Fold | Best epoch | ACC | Probability Macro-AUC |
|---:|---:|---:|---:|
| 0 | 78 | 0.933333 | 0.941677 |
| 1 | 98 | 0.916667 | 0.933100 |
| 2 | 133 | 0.916667 | 0.982257 |
| 3 | 60 | 0.916667 | 0.990230 |
| 4 | 288 | 0.883333 | 0.949626 |
| 5 | 80 | 0.916667 | 0.960144 |
| 6 | 118 | 0.933333 | 0.984966 |
| 7 | 174 | 0.900000 | 0.952273 |
| 8 | 283 | 0.864407 | 0.941191 |
| 9 | 115 | 0.932203 | 0.970311 |

Mean ± sample SD:

| Metric | Mean ± SD | Pooled OOF |
|---|---:|---:|
| acc | 0.911328 ± 0.022686 | 0.911371 |
| macro_f1 | 0.901313 ± 0.035391 | 0.901212 |
| bacc | 0.902401 ± 0.036730 | 0.901742 |
| probability_macro_auc | 0.960578 ± 0.020320 | 0.955104 |
| weighted_f1 | 0.911276 ± 0.022633 | 0.911415 |

Pooled confusion matrix (rows true, columns predicted; AD/CN/SMCI):

```text
[63, 0, 9]
[0, 190, 19]
[10, 15, 292]
```

### RG

| Fold | Best epoch | ACC | Probability Macro-AUC |
|---:|---:|---:|---:|
| 0 | 144 | 0.916667 | 0.966793 |
| 1 | 281 | 0.900000 | 0.950466 |
| 2 | 172 | 0.900000 | 0.948921 |
| 3 | 191 | 0.916667 | 0.960048 |
| 4 | 112 | 0.900000 | 0.949246 |
| 5 | 226 | 0.950000 | 0.981626 |
| 6 | 133 | 0.933333 | 0.986013 |
| 7 | 114 | 0.883333 | 0.906777 |
| 8 | 190 | 0.898305 | 0.962278 |
| 9 | 57 | 0.949153 | 0.987521 |

Mean ± sample SD:

| Metric | Mean ± SD | Pooled OOF |
|---|---:|---:|
| acc | 0.914746 ± 0.022825 | 0.914716 |
| macro_f1 | 0.904985 ± 0.028812 | 0.905364 |
| bacc | 0.899318 ± 0.045643 | 0.898863 |
| probability_macro_auc | 0.959969 ± 0.023840 | 0.953153 |
| weighted_f1 | 0.914273 ± 0.023165 | 0.914550 |

Pooled confusion matrix (rows true, columns predicted; AD/CN/SMCI):

```text
[61, 1, 10]
[0, 194, 15]
[6, 19, 292]
```

## Paired hard-classification comparisons

| Comparison | New only correct | Other only correct | Discordant | Exact McNemar p |
|---|---:|---:|---:|---:|
| R vs SEPS-Q | 24 | 23 | 47 | 1 |
| G vs SEPS-Q | 15 | 23 | 38 | 0.255875 |
| RG vs SEPS-Q | 21 | 27 | 48 | 0.470879 |
| R vs Original Query | 19 | 21 | 40 | 0.874629 |

## Best-arm deltas and target checks

Best arm: **R**

| Comparison | ΔACC | ΔMacro-F1 | ΔBACC | ΔProbability Macro-AUC | ΔWeighted-F1 |
|---|---:|---:|---:|---:|---:|
| R − Original Query | -0.003344 | +0.003315 | +0.001793 | -0.000638 | -0.003341 |
| R − SEPS-Q | +0.001672 | +0.003527 | +0.007664 | +0.002121 | +0.001794 |

- Exceeds Original Query ACC 0.9297659: **False**
- Macro-F1 at least approximately 0.914: **True**
- BACC at least approximately 0.914: **True**
- Parameter count below Original Query: **True**
- Learned graph gamma used (G/RG): **{"G": true, "RG": true}**
- Class graphs differ (G/RG): **{"G": true, "RG": true}**
- Class modality weights differ (R/RG): **{"R": true, "RG": true}**

## Minimal fold0 mechanism report

### R

```json
{
  "arm": "R",
  "reader": {
    "rank": 8,
    "class": {
      "AD": {
        "mean_residual_norm": 6.524177074432373,
        "mean_residual_to_shared_norm_ratio": 0.38158872723579407,
        "mean_modality_weights": [
          0.2288227379322052,
          0.19710145890712738,
          0.110527902841568,
          0.20282411575317383,
          0.1203346848487854,
          0.140389084815979
        ]
      },
      "CN": {
        "mean_residual_norm": 19.806156158447266,
        "mean_residual_to_shared_norm_ratio": 1.197391390800476,
        "mean_modality_weights": [
          0.09778828918933868,
          0.12582090497016907,
          0.24112100899219513,
          0.1140953078866005,
          0.23797732591629028,
          0.18319712579250336
        ]
      },
      "SMCI": {
        "mean_residual_norm": 4.988036155700684,
        "mean_residual_to_shared_norm_ratio": 0.298297256231308,
        "mean_modality_weights": [
          0.10616571456193924,
          0.13079528510570526,
          0.22981902956962585,
          0.12096258252859116,
          0.23110675811767578,
          0.18115060031414032
        ]
      }
    },
    "pairwise_weight_difference": {
      "AD_vs_CN": {
        "mean_absolute_difference": 0.10917150229215622,
        "mean_l1_difference": 0.6550289988517761,
        "maximum_absolute_difference": 0.350109338760376
      },
      "AD_vs_SMCI": {
        "mean_absolute_difference": 0.10148970782756805,
        "mean_l1_difference": 0.6089382171630859,
        "maximum_absolute_difference": 0.327883780002594
      },
      "CN_vs_SMCI": {
        "mean_absolute_difference": 0.007931390777230263,
        "mean_l1_difference": 0.04758834093809128,
        "maximum_absolute_difference": 0.04620197415351868
      }
    },
    "weights_identical": {
      "AD_vs_CN": false,
      "AD_vs_SMCI": false,
      "CN_vs_SMCI": false
    },
    "all_class_readers_identical": false
  }
}
```

### G

```json
{
  "arm": "G",
  "graph": {
    "gamma": {
      "AD": -0.07975108176469803,
      "CN": 0.1825132817029953,
      "SMCI": -0.17608124017715454
    },
    "per_class_graph": {
      "AD": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1786,
        "density": 0.010005434082340352,
        "average_degree": 5.9732441902160645,
        "self_loop_only_nodes": 2,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9792612791061401,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      },
      "CN": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1808,
        "density": 0.010128681310678253,
        "average_degree": 6.046822547912598,
        "self_loop_only_nodes": 3,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9815076589584351,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      },
      "SMCI": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1770,
        "density": 0.009915799734458244,
        "average_degree": 5.919732570648193,
        "self_loop_only_nodes": 5,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9529248476028442,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      }
    },
    "edge_jaccard": {
      "AD_vs_CN": {
        "intersection_edges": 1594,
        "union_edges": 2000,
        "edge_jaccard": 0.797
      },
      "AD_vs_SMCI": {
        "intersection_edges": 1538,
        "union_edges": 2018,
        "edge_jaccard": 0.7621407333994054
      },
      "CN_vs_SMCI": {
        "intersection_edges": 1621,
        "union_edges": 1957,
        "edge_jaccard": 0.828308635666837
      }
    },
    "representation": [
      {
        "class": "AD",
        "mean_pre_to_post_cosine": 0.7220678329467773,
        "mean_offdiag_cosine_pre": 0.4296830892562866,
        "mean_offdiag_cosine_post": 0.22700417041778564,
        "offdiag_cosine_delta": -0.20267891883850098,
        "obvious_oversmoothing": false
      },
      {
        "class": "CN",
        "mean_pre_to_post_cosine": 0.714869499206543,
        "mean_offdiag_cosine_pre": 0.31484153866767883,
        "mean_offdiag_cosine_post": 0.13496416807174683,
        "offdiag_cosine_delta": -0.179877370595932,
        "obvious_oversmoothing": false
      },
      {
        "class": "SMCI",
        "mean_pre_to_post_cosine": 0.7217323184013367,
        "mean_offdiag_cosine_pre": 0.26310327649116516,
        "mean_offdiag_cosine_post": 0.059417616575956345,
        "offdiag_cosine_delta": -0.20368565991520882,
        "obvious_oversmoothing": false
      }
    ],
    "graphs_identical": {
      "AD_vs_CN": false,
      "AD_vs_SMCI": false,
      "CN_vs_SMCI": false
    },
    "all_graphs_identical": false,
    "obvious_oversmoothing": false
  }
}
```

### RG

```json
{
  "arm": "RG",
  "reader": {
    "rank": 8,
    "class": {
      "AD": {
        "mean_residual_norm": 9.455065727233887,
        "mean_residual_to_shared_norm_ratio": 1.1642111539840698,
        "mean_modality_weights": [
          0.2042713463306427,
          0.20812977850437164,
          0.11857329308986664,
          0.16152417659759521,
          0.13914191722869873,
          0.16835950314998627
        ]
      },
      "CN": {
        "mean_residual_norm": 26.35803985595703,
        "mean_residual_to_shared_norm_ratio": 3.098822593688965,
        "mean_modality_weights": [
          0.17858953773975372,
          0.17737741768360138,
          0.15111298859119415,
          0.1676560640335083,
          0.15726478397846222,
          0.16799919307231903
        ]
      },
      "SMCI": {
        "mean_residual_norm": 8.182731628417969,
        "mean_residual_to_shared_norm_ratio": 0.9859378337860107,
        "mean_modality_weights": [
          0.1946694552898407,
          0.1983424425125122,
          0.131779283285141,
          0.16210508346557617,
          0.14889048039913177,
          0.16421321034431458
        ]
      }
    },
    "pairwise_weight_difference": {
      "AD_vs_CN": {
        "mean_absolute_difference": 0.031810179352760315,
        "mean_l1_difference": 0.1908610463142395,
        "maximum_absolute_difference": 0.19275152683258057
      },
      "AD_vs_SMCI": {
        "mean_absolute_difference": 0.011054079048335552,
        "mean_l1_difference": 0.06632446497678757,
        "maximum_absolute_difference": 0.05711144208908081
      },
      "CN_vs_SMCI": {
        "mean_absolute_difference": 0.02214420959353447,
        "mean_l1_difference": 0.13286523520946503,
        "maximum_absolute_difference": 0.13564008474349976
      }
    },
    "weights_identical": {
      "AD_vs_CN": false,
      "AD_vs_SMCI": false,
      "CN_vs_SMCI": false
    },
    "all_class_readers_identical": false
  },
  "graph": {
    "gamma": {
      "AD": 0.21931429207324982,
      "CN": -0.22960016131401062,
      "SMCI": 0.18645957112312317
    },
    "per_class_graph": {
      "AD": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1627,
        "density": 0.0091146927502619,
        "average_degree": 5.441471576690674,
        "self_loop_only_nodes": 12,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9569880366325378,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      },
      "CN": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1662,
        "density": 0.009310767886254012,
        "average_degree": 5.558528423309326,
        "self_loop_only_nodes": 8,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9909219145774841,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      },
      "SMCI": {
        "node_count": 598,
        "top_k": 8,
        "undirected_edge_count": 1621,
        "density": 0.00908107986980611,
        "average_degree": 5.421404838562012,
        "self_loop_only_nodes": 9,
        "symmetry_error": 0.0,
        "nonedge_max_abs": 0.0,
        "minimum_retained_edge_weight": 0.9432969093322754,
        "minimum_self_loop": 1.0,
        "normalized_symmetry_error": 2.9802322387695312e-08,
        "all_finite": true
      }
    },
    "edge_jaccard": {
      "AD_vs_CN": {
        "intersection_edges": 422,
        "union_edges": 2867,
        "edge_jaccard": 0.14719218695500524
      },
      "AD_vs_SMCI": {
        "intersection_edges": 833,
        "union_edges": 2415,
        "edge_jaccard": 0.34492753623188405
      },
      "CN_vs_SMCI": {
        "intersection_edges": 574,
        "union_edges": 2709,
        "edge_jaccard": 0.21188630490956073
      }
    },
    "representation": [
      {
        "class": "AD",
        "mean_pre_to_post_cosine": 0.9539943933486938,
        "mean_offdiag_cosine_pre": 0.4885885715484619,
        "mean_offdiag_cosine_post": 0.48764193058013916,
        "offdiag_cosine_delta": -0.0009466409683227539,
        "obvious_oversmoothing": false
      },
      {
        "class": "CN",
        "mean_pre_to_post_cosine": 0.9352898597717285,
        "mean_offdiag_cosine_pre": 0.09135802835226059,
        "mean_offdiag_cosine_post": 0.08961418271064758,
        "offdiag_cosine_delta": -0.0017438456416130066,
        "obvious_oversmoothing": false
      },
      {
        "class": "SMCI",
        "mean_pre_to_post_cosine": 0.9297066926956177,
        "mean_offdiag_cosine_pre": 0.11547883599996567,
        "mean_offdiag_cosine_post": 0.10174596309661865,
        "offdiag_cosine_delta": -0.013732872903347015,
        "obvious_oversmoothing": false
      }
    ],
    "graphs_identical": {
      "AD_vs_CN": false,
      "AD_vs_SMCI": false,
      "CN_vs_SMCI": false
    },
    "all_graphs_identical": false,
    "obvious_oversmoothing": false
  }
}
```

## Integrity

- Initialization raw-logit equivalence passed for B/R/G/RG at max absolute difference ≤1e-6.
- Graph symmetry, strict zero nonedges, self-loops, finite values, edge counts/density/degrees, and self-loop-only nodes were validated.
- R gradients, gamma gradients, and post-gamma graph-transform/scorer gradients were finite and active.
- RG fold0 CUDA three-epoch smoke and checkpoint readback passed before formal training.
- Every arm contains ten independently trained fresh models and exactly one OOF prediction for each of 598 subjects.
- Every best checkpoint was read back and its predictions and metrics reproduced before the fold was finalized.

## Decision

全部低于Original：本路线当前不能替代Original。

No rank search, k search, multi-seed run, ensemble, TabPFN-guided graph, second dataset, new loss, or alternative GNN head was run.

## Artifacts

- Runner: `scripts/run_class_private_evidence_graph_v1.py`
- Smoke and validation: `experiments/class_private_evidence_graph_v1/smoke_rg/`
- R/G/RG ten-fold results: `experiments/class_private_evidence_graph_v1/{r,g,rg}/tenfold_seed0/`
- Final JSON/CSV and paired rows: `experiments/class_private_evidence_graph_v1/final/`
