# V7 Publishable Insights Report

Completed evaluated runs: 22

## Clean Main Runs

| seed | variant | blimp_accuracy | syntaxgym_accuracy | fillergap_accuracy | naturalstories_delta_r2_gate_flexible_surprisal | naturalstories_cv_by_story_delta_r2_gate_flexible_surprisal |
| --- | --- | --- | --- | --- | --- | --- |
| 23.0000 | anira_emergent | 0.9360 | 0.8368 | 0.9981 | 0.0029 | 0.0027 |
| 23.0000 | cheap_only | 0.9337 | 0.8542 | 0.9981 | 0.0000 | 0.0000 |
| 23.0000 | dense_gru | 0.9347 | 0.8472 | 0.9981 | 0.0000 | -0.0000 |
| 23.0000 | etb | 0.8949 | 0.7326 | 0.9819 | 0.0012 | 0.0037 |
| 23.0000 | generic_dynamic | 0.9335 | 0.8438 | 0.9956 | 0.0032 | 0.0027 |
| 23.0000 | punctuation_only | 0.9349 | 0.8403 | 0.9981 | 0.0095 | 0.0061 |
| 23.0000 | random_matched | 0.9325 | 0.8368 | 1.0000 | 0.0000 | 0.0000 |
| 29.0000 | anira_emergent | 0.9281 | 0.8368 | 0.9962 | 0.0004 | -0.0020 |
| 29.0000 | cheap_only | 0.9326 | 0.8542 | 0.9981 | 0.0000 | -0.0000 |
| 29.0000 | dense_gru | 0.9357 | 0.8472 | 0.9975 | 0.0000 | 0.0000 |
| 29.0000 | etb | 0.9246 | 0.7847 | 0.9962 | 0.0047 | 0.0088 |
| 29.0000 | generic_dynamic | 0.9355 | 0.8368 | 0.9962 | 0.0029 | 0.0015 |
| 29.0000 | punctuation_only | 0.9319 | 0.8125 | 0.9962 | 0.0105 | 0.0095 |
| 29.0000 | random_matched | 0.9346 | 0.8333 | 0.9975 | 0.0000 | 0.0000 |
| 31.0000 | etb | 0.9297 | 0.7986 | 0.9969 | 0.0079 | 0.0141 |
| 31.0000 | generic_dynamic | 0.9364 | 0.8229 | 0.9981 | 0.0040 | 0.0068 |
| 31.0000 | punctuation_only | 0.9351 | 0.8125 | 0.9981 | 0.0096 | 0.0063 |

## ETB Minus Baselines

| seed | baseline | etb_minus_baseline_blimp_accuracy | etb_minus_baseline_syntaxgym_accuracy | etb_minus_baseline_fillergap_accuracy | etb_minus_baseline_naturalstories_delta_r2_gate_flexible_surprisal |
| --- | --- | --- | --- | --- | --- |
| 23.0000 | anira_emergent | -0.0410 | -0.1042 | -0.0162 | -0.0017 |
| 23.0000 | cheap_only | -0.0387 | -0.1215 | -0.0162 | 0.0012 |
| 23.0000 | dense_gru | -0.0398 | -0.1146 | -0.0162 | 0.0012 |
| 23.0000 | generic_dynamic | -0.0386 | -0.1111 | -0.0137 | -0.0020 |
| 23.0000 | punctuation_only | -0.0399 | -0.1076 | -0.0162 | -0.0083 |
| 23.0000 | random_matched | -0.0376 | -0.1042 | -0.0181 | 0.0012 |
| 29.0000 | anira_emergent | -0.0035 | -0.0521 | 0.0000 | 0.0043 |
| 29.0000 | cheap_only | -0.0081 | -0.0694 | -0.0019 | 0.0047 |
| 29.0000 | dense_gru | -0.0112 | -0.0625 | -0.0013 | 0.0047 |
| 29.0000 | generic_dynamic | -0.0110 | -0.0521 | 0.0000 | 0.0018 |
| 29.0000 | punctuation_only | -0.0073 | -0.0278 | 0.0000 | -0.0058 |
| 29.0000 | random_matched | -0.0101 | -0.0486 | -0.0013 | 0.0047 |
| 31.0000 | generic_dynamic | -0.0067 | -0.0243 | -0.0013 | 0.0039 |
| 31.0000 | punctuation_only | -0.0054 | -0.0139 | -0.0013 | -0.0017 |

## RT Supervision Upper Bound

| seed | rt_supervised_minus_main_blimp_accuracy | rt_supervised_minus_main_syntaxgym_accuracy | rt_supervised_minus_main_fillergap_accuracy | rt_supervised_minus_main_naturalstories_delta_r2_gate_flexible_surprisal |
| --- | --- | --- | --- | --- |
| 23.0000 | 0.0001 | -0.0312 | 0.0100 | 0.0075 |
| 29.0000 | 0.0071 | 0.0625 | 0.0000 | 0.0094 |
| 31.0000 | -0.0294 | -0.1528 | -0.0137 | 0.0026 |

## Natural Stories Uncertainty

| condition | seed | variant | observed_delta_r2_flexible | bootstrap_ci95_low | bootstrap_ci95_high | permutation_null_mean | permutation_p_one_sided |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_aux | 23.0000 | etb | 0.0000 | 0.0000 | 0.0022 | 0.0005 | 0.9271 |
| no_contrastive | 23.0000 | etb | 0.0000 | 0.0000 | 0.0025 | 0.0005 | 0.9071 |
| rt_supervised_upper_bound | 23.0000 | etb | 0.0087 | 0.0016 | 0.0229 | 0.0005 | 0.0010 |
| rt_supervised_upper_bound | 29.0000 | etb | 0.0140 | 0.0048 | 0.0286 | 0.0005 | 0.0010 |
| rt_supervised_upper_bound | 31.0000 | etb | 0.0105 | 0.0027 | 0.0231 | 0.0005 | 0.0010 |
| main | 23.0000 | anira_emergent | 0.0029 | 0.0001 | 0.0101 | 0.0004 | 0.0160 |
| main | 23.0000 | cheap_only | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 23.0000 | dense_gru | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 23.0000 | etb | 0.0012 | 0.0000 | 0.0064 | 0.0006 | 0.1588 |
| main | 23.0000 | generic_dynamic | 0.0032 | 0.0001 | 0.0110 | 0.0004 | 0.0050 |
| main | 23.0000 | punctuation_only | 0.0095 | 0.0014 | 0.0259 | 0.0005 | 0.0010 |
| main | 23.0000 | random_matched | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 29.0000 | anira_emergent | 0.0004 | 0.0000 | 0.0044 | 0.0005 | 0.3666 |
| main | 29.0000 | cheap_only | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 29.0000 | dense_gru | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 29.0000 | etb | 0.0047 | 0.0004 | 0.0144 | 0.0005 | 0.0020 |
| main | 29.0000 | generic_dynamic | 0.0029 | 0.0001 | 0.0109 | 0.0005 | 0.0110 |
| main | 29.0000 | punctuation_only | 0.0105 | 0.0018 | 0.0262 | 0.0005 | 0.0010 |
| main | 29.0000 | random_matched | 0.0000 | -0.0000 | 0.0000 | 0.0000 | 1.0000 |
| main | 31.0000 | etb | 0.0079 | 0.0011 | 0.0226 | 0.0005 | 0.0010 |

## Interpretation Checklist

- Treat RT-supervised runs as an upper bound or leakage stress test, not the main claim.
- Main ETB evidence requires no RT gate training, held-out Natural Stories, and matched/flexible surprisal controls.
- If generic_dynamic or other simple baselines beat ETB, frame the result as a failure-analysis/strong-baseline finding.
- Report seed-level variability and uncertainty before making claims about small Natural Stories deltas.
