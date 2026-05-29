# Phase 10 Candidate Scorecard

Target: RBCSmart (`cost <= 17884.3`, `ev_min >= 0.99`,
`ev_within_tolerance >= 0.4`, no electrical violations).

BAU ratios are marked unavailable when BAU export was intentionally disabled.

Verdict counts:

- FAIL_EV_MIN: 21
- FAIL_EV_TOL: 3

| algorithm | recipe | seed | steps | verdict | cost_eur | ev_min_acceptable_feasible_rate | ev_within_tolerance_rate | battery_throughput_kwh | v2g_export_kwh | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MADDPG | w5_bc | 123 | 4096 | FAIL_EV_MIN | 8792.248 | 0.989 | 0.256 | 34592.588 | 1507.352 | 492.733 |
| MATD3 | w5_plain | 123 | 4096 | FAIL_EV_MIN | 9243.894 | 0.879 | 0.065 | 12956.425 | 5790.527 | 458.697 |
| MADDPG | w5_plain | 123 | 4096 | FAIL_EV_MIN | 9251.905 | 0.879 | 0.065 | 13153.804 | 5786.348 | 415.687 |
| MADDPG | w5_bc_light | 789 | 8760 | FAIL_EV_MIN | 18627.087 | 0.977 | 0.328 | 66714.046 | 5469.518 | 1015.312 |
| MATD3 | w5_bc_light | 456 | 8760 | FAIL_EV_MIN | 18943.327 | 0.979 | 0.201 | 63998.385 | 4199.088 | 1023.908 |
| MADDPG | w5_bc_light | 456 | 8760 | FAIL_EV_MIN | 19512.700 | 0.978 | 0.219 | 67429.437 | 3561.717 | 1074.747 |
| MATD3 | w5_bc_light | 789 | 8760 | FAIL_EV_MIN | 19604.850 | 0.980 | 0.213 | 68120.024 | 4041.619 | 1027.107 |
| MATD3 | w5_bc_light | 2025 | 8760 | FAIL_EV_MIN | 19667.848 | 0.986 | 0.230 | 61904.401 | 4072.737 | 1034.671 |
| MATD3 | w5_bc_light | 123 | 8760 | FAIL_EV_MIN | 19771.694 | 0.988 | 0.223 | 65958.436 | 3716.153 | 1054.031 |
| MATD3 | w5_bc_light | 2024 | 8760 | FAIL_EV_MIN | 19799.838 | 0.981 | 0.221 | 71191.288 | 5437.205 | 1035.581 |
| MADDPG | w5_bc_light | 2025 | 8760 | FAIL_EV_MIN | 19842.099 | 0.988 | 0.218 | 68493.186 | 3551.870 | 1060.889 |
| MADDPG | w5_plain | 456 | 8760 | FAIL_EV_MIN | 19997.049 | 0.938 | 0.050 | 13313.193 | 6077.913 | 862.302 |
| MATD3 | w5_plain | 456 | 8760 | FAIL_EV_MIN | 20055.706 | 0.940 | 0.050 | 13054.275 | 5869.758 | 963.596 |
| MATD3 | w5_plain | 123 | 8760 | FAIL_EV_MIN | 20110.873 | 0.943 | 0.049 | 13068.324 | 5793.747 | 958.561 |
| MATD3 | w5_plain | 2025 | 8760 | FAIL_EV_MIN | 20183.116 | 0.947 | 0.051 | 13657.376 | 5717.324 | 963.899 |
| MADDPG | w5_plain | 123 | 8760 | FAIL_EV_MIN | 20281.963 | 0.943 | 0.049 | 12968.222 | 5789.519 | 906.449 |
| MADDPG | w5_plain | 2025 | 8760 | FAIL_EV_MIN | 20618.001 | 0.946 | 0.051 | 14071.087 | 5722.582 | 930.194 |
| MATD3 | w5_plain | 2024 | 8760 | FAIL_EV_MIN | 20989.683 | 0.946 | 0.053 | 13424.342 | 5457.915 | 1066.714 |
| MATD3 | w5_plain | 789 | 8760 | FAIL_EV_MIN | 21042.654 | 0.944 | 0.050 | 12833.307 | 5646.156 | 1028.211 |
| MADDPG | w5_plain | 789 | 8760 | FAIL_EV_MIN | 21076.530 | 0.944 | 0.050 | 14284.776 | 5641.121 | 901.520 |
| MADDPG | w5_plain | 2024 | 8760 | FAIL_EV_MIN | 21160.336 | 0.946 | 0.053 | 15621.506 | 5458.162 | 916.052 |
| MATD3 | w5_bc | 123 | 4096 | FAIL_EV_TOL | 8888.708 | 0.991 | 0.217 | 32320.192 | 865.489 | 474.037 |
| MADDPG | w5_bc_light | 2024 | 8760 | FAIL_EV_TOL | 19544.530 | 0.995 | 0.282 | 67627.000 | 5182.699 | 1040.043 |
| MADDPG | w5_bc_light | 123 | 8760 | FAIL_EV_TOL | 20018.595 | 0.996 | 0.208 | 68356.432 | 3436.637 | 1034.697 |
