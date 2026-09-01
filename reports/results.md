# Ring Sentinel — Results

| Metric | A_rules_only | B_lgbm_base | D_plus_ring_features | F_plus_calibration |
|---|---|---|---|---|
| PR-AUC | 0.0331 | 0.5122 | 0.5712 | 0.5581 |
| Recall @ 0.1% FPR | 0.1214 | 0.2316 | 0.2728 | 0.3176 |
| Recall @ 0.5% FPR | 0.1214 | 0.3657 | 0.4332 | 0.4439 |
| Recall @ 1.0% FPR | 0.1214 | 0.4161 | 0.4859 | 0.5050 |
| Recall @ 5.0% FPR | 0.1214 | 0.6228 | 0.6770 | 0.7043 |
| Precision @ k | 0.0215 | 0.4666 | 0.5122 | 0.5124 |
| Brier score | 0.0428 | 0.0225 | 0.0206 | 0.0207 |
| ECE | 0.0679 | 0.0081 | 0.0048 | 0.0035 |
| ROC-AUC* | 0.4779 | 0.8999 | 0.9166 | 0.9162 |

\*ROC-AUC is reported for reference only. At a 3.4% base rate, ROC-AUC is dominated by the large true-negative population and looks high even for a mediocre detector; PR-AUC and recall@FPR are the metrics that actually distinguish models on this track.
