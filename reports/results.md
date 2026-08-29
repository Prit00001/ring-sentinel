# Ring Sentinel — Results

| Metric | A_rules_only | B_lgbm_base | D_plus_ring_features | F_plus_calibration |
|---|---|---|---|---|
| PR-AUC | 0.0331 | 0.5122 | 0.5069 | 0.4961 |
| Recall @ 0.1% FPR | 0.1822 | 0.2316 | 0.2364 | 0.2408 |
| Recall @ 0.5% FPR | 0.1822 | 0.3657 | 0.3642 | 0.3706 |
| Recall @ 1.0% FPR | 0.1822 | 0.4161 | 0.4131 | 0.4380 |
| Recall @ 5.0% FPR | 0.1822 | 0.6228 | 0.6136 | 0.6327 |
| Precision @ k | 0.0500 | 1.0000 | 1.0000 | 1.0000 |
| Brier score | 0.0454 | 0.0225 | 0.0226 | 0.0229 |
| ECE | 0.0780 | 0.0081 | 0.0066 | 0.0044 |
| ROC-AUC* | 0.4807 | 0.8999 | 0.8991 | 0.8988 |

\*ROC-AUC is reported for reference only. At a 3.4% base rate, ROC-AUC is dominated by the large true-negative population and looks high even for a mediocre detector; PR-AUC and recall@FPR are the metrics that actually distinguish models on this track.
