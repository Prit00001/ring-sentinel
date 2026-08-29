# Ring Sentinel — Model Card

---

## Model Details

**Model Name:** Ring Sentinel  
**Version:** 0.1.0  
**Type:** Gradient-boosted decision tree (LightGBM) with component-level aggregation  
**Task:** Coordinated fraud ring detection (classification → case scoring → cost-optimal decision)  
**Input:** Transaction + identity + graph features  
**Output:** Fraud probability (transaction), ring score (case), decision (allow/block/review)

---

## Training Data

**Primary Dataset:** IEEE-CIS Fraud Detection (Vesta)
- **Source:** https://www.kaggle.com/competitions/ieee-fraud-detection
- **Rows:** 590,540 transactions (training split only; test files have no public labels)
- **Fraud Rate:** 20,663 positives / 590,540 = 3.5% base rate
- **Time Span:** 0–40 days (synthetic timedelta in seconds from unstated epoch)
- **Temporal Split:** Train 60% / val 20% / test 20% by TransactionDT
- **Embargo:** 1-day gap between splits to kill boundary leakage

**Secondary Dataset (Ablation Only):** Bank Account Fraud, NeurIPS 2022
- **Source:** https://github.com/feedzai/bank-account-fraud
- **Rows:** 6M instances across 6 variants
- **Use:** Calibration curve, cost-model ablation; no headline metrics
- **License:** CC-BY-NC (non-commercial research and portfolio projects only)

---

## Feature Groups (Ablation Rows)

| Row | Configuration | Features | Notes |
|---|---|---|---|
| A | Rules only | 3 hand-crafted thresholds | Baseline |
| B | LGBM, raw features | amount, C*, D*, M*, V* (pruned), identity presence | No aggregates |
| C | B + causal aggregates | uid_*, card_*, addr_*, device_* expanding-window features | No graph |
| D | C + ring graph features | comp_*, uid_degree, device_degree, addr_degree | **Differentiator** |
| E | D + case-level aggregation | Noisy-OR of member transaction scores | Ring score |
| F | E + isotonic calibration | Post-hoc probability recalibration | Matches cost model |

---

## Limitations

### Known

1. **UID is a fingerprint, not a person.** Reconstruction from card + address + D1 is a 
   community standard (first-place IEEE-CIS writeup, Amazon FDB) but is imperfect. 
   Two cardholders can collide; one person can fragment across multiple UIDs (e.g., 
   after a move, after a card reissue).

2. **Identity coverage is sparse.** Device data (DeviceInfo, id_30..id_33) appears in 
   ~24% of rows. Graph statistics computed on sparse entity types (low coverage) are 
   weighted against dense ones (high coverage) in interpretation. A coverage table 
   (`reports/entity_coverage.md`) is published before any graph metric.

3. **V-block is internally correlated.** Correlation pruning drops one of any pair with 
   |r| > 0.95, keeping the higher-coverage member. ~30% of V columns are pruned. 
   Correlation measured on a 100k-row sample of the training split only.

4. **Amounts are USD; cost model uses fixed exchange rate.** `USD_INR: 87.0` is a 
   modeling convention (ensures reproducibility) and NOT a market rate. Actual 
   deployments would use live FX or transaction-time rates.

5. **Label maturation lag is fixed at 30 days.** Chargebacks arrive on a spectrum 
   depending on region, card network, and acquirer policies. This is a best guess; 
   a production system would calibrate against real dispute timelines.

6. **Scheme monitoring thresholds are not defended.** Visa consolidated fraud/dispute 
   programmes in 2025; Mastercard runs separate excessive-fraud and excessive-chargeback 
   programmes. Thresholds change and vary by region and acquirer. We quote no specific 
   numbers and leave `scheme_penalty_risk` at zero in the cost model. A production 
   system would verify current thresholds against the schemes' published documents 
   before baking them in.

7. **Graph is rebuilt daily, not in real time.** A true streaming fraud detector would 
   update components as transactions arrive. This batch-daily architecture suits offline 
   review queues; it would not be adequate for instant transaction blocking.

8. **Review accuracy is assumed constant.** `review_accuracy: 0.85` assumes every 
   analyst catches fraud at the same rate. Actual accuracy varies by analyst, fatigue, 
   case complexity, and other factors. A production system would track per-analyst and 
   per-case-type accuracy.

9. **No handling of account takeover or refund fraud.** This model targets card-not-present 
   abuse rings (coordinated testing, account stuffing, etc.). Account takeover (compromised 
   password) and refund fraud (legitimate transaction + false dispute) have different 
   signatures and would need separate models or feature layers.

10. **Test set touched once.** The val split was used for calibration threshold fitting. 
    If you tune anything on val *after* calibration, you have contaminated the validation 
    split and should discard the run and start over. Model card will state if this happened.

---

## Intended Use

**Primary:** Offline review queue prioritization for a payments processor or risk team.  
Input: Transactions settled in the past 24 hours, grouped by component.  
Output: A ranked queue of cases for analyst review, with brief explanations.

**Constraints:**
- Do not use for instant transaction blocking without human review.
- Do not deploy without tuning the cost model to your actual costs.
- Do not use without verifying scheme monitoring thresholds against current network rules.
- Do not use for identity verification or KYC decisions.

---

## Performance

> **Results section is empty until `make repro` completes.**

Placeholder for:
- PR-AUC, recall @ fixed FPRs, precision @ k
- Calibration (Brier score, ECE) before and after isotonic regression
- Case-level metrics (distinct from transaction-level)
- Expected rupee cost under the chosen policy
- Ablation table (one row per feature layer)
- Slices: recall by amount band, hour of day, device coverage, new vs returning account
- Drift: PSI on top 15 features between train and test windows

---

## Training Procedure

1. **Temporal split:** Sort by TransactionDT; split at 60/20/20 boundaries; drop 1-day embargo.
2. **Entity resolution:** Reconstruct UID (card1/2/3/5 + addr1/2 + D1n); extract device, email, addr, card nodes.
3. **Graph:** Rebuild daily using [d − 14 days, d); hub prune (max degree 200); union-find.
4. **Features:**
   - Base: amount, hour, ProductCD, C*, D*, M*, V-pruned, identity presence
   - Causal aggregates: UID/card/device/addr expanding windows, label-maturation-lagged
   - Ring: component-level (comp_size, comp_n_uid, comp_velocity_*, etc.)
5. **Base model:** LightGBM
   - Objective: binary
   - Scale_pos_weight: 1.0 (NOT rebalanced; probability scale matters downstream)
   - Early stopping on validation split
   - Seed: fixed (1337)
6. **Calibration:** Isotonic regression fitted on validation split only (disjoint from test).
7. **Case aggregation:** Noisy-OR of member transaction scores (capped at 50).
8. **Decision:** Argmin expected cost over {allow, block, review}.

---

## Hyperparameters

All in `config/base.yaml`, `config/features.yaml`, `config/costs.yaml`.

**LightGBM:**
```yaml
num_leaves: 96
learning_rate: 0.03
feature_fraction: 0.7
bagging_fraction: 0.8
lambda_l2: 5.0
n_estimators: 3000
early_stopping_rounds: 200
```

**Graph:**
```yaml
lookback_days: 14
max_entity_degree: 200
min_entity_degree: 2
```

**Features:**
```yaml
label_lag_days: 30
smoothing_strength: 20.0
small_amount_usd: 5.0
```

---

## Bias and Fairness

IEEE-CIS provides no protected attributes (age, gender, etc.). Bank Account Fraud (BAF) 
does (age group, income percentile, employment status). 

**Planned ablation:** Slice recall by age group and income percentile on BAF to surface 
any disparities. Results will be published in `reports/slices.md`.

---

## Ethical Considerations

**Defense-only.** This project contains no component that generates fraudulent traffic, 
evades detectors, or documents bypass techniques. It is designed to *defend* against 
coordinated abuse, not to enable it. No cardholder PII or real personal data is used.

---

## LLM Component (Narrator)

**Provider:** Groq  
**Narrator Model:** openai/gpt-oss-120b (checked live on [DATE])  
**Judge Model:** qwen/qwen3.6-27b (checked live on [DATE])  

**What it does:** Converts evidence JSON into a 4-section brief (summary, evidence, counter, 
next step) for an analyst to read.

**What it does NOT do:** It does not make the fraud/legit decision. The policy engine (deterministic) 
decides; the LLM narrates the supporting evidence.

**Grounding evaluation:**
- Numeric fidelity: every number in the brief must appear in the evidence JSON (target 1.00)
- Claim support: yes/no/partial for each bullet against evidence (measured vs hand labels)
- Counter-section presence: should always be present (catches prompt violation)

**Degradation path:** If GROQ_API_KEY is absent and cache misses, falls back to a 
deterministic template. System still works (no API call = no decision loss).

---

## Versioning and Reproducibility

**Seed:** 1337 (fixed across all random operations)  
**Checksums:** `data/CHECKSUMS.txt` records SHA256 of every raw input file  
**Artifacts:** Cached Groq responses in `artifacts/narratives/` (committed)  
**Configuration:** Every number lives in `config/*.yaml`  

**Reproduction:** `make repro` regenerates every metric in README from committed artefacts.

---

## Citation

If you use this work, cite the build spec and the underlying datasets:

```bibtex
@inproceedings{ring-sentinel-2024,
  title   = {Ring Sentinel: Detecting Coordinated Card-Not-Present Fraud Rings},
  author  = {[Your Name]},
  journal = {Razorpay AI Buildathon, Track 2},
  year    = {2024},
  note    = {Defense-only fraud risk system on IEEE-CIS dataset}
}

@inproceedings{ieee-cis-2019,
  title   = {IEEE-CIS Fraud Detection},
  author  = {{IEEE Computational Intelligence Society} and {Vesta Corporation}},
  booktitle = {Kaggle},
  year    = {2019}
}

@article{jesus-2022,
  title   = {Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author  = {Jesus, Sérgio and others},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```

---

**Questions?** See README.md, tests/test_leakage.py, config/*.yaml.
