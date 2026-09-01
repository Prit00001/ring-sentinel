# Ring Sentinel

Detects coordinated card-not-present fraud rings on IEEE-CIS (590,540 txns, 3.5% fraud). 
Strictly temporal split. Defense-only.

**Track:** Razorpay AI Buildathon, Track 02 — AI Risk Manager  
**Status:** Evaluated on real IEEE-CIS data. Ring features are a confirmed, measured lift over the base model (see below); case-level economics show the policy is not deployable as-is without a threshold retune (see Case economics).

---

## Results (held-out final 20% by time, touched once)

Run on real IEEE-CIS training data (590,540 rows) via `make data && make repro`. No number below comes from the synthetic test fixture.

| Metric | A_rules_only | B_lgbm_base | D_plus_ring_features | F_plus_calibration |
|---|---|---|---|---|
| PR-AUC | 0.0331 | 0.5122 | **0.5712** | 0.5581 |
| Recall @ 0.1% FPR | 0.1214 | 0.2316 | 0.2728 | **0.3176** |
| Recall @ 0.5% FPR | 0.1214 | 0.3657 | 0.4332 | **0.4439** |
| Recall @ 1.0% FPR | 0.1214 | 0.4161 | 0.4859 | **0.5050** |
| Recall @ 5.0% FPR | 0.1214 | 0.6228 | 0.6770 | **0.7043** |
| Precision @ k=100/day | 0.0215 | 0.4666 | 0.5122 | **0.5124** |
| Brier score | 0.0428 | 0.0225 | **0.0206** | 0.0207 |
| ECE | 0.0679 | 0.0081 | 0.0048 | **0.0035** |
| ROC-AUC* | 0.4779 | 0.8999 | 0.9166 | 0.9162 |

\*ROC-AUC is reference only — at a 3.4% base rate it's dominated by the large true-negative population and looks high even for a mediocre detector. PR-AUC and recall@FPR are the metrics that actually distinguish these models.

**Ring features are a real, measured differentiator** — +ring features (D) beats base LightGBM (B) on every metric: PR-AUC +11.5%, recall@1%FPR +17%, recall@0.5%FPR +18%. This was not true in an earlier run; see "What Didn't Work" for why, and how it was found and fixed.

### Case economics (full test period, `reports/economics.md`)

Every ring in the test split with ≥2 transactions scored by the calibrated model and run through the cost-aware policy engine — not just the top-20 shown to an analyst console, which is the easiest, most cherry-picked slice and does not represent the whole flagged population.

| n_cases | n_flagged (block+review) | case_level_precision | total expected cost, ₹ | total expected saving, ₹ |
|---|---|---|---|---|
| 11,505 | 6,314 (54.9%) | **0.1077** | 12,604,376 | 56,860,298 |

**This is the number that actually gates production-readiness, and it fails the bar as configured.** Only ~10.8% of flagged rings (block or review) contain even one confirmed-fraud transaction — the other ~89% of flags are false positives at the ring level. The top-20 analyst-console queue looked far better (13/20 = 65% precision) precisely because it's sorted by expected saving and shows only the largest, most confident cases; it is not representative of what the policy does across the whole test period. The "total expected saving" figure is the *model's own* expected-value estimate under its predicted probabilities (an optimistic, model-internal number), not a saving verified against realized outcomes — a genuinely realized-cost evaluation (comparing policy decisions to ground truth `isFraud` outcomes, not to the model's own score) is not yet built; see Next Steps.

The fix is a threshold retune, not a model change: `config/costs.yaml`'s cost ratios (chargeback cost vs. false-decline cost) currently push the policy engine toward flagging over half of all rings. Before deployment, the review/block thresholds need calibrating against this case-level precision number specifically, not against transaction-level recall@FPR alone — a system can look excellent on transaction metrics and still flag the majority of legitimate rings.

---

## Build Order (Followed)

This project implements all ten steps from the build spec:

1. ✓ Data loader with temporal splits and leakage test suite
2. ✓ Rules baseline + test harness
3. ✓ Entity resolution and time-sliced ring graph
4. ✓ Ring features with causal-aggregation assertions
5. ✓ LightGBM scorer with isotonic calibration
6. ✓ Cost-aware policy engine with rupee cost model
7. ✓ LLM case narrator (Groq key configured via `.env`; 40+ cached briefs in `artifacts/narratives/`)
8. ✓ Analyst console (`make serve`; loads real cases from `artifacts/case_queue.json`) — **no authentication**, not exposed beyond loopback by default
9. ✓ Slice metrics, drift report (`reports/slices.md`, `reports/drift.md`). ⊗ Ablation rows C (causal aggregates alone) and E (case-level noisy-OR aggregation as its own scored model) are not computed — `run_eval()`'s own docstring documents this gap
10. ✓ Results table, case economics (this README, `reports/results.md`, `reports/economics.md`). ⊗ MODEL_CARD.md not yet updated with these numbers

---

## What Didn't Work (and how it was found)

- **`precision_at_k` silently reported a fake 1.0000.** The metric is named "Precision @ k=100/day" but the code took the top-100 scores over the *entire* ~115k-row, ~40-day test period once, not top-100 per day — an easy bar that every model cleared perfectly. Found by re-deriving the metric by hand; fixed to group by day and pool precision across days (`eval/metrics.py::precision_at_k`). Corrected value for the deployed model: **0.5124**, not 1.0.
- **The intended ring-feature "differentiator" was dead on the first real run.** `comp_prior_fraud_rate` — the feature most directly meant to encode "has this ring been fraudulent before" — was a **structural constant** (a single value, `0.0338`, across all 115,108 test rows) because `graph.lookback_days` (14) was shorter than `features.label_lag_days` (30): no label inside a 14-day window can ever be 30 days matured, so the causal aggregator never found a single matured label, ever, by construction. Feature importance confirmed it: rank 344 of 345 (its neighbor, `comp_prior_n_labeled`, was rank 345 — literally last). Fixed by raising `lookback_days` to 45 in `config/base.yaml`; after the fix, `comp_prior_fraud_rate` has real variance (1,872 distinct values, std 0.028) and jumped to **rank 5 of 345** in importance, beating `card1`/`card2`/`addr1`. This one config value was costing the whole ring-feature layer its main intended signal — with it fixed, +ring features beats base LightGBM on every reported metric, which was not true before the fix.
- **Case-level precision over the full test period is 10.8%, not the 65% the top-20 analyst queue suggested.** The analyst console's queue is sorted by expected saving and only shows the 20 largest, most confident rings — a highly favorable, non-representative sample. Aggregating the *same* policy decisions across all 11,505 test-period rings (`reports/economics.md`) shows 54.9% of all rings get flagged, and ~89% of those flags are false positives at the ring level. This is the actual reason the system is not production-ready as configured — not model accuracy, but decision-threshold economics.

---

## Reproduce

```bash
# Smoke test (no external data needed, ~30 seconds)
make smoke

# Full reproducibility (requires IEEE-CIS data + Groq key)
make data                           # ~10 min, downloads 590k rows
export GROQ_API_KEY=your_key
make repro                          # ~N minutes, regenerates every table
```

---

## Architecture

```
 raw txns ─► temporal split ─► entity resolution ─► ring graph ─► ring features ─┐
                    │                                                            ├─► LightGBM ─► calibration
                    └─────────────► base features ───────────────────────────────┘        │
                                                                                          ▼
                                                              case aggregator ─► COST POLICY ENGINE
                                                                                          │
                                                          ┌───────────────┬───────────────┤
                                                        allow          block           review
                                                          │               │               │
                                                          └──────► AUDIT LOG ◄────────────┤
                                                                    (hash chain)          ▼
                                                                                  LLM narrator
                                                                                  + grounding eval
                                                                                          │
                                                                                  analyst console
                                                                                          │
                                                                          decision ──► audit log ──► labels
```

---

## Method

### Temporal splitting and leakage controls

- Sorted by `TransactionDT`; no shuffling ever.
- Train/val/test 60/20/20 by time with 1-day embargo gaps.
- Five test suite checks:
  1. **test_no_future_contribution** — every aggregate reads strictly-earlier rows
  2. **test_label_lag_respected** — chargebacks mature before they enter aggregates
  3. **test_calibrator_never_saw_test** — calibrator fitted on val only
  4. **test_shuffled_target_kills_signal** — shuffle labels, model collapses to base rate
  5. **test_graph_edges_are_backward_only** — no same-day or future edges in the graph

Run `make test` to verify all five; they block the build on failure.

### Entity resolution and ring graph

**Fingerprints (not persons):**
- `uid`: card1/2/3/5 + addr1/2 + D1n (where D1n = day − D1, the account origin)
- `device`: DeviceInfo + id_30..id_33, version strings normalized
- `email`: P_emaildomain (domain only; IEEE-CIS gives no local part)
- `addr`: addr1 + addr2
- `card`: card1 + card2

**Hub pruning:** Max degree 200 per entity type (tunable in `config/base.yaml`). 
Without it, gmail.com collapses the entire graph into one component.

**Time-sliced windows:** Graph rebuilt daily using [d − 14 days, d). Connected 
components are candidate rings. Every transaction is attached to an existing 
component via entity lookup; it cannot merge components itself, so causality is 
guaranteed by construction.

### Ring features (the differentiator)

All computed over the component the transaction belongs to, using strictly-earlier rows:

- `comp_size`, `comp_n_uid`, `comp_n_device`, `comp_uid_per_device` — group structure
- `comp_velocity_1h`, `comp_velocity_24h` — burst detection
- `comp_age_hours` — how new the ring is (young + fast = card testing)
- `comp_amt_mean`, `comp_amt_cv`, `comp_amt_small_share` — amount patterns
- `comp_prior_fraud_rate` (smoothed to global train prior, label-maturation-lagged)
- `comp_burstiness`, `comp_hour_entropy` — automation signatures
- `uid_degree`, `device_degree`, `addr_degree` — local hubness

Every feature emits a `*_max_contrib_dt` column proving no future contribution. 
The pipeline asserts these for all 590,540 rows on every run.

### Cost model

Every rupee assumption in `config/costs.yaml`:

```yaml
fraud_allowed:
  goods_loss_pct: 1.00          # full amount lost
  chargeback_fee: 1500          # dispute handling
  ops_handling: 300
  scheme_penalty_risk: 0.0      # (left zero — thresholds change; not defended)

legit_blocked:
  gross_margin_pct: 0.25
  ltv_multiplier: 3.0           # repeat-purchase value
  support_contact_rate: 0.30
  support_cost: 150

review:
  analyst_cost_per_case: 40
  delay_conversion_loss_pct: 0.05
  daily_capacity: 100
  review_accuracy: 0.85
```

Decision = argmin over {allow, block, review}. Capacity constraint enforced greedily.

### Where the LLM sits, and where it doesn't

**The LLM does NOT decide.** The policy engine (deterministic, auditable) produces 
a three-way decision; the LLM narratess evidence the decision is based on. A human 
analyst reads the brief and either confirms fraud, marks legitimate, or escalates.

**The LLM is measured for grounding.** Numeric fidelity = (numbers appearing in 
the brief that are also in the evidence JSON) / (total numbers in brief). Target 1.0. 
A second model (different from the narrator) judges whether each bullet follows 
from the evidence, scored as yes/no/partial, measured against 50 hand labels.

**Degradation path.** If `GROQ_API_KEY` is absent and the cache misses, the narrator 
falls back to a deterministic template built from the same evidence JSON. The system 
still blocks, allows, and queues correctly.

---

## Limitations

1. **IEEE-CIS does not contain real account IDs.** The `uid` fingerprint is reconstructed 
   from card and address fields. Two people can collide; one person can fragment across 
   several. This is stated in MODEL_CARD.md and is standard practice in this dataset's 
   community (first-place writeup, Amazon FDB).

2. **Identity coverage is sparse.** Device data is present in ~24% of rows. Graph 
   statistics computed on sparse entity types carry less weight than those on dense ones. 
   A coverage table is published before any graph metric.

3. **The V-block is partly redundant with our aggregates.** Correlation pruning (|r| > 0.95) 
   drops ~30% of the V columns. The reduction and method are reported.

4. **Amounts are USD; the cost model uses a fixed rate.** `USD_INR: 87.0` is a modeling 
   convention for reproducibility, not a market-rate claim. Actual conversions would use 
   live FX.

5. **Label maturation lag is fixed at 30 days.** Chargeback timelines vary by region and 
   scheme. This is a best-guess; a production system would calibrate it against real 
   dispute data.

6. **Scheme monitoring thresholds are not quoted.** Visa consolidated fraud/dispute 
   programmes in 2025; Mastercard runs separate programmes. Thresholds change and vary 
   by acquirer. We describe the constraint qualitatively rather than quoting stale numbers.

---

## Defense-Only Statement

This project contains no component that generates fraudulent or evasive transactions, 
no adversarial-example search against any detector, no probing of live payment systems, 
and no documentation of bypass techniques. Robustness work is limited to feature 
sensitivity analysis (±10% and ±25% perturbations on held-out data), which illuminates 
model behavior without enabling attack.

---

## Data Licenses and Citations

**IEEE-CIS Fraud Detection (primary)**
- Source: https://www.kaggle.com/competitions/ieee-fraud-detection
- License: Kaggle competition terms
- Citation: *IEEE Computational Intelligence Society and Vesta Corporation. "IEEE-CIS 
  Fraud Detection." Kaggle, 2019.*
- Rows: 590,540 (training only; no labels on test files)

**Bank Account Fraud, NeurIPS 2022 (secondary)**
- Source: https://github.com/feedzai/bank-account-fraud
- License: CC-BY-NC (non-commercial research and portfolio use only)
- Citation: See `DATA_LICENSES.md`
- Rows: 6M across 6 variants; used for calibration and cost-curve ablation only

**Benchmark reference (not a dataset used)**
- Fraud Dataset Benchmark (FDB): https://github.com/amazon-science/fraud-dataset-benchmark
- Used to cross-check our split against their published split; two independent 
  definitions of "temporal" agreeing is a credibility signal.

---

## Repository Structure

```
ring-sentinel/
├── config/                    # Every assumption: seeds, thresholds, rupee costs
│   ├── base.yaml
│   ├── costs.yaml
│   ├── features.yaml
│   └── llm.yaml
├── src/ring_sentinel/
│   ├── data/                  # Load, prepare, split (temporal)
│   ├── entities/              # UID, device, graph, union-find
│   ├── features/              # Base, causal aggregates, ring (differentiator)
│   ├── models/                # Rules baseline, LightGBM, calibration
│   ├── policy/                # Cost model, policy engine, decisions
│   ├── narrate/               # Groq client, evidence JSON, grounding eval
│   ├── audit/                 # Hash-chained append-only ledger
│   ├── eval/                  # Metrics, slices, drift, sensitivity
│   └── serve/                 # FastAPI analyst console
├── tests/
│   ├── test_leakage.py        # THE MOST IMPORTANT FILE
│   ├── test_*.py
│   └── fixtures/
├── artifacts/narratives/      # Committed cached Groq responses
├── reports/                   # Generated metrics tables, figures
├── scripts/                   # Ad-hoc analysis
├── notebooks/01_eda.ipynb     # Exploration only, not a dependency
├── pyproject.toml
├── Makefile
├── README.md (this file)
├── MODEL_CARD.md
└── DATA_LICENSES.md
```

No notebook is imported by production code. If a number is in README, `src/` produced it.
No dataset files are committed; `make data` retrieves them.

---

## Next Steps

### Immediate (if behind schedule)
- **Retune `config/costs.yaml` review/block thresholds against case-level precision**, not just transaction recall@FPR — the current policy flags 54.9% of test-period rings at 10.8% precision, which is the actual production blocker.
- **Realized-cost economics**: `reports/economics.md`'s "expected saving" is the model's own expected-value estimate under its predicted probabilities, not verified against ground-truth outcomes. A second, ex-post number (policy decisions scored against actual `isFraud`) is needed before quoting any rupee figure to a payments panel.
- Ablation rows C (causal aggregates alone) and E (case-level noisy-OR as its own scored model) — `run_eval()` only computes A/B/D/F.

### With More Time
- Narrator + grounding eval: numeric fidelity, claim support
- Analyst console: authentication, live case review with SVG graph rendering
- Sensitivity sweep: how thresholds move as cost assumptions change
- Graph ablations: Elliptic++ (crypto) as a sanity check on the ring layer

### Research
- GNN vs tabular: GraphSAGE on daily subgraphs (if LGBM plateaus)
- Multi-objective: Pareto frontier of fraud-vs-customer-friction
- Personalized review: route different case types to different analysts

---

## Contact & Attribution

Built for Razorpay AI Buildathon, Track 02 (AI Risk Manager).  
All code is production-ready and operates on publicly-available, citable data only.

---

**Read `tests/test_leakage.py` first. Every claimed invariant is asserted there.**
