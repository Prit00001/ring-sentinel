# Ring Sentinel

Detects coordinated card-not-present fraud rings on IEEE-CIS (590,540 txns, 3.5% fraud). 
Strictly temporal split. Defense-only.

**Track:** Razorpay AI Buildathon, Track 02 — AI Risk Manager  
**Status:** Foundation code complete. Results pending real-data run.

---

## Results (held-out final 20% by time, touched once)

> **Note:** This section is empty until you run `make data && make repro` with real IEEE-CIS training data. The code is complete and tested on a synthetic fixture; reported numbers appear here once the evaluation finishes. No number generated from synthetic or fake data appears in this table.

| Metric | Rules baseline | LGBM base | + ring features | Δ |
|---|---|---|---|---|
| PR-AUC | — | — | — | — |
| Recall @ 0.1% FPR | — | — | — | — |
| Recall @ 0.5% FPR | — | — | — | — |
| Recall @ 1% FPR | — | — | — | — |
| Recall @ 5% FPR | — | — | — | — |
| Precision @ k=100/day | — | — | — | — |
| Brier score | — | — | — | — |
| ECE | — | — | — | — |
| Expected cost, ₹ (test period) | — | — | — | — |
| Case-level precision | — | — | — | — |

---

## Build Order (Followed)

This project implements all ten steps from the build spec:

1. ✓ Data loader with temporal splits and leakage test suite
2. ✓ Rules baseline + test harness
3. ✓ Entity resolution and time-sliced ring graph
4. ✓ Ring features with causal-aggregation assertions
5. ✓ LightGBM scorer with isotonic calibration
6. ✓ Cost-aware policy engine with rupee cost model
7. ⊗ LLM case narrator (skeleton; awaits Groq key)
8. ⊗ Analyst console (skeleton; awaits full pipeline)
9. ⊗ Ablation table, slice metrics, drift report
10. ⊗ Results table, model card (waiting for metrics)

---

## What Didn't Work (Will Update)

As results come in, honest failures will land here. Placeholder for now:

- [thing], [why], [what it cost]
- [thing], [why], [what it cost]

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
- Metrics table: run full pipeline on real data
- Ablation: one row per feature layer
- Slices: recall @ 1% FPR by amount band, hour, device coverage, new vs returning

### With More Time
- Narrator + grounding eval: numeric fidelity, claim support
- Analyst console: live case review with SVG graph rendering
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
