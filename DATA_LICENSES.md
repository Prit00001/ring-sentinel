# Data Licenses and Citations

This project uses **only public, citable datasets with published licenses.** No real 
cardholder data, no scraping, no proprietary information.

---

## 1. IEEE-CIS Fraud Detection (Primary Dataset)

| | |
|---|---|
| **Official Name** | IEEE-CIS Fraud Detection |
| **Provider** | IEEE Computational Intelligence Society & Vesta Corporation |
| **Source** | https://www.kaggle.com/competitions/ieee-fraud-detection |
| **Mirror** | https://ieee-dataport.org/documents/ieee-cis-fraud-detection |
| **License** | Kaggle competition terms (free use, academic/portfolio OK; see below) |
| **Rows** | 590,540 transactions in `train_transaction.csv` |
| **Positives** | 20,663 fraud labels (3.5% base rate) |
| **Features** | 431 total: 394 transaction + 37 identity |
| **Time Column** | `TransactionDT` (seconds from unstated epoch, ~40 days) |
| **Label** | `isFraud` binary |
| **Data Type** | Real payment transaction data, anonymized by Vesta |

**Citation:**

```bibtex
@dataset{ieee_cis_fraud_2019,
  title     = {IEEE-CIS Fraud Detection},
  author    = {{IEEE Computational Intelligence Society} and {Vesta Corporation}},
  year      = {2019},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/competitions/ieee-fraud-detection}
}
```

**Plain text:** IEEE Computational Intelligence Society and Vesta Corporation. 
"IEEE-CIS Fraud Detection." Kaggle, 2019.

**Download:** Requires accepting Kaggle competition rules. Terms state use is free 
for non-commercial and academic purposes. This project is academic/portfolio.

---

## 2. Bank Account Fraud Dataset, NeurIPS 2022 (Secondary, Ablation Only)

| | |
|---|---|
| **Official Name** | Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation |
| **Provider** | Feedzai Research |
| **GitHub** | https://github.com/feedzai/bank-account-fraud |
| **Kaggle Mirror** | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 |
| **Paper (arXiv)** | https://arxiv.org/abs/2211.13358 |
| **Paper (NeurIPS)** | https://proceedings.neurips.cc/paper_files/paper/2022/file/d9696563856bd350e4e7ac5e5812f23c-Paper-Datasets_and_Benchmarks.pdf |
| **OpenReview** | https://openreview.net/forum?id=UrAYT2QwOX8 |
| **Datasheet** | https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf |
| **License** | Creative Commons CC-BY-NC (non-commercial) |
| **Rows** | 6,000,000 instances across 6 variants (Base, I–V) |
| **Task** | Account-opening fraud, not payment fraud |
| **Data Type** | Synthetic (CTGAN trained on anonymized real data, differential privacy added) |
| **Protected Attributes** | age group, employment status, income percentile |

**Citation:**

```bibtex
@article{jesus2022,
  title   = {Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation},
  author  = {Jesus, Sérgio and Pombal, José and Alves, Duarte and Cruz, André 
             and Saleiro, Pedro and Ribeiro, Rita P. and Gama, João and Bizarro, Pedro},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022},
  volume  = {35}
}
```

**Plain text:** Jesus, Sérgio, et al. "Turning the Tables: Biased, Imbalanced, Dynamic 
Tabular Datasets for ML Evaluation." *Advances in Neural Information Processing 
Systems*, 2022.

**License Note:** BAF is published under CC-BY-NC. This project is non-commercial 
research and portfolio work, so use is permitted. If this were a commercial product, 
BAF results would need to be excluded or licensed separately.

**Use in Ring Sentinel:** BAF is used for calibration curve fitting and cost-model 
ablation only. No headline metrics are reported from BAF. The primary result set is 
on IEEE-CIS.

---

## 3. Fraud Dataset Benchmark (Reference Only, Not a Dataset)

| | |
|---|---|
| **Official Name** | Fraud Dataset Benchmark |
| **Provider** | Amazon Science |
| **GitHub** | https://github.com/amazon-science/fraud-dataset-benchmark |
| **Paper** | https://arxiv.org/abs/2208.14417 |
| **Landing Page** | https://www.amazon.science/code-and-datasets/fdb-fraud-dataset-benchmark |
| **Use** | Validation of temporal split; NOT a reported dataset |

FDB is used as a credibility cross-check: our temporal split on IEEE-CIS is compared 
against FDB's published 95/5 split. If two independently-defined temporal splits 
agree on findings, that is a signal of robustness.

**Citation:**

```bibtex
@article{fdb2022,
  title   = {Fraud Dataset Benchmark: Unified Evaluation of Fraud Detection Algorithms},
  author  = {[Authors]},
  journal = {arXiv preprint},
  year    = {2022},
  url     = {https://arxiv.org/abs/2208.14417}
}
```

---

## Summary Table

| Dataset | Role | License | Commercial Use | Used in Results |
|---|---|---|---|---|
| IEEE-CIS | Primary | Kaggle terms | Academic/portfolio OK | ✓ Yes |
| BAF | Secondary | CC-BY-NC | Non-commercial only | ⊗ Calibration only |
| FDB | Reference | Apache 2.0 | Free | ⊗ No (cross-check only) |

---

## How to Get the Data

### IEEE-CIS

```bash
# Step 1: Accept rules on Kaggle (required by their API)
# https://www.kaggle.com/competitions/ieee-fraud-detection/rules

# Step 2: Install Kaggle CLI and credentials
pip install kaggle
# Create ~/.kaggle/kaggle.json with your API token (chmod 600)

# Step 3: Download
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee
unzip -q data/raw/ieee/\*.zip -d data/raw/ieee
```

### BAF

```bash
# Via Kaggle
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 \
  -p data/raw/baf
unzip -q data/raw/baf/\*.zip -d data/raw/baf

# Or via GitHub
git clone https://github.com/feedzai/bank-account-fraud.git data/raw/baf_github
```

### Verification

After download, verify file integrity:

```bash
make data  # Automatically checksums; compare with data/CHECKSUMS.txt
```

---

## Ethical Use

- **No real PII.** IEEE-CIS data is anonymized by Vesta. No real cardholder names, 
  numbers, addresses, or email addresses appear in the dataset.
- **Non-redistributable.** Dataset files are in `.gitignore` and are never committed. 
  Users download them directly from Kaggle under their own account.
- **No generation of fraud.** This project contains no generator of synthetic fraudulent 
  traffic or evasion techniques.
- **Attribution.** Every dataset source is cited with a link so a judge can download 
  the exact same bytes.

---

## Questions?

- **IEEE-CIS:** See the competition page, linked above.
- **BAF:** See the GitHub repo and NeurIPS paper.
- **This project:** See README.md and MODEL_CARD.md.
