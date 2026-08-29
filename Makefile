.PHONY: help data repro test smoke serve verify-ledger clean check-models features train eval

help:
	@echo "Ring Sentinel — Razorpay AI Buildathon Track 2"
	@echo ""
	@echo "Targets:"
	@echo "  make data          Download IEEE-CIS + BAF datasets (requires Kaggle credentials)"
	@echo "  make smoke         Run pipeline on fixture; reports/ generated"
	@echo "  make test          pytest; includes leakage tests"
	@echo "  make repro         Reproduce all reported metrics from README (requires data + keys)"
	@echo "  make serve         Start FastAPI analyst console on 127.0.0.1:8000"
	@echo "                     (HOST=0.0.0.0 make serve to expose beyond loopback)"
	@echo "  make verify-ledger Walk the audit ledger's hash chain and report pass/fail"
	@echo "  make check-models  Verify Groq model IDs against live /models endpoint"
	@echo "  make clean         Delete data/, artifacts/, reports/"
	@echo ""

PYTHON := python3
SHELL := /bin/bash
VENV := .venv
SRC := src/ring_sentinel
# Loopback by default: the analyst console has no authentication yet, so
# binding wider than 127.0.0.1 hands write access (recording fraud
# decisions) to anyone who can reach the port. Override explicitly if you
# actually want that, e.g. `HOST=0.0.0.0 make serve`.
HOST ?= 127.0.0.1
PORT ?= 8000

# ============================================================================
# DATA
# ============================================================================

data:
	@echo "Downloading IEEE-CIS + BAF datasets..."
	@echo "This requires Kaggle credentials. First:"
	@echo "  1. Accept rules at https://www.kaggle.com/competitions/ieee-fraud-detection/rules"
	@echo "  2. Place ~/.kaggle/kaggle.json (chmod 600)"
	@echo ""
	mkdir -p data/raw/ieee data/raw/baf
	@if [ -f data/raw/ieee/train_transaction.csv ] && [ -f data/raw/ieee/train_identity.csv ]; then \
		echo "✓ data/raw/ieee already has train_transaction.csv + train_identity.csv — skipping download"; \
	else \
		kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee && \
		unzip -q data/raw/ieee/\*.zip -d data/raw/ieee; \
	fi
	@if ls data/raw/baf/*.csv >/dev/null 2>&1; then \
		echo "✓ data/raw/baf already has CSVs — skipping download"; \
	else \
		kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw/baf && \
		unzip -q data/raw/baf/\*.zip -d data/raw/baf; \
	fi
	$(PYTHON) -c "from src.ring_sentinel.data.load import write_checksums; \
		from src.ring_sentinel.config import repo_root; \
		write_checksums(repo_root()/'data'/'raw'/'ieee', repo_root()/'data'/'CHECKSUMS.txt')"
	@echo "✓ Datasets downloaded and checksummed"

# ============================================================================
# TEST & SMOKE
# ============================================================================

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:
	@echo "Smoke test: pipeline on fixture (no external data)..."
	mkdir -p reports/_smoke data/interim data/processed
	$(PYTHON) -c "from tests.fixtures.make_fixture import make_fixture; \
		from src.ring_sentinel.data.load import prepare; \
		from src.ring_sentinel.data.split import temporal_split; \
		from src.ring_sentinel.entities.resolve import add_entities; \
		from src.ring_sentinel.features.base import build_base_features; \
		from src.ring_sentinel.features.ring import build_ring_features; \
		df = prepare(make_fixture()); \
		df = add_entities(df); \
		s = temporal_split(df); \
		feat, v_keep, email_maps = build_base_features(df, s.train); \
		ring_feat, stats = build_ring_features(df, 0.035); \
		print(f'✓ Smoke OK: {len(df)} rows, {feat.shape[1]} base features, {ring_feat.shape[1]} ring features')"

# ============================================================================
# FEATURES / TRAIN / EVAL (real IEEE-CIS data — run `make data` first)
# ============================================================================
#
# Each stage pickles its output to artifacts/ so `make train` and `make eval`
# don't have to re-run the (slow) feature build. Pickles are gitignored
# (artifacts/*.pickle) — they are intermediate build state, not a
# reproducibility artifact; `make repro` regenerates them from committed
# data every time.

features:
	@echo "Building base + ring features on the full IEEE-CIS train file..."
	@test -f data/raw/ieee/train_transaction.csv || \
		(echo "Missing data/raw/ieee/train_transaction.csv — run 'make data' first." && exit 1)
	mkdir -p artifacts
	$(PYTHON) -c "\
import warnings, pickle; warnings.filterwarnings('ignore'); \
from src.ring_sentinel.data.load import load_raw; \
from src.ring_sentinel import pipeline; \
df = load_raw(); \
print(f'Loaded {len(df)} rows'); \
bundle = pipeline.run_features(df); \
print(f'base={bundle.base_feat.shape} ring={bundle.ring_feat.shape} ' \
      f'train/val/test={len(bundle.splits.train)}/{len(bundle.splits.val)}/{len(bundle.splits.test)}'); \
pickle.dump(bundle, open('artifacts/feature_bundle.pickle', 'wb'))"
	@echo "✓ Features built → artifacts/feature_bundle.pickle"

train: features
	@echo "Training rules baseline + LightGBM (base, +ring) + isotonic calibration..."
	$(PYTHON) -c "\
import warnings, pickle; warnings.filterwarnings('ignore'); \
from src.ring_sentinel import pipeline; \
bundle = pickle.load(open('artifacts/feature_bundle.pickle', 'rb')); \
result = pipeline.run_train(bundle); \
print(f'scorer_full trees: {result.scorer_full.bst.num_trees()}'); \
pickle.dump(result, open('artifacts/train_result.pickle', 'wb'))"
	@echo "✓ Trained → artifacts/train_result.pickle"

eval: train
	@echo "Evaluating on the held-out test split, writing reports/*.md..."
	mkdir -p reports figures
	$(PYTHON) -c "\
import warnings, pickle; warnings.filterwarnings('ignore'); \
from src.ring_sentinel import pipeline; \
bundle = pickle.load(open('artifacts/feature_bundle.pickle', 'rb')); \
result = pickle.load(open('artifacts/train_result.pickle', 'rb')); \
out = pipeline.run_eval(bundle, result, out_dir='reports'); \
print('Wrote reports/results.md, reports/slices.md, reports/drift.md'); \
print(out['note']); \
cases = pipeline.build_case_queue(bundle, result, top_n=20, out_path='artifacts/case_queue.json'); \
print(f'Wrote {len(cases)} real cases → artifacts/case_queue.json (make serve picks these up automatically)')"
	@echo "✓ Eval complete → reports/results.md"

# ============================================================================
# REPRODUCE
# ============================================================================

repro: test data features train eval
	@echo ""
	@echo "✓ make repro complete: reports/results.md, slices.md, drift.md regenerated"
	@echo "  from data/raw/ieee (see data/CHECKSUMS.txt) and config/*.yaml."
	@echo "Verify ledger chain: make verify-ledger"

# ============================================================================
# SERVE
# ============================================================================

serve:
	@echo "Starting analyst console on http://$(HOST):$(PORT)"
	$(PYTHON) -m uvicorn src.ring_sentinel.serve.api:app --reload --host $(HOST) --port $(PORT)

verify-ledger:
	$(PYTHON) -c "\
from pathlib import Path; \
from src.ring_sentinel.audit.ledger import verify; \
from src.ring_sentinel.config import load_config; \
p = load_config().path('ledger'); \
ok = verify(p); \
print(f'Ledger {p}: {\"VALID\" if ok else \"BROKEN — hash chain does not verify\"}'); \
raise SystemExit(0 if ok else 1)"

check-models:
	$(PYTHON) -c "import requests; \
		r = requests.get('https://api.groq.com/openai/v1/models', \
			headers={'Authorization': f'Bearer $${GROQ_API_KEY:?Set GROQ_API_KEY}'}); \
		from datetime import datetime; \
		print(f'Groq models live {datetime.now().isoformat()}:\n'); \
		for m in r.json()['data']: \
			print(f'  {m[\"id\"]}'); \
		print('\nUpdate config/llm.yaml narrator/judge model IDs')"

# ============================================================================
# CLEANUP
# ============================================================================

clean:
	rm -rf data/interim data/processed data/CHECKSUMS.txt
	rm -rf artifacts/narratives reports
	rm -f artifacts/feature_bundle.pickle artifacts/train_result.pickle
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"

.DEFAULT_GOAL := help
