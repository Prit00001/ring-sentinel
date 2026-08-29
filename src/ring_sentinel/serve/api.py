"""FastAPI analyst console for case review (build spec section 11).

Server-rendered with Jinja2 — no React, no build step, `make serve` and it
works. The review queue is server-rendered, decisions are recorded through
audit.ledger's hash-chained log (never as a bare JSON echo), and the
decision value is a closed Enum rather than an unconstrained string.

CASE_STORE loads real cases from artifacts/case_queue.json when that file
exists (written by `make eval` via pipeline.build_case_queue — real
transactions from the held-out test split, rolled up to case level via
noisy-OR). If it doesn't exist yet (no `make repro` has run), the console
falls back to two illustrative demo cases, clearly banner-labelled as such so
nobody mistakes them for a result.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..audit.ledger import LedgerEntry, append as ledger_append
from ..config import load_config
from ..narrate.evidence import (
    ComponentSummary,
    Contribution,
    PriorOutcomes,
    SharedEntity,
    build_evidence,
)
from ..narrate.narrator import narrate_and_evaluate

log = logging.getLogger(__name__)

app = FastAPI(
    title="Ring Sentinel",
    description="Fraud analyst console (defense-only)",
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class Decision(str, Enum):
    block = "block"
    allow = "allow"
    escalate = "escalate"


def _demo_case_store() -> dict:
    """A handful of illustrative cases so `make serve` has something to show
    without requiring a trained model or downloaded IEEE-CIS data. Marked
    demo_mode=True everywhere it's rendered — these numbers are never used in
    any reported metric."""
    return {
        "c_00417": {
            "case_id": "c_00417", "score": 0.87, "decision": "review",
            "expected_cost": 41200, "expected_saving": 41200,
            "n_uid": 19, "n_device": 2, "n_transactions": 34,
            "amount_total_inr": 47850, "amount_cv": 0.08, "age_hours": 31,
            "velocity_24h": 27, "day_start": 141, "day_end": 155,
            "shared_entity": {"type": "device", "value": "device_a91f", "n_uid": 17},
            "top_feature": {"feature": "comp_uid_per_device", "value": 9.5, "shap": 1.82},
            "n_labeled": 6, "n_fraud": 4,
            "analyst_decision": None, "decided_at": None,
        },
        "c_00512": {
            "case_id": "c_00512", "score": 0.34, "decision": "allow",
            "expected_cost": 3200, "expected_saving": 400,
            "n_uid": 2, "n_device": 1, "n_transactions": 4,
            "amount_total_inr": 5600, "amount_cv": 0.41, "age_hours": 96,
            "velocity_24h": 1, "day_start": 160, "day_end": 161,
            "shared_entity": {"type": "addr", "value": "addr1_204", "n_uid": 2},
            "top_feature": {"feature": "comp_amt_cv", "value": 0.41, "shap": 0.22},
            "n_labeled": 1, "n_fraud": 0,
            "analyst_decision": None, "decided_at": None,
        },
    }


def _case_queue_path() -> Path:
    try:
        return load_config().root / "artifacts" / "case_queue.json"
    except Exception:
        return Path("artifacts/case_queue.json")


def _load_case_store() -> tuple[dict, bool]:
    """Returns (cases_by_id, is_demo_data)."""
    path = _case_queue_path()
    if path.exists():
        try:
            cases = json.loads(path.read_text())
            if cases:
                return {c["case_id"]: c for c in cases}, False
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Failed to load %s (%s) — falling back to demo cases.", path, exc)
    return _demo_case_store(), True


CASE_STORE, USING_DEMO_DATA = _load_case_store()


def _ledger_path() -> Path:
    try:
        return load_config().path("ledger")
    except Exception:
        # Config not resolvable in a bare test/CLI context — fall back to a
        # path relative to the working directory rather than crashing.
        return Path("artifacts/ledger.jsonl")


def _case_evidence(case: dict) -> dict:
    return build_evidence(
        case_id=case["case_id"],
        day_start=case["day_start"],
        day_end=case["day_end"],
        case_score=case["score"],
        decision=case["decision"],
        expected_cost_inr={case["decision"]: case["expected_cost"]},
        component=ComponentSummary(
            n_transactions=case["n_transactions"], n_uid=case["n_uid"],
            n_device=case["n_device"],
            uid_per_device=(case["n_uid"] / case["n_device"]) if case["n_device"] else 0.0,
            amount_total_inr=case["amount_total_inr"], amount_cv=case["amount_cv"],
            age_hours=case["age_hours"], velocity_24h=case["velocity_24h"],
        ),
        shared_entities=[SharedEntity(**case["shared_entity"])],
        top_contributions=[Contribution(**case["top_feature"])],
        prior_outcomes=PriorOutcomes(
            n_labeled=case["n_labeled"], n_fraud=case["n_fraud"], label_lag_days=30
        ),
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    cases = sorted(CASE_STORE.values(), key=lambda c: c.get("expected_saving", c["expected_cost"]), reverse=True)
    return templates.TemplateResponse(
        request, "index.html", {"cases": cases, "demo_mode": USING_DEMO_DATA}
    )


@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, case_id: str):
    case = CASE_STORE.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")

    evidence = _case_evidence(case)
    llm_cfg = load_config().llm
    result = narrate_and_evaluate(evidence, llm_cfg=llm_cfg)

    return templates.TemplateResponse(
        request, "case_detail.html", {
            "case": case,
            "brief": result["brief"],
            "used_template": result["used_template"],
            "grounding": result["grounding"],
            "evidence_json": json.dumps(evidence, indent=2, sort_keys=True),
        },
    )


@app.post("/cases/{case_id}/decide")
def decide(case_id: str, decision: Decision = Form(...)):
    case = CASE_STORE.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")

    ts = datetime.now(timezone.utc).isoformat()
    entry = LedgerEntry(
        ts=ts,
        case_id=case_id,
        actor="analyst",
        action="decide",
        decision=decision.value,
        expected_cost=float(case["expected_cost"]),
        prev_hash="",  # filled in by ledger.append from the on-disk chain tail
        score=float(case["score"]),
        policy_version="demo",
    )
    ledger_append(_ledger_path(), entry)

    case["analyst_decision"] = decision.value
    case["decided_at"] = ts

    return RedirectResponse(url=f"/cases/{case_id}", status_code=303)


@app.get("/api/cases")
def list_cases():
    """JSON view of the queue, for smoke tests and tooling — the HTML console
    at `/` is the console the spec asks for."""
    cases = sorted(CASE_STORE.values(), key=lambda c: c.get("expected_saving", c["expected_cost"]), reverse=True)
    return JSONResponse({"cases": cases, "total": len(cases)})
