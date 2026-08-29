"""Build the evidence JSON handed to the narrator (build spec 10.2).

This is the ONLY input the narrator ever sees. Every number in the brief it
writes must trace back to a field here — that is what grounding.py checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentSummary:
    n_transactions: int
    n_uid: int
    n_device: int
    uid_per_device: float
    amount_total_inr: float
    amount_cv: float
    age_hours: float
    velocity_24h: float


@dataclass
class SharedEntity:
    type: str
    value: str
    n_uid: int


@dataclass
class Contribution:
    feature: str
    value: float
    shap: float


@dataclass
class PriorOutcomes:
    n_labeled: int
    n_fraud: int
    label_lag_days: int


def build_evidence(
    case_id: str,
    day_start: int,
    day_end: int,
    case_score: float,
    decision: str,
    expected_cost_inr: dict,
    component: ComponentSummary,
    shared_entities: list[SharedEntity],
    top_contributions: list[Contribution],
    prior_outcomes: PriorOutcomes,
) -> dict:
    """Assemble the evidence dict exactly matching the spec 10.2 schema."""
    return {
        "case_id": case_id,
        "window": {"day_start": day_start, "day_end": day_end},
        "case_score": round(float(case_score), 4),
        "decision": decision,
        "expected_cost_inr": {k: round(float(v), 2) for k, v in expected_cost_inr.items()},
        "component": {
            "n_transactions": component.n_transactions,
            "n_uid": component.n_uid,
            "n_device": component.n_device,
            "uid_per_device": round(component.uid_per_device, 4),
            "amount_total_inr": round(component.amount_total_inr, 2),
            "amount_cv": round(component.amount_cv, 4),
            "age_hours": round(component.age_hours, 2),
            "velocity_24h": component.velocity_24h,
        },
        "shared_entities": [
            {"type": e.type, "value": e.value, "n_uid": e.n_uid} for e in shared_entities
        ],
        "top_contributions": [
            {"feature": c.feature, "value": c.value, "shap": round(c.shap, 4)}
            for c in top_contributions
        ],
        "prior_outcomes": {
            "n_labeled": prior_outcomes.n_labeled,
            "n_fraud": prior_outcomes.n_fraud,
            "label_lag_days": prior_outcomes.label_lag_days,
        },
    }


def flatten_numeric_values(evidence: dict) -> set[str]:
    """Every numeric value in the evidence JSON, as normalised strings.

    Used by grounding.py's numeric-fidelity check: any number the narrator
    writes that is not in this set was not actually in the evidence.
    """
    out: set[str] = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.add(_normalise_number(v))
        elif isinstance(v, dict):
            for vv in v.values():
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)

    walk(evidence)
    return out


def _normalise_number(v) -> str:
    """1500, 1500.0, and 1500.00 must all compare equal to a narrated "1500"."""
    f = float(v)
    if f == int(f):
        return str(int(f))
    return f"{f:g}"
