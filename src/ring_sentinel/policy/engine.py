"""Cost-aware policy engine. Argmin expected rupee loss under capacity constraint."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Decision:
    case_id: str
    case_score: float
    expected_cost_allow: float
    expected_cost_block: float
    expected_cost_review: float
    decision: str  # "allow", "block", or "review"

    def best_action_unconstrained(self) -> str:
        costs = {
            "allow": self.expected_cost_allow,
            "block": self.expected_cost_block,
            "review": self.expected_cost_review,
        }
        return min(costs, key=costs.get)

    def cost(self) -> float:
        return {
            "allow": self.expected_cost_allow,
            "block": self.expected_cost_block,
            "review": self.expected_cost_review,
        }[self.decision]


def expected_costs(
    case_score: float,
    amount_inr: float,
    costs_cfg: dict,
) -> tuple[float, float, float]:
    """
    E[cost | allow]  = p * (amt * goods_loss_pct + chargeback_fee + ops_handling)
    E[cost | block]  = (1 - p) * (amt * gross_margin_pct * ltv_multiplier
                                  + support_contact_rate * support_cost)
    E[cost | review] = analyst_cost_per_case
                       + (1 - p) * amt * gross_margin_pct * delay_conversion_loss_pct
                       + p * (1 - review_accuracy) * (amt * goods_loss_pct + chargeback_fee)
    """
    p = case_score
    fa = costs_cfg["fraud_allowed"]
    lb = costs_cfg["legit_blocked"]
    rev = costs_cfg["review"]

    c_allow = p * (
        amount_inr * fa["goods_loss_pct"] +
        fa["chargeback_fee"] +
        fa["ops_handling"] +
        fa.get("scheme_penalty_risk", 0.0)
    )

    c_block = (1 - p) * (
        amount_inr * lb["gross_margin_pct"] * lb["ltv_multiplier"] +
        lb["support_contact_rate"] * lb["support_cost"]
    )

    review_accuracy = rev.get("review_accuracy", 0.85)
    c_review = (
        rev["analyst_cost_per_case"] +
        (1 - p) * amount_inr * lb["gross_margin_pct"] * rev["delay_conversion_loss_pct"] +
        p * (1 - review_accuracy) * (
            amount_inr * fa["goods_loss_pct"] + fa["chargeback_fee"]
        )
    )

    return float(c_allow), float(c_block), float(c_review)


def make_decisions(
    cases: pd.DataFrame,
    costs_cfg: dict,
    capacity_per_day: int | None = None,
) -> list[Decision]:
    """
    Apply the policy to a set of cases. cases must have:
      case_id, case_score, amount_inr, (optional) day
    """
    decisions = []
    for _idx, row in cases.iterrows():
        case_id = str(row["case_id"])
        score = float(row["case_score"])
        amt = float(row["amount_inr"])

        c_a, c_b, c_r = expected_costs(score, amt, costs_cfg)

        best = min(
            ("allow", c_a),
            ("block", c_b),
            ("review", c_r),
            key=lambda x: x[1],
        )[0]

        decisions.append(Decision(
            case_id=case_id,
            case_score=score,
            expected_cost_allow=c_a,
            expected_cost_block=c_b,
            expected_cost_review=c_r,
            decision=best,
        ))

    # Capacity constraint: enforce daily review limit.
    if capacity_per_day and "day" in cases.columns:
        by_day = {}
        for dec in decisions:
            day = int(cases.loc[cases["case_id"] == dec.case_id, "day"].iloc[0])
            by_day.setdefault(day, []).append(dec)

        for day_decs in by_day.values():
            review_decs = [d for d in day_decs if d.decision == "review"]
            if len(review_decs) > capacity_per_day:
                # Keep top-k reviews by expected cost saving, push remainder to
                # their next-best action.
                review_decs.sort(
                    key=lambda d: min(d.expected_cost_allow, d.expected_cost_block)
                )
                for dec in review_decs[capacity_per_day:]:
                    if dec.expected_cost_block < dec.expected_cost_allow:
                        dec.decision = "block"
                    else:
                        dec.decision = "allow"

    return decisions
