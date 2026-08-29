"""Narrator + grounding tests.

No network access and no GROQ_API_KEY are required: these exercise the
template-fallback path (which build spec 10.0's "degradation path" makes a
first-class, always-available mode) and the deterministic numeric-fidelity
check, which is the free half of grounding evaluation.
"""

from __future__ import annotations

from ring_sentinel.narrate.evidence import (
    Contribution,
    ComponentSummary,
    PriorOutcomes,
    SharedEntity,
    build_evidence,
)
from ring_sentinel.narrate.grounding import evaluate_grounding, numeric_fidelity
from ring_sentinel.narrate.narrator import narrate_evidence


def _sample_evidence() -> dict:
    return build_evidence(
        case_id="c_00417",
        day_start=141,
        day_end=155,
        case_score=0.87,
        decision="review",
        expected_cost_inr={"allow": 41200, "block": 8900, "review": 640},
        component=ComponentSummary(
            n_transactions=34, n_uid=19, n_device=2, uid_per_device=9.5,
            amount_total_inr=47850, amount_cv=0.08, age_hours=31, velocity_24h=27,
        ),
        shared_entities=[
            SharedEntity(type="device", value="device_a91f", n_uid=17),
            SharedEntity(type="addr", value="addr1_204", n_uid=12),
        ],
        top_contributions=[
            Contribution(feature="comp_uid_per_device", value=9.5, shap=1.82),
            Contribution(feature="comp_amt_cv", value=0.08, shap=1.14),
        ],
        prior_outcomes=PriorOutcomes(n_labeled=6, n_fraud=4, label_lag_days=30),
    )


def test_narrator_degrades_to_template_with_no_llm_config():
    """No llm_cfg (e.g. no GROQ_API_KEY at all) must never raise."""
    evidence = _sample_evidence()
    brief, used_template = narrate_evidence(evidence, llm_cfg=None)
    assert used_template is True
    for key in ("summary", "evidence", "counter", "next_step"):
        assert key in brief
    assert brief["counter"], "COUNTER section is mandatory per the narrator system prompt"


def test_template_brief_is_fully_grounded():
    """Every number the template writes comes straight from the evidence, so
    numeric fidelity must be exactly 1.0 — the fallback path must never be
    the reason a repro run reports hallucinated numbers."""
    evidence = _sample_evidence()
    brief, _ = narrate_evidence(evidence, llm_cfg=None)
    result = numeric_fidelity(brief, evidence)
    assert result["numeric_fidelity"] == 1.0, result["unsupported"]


def test_numeric_fidelity_catches_a_hallucinated_number():
    """A guard that cannot fail is not a guard."""
    evidence = _sample_evidence()
    bad_brief = {
        "summary": "This case involves 9999 transactions, which is unusually high.",
        "evidence": ["Component has 34 prior transactions."],
        "counter": ["Could be a legitimate business."],
        "next_step": "Escalate.",
    }
    result = numeric_fidelity(bad_brief, evidence)
    assert result["numeric_fidelity"] < 1.0
    assert any(num == "9999" for _section, num in result["unsupported"])


def test_evaluate_grounding_without_judge_reports_none_for_claim_support():
    evidence = _sample_evidence()
    brief, _ = narrate_evidence(evidence, llm_cfg=None)
    result = evaluate_grounding(brief, evidence, judge_call=None)
    assert result["unsupported_claim_rate"] is None
    assert result["numeric_fidelity"] == 1.0
    assert result["counter_present"] is True


def test_evaluate_grounding_with_a_stub_judge():
    """judge_call is injected so this needs no network access; it exercises
    the claim-support wiring end to end with a deterministic stand-in."""
    evidence = _sample_evidence()
    brief, _ = narrate_evidence(evidence, llm_cfg=None)

    def stub_judge(_system: str, _user: str) -> str:
        return "yes"

    result = evaluate_grounding(brief, evidence, judge_call=stub_judge)
    assert result["unsupported_claim_rate"] == 0.0
    assert all(v["verdict"] == "yes" for v in result["verdicts"])
