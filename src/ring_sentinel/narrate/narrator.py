"""LLM case narrator (build spec 10). It does not decide anything.

Input is the fully-formed evidence JSON from evidence.py. Output is a brief
for a human analyst, written by config.llm.narrator.model on Groq. If
GROQ_API_KEY is absent and the disk cache misses, this degrades to a
deterministic template built from the same evidence JSON — the system still
blocks, allows, and queues correctly; only the prose is worse. That
degradation path is the "one failure handled gracefully" the track asks for.
"""

from __future__ import annotations

import json
import logging

from .client import NoGroqKey, complete
from .grounding import evaluate_grounding

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write case briefs for a payments fraud analyst.

You will receive a JSON evidence object. Write a brief with exactly these
sections:

SUMMARY      — two sentences on what the pattern appears to be.
EVIDENCE     — 3 to 5 bullets, each citing a specific field from the JSON.
COUNTER      — 1 to 2 bullets giving the strongest innocent explanation.
NEXT STEP    — one concrete action the analyst can take.

HARD RULES
- Every number you write must appear verbatim in the JSON. Do not compute,
  round, convert, or infer new numbers.
- Do not state or imply a final verdict. The analyst decides.
- Do not name a fraud typology unless the evidence fields directly support it.
- If a field is absent, say the evidence is unavailable. Never fill a gap.
- The COUNTER section is mandatory and must be a real alternative
  explanation, not a token hedge.

Return JSON: {"summary": str, "evidence": [str], "counter": [str],
"next_step": str}"""


def _template_brief(evidence: dict) -> dict:
    """Deterministic fallback: no LLM, no network, always available.

    Every number here is read directly from the evidence dict, so this brief
    is grounded by construction (numeric_fidelity should always be 1.0).
    """
    comp = evidence.get("component", {})
    shared = evidence.get("shared_entities", [])
    top = evidence.get("top_contributions", [])

    evidence_bullets = [
        f"Component has {comp.get('n_transactions')} prior transactions across "
        f"{comp.get('n_uid')} UIDs and {comp.get('n_device')} devices "
        f"(uid_per_device={comp.get('uid_per_device')}).",
        f"Amount coefficient of variation is {comp.get('amount_cv')} over a total of "
        f"₹{comp.get('amount_total_inr')}.",
        f"24h velocity is {comp.get('velocity_24h')} transactions; component age is "
        f"{comp.get('age_hours')} hours.",
    ]
    if shared:
        s = shared[0]
        evidence_bullets.append(
            f"Shared {s.get('type')} '{s.get('value')}' is common to {s.get('n_uid')} UIDs."
        )
    if top:
        t = top[0]
        evidence_bullets.append(
            f"Top contributing feature is {t.get('feature')}={t.get('value')} "
            f"(SHAP={t.get('shap')})."
        )

    return {
        "summary": (
            f"Case {evidence.get('case_id')} scored {evidence.get('case_score')}; "
            f"policy decision is '{evidence.get('decision')}'. This is a templated "
            "summary — no LLM call was made."
        ),
        "evidence": evidence_bullets[:5],
        "counter": [
            "High velocity and multiple UIDs on shared devices can also reflect "
            "legitimate shared-device households or business batch purchasing.",
            "Device or address sharing alone does not establish a single operator.",
        ],
        "next_step": "Review the component subgraph and top transactions before deciding.",
    }


def narrate_evidence(
    evidence: dict,
    llm_cfg: dict | None = None,
    force_template: bool = False,
) -> tuple[dict, bool]:
    """Return (brief, used_template). used_template is False only on a real
    Groq completion — surfaced so callers/tests can assert the fallback path
    actually triggers rather than silently masking a config problem.
    """
    if force_template or llm_cfg is None:
        return _template_brief(evidence), True

    narrator_cfg = llm_cfg["narrator"]
    user_msg = json.dumps(evidence, sort_keys=True)

    try:
        content, _usage = complete(
            system=SYSTEM_PROMPT,
            user=user_msg,
            model=narrator_cfg["model"],
            cache_dir=llm_cfg["cache_dir"],
            max_retries=int(llm_cfg.get("max_retries", 5)),
            retry_base_delay_sec=float(llm_cfg.get("retry_base_delay_sec", 1.0)),
            request_timeout_sec=float(llm_cfg.get("request_timeout_sec", 60.0)),
            temperature=float(narrator_cfg.get("temperature", 0.2)),
            max_completion_tokens=int(narrator_cfg.get("max_completion_tokens", 900)),
        )
        brief = json.loads(content)
        for key in ("summary", "evidence", "counter", "next_step"):
            if key not in brief:
                raise ValueError(f"narrator response missing required key '{key}'")
        return brief, False
    except NoGroqKey:
        if not llm_cfg.get("allow_template_fallback", True):
            raise
        log.info("No GROQ_API_KEY and cache miss — degrading to template narrator.")
        return _template_brief(evidence), True
    except Exception as exc:  # noqa: BLE001 - any Groq/JSON failure degrades gracefully
        log.warning("Narrator call failed (%s) — degrading to template narrator.", exc)
        return _template_brief(evidence), True


def narrate_and_evaluate(evidence: dict, llm_cfg: dict | None = None) -> dict:
    """Convenience wrapper: narrate, then grade the brief's grounding.

    The judge call is only wired up when llm_cfg is supplied AND a key is
    available; otherwise grounding falls back to numeric-fidelity-only,
    which needs no network access and is still a real measurement.
    """
    brief, used_template = narrate_evidence(evidence, llm_cfg)

    judge_call = None
    if llm_cfg is not None and not used_template:
        judge_cfg = llm_cfg["judge"]

        def judge_call(system: str, user: str) -> str:
            content, _usage = complete(
                system=system,
                user=user,
                model=judge_cfg["model"],
                cache_dir=llm_cfg["cache_dir"],
                max_retries=int(llm_cfg.get("max_retries", 5)),
                temperature=float(judge_cfg.get("temperature", 0.0)),
                max_completion_tokens=int(judge_cfg.get("max_completion_tokens", 300)),
            )
            return content

    grounding = evaluate_grounding(brief, evidence, judge_call=judge_call)
    return {"brief": brief, "used_template": used_template, "grounding": grounding}


def evaluate_grounding_simple(brief: dict, evidence: dict) -> dict:
    """Backwards-compatible numeric-fidelity-only entry point."""
    return evaluate_grounding(brief, evidence, judge_call=None)
