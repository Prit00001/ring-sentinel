"""Grounding evaluation for narrator briefs (build spec 10.4).

Two layers:
  1. Numeric fidelity — deterministic, free, no API call. Every numeric token
     in the brief must appear in the evidence JSON's flattened value set.
  2. Claim support — LLM-as-judge, one call per bullet, using a model that
     MUST differ from the narrator (config.llm.judge.model). A model is
     lenient about its own phrasing, so judging with the narrator's own ID
     gives correlated errors and an inflated score.
"""

from __future__ import annotations

import json
import re

from .evidence import flatten_numeric_values

# Numbers only — never a digit that's part of a larger identifier token like
# "c_00417", "device_a91f", or "24h" (velocity_24h). Lookarounds exclude any
# adjacent letter, digit, or underscore so those don't get misread as
# hallucinated figures.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d[\d,]*\.?\d*(?![A-Za-z0-9_])")

JUDGE_SYSTEM_PROMPT = """You are checking one bullet point from a fraud case brief \
against the evidence JSON it was written from.

Question: does this claim follow from the evidence JSON alone, with no \
outside knowledge or inference beyond what the fields directly state?

Answer with exactly one word: yes, no, or partial."""


def _extract_numbers(text: str) -> list[str]:
    out = []
    for m in _NUMBER_RE.finditer(text):
        tok = m.group().replace(",", "")
        if tok in ("", "-", "."):
            continue
        try:
            f = float(tok)
        except ValueError:
            continue
        out.append(str(int(f)) if f == int(f) else f"{f:g}")
    return out


def numeric_fidelity(brief: dict, evidence: dict) -> dict:
    """Fraction of numbers in the brief that are traceable to the evidence.

    Returns {"numeric_fidelity": float in [0,1] or 1.0 if no numbers were
    written, "unsupported": [ (section, number), ... ]}.
    """
    allowed = flatten_numeric_values(evidence)
    unsupported: list[tuple[str, str]] = []
    total = 0

    for section in ("summary", "counter", "next_step"):
        text = brief.get(section)
        if isinstance(text, list):
            text = " ".join(text)
        for num in _extract_numbers(text or ""):
            total += 1
            if num not in allowed:
                unsupported.append((section, num))

    for bullet in brief.get("evidence", []) or []:
        for num in _extract_numbers(bullet):
            total += 1
            if num not in allowed:
                unsupported.append(("evidence", num))

    fidelity = 1.0 if total == 0 else 1.0 - (len(unsupported) / total)
    return {"numeric_fidelity": fidelity, "total_numbers": total, "unsupported": unsupported}


def claim_support(brief: dict, evidence: dict, judge_call) -> dict:
    """LLM-as-judge claim support, one call per evidence bullet.

    judge_call(system, user) -> str is injected so this stays testable
    without a network dependency; narrator.py wires it to client.complete
    with config.llm.judge.model.
    """
    bullets = list(brief.get("evidence", []) or [])
    verdicts = []
    for bullet in bullets:
        user = (
            f"EVIDENCE JSON:\n{json.dumps(evidence, sort_keys=True)}\n\n"
            f"CLAIM:\n{bullet}"
        )
        raw = judge_call(JUDGE_SYSTEM_PROMPT, user).strip().lower()
        verdict = "partial"
        if raw.startswith("yes"):
            verdict = "yes"
        elif raw.startswith("no"):
            verdict = "no"
        verdicts.append({"claim": bullet, "verdict": verdict})

    unsupported = sum(1 for v in verdicts if v["verdict"] == "no")
    rate = 0.0 if not verdicts else unsupported / len(verdicts)
    return {"unsupported_claim_rate": rate, "verdicts": verdicts}


def evaluate_grounding(brief: dict, evidence: dict, judge_call=None) -> dict:
    """Full grounding report. judge_call is optional — omit it to skip the
    LLM-judge layer (e.g. no GROQ_API_KEY) and report numeric fidelity only.
    """
    result = numeric_fidelity(brief, evidence)
    if judge_call is not None:
        result.update(claim_support(brief, evidence, judge_call))
    else:
        result["unsupported_claim_rate"] = None
        result["verdicts"] = None
    result["counter_present"] = bool(brief.get("counter"))
    return result
