"""Transaction scores -> case score (build spec section 8).

Noisy-OR over member transaction probabilities: a ring of forty weak signals
can outrank a single moderate one, which is the point of working cases
instead of isolated transactions. The cap stops a large component saturating
to 1.0 on volume alone.
"""

from __future__ import annotations

import numpy as np


def case_score(probs, cap: int = 50) -> float:
    """1 - prod(1 - p_i), computed in log-space.

    Computing prod([1-x for x in p]) directly underflows to exactly 0.0 once
    a handful of members have even moderately high probability — a 20-member
    ring with most scores around 0.3-0.5 already underflows float64. That
    silently turns every well-populated ring's score into an identical,
    maximally-overconfident 1.0, which then zeroes out every downstream cost
    computed from (1 - score). log1p/expm1 keep the precision that plain
    multiplication throws away, so a "very confident, 40-member ring" and an
    "absolutely certain, 40-member ring" don't collapse to the same case
    score just because the naive formula ran out of floating-point range.
    """
    p = sorted(probs, reverse=True)[:cap]
    if not p:
        return 0.0
    if any(x >= 1.0 for x in p):
        return 1.0  # a member the model is certain about makes the case certain
    log_survival = sum(np.log1p(-x) for x in p)  # sum(log(1 - p_i)), stable near p_i -> 1
    return float(-np.expm1(log_survival))         # 1 - exp(log_survival)
