"""Causal (strictly-backward) aggregation, with the proof attached.

The IEEE-CIS dataset does not contain future information by default. People
introduce it by aggregating over an entity without a time bound. That is the
single most common way a strong-looking fraud model turns out to be worthless,
and the winning team called it out explicitly.

Two guarantees are enforced here, mechanically:

  1. TIME BOUND. Every aggregate for row r is computed from rows whose
     TransactionDT is STRICTLY LESS than r's. Rows sharing an identical
     timestamp do not contribute to each other. Each aggregator emits a
     `*_max_contrib_dt` column recording the maximum timestamp that actually
     fed the aggregate, so the assertion is exact over every row rather than a
     sampled spot-check.

  2. LABEL MATURATION. A label is not knowable the instant a transaction
     happens. Chargebacks arrive with a delay. A prior row's label may enter an
     aggregate for row r only if
         prior.TransactionDT + LABEL_LAG_DAYS * 86400 <= r.TransactionDT
     Ten lines, and it is the difference between a number you can defend and a
     number you cannot.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import load_config

NO_PRIOR = -1.0


def smoothed_rate(n_fraud: float, n_labeled: float, prior: float, strength: float = 20.0) -> float:
    """Empirical-Bayes shrink toward the global training prior.

    With n_labeled = 0 this returns the prior exactly, which is the honest
    answer for an entity we have never seen resolve.
    """
    return (n_fraud + strength * prior) / (n_labeled + strength)


def shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def fano_factor(intervals: list[float]) -> float:
    """Fano factor of inter-arrival times: variance / mean.

    ~1 for a Poisson (human-ish) arrival process, near 0 for a metronome
    (automation), high for bursty. Reported as -1 when undefined.
    """
    if len(intervals) < 2:
        return NO_PRIOR
    a = np.asarray(intervals, dtype=float)
    m = a.mean()
    if m <= 0:
        return NO_PRIOR
    return float(a.var() / m)


@dataclass
class Accumulator:
    """Running state for one entity or one component. Prior rows only."""

    dts: list[int] = field(default_factory=list)          # ascending
    amts: list[float] = field(default_factory=list)
    n: int = 0
    amt_sum: float = 0.0
    amt_sq: float = 0.0
    small_count: int = 0
    first_dt: int | None = None
    last_dt: int | None = None
    hour_counts: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.int64))
    label_dts: list[int] = field(default_factory=list)     # ascending
    label_cum_fraud: list[int] = field(default_factory=list)  # prefix sums aligned to label_dts
    uids: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    addrs: set = field(default_factory=set)
    cards: set = field(default_factory=set)
    entity_first_seen: dict = field(default_factory=dict)  # (etype, value) -> dt

    # ---- reads (must never mutate) ----

    def count_before(self, t: int, window_sec: int) -> int:
        """Prior rows with dt in [t - window_sec, t)."""
        lo = bisect.bisect_left(self.dts, t - window_sec)
        hi = bisect.bisect_left(self.dts, t)
        return hi - lo

    def matured_labels(self, t: int, lag_sec: int) -> tuple[int, int]:
        """(n_labeled, n_fraud) among prior rows whose labels have matured by t."""
        cutoff = t - lag_sec
        k = bisect.bisect_right(self.label_dts, cutoff)
        if k == 0:
            return 0, 0
        return k, self.label_cum_fraud[k - 1]

    def max_contrib_dt(self) -> int:
        return self.last_dt if self.last_dt is not None else -1

    def amt_mean(self) -> float:
        return self.amt_sum / self.n if self.n else NO_PRIOR

    def amt_cv(self) -> float:
        """Coefficient of variation. Card testing uses near-identical amounts,
        which drives this close to zero — that is the signal."""
        if self.n < 2:
            return NO_PRIOR
        mean = self.amt_sum / self.n
        if mean <= 0:
            return NO_PRIOR
        var = max(self.amt_sq / self.n - mean * mean, 0.0)
        return float(math.sqrt(var) / mean)

    def burstiness(self) -> float:
        if len(self.dts) < 3:
            return NO_PRIOR
        d = np.diff(np.asarray(self.dts, dtype=float))
        return fano_factor(d.tolist())

    def hour_entropy(self) -> float:
        if self.n < 2:
            return NO_PRIOR
        return shannon_entropy(self.hour_counts)

    def new_entity_ratio(self, t: int, window_sec: int = 86400) -> float:
        if not self.entity_first_seen:
            return NO_PRIOR
        recent = sum(1 for dt in self.entity_first_seen.values() if dt >= t - window_sec)
        return recent / len(self.entity_first_seen)

    # ---- write (called only AFTER the row's own features are emitted) ----

    def add(
        self,
        dt: int,
        amt: float,
        hour: int,
        label: int | None,
        small_threshold: float,
        uid=None,
        device=None,
        addr=None,
        card=None,
    ) -> None:
        self.dts.append(dt)
        self.amts.append(amt)
        self.n += 1
        self.amt_sum += amt
        self.amt_sq += amt * amt
        if amt < small_threshold:
            self.small_count += 1
        if self.first_dt is None:
            self.first_dt = dt
        self.last_dt = dt
        self.hour_counts[hour % 24] += 1

        if label is not None:
            self.label_dts.append(dt)
            prev = self.label_cum_fraud[-1] if self.label_cum_fraud else 0
            self.label_cum_fraud.append(prev + int(label))

        for etype, val in (("uid", uid), ("device", device), ("addr", addr), ("card", card)):
            if val is None or val is pd.NA:
                continue
            if etype == "uid":
                self.uids.add(val)
            elif etype == "device":
                self.devices.add(val)
            elif etype == "addr":
                self.addrs.add(val)
            else:
                self.cards.add(val)
            self.entity_first_seen.setdefault((etype, val), dt)


def iter_time_groups(dt: np.ndarray):
    """Yield (start, end) slices of rows sharing an identical TransactionDT.

    Emitting a whole tie-group before updating any of its members is what makes
    the time bound STRICT rather than <=. Without this, two transactions at the
    same second see each other.
    """
    n = len(dt)
    i = 0
    while i < n:
        j = i + 1
        while j < n and dt[j] == dt[i]:
            j += 1
        yield i, j
        i = j


def expanding_entity_features(
    df: pd.DataFrame,
    entity_col: str,
    prefix: str,
    train_prior: float,
    cfg=None,
    with_fraud_rate: bool = True,
) -> pd.DataFrame:
    """Strictly-prior expanding aggregates keyed on one entity column.

    This is ablation row C — entity-level aggregates with no graph at all.
    """
    cfg = cfg or load_config()
    fc = cfg.base["features"]
    lag_sec = int(fc["label_lag_days"]) * 86400
    strength = float(fc["smoothing_strength"])
    small_threshold = float(cfg.base["graph"]["small_amount_usd"])
    windows = list(fc["velocity_windows_sec"])

    dt = df["TransactionDT"].to_numpy(dtype=np.int64)
    amt = df["TransactionAmt"].to_numpy(dtype=float)
    hour = df["hour"].to_numpy(dtype=np.int64)
    keys = df[entity_col].to_numpy(dtype=object)
    has_label = "isFraud" in df.columns
    labels = df["isFraud"].to_numpy(dtype=np.int64) if has_label else None

    n = len(df)
    out = {
        f"{prefix}_prior_count": np.zeros(n, dtype=np.float32),
        f"{prefix}_prior_amt_mean": np.full(n, NO_PRIOR, dtype=np.float32),
        f"{prefix}_prior_amt_cv": np.full(n, NO_PRIOR, dtype=np.float32),
        f"{prefix}_age_hours": np.full(n, NO_PRIOR, dtype=np.float32),
        f"{prefix}_max_contrib_dt": np.full(n, -1, dtype=np.int64),
    }
    for w in windows:
        label = "1h" if w == 3600 else ("24h" if w == 86400 else f"{w}s")
        out[f"{prefix}_velocity_{label}"] = np.zeros(n, dtype=np.float32)
    if with_fraud_rate and has_label:
        out[f"{prefix}_prior_fraud_rate"] = np.full(n, train_prior, dtype=np.float32)
        out[f"{prefix}_prior_n_labeled"] = np.zeros(n, dtype=np.float32)
        out[f"{prefix}_label_max_contrib_dt"] = np.full(n, -1, dtype=np.int64)

    acc: dict[object, Accumulator] = {}

    for lo, hi in iter_time_groups(dt):
        t = int(dt[lo])
        # ---- emit for every row in the tie-group, from state that contains
        # ---- only rows with strictly smaller dt
        for i in range(lo, hi):
            k = keys[i]
            if k is None or k is pd.NA:
                continue
            a = acc.get(k)
            if a is None or a.n == 0:
                continue
            out[f"{prefix}_prior_count"][i] = a.n
            out[f"{prefix}_prior_amt_mean"][i] = a.amt_mean()
            out[f"{prefix}_prior_amt_cv"][i] = a.amt_cv()
            out[f"{prefix}_age_hours"][i] = (t - a.first_dt) / 3600.0
            out[f"{prefix}_max_contrib_dt"][i] = a.max_contrib_dt()
            for w in windows:
                lbl = "1h" if w == 3600 else ("24h" if w == 86400 else f"{w}s")
                out[f"{prefix}_velocity_{lbl}"][i] = a.count_before(t, w)
            if with_fraud_rate and has_label:
                n_lab, n_fr = a.matured_labels(t, lag_sec)
                out[f"{prefix}_prior_fraud_rate"][i] = smoothed_rate(
                    n_fr, n_lab, train_prior, strength
                )
                out[f"{prefix}_prior_n_labeled"][i] = n_lab
                if n_lab:
                    kk = bisect.bisect_right(a.label_dts, t - lag_sec)
                    out[f"{prefix}_label_max_contrib_dt"][i] = a.label_dts[kk - 1]

        # ---- now update state with the tie-group
        for i in range(lo, hi):
            k = keys[i]
            if k is None or k is pd.NA:
                continue
            acc.setdefault(k, Accumulator()).add(
                dt=int(dt[i]),
                amt=float(amt[i]),
                hour=int(hour[i]),
                label=int(labels[i]) if has_label else None,
                small_threshold=small_threshold,
            )

    return pd.DataFrame(out, index=df.index)


# --------------------------------------------------------------------------
# Assertions. These are imported by tests/ AND called inline by the pipeline,
# so a leak fails the build, not just the test run.
# --------------------------------------------------------------------------


def assert_strictly_prior(frame: pd.DataFrame, dt: np.ndarray, columns=None) -> None:
    """Every *_max_contrib_dt must be strictly less than the row's own dt."""
    cols = columns or [c for c in frame.columns if c.endswith("_max_contrib_dt")]
    if not cols:
        raise AssertionError(
            "No *_max_contrib_dt columns found — the causality proof is missing. "
            "Every aggregator must emit one."
        )
    for c in cols:
        v = frame[c].to_numpy()
        bad = (v >= 0) & (v >= dt)
        if bad.any():
            i = int(np.argmax(bad))
            raise AssertionError(
                f"CAUSALITY VIOLATION in {c}: row {i} has contributing timestamp "
                f"{v[i]} but its own TransactionDT is {dt[i]}. "
                "An aggregate saw the present or the future."
            )


def assert_label_lag_respected(frame: pd.DataFrame, dt: np.ndarray, lag_days: int) -> None:
    """No label may enter an aggregate before it would have matured."""
    lag_sec = lag_days * 86400
    cols = [c for c in frame.columns if c.endswith("_label_max_contrib_dt")]
    if not cols:
        return
    for c in cols:
        v = frame[c].to_numpy()
        bad = (v >= 0) & (v + lag_sec > dt)
        if bad.any():
            i = int(np.argmax(bad))
            raise AssertionError(
                f"LABEL LAG VIOLATION in {c}: row {i} used a label from t={v[i]}, "
                f"which matures at {v[i] + lag_sec}, but the row's own time is {dt[i]}. "
                f"That chargeback had not arrived yet."
            )


def drop_proof_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip the *_max_contrib_dt proof columns before training.

    They are diagnostics, not features. A raw timestamp handed to a GBDT is a
    trivially leaky feature in its own right.
    """
    drop = [c for c in frame.columns if c.endswith("_max_contrib_dt")]
    return frame.drop(columns=drop)
