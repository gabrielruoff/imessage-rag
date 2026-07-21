"""Shared robust-statistics primitives: Theil-Sen trend + bootstrap CI,
Kaplan-Meier survival estimators, ECDF placement, robust outlier scoring
(modified z / MAD), and CUSUM change-point detection.

Deliberately dependency-free: pure stdlib (`math`, `statistics`, `random`),
deterministic, no numpy, no dataclasses, and NO import of any app module that
pulls in `config`/Ollama/sqlite. That keeps this a foundation other pipelines
can import without dragging in the whole application.

Every function is TOTAL over its stated domain: on degenerate input it returns
`None` (or, where the signature is a list, an empty list) and never raises.

Critical modeling note (load-bearing for downstream consumers)
-------------------------------------------------------------
Do NOT reuse the censored latency path from `get_response_dynamics`: it drops
gaps `>= INITIATION_GAP_HOURS`, which biases a dying tie's latency *downward*
(the longest silences -- exactly the signal that a relationship is fading --
are thrown away). This module is the intended replacement: model long/absent
replies as right-censored observations and use the survival estimators here
(`kaplan_meier_median`, `frac_over_threshold`) instead of truncating the tail.
"""

import math
import random
import statistics

# Iglewicz-Hoaglin consistency constant: 0.6745 == the 0.75 quantile of the
# standard normal, so modified z reduces to a standard z for normal data.
_MOD_Z_CONST = 0.6745


def _quantile(values, q):
    """Linear-interpolation quantile (numpy 'linear' / type 7). `values` must be
    non-empty; `q` in [0, 1] (clamped). Returns a float, or None if empty."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    q = min(1.0, max(0.0, q))
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def theil_sen(points):
    """Theil-Sen robust line fit. `points` is a list of (x, y).

    Returns (slope, intercept) -- slope is the median of all pairwise slopes
    over point pairs with distinct x; intercept is the median of
    y_i - slope * x_i. Returns None if fewer than 2 distinct x values (no
    pairwise slope is defined).
    """
    if not points or len(points) < 2:
        return None
    slopes = []
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            if xi != xj:
                slopes.append((yj - yi) / (xj - xi))
    if not slopes:
        return None
    slope = statistics.median(slopes)
    intercept = statistics.median([y - slope * x for x, y in points])
    return (float(slope), float(intercept))


def robust_slope(points, *, n_boot=2000, alpha=0.05, seed=0):
    """Theil-Sen slope with a seeded bootstrap confidence interval.

    Returns {"slope", "ci_low", "ci_high", "significant"} where the point
    estimate is the Theil-Sen slope over all `points`, the CI is the
    (alpha/2, 1 - alpha/2) percentile interval of the Theil-Sen slope across
    `n_boot` resamples (drawn with replacement using `random.Random(seed)`),
    and `significant` is True iff that CI excludes 0. Deterministic for a given
    seed. Returns None if fewer than 4 points, or if the slope is undefined
    (fewer than 2 distinct x).
    """
    if not points or len(points) < 4:
        return None
    base = theil_sen(points)
    if base is None:
        return None
    slope = base[0]

    rng = random.Random(seed)
    n = len(points)
    boot = []
    for _ in range(n_boot):
        sample = [points[rng.randrange(n)] for _ in range(n)]
        fit = theil_sen(sample)
        if fit is not None:
            boot.append(fit[0])
    if not boot:
        return None

    ci_low = _quantile(boot, alpha / 2.0)
    ci_high = _quantile(boot, 1.0 - alpha / 2.0)
    significant = ci_low > 0.0 or ci_high < 0.0
    return {
        "slope": float(slope),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(significant),
    }


def _km_steps(durations, censored):
    """Kaplan-Meier survival steps at each distinct event time.

    Returns a list of (event_time, S_after) ascending by time, where S_after is
    the survival estimate immediately after that time. Right-censored
    observations (censored[i] is True) contribute to the risk set but are not
    events. Returns None on invalid input (empty or mismatched lengths) and []
    when there are no events at all (all censored).
    """
    if not durations or censored is None or len(durations) != len(censored):
        return None
    obs = list(zip(durations, censored))
    distinct = sorted({d for d, _ in obs})
    events_at = {}
    all_at = {}
    for d, c in obs:
        all_at[d] = all_at.get(d, 0) + 1
        if not c:
            events_at[d] = events_at.get(d, 0) + 1
    if not events_at:
        return []

    at_risk = len(obs)
    surv = 1.0
    steps = []
    for t in distinct:
        d_i = events_at.get(t, 0)
        if d_i > 0 and at_risk > 0:
            surv *= 1.0 - d_i / at_risk
            steps.append((t, surv))
        at_risk -= all_at[t]  # events + censored at t leave the risk set
    return steps


def kaplan_meier_median(durations, censored):
    """Right-censored median duration via Kaplan-Meier: the smallest time at
    which survival S(t) <= 0.5. Returns None if that level is never reached
    (including the all-censored case) or on invalid input."""
    steps = _km_steps(durations, censored)
    if not steps:  # None (invalid) or [] (all censored)
        return None
    for t, surv in steps:
        if surv <= 0.5:
            return float(t)
    return None


def frac_over_threshold(durations, censored, q):
    """KM-estimated P(duration > q) == S(q), the survival estimate at q.

    Returns a float in [0, 1], or None if inestimable: invalid input, or no
    events at all (all censored -- the survival curve is flat at 1 and carries
    no real information, so we refuse to report a spurious 1.0)."""
    steps = _km_steps(durations, censored)
    if not steps:
        return None
    surv = 1.0
    for t, s in steps:
        if t <= q:
            surv = s
        else:
            break
    return float(surv)


def ecdf_percentile(value, sample):
    """Empirical CDF placement of `value`: the fraction of `sample` that is
    <= value, in [0, 1]. Returns None on an empty sample."""
    if not sample:
        return None
    count = sum(1 for x in sample if x <= value)
    return count / len(sample)


def winsorize(series, limits=(0.05, 0.95)):
    """Clamp each value into [Q(lower), Q(upper)], where `limits` are the lower
    and upper quantile positions (defaults: 5th and 95th percentiles). Returns a
    new list (empty for an empty input)."""
    if not series:
        return []
    lo = _quantile(series, limits[0])
    hi = _quantile(series, limits[1])
    if lo > hi:
        lo, hi = hi, lo
    return [min(max(float(x), lo), hi) for x in series]


def winsorized_log_gap_ecdf(value, sample, *, limits=(0.05, 0.95)):
    """ECDF placement of `value` against `sample` in winsorized log space:
    log-transform the (positive) sample, winsorize its tails, then place
    log(value) via `ecdf_percentile`. Intended for heavy-tailed reply-gap
    distributions, where the raw ECDF is dominated by a few huge silences.

    Returns a float in [0, 1], or None if the sample has no positive values."""
    positives = [v for v in sample if v > 0]
    if not positives:
        return None
    logs = [math.log(v) for v in positives]
    winsorized = winsorize(logs, limits)
    log_value = math.log(value) if value > 0 else float("-inf")
    return ecdf_percentile(log_value, winsorized)


def modified_z(series):
    """Iglewicz-Hoaglin modified z-score per point:
    0.6745 * (x_i - median) / MAD, where MAD == median(|x_i - median|).

    Returns a list of scores, or None on degenerate input: an empty series, or
    MAD == 0 (a constant-enough series). We refuse to epsilon-divide -- MAD == 0
    means "no robust spread", so no meaningful z exists."""
    if not series:
        return None
    med = statistics.median(series)
    mad = statistics.median([abs(x - med) for x in series])
    if mad == 0:
        return None
    return [_MOD_Z_CONST * (x - med) / mad for x in series]


def leave_window_out_baseline(series, window):
    """Robust baseline (median, MAD) computed over the points OUTSIDE the
    half-open index range [start, end) given by `window`. Lets a consumer score
    a candidate window against the rest of the series without that window
    contaminating its own baseline. Returns None if fewer than 4 points remain
    outside the window (too few for a stable baseline)."""
    if not series:
        return None
    start, end = window
    kept = [x for i, x in enumerate(series) if i < start or i >= end]
    if len(kept) < 4:
        return None
    med = statistics.median(kept)
    mad = statistics.median([abs(x - med) for x in kept])
    return (float(med), float(mad))


def cusum_changepoints(series, *, k, h):
    """Two-sided tabular CUSUM change-point detection on a MAD-standardized
    series. Each point is standardized as z = (x - median) / MAD, then the
    upper/lower CUSUM accumulators run with allowance `k` (the drift the
    statistic tolerates) and decision threshold `h` -- both expressed in
    MAD-standardized units. Returns the list of indices where either
    accumulator strictly exceeds `h` (the accumulators reset to 0 after each
    detection). Returns [] on degenerate input (empty series or MAD == 0)."""
    if not series:
        return []
    med = statistics.median(series)
    mad = statistics.median([abs(x - med) for x in series])
    if mad == 0:
        return []
    changepoints = []
    s_hi = 0.0
    s_lo = 0.0
    for i, x in enumerate(series):
        z = (x - med) / mad
        s_hi = max(0.0, s_hi + z - k)
        s_lo = max(0.0, s_lo - z - k)
        if s_hi > h or s_lo > h:
            changepoints.append(i)
            s_hi = 0.0
            s_lo = 0.0
    return changepoints
