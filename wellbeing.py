"""Deterministic wellbeing trajectory / change-point core (RES-49 / W1).

Given per-marker monthly time series for a SINGLE conversational identity
(a `member_key` -- self, a contact, whoever the caller scopes to), decide
whether that identity's recent language + behavior shows a *sustained,
cross-family* departure from its own history. This is the deterministic,
hermetically testable heart of the wellbeing signal: no LLM, no Ollama, no
sqlite, no `config`. Every tunable is a keyword argument with the pinned
default; W2 wires the corresponding `WELLBEING_*` config vars through and does
all of the SQL. This module never talks to a database and never decides how a
result reaches a human -- W3 owns delivery.

Design philosophy: false positives are the enemy. A person joking, venting, or
having one loud month must NOT trip the wire. The gates are therefore
deliberately conservative and stack multiplicatively:

  1. Per-marker robustness -- values are winsorized before scoring, so a single
     wild month cannot dominate (F2 `winsorize`; note its `limits` are QUANTILE
     POSITIONS, not per-tail fractions).
  2. Family capping -- each family's composite is the mean of its members'
     directional modified-z scores, with every member clamped to +/-`family_cap`
     so no lone marker can carry a family. `family_cap` (2.0) is intentionally
     BELOW `z_threshold` (3.5): a family only crosses threshold once its capped
     composite stands out sharply against the person's OWN quiet baseline
     (leave-window-out), which in practice needs several corroborating markers.
     A single-marker family (see `self_focus`) therefore essentially cannot
     reach threshold on its own -- corroboration, by construction. (Cross-issue
     note for W2/W3: this is why `self_focus` rarely "departs" independently.)
  3. Change-point + run gating -- a candidate window must contain a CUSUM
     change-point AND stay elevated (composite >= 50% of `z_threshold`) for at
     least `min_run` consecutive months.
  4. Leave-window-out baseline -- the final peak modified-z is scored against a
     baseline that EXCLUDES the candidate window, so an anomaly can never
     inflate its own denominator. A constant kept region yields MAD==0; per F2
     that is NOT None, so we drop the marker / decline the window rather than
     epsilon-divide.
  5. Baseline sufficiency -- at least `min_baseline_months` (12) qualifying
     baseline months are required.
  6. Cross-family concordance -- firing requires >=1 LEXICAL family AND the
     BEHAVIORAL family to depart in overlapping windows. Lexical spikes alone
     (sarcasm, venting) do not fire; behavioral change alone does not fire.

Markers, families, directions (PINNED). "Direction" is the DEPARTURE direction
-- the concerning way a marker moves. Every marker's modified-z is oriented so
that a departure is positive, so a benign-direction excursion produces a
negative composite and never fires.

  Lexical family `self_focus`:
    - i_count rate            UP    (first-person-singular focus rising)
  Lexical family `absolutist_negation`:
    - absolutist_count rate   UP
    - fw_negation_rate        UP
  Behavioral family `behavioral`:
    - late_night_share        UP    (more messaging in the small hours)
    - initiation_share        DOWN  (outbound session-initiation falling)
    - laugh_rate              DOWN  (laugh_count / n_words falling)
    - emoji_rate              DOWN

`question_count` and `we_count` are deliberately NOT markers: question-rate has
an ambiguous direction (curiosity vs. anxious reassurance-seeking read the
same), and we-talk is dyad-confounded (it tracks the specific relationship far
more than the individual).

Marker values arrive ALREADY FORMED and ALREADY FILTERED. W2 sums each raw
count across the scoped lexicon rows for a month and divides by that month's
summed `n_words` to form the rate, and drops any marker with fewer than
`WELLBEING_MIN_MARKER_HITS` (30) lifetime raw hits BEFORE calling in -- the raw
counts never reach this module, so the hit floor cannot live here. The pinned
signature of `analyze_trajectory` therefore takes NO `min_marker_hits`
parameter; the constant below documents the value W2 enforces. Marker values
with `None` counts (a not-yet-rebuilt DB reads NULL for the F1 columns) must be
coalesced/skipped by W2 as well.

Scope-agnostic: the caller supplies whichever identity's series it wants. This
module hard-codes no self filter and no direction changes.

Dependency-free: imports only `stats` (itself pure stdlib) and `re`. A clean
`import wellbeing` pulls in no `config`, Ollama, or sqlite (asserted in tests).
"""

import re

from stats import (
    cusum_changepoints,
    leave_window_out_baseline,
    modified_z,
    winsorize,
)

__all__ = [
    "analyze_trajectory",
    "support_block",
    "sanitize_condition_names",
    "MARKERS",
    "FAMILY_TYPE",
    "WELLBEING_MIN_MARKER_HITS",
]

# Iglewicz-Hoaglin consistency constant (0.75-quantile of the standard normal),
# so a leave-window-out modified-z reduces to a standard z for normal data.
# Mirrors stats._MOD_Z_CONST; kept local rather than importing a private name.
_MOD_Z_CONST = 0.6745

# Documented ONLY: the lifetime raw-hit floor W2 applies upstream (a marker with
# fewer real hits is too sparse to trust). Not consumed here -- the raw counts
# never reach this module. Named so W2 has a single canonical default to wire.
WELLBEING_MIN_MARKER_HITS = 30

# marker name -> (family, departure-direction). "up" means a rising value is the
# concerning departure; "down" means a falling value is. W2 forms the series
# under exactly these keys.
MARKERS = {
    "i_count": ("self_focus", "up"),
    "absolutist_count": ("absolutist_negation", "up"),
    "fw_negation_rate": ("absolutist_negation", "up"),
    "late_night_share": ("behavioral", "up"),
    "initiation_share": ("behavioral", "down"),
    "laugh_rate": ("behavioral", "down"),
    "emoji_rate": ("behavioral", "down"),
}

# family -> family type. Concordance requires a departure in a "lexical" family
# AND the "behavioral" family, in overlapping windows.
FAMILY_TYPE = {
    "self_focus": "lexical",
    "absolutist_negation": "lexical",
    "behavioral": "behavioral",
}

# The fixed, direction-independent support text (see support_block). Assumes no
# specific condition and names none -- it is safe to show for ANY sustained
# departure regardless of which markers moved.
_SUPPORT_BLOCK = (
    "You don't have to carry a hard stretch on your own. If the last while has "
    "felt heavier than usual, it can help to say so out loud to someone you "
    "trust -- a friend, a family member, or your doctor.\n"
    "\n"
    "If you'd like to talk with someone right now, free and confidential "
    "support is available any time:\n"
    "  - US: call or text 988 (Suicide & Crisis Lifeline)\n"
    "  - US: text HOME to 741741 (Crisis Text Line)\n"
    "  - Outside the US: https://findahelpline.com lists free lines by country\n"
    "\n"
    "Reaching out is a sign of strength. Whatever you are facing, you deserve "
    "support."
)

# The neutral sentence that replaces any sentence naming a clinical condition.
_CONDITION_FALLBACK = "a sustained shift in your language patterns"

# Denylist of clinical condition names (PINNED initial set; extending is
# expected). Lower-cased; matched case-insensitively at word boundaries so
# "OCD"/"ADHD"/"PTSD" and capitalized forms are caught. The sanitizer redacts
# the whole SENTENCE containing any hit -- never coin a diagnosis for the user.
_CONDITION_DENYLIST = frozenset({
    "depression", "depressed", "anxiety", "anxious", "bipolar", "ocd", "adhd",
    "ptsd", "trauma", "disorder", "diagnosis", "manic", "mania", "clinical",
    "symptomatic", "episode",
})

_DENY_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, _CONDITION_DENYLIST))) + r")\b",
    re.IGNORECASE,
)

# Split text into alternating [content, delimiter, content, delimiter, ...].
# A delimiter is a run of sentence punctuation with trailing whitespace, or a
# run of newlines -- so a hit anywhere in a sentence takes the whole sentence.
_SENT_SPLIT = re.compile(r"([.!?]+\s*|\n+)")


def support_block():
    """Return the fixed, code-emitted, direction-independent support text.

    A constant: identical every call, names no condition, and makes no
    assumption about which way any marker moved. W3 decides if/how it reaches
    the user; this module only owns the words.
    """
    return _SUPPORT_BLOCK


def sanitize_condition_names(text):
    """Redact clinical-condition names from `text` at SENTENCE granularity.

    Any sentence containing a denylisted word (case-insensitive, whole-word) is
    replaced wholesale with the neutral templated fallback -- not just the word,
    because the surrounding clause usually carries the same claim ("...looks
    like depression" -> the whole sentence goes). Returns
    `(clean_text, was_redacted)`; `was_redacted` is True iff at least one
    sentence was replaced. `None`/empty input passes through unchanged.
    """
    if not text:
        return (text, False)
    parts = _SENT_SPLIT.split(text)
    redacted = False
    for i in range(0, len(parts), 2):  # even indices are sentence content
        if _DENY_RE.search(parts[i]):
            parts[i] = _CONDITION_FALLBACK
            redacted = True
    return ("".join(parts), redacted)


def _orient(direction):
    """+1 if a rising value is the departure, -1 if a falling value is."""
    return 1.0 if direction == "up" else -1.0


def _clamp(x, cap):
    return max(-cap, min(cap, x))


def _member_z(values, direction, winsor_limits):
    """Oriented per-point modified-z for one marker's ordered value list.

    Winsorize (tame single-month spikes), take the modified-z, then orient so a
    departure is positive. Returns a list aligned to `values`, or None if the
    marker is degenerate -- empty, or MAD==0 (a constant-enough series): per F2
    that marker is dropped, never epsilon-divided.
    """
    z = modified_z(winsorize(values, winsor_limits))
    if z is None:
        return None
    orient = _orient(direction)
    return [orient * zi for zi in z]


def analyze_trajectory(marker_series, *, z_threshold=3.5, min_run=3,
                       min_baseline_months=12, family_cap=2.0,
                       winsor_limits=(0.05, 0.95), cusum_k=0.5, cusum_h=4.0):
    """Detect a sustained, cross-family wellbeing departure for one identity.

    Parameters
    ----------
    marker_series : dict[str, list[tuple[str, float]]]
        marker name (a key of `MARKERS`) -> list of (month 'YYYY-MM', value).
        Values are already-formed rates/shares, already hit-floor filtered by
        W2 (see module docstring). Unknown keys and empty lists are ignored.
    z_threshold, min_run, min_baseline_months, family_cap, winsor_limits,
    cusum_k, cusum_h :
        The pinned gates. `family_cap` clamps each member z BEFORE the family
        mean (so the composite is bounded to +/-`family_cap`); the window's
        peak modified-z is then scored against a leave-window-out baseline.

    Returns
    -------
    dict with keys:
        "flagged": bool -- a concordant (lexical + behavioral) departure fired.
        "window": [start_month, end_month] | None -- span of the firing episode.
        "peak_z": float | None -- strongest contributing family's peak
            leave-window-out modified-z.
        "families": {family: {"departed": bool, "direction": str,
            "markers": {marker: {"direction","peak_z","n_months"}}}}.
        "baseline_months": int -- baseline months backing the decision (the
            binding contributor's, when flagged; else the months of history
            available).
        "dropped_markers": [marker, ...] -- known markers dropped as degenerate
            (empty or MAD==0). Sorted.

    Total over its domain: degenerate / empty input returns an un-flagged result
    and never raises.
    """
    # --- 1. Keep only known, non-empty markers; build the shared month axis. ---
    known = {name: vals for name, vals in (marker_series or {}).items()
             if name in MARKERS and vals}
    months = sorted({m for vals in known.values() for m, _ in vals})
    month_index = {m: i for i, m in enumerate(months)}
    n = len(months)

    # --- 2. Per-marker oriented member-z, mapped onto the axis. ---
    dropped_markers = []
    marker_z = {}      # name -> {axis_index: oriented_z}
    marker_meta = {}   # name -> {"direction","peak_z","n_months"}
    for name, vals in known.items():
        _, direction = MARKERS[name]
        by_month = {m: float(v) for m, v in vals}      # de-dup; W2 gives uniques
        ordered = sorted(by_month)
        z = _member_z([by_month[m] for m in ordered], direction, winsor_limits)
        if z is None:                                  # empty or MAD==0 -> drop
            dropped_markers.append(name)
            continue
        zmap = {month_index[m]: zi for m, zi in zip(ordered, z)}
        marker_z[name] = zmap
        marker_meta[name] = {
            "direction": direction,
            "peak_z": max(zmap.values()) if zmap else None,
            "n_months": len(zmap),
        }

    # --- 3. Capped family composites over the axis (None where no member). ---
    fam_members = {fam: [name for name in marker_z if MARKERS[name][0] == fam]
                   for fam in FAMILY_TYPE}
    fam_composite = {}
    for fam, members in fam_members.items():
        series = []
        for i in range(n):
            vals = [_clamp(marker_z[name][i], family_cap)
                    for name in members if i in marker_z[name]]
            series.append(sum(vals) / len(vals) if vals else None)
        fam_composite[fam] = series

    # --- 4. Per-family departing windows (elevated runs w/ a change-point,
    #        scored leave-window-out). ---
    fam_departures = {fam: _family_departures(
        fam_composite[fam], z_threshold=z_threshold, min_run=min_run,
        min_baseline_months=min_baseline_months, cusum_k=cusum_k, cusum_h=cusum_h)
        for fam in FAMILY_TYPE}

    # --- 5. Cross-family concordance: a lexical AND the behavioral family
    #        departing in overlapping windows. Pick the strongest such pair. ---
    lexical = [(f, d) for f, deps in fam_departures.items()
               if FAMILY_TYPE[f] == "lexical" for d in deps]
    behavioral = [(f, d) for f, deps in fam_departures.items()
                  if FAMILY_TYPE[f] == "behavioral" for d in deps]
    best = None  # (peak_z, union_start_idx, union_end_idx, baseline_months)
    for _lf, ld in lexical:
        for _bf, bd in behavioral:
            if max(ld["start"], bd["start"]) > min(ld["end"], bd["end"]):
                continue  # windows do not overlap in time -> not concordant
            us, ue = min(ld["start"], bd["start"]), max(ld["end"], bd["end"])
            if ue - us + 1 < min_run:
                continue
            pk = max(ld["peak_z"], bd["peak_z"])
            base = min(ld["baseline_months"], bd["baseline_months"])
            if best is None or pk > best[0]:
                best = (pk, us, ue, base)

    # --- 6. Family report blocks. ---
    families = {}
    for fam in FAMILY_TYPE:
        departed = bool(fam_departures[fam])
        defined = [v for v in fam_composite[fam] if v is not None]
        if departed:
            direction = "concerning"
        elif defined and min(defined) <= -0.75 * family_cap and -min(defined) > max(defined):
            direction = "benign"     # strongest excursion is the non-concerning way
        elif defined:
            direction = "stable"
        else:
            direction = "absent"
        families[fam] = {
            "departed": departed,
            "direction": direction,
            "markers": {name: dict(marker_meta[name]) for name in fam_members[fam]},
        }

    flagged = best is not None
    return {
        "flagged": flagged,
        "window": [months[best[1]], months[best[2]]] if flagged else None,
        "peak_z": best[0] if flagged else None,
        "families": families,
        "baseline_months": best[3] if flagged else n,
        "dropped_markers": sorted(dropped_markers),
    }


def _family_departures(series, *, z_threshold, min_run, min_baseline_months,
                       cusum_k, cusum_h):
    """Departing windows for one family's capped composite `series`.

    `series` is over the full month axis, with None where the family had no
    member that month. We compact to defined months, find maximal contiguous
    runs where the composite stays >= 50% of `z_threshold`, keep only runs that
    are >= `min_run` long AND contain a CUSUM change-point, then score each
    surviving run's peak against a leave-window-out baseline (which excludes the
    run itself, so the anomaly cannot inflate its own denominator). A run
    "departs" when peak modified-z >= `z_threshold` with >= `min_baseline_months`
    qualifying baseline months and a non-zero baseline MAD.

    Returns a list of {"start","end","peak_z","baseline_months"} with start/end
    as INCLUSIVE axis indices, ordered by discovery.
    """
    axis_idx = [i for i, v in enumerate(series) if v is not None]
    comp = [series[i] for i in axis_idx]
    if len(comp) < min_baseline_months + min_run:
        return []

    changepoints = set(cusum_changepoints(comp, k=cusum_k, h=cusum_h))
    bar = 0.5 * z_threshold
    departures = []
    j, length = 0, len(comp)
    while j < length:
        if comp[j] < bar:
            j += 1
            continue
        start = j
        while j < length and comp[j] >= bar:
            j += 1
        end = j  # compact half-open [start, end)
        if end - start < min_run:
            continue
        if not any(start <= c < end for c in changepoints):
            continue
        base = leave_window_out_baseline(comp, (start, end))
        if base is None:
            continue
        med, mad = base
        if mad == 0:                       # constant kept region -> never divide
            continue
        baseline_months = len(comp) - (end - start)
        if baseline_months < min_baseline_months:
            continue
        peak_z = max(_MOD_Z_CONST * (comp[k] - med) / mad for k in range(start, end))
        if peak_z >= z_threshold:
            departures.append({
                "start": axis_idx[start],
                "end": axis_idx[end - 1],
                "peak_z": peak_z,
                "baseline_months": baseline_months,
            })
    return departures
