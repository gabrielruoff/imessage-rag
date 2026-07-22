"""S13 recipes: relationship drift / fade-out watchlist (get_fadeout_watchlist).

Ground truth is the drift read tool's own deterministic output (dyad_drift is
built LLM-free by drift.py). The watchlist is the corroborated, mandatory-complete
tier ranked by drift_score; a faithful reply conveys who is cooling/dormant with
the tool's channel caveat ('based only on iMessage'), never inventing a fade-out
or asserting certainty the drift score does not carry. n_watchlist is advisory.

DRIFT_ENABLED gates the three drift tools' registration, so these recipes (and
the S13 cases) only route when the flag is on -- see the S13 suite notes."""

from ._base import _qt, _v, recipe


@recipe("fadeout_watchlist")
def fadeout_watchlist(roster):
    """The corroborated fade-out watchlist get_fadeout_watchlist surfaces.
    n_watchlist is the count in the mandatory-complete tier; raw carries the
    contacts + phases so a grader can check the reply names real drifting ties
    and honors the channel caveat rather than manufacturing a cooling contact."""
    qt = _qt()
    r = qt.get_fadeout_watchlist()
    wl = r.get("watchlist") or []
    return {
        "values": {"n_watchlist": _v(len(wl), "exact")},
        "raw": {
            "n_watchlist": len(wl),
            "contacts": [w.get("contact") for w in wl],
            "phases": sorted({w.get("phase") for w in wl if w.get("phase")}),
            "order": r.get("order"),
            "n_insufficient_signal": len(r.get("insufficient_signal") or []),
        },
    }
