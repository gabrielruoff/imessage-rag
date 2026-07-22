"""S11 recipes: conflict episodes (the deterministic finder's output).

Ground truth comes from find_conflicts itself -- the finder is deterministic,
so a faithful reply must convey the tool's own scoped counts, present the
signals as tentative receipts (not a verdict), and NEVER assign a
role/aggressor: the finder has no such field by design (C2). n_conflicts is
advisory numeric evidence; the routing verdict and the faithfulness watch in
grader_notes carry the case."""

from ._base import _qt, _v, recipe


@recipe("conflicts_for_contact")
def conflicts_for_contact(roster, scope):
    """find_conflicts scoped to a contact's direct (1:1) chats. n_conflicts is
    the finder's total_matching -- the population of tense stretches it reports,
    not a capped sample. raw carries the top episode's deterministic signals and
    tone corroboration so a grader can check the reply frames them honestly
    (tentative, with receipts) and invents no fault."""
    qt = _qt()
    r = qt.find_conflicts(contact_id=scope["contact_id"], limit=50)
    top = r["rows"][0] if r["rows"] else None
    return {
        "values": {"n_conflicts": _v(r["total_matching"], "exact")},
        "raw": {
            "total_matching": r["total_matching"],
            "returned": r["returned"],
            "top_first_date": (top or {}).get("first_date"),
            "top_signals": (top or {}).get("signals"),
            "top_corroboration": (top or {}).get("corroboration"),
        },
    }
