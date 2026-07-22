"""S12 recipes: the Loose Ends / nudge digest (loose_threads read tools).

get_loose_threads returns a short, urgency-ranked digest capped at
NUDGE_MAX_RESULTS and carries NO population total -- so ground truth here is
what the model actually sees (n_shown + the surfaced kinds/contacts). A faithful
reply speaks only to the surfaced nudges; it must not claim a grand total it was
never given, nor invent a thread. n_shown is advisory numeric evidence."""

from ._base import _qt, _v, recipe


@recipe("loose_threads_digest")
def loose_threads_digest(roster, kind=None, status="open"):
    """The digest get_loose_threads surfaces for the given status/kind. n_shown
    is the count of real nudge rows returned (the trailing dormant-tie notice,
    if any, is not a nudge and is dropped). raw carries the kinds and resolved
    contacts so a grader can check the reply represents them without fabricating
    threads or a total the tool never reported."""
    qt = _qt()
    rows = [r for r in qt.get_loose_threads(status=status, kind=kind)
            if "notice" not in r and not r.get("digest_meta")]
    return {
        "values": {"n_shown": _v(len(rows), "exact")},
        "raw": {
            "n_shown": len(rows),
            "kinds": sorted({r["kind"] for r in rows}),
            "contacts": [r.get("contact") for r in rows],
        },
    }
