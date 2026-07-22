"""S14 recipes: per-person voiceprint (get_voiceprint).

Ground truth is the deterministic voiceprint (person_voiceprint, built LLM-free
by voiceprint.py). A faithful reply describes the surfaced stylometry (lowercase
ratio, burst size, openers/closers) without inventing features, and -- for a
low-volume counterpart -- carries the tool's own low-confidence signal rather
than presenting sparse stylometry as settled. n_messages/confidence are the
gradeable population signals."""

from ._base import _qt, _v, recipe


@recipe("voiceprint_for")
def voiceprint_for(roster, scope=None):
    """get_voiceprint for self (scope=None) or a scoped counterpart. raw carries
    n_messages + confidence + the headline features so a grader can check the
    reply is faithful to the profile and honors the confidence tier. When scope
    is a sub-floor counterpart the tool returns confidence='low' (or declines);
    the recipe records whichever so the case can assert honest framing."""
    qt = _qt()
    kwargs = {}
    if scope is not None:
        kwargs["contact_id"] = scope["contact_id"]
    r = qt.get_voiceprint(**kwargs)
    feats = r.get("features") or {}
    return {
        "values": {"n_messages": _v(r.get("n_messages"), "round")},
        "raw": {
            "n_messages": r.get("n_messages"),
            "n_months": r.get("n_months"),
            "confidence": r.get("confidence"),
            "scope": r.get("scope"),
            "declined": bool(r.get("declined")) or r.get("available") is False,
            "lowercase_ratio": feats.get("lowercase_ratio"),
            "burst_size": feats.get("burst_size"),
            "openers": (r.get("openers") or [])[:3],
            "closers": (r.get("closers") or [])[:3],
        },
    }
