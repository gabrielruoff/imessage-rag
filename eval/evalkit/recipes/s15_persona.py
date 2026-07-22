"""S15 recipes: persona drafting -- the deterministic style-match check.

check_style_match is LLM-free (LSM/function-word distance of a candidate draft
against the owner's voiceprint), so its distance + top-3 divergences are exact
ground truth. A faithful reply reports the tool's own verdict and divergence
reasons, never inventing a stylistic critique the tool did not compute. The draft
text is passed as a param and MUST match the string embedded in the S15.1 prompt
verbatim -- the model can only be graded against the draft it was actually shown.

(draft_reply, S15.2, GENERATES via Ollama, so it has no deterministic GT and is
graded by grader_notes -- its quality_watch is that the draft is a real reply,
not a stripped single-token fragment.)"""

from ._base import _qt, _v, recipe


@recipe("style_match")
def style_match(roster, draft):
    """check_style_match(draft) against the owner ('me'). raw carries the exact
    distance and the top divergence features so a grader can confirm the reply
    conveys the tool's verdict (match / not) and its real reasons."""
    qt = _qt()
    r = qt.check_style_match(draft=draft)
    divs = r.get("top_divergences") or r.get("divergences") or []
    return {
        "values": {"distance": _v(r.get("distance"), "round")},
        "raw": {
            "distance": r.get("distance"),
            "voiceprint_confidence": r.get("voiceprint_confidence"),
            "n_features_compared": r.get("n_features_compared"),
            "top_divergence_features": [d.get("feature") for d in divs[:3]],
        },
    }
