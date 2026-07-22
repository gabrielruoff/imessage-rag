"""Inter-rater agreement + detector precision math for the eval-gate harness
(RES-70 / F4). Stdlib-only, pure functions over paired ``(gold, predicted)``
label lists -- no config, no DB, no Ollama -- so the three measurement gates
(kappa conflict labels, dropped_plan precision, wellbeing narration) share one
tested implementation. Callers pass the sample-size floors in; this module
owns the arithmetic, not the policy.
"""

from collections import Counter


def _check_pair(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError(
            f"gold/predicted length mismatch: {len(gold)} != {len(predicted)}"
        )


def cohens_kappa(gold, predicted):
    """Cohen's kappa over paired labels (any hashable label space).

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e
    is the agreement expected from the two raters' marginal label frequencies.
    Returns ``None`` when kappa is undefined: no items, or a degenerate case
    where chance agreement is already perfect (p_e == 1, e.g. both raters used
    a single label for every item) -- those carry no information and must not
    read as a passing 1.0.
    """
    _check_pair(gold, predicted)
    n = len(gold)
    if n == 0:
        return None
    agree = sum(1 for g, p in zip(gold, predicted) if g == p)
    p_o = agree / n
    gc, pc = Counter(gold), Counter(predicted)
    labels = set(gc) | set(pc)
    p_e = sum((gc.get(c, 0) / n) * (pc.get(c, 0) / n) for c in labels)
    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def kappa_per_class(gold, predicted, classes=None, min_labels=None):
    """One-vs-rest Cohen's kappa for each class in ``classes`` (defaulting to
    every label seen). Each class is scored as a binary problem (this class vs
    everything else), which is how a per-class gate reads a multi-class rater.

    Small-sample guard: a class with fewer than ``min_labels`` *gold*
    instances (its support) returns ``None`` rather than a kappa computed from
    too few examples -- the caller treats ``None`` as "not cleared", never as a
    pass. ``min_labels=None`` disables the guard.
    """
    _check_pair(gold, predicted)
    if classes is None:
        classes = sorted(set(gold) | set(predicted))
    out = {}
    for c in classes:
        support = sum(1 for g in gold if g == c)
        if min_labels is not None and support < min_labels:
            out[c] = None
            continue
        g_bin = [1 if g == c else 0 for g in gold]
        p_bin = [1 if p == c else 0 for p in predicted]
        out[c] = cohens_kappa(g_bin, p_bin)
    return out


def precision_recall(gold, predicted, positive):
    """Binary precision/recall for a detector task: ``positive`` is the label
    counted as a hit. Returns ``{precision, recall, tp, fp, fn, support}``,
    with ``precision``/``recall`` set to ``None`` when their denominator is
    zero (no predicted positives / no gold positives -- undefined, not zero).
    ``support`` is the number of gold positives, the sample-size the caller
    guards on.
    """
    _check_pair(gold, predicted)
    tp = sum(1 for g, p in zip(gold, predicted) if p == positive and g == positive)
    fp = sum(1 for g, p in zip(gold, predicted) if p == positive and g != positive)
    fn = sum(1 for g, p in zip(gold, predicted) if p != positive and g == positive)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }
