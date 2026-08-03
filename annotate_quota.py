"""Largest-remainder apportionment: the one home for "split N slots across
chats, weighted, capped by what each chat can actually hold".

Two callers, two questions, one loop:

  * annotate_sampling._allocate       -- how many SEEDS to draw per chat,
                                         weighted by each chat's PENDING count
  * annotate_distill._corpus_quotas   -- how many seed-INDEX slots per chat,
                                         weighted by each chat's share of the
                                         CORPUS (deliberately not of the pool:
                                         a pool-proportional draw reproduces a
                                         lopsided pool exactly)

They used to be two copies of the same twenty lines, and the copies had already
drifted where it mattered most -- see `apportion`'s note on the degenerate tail.

Pure and deterministic: no I/O, no RNG, no config. Remainders are compared as
exact integers, never floats, so a tie-break can never drift with rounding.
"""


def apportion(room, weights, need, tie_key):
    """Split `need` across the keys of `room`, weighted, capped at room[key].

    `room`    {key: capacity}. Keys with capacity <= 0 take part in nothing but
              still appear in the result (with 0), so a caller's key set comes
              back intact.
    `weights` {key: weight} for a STATIC weighting (each weight floored at 1, so
              a key the weight source has never heard of cannot become
              unreachable), or None to weight by each key's REMAINING capacity.
    `need`    slots to hand out. The result sums to min(need, total capacity).
    `tie_key` sort key over the keys, deciding who takes a rounding remainder
              first. Must impose a TOTAL order, or the result is not
              reproducible.

    Largest-remainder rounding, iterated: capacity-starved keys hand their
    leftover back and the next pass redistributes it over the keys that still
    have room, so the quotas sum to exactly min(need, capacity) rather than
    landing short.

    The degenerate tail is load-bearing and is why this function exists. A
    pathological weight/capacity combination can round every share to zero, at
    which point the loop makes no progress. The two copies of this code
    disagreed about what to do then: one handed the remainder out in a fixed
    order (allocating everything, as the contract above promises); the other
    called the state unreachable and simply broke, silently returning a SHORT
    allocation. The fixed-order hand-out is what shipped here, because a caller
    that asked for `need` slots and has the capacity for them should get them,
    and because "unreachable" is a claim about one caller's weighting, not about
    this loop.
    """
    quota = dict.fromkeys(room, 0)
    room = {k: n for k, n in room.items() if n > 0}
    remaining = min(need, sum(room.values()))
    while remaining > 0 and room:
        w = {k: max(1, room[k] if weights is None else weights.get(k, 0))
             for k in room}
        total = sum(w.values())
        parts = []
        for key, cap in room.items():
            num = remaining * w[key]
            parts.append([key, min(num // total, cap), num % total])
        # max(0, ...) only so a negative slice can never silently hand the
        # remainder to the *smallest* fractions; the sum cannot exceed remaining.
        extra = max(0, remaining - sum(p[1] for p in parts))
        # Hand the rounding remainder to the biggest fractional parts first.
        for part in sorted(parts, key=lambda p: (-p[2], tie_key(p[0])))[:extra]:
            if part[1] < room[part[0]]:
                part[1] += 1
        progressed = False
        for key, add, _rem in parts:
            if not add:
                continue
            quota[key] += add
            remaining -= add
            room[key] -= add
            progressed = True
        room = {k: n for k, n in room.items() if n > 0}
        if not progressed:
            # See the docstring: hand the rest out in a fixed order rather than
            # spin -- or, as one of the two originals did, return short.
            for key in sorted(room, key=tie_key):
                if remaining <= 0:
                    break
                take = min(remaining, room[key])
                quota[key] += take
                remaining -= take
            break
    return quota
